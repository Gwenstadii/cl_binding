"""Preserve and verify complete rows from an interrupted run; never select by outcome."""

from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import struct

import anchor_protocol as spec


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def row_text(row):
    return json.dumps(row,allow_nan=False,separators=(",",":"))+"\n"


def key_text(key):
    return json.dumps(list(key),separators=(",",":"))


def decoded_rows(path):
    with gzip.open(path,"rt",encoding="utf-8") as stream:
        while True:
            try:
                text = stream.readline()
            except EOFError:
                return
            if not text:
                return
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                if not text.endswith("\n"):
                    return
                raise
            assert row_text(row) == text
            yield row,sha256(text.encode()).hexdigest()


def prepare_recovery(original,current):
    old = json.loads((original/"protocol.json").read_text())
    new = json.loads((current/"protocol.json").read_text())
    for k in ("lines","ids","goals","anchors","gains","windows","arms","jobs","all_jobs","expected_new","expected_total",
              "checkpoints","cases_sha256","timing_protocol_sha256","timing_source_sha256","timing_spec_sha256","base_source_sha256","baseline_source_sha256"):
        assert old[k] == new[k],k
    assert digest(original/"anchor_check.py") == old["source_sha256"]
    cases = {r["init"]:r for r in json.loads((current/"cases.json").read_text())}
    files,total = {},0
    for line in spec.LINES:
        for lane in range(8):
            path = original/f"{line}_{lane}.jsonl.gz"
            if not path.exists():
                continue
            expected = spec.expected_keys(list(spec.IDS)[lane::8])
            hashes = {}
            for row,h in decoded_rows(path):
                key = spec.key(row)
                assert row["line"] == line and key in expected and key_text(key) not in hashes
                result = spec.temporal.score(row["trace"],new["geometry"],cases[row["init"]]["z0"])
                assert all(row[k] == v for k,v in result.items())
                for block,expected_hash in enumerate(row["action_hashes"]):
                    flat = [x for a in row["trace"]["actions"][block*10:block*10+10] for x in a]
                    assert sha256(struct.pack("<70f",*flat)).hexdigest() == expected_hash
                hashes[key_text(key)] = h
            files[path.name] = dict(sha256=digest(path),rows=len(hashes),row_sha256=hashes)
            total += len(hashes)
    assert total == 804
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),original_run=str(original),
        original_protocol_sha256=digest(original/"protocol.json"),original_source_sha256=old["source_sha256"],
        recovery_source_sha256=digest(Path(__file__)),verified_complete_original_rows=total,
        missing_grid_episodes=spec.EXPECTED_NEW-total,files=files,
        change="BF16 storage hashing fix and idempotent row reuse only; checkpoints, gains, masks, sampling and scoring unchanged",
        replay_rule="one complete cached delta episode per previously active lane is replayed bitwise; technical repeats excluded from grid n")
    dest = current/"recovery_manifest.json"
    assert not dest.exists()
    dest.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


def load_cache(current,line,lane):
    path = current/"recovery_manifest.json"
    if not path.exists():
        return {},{},None
    manifest = json.loads(path.read_text())
    name = f"{line}_{lane}.jsonl.gz"
    if name not in manifest["files"]:
        return {},{},None
    source = Path(manifest["original_run"])/name
    expected = manifest["files"][name]
    assert digest(source) == expected["sha256"]
    cache,hashes = {},{}
    for row,h in decoded_rows(source):
        key = spec.key(row)
        assert h == expected["row_sha256"][key_text(key)]
        cache[key] = row
        hashes[key_text(key)] = h
    assert len(cache) == expected["rows"]
    return cache,hashes,expected["sha256"]
