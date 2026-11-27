"""Audit paired endpoints and the pretarget boundary of native K18 rollouts."""

import hashlib
import json
from pathlib import Path


RUN = Path(__file__).resolve().parent
K14 = RUN.parent / "k14_snapshot_batch1_20261108_01"
SNAPSHOTS = K14 / "snapshots/snapshots.json"
EXPECTED_SNAPSHOT_SHA256 = "0ee8b429c5098aeb30ea2404968c1db068b894fe57407a47c5c92ac78329db5f"


def rows(files):
    for path in sorted(files):
        yield from (json.loads(line) for line in path.read_text().splitlines())


def main():
    digest = hashlib.sha256(SNAPSHOTS.read_bytes()).hexdigest()
    assert digest == EXPECTED_SNAPSHOT_SHA256
    snap = {r["init"]: r for r in json.loads(SNAPSHOTS.read_text())["records"] if 20 <= r["init"] < 40}
    assert len(snap) == 20
    assert all(r["checks"][str(r["boundary"])]["pretarget"] and all(r["rebuild_x3"]) for r in snap.values())
    old_reference = {
        (r["group"], r["goal"], r["init"]): r for r in rows((K14 / "episodes").glob("w[0-9].jsonl"))
        if r["group"] in ("R_full", "A_nat") and r["goal"] in "AB"
    }
    native = list(rows(RUN.glob("native_[0-9]*_w*.jsonl")))
    expected = {(s, g, i) for s in (0, 1000, 3000, 6000, 12000) for g in "ABN" for i in snap}
    assert len(native) == len(expected) and {(r["step"], r["goal"], r["init"]) for r in native} == expected
    checks = {0: {"compared": 0, "mismatches": []}, 12000: {"compared": 0, "mismatches": []}}
    for r in native:
        if r["goal"] == "N":
            continue
        assert r["first_step"] is None or r["first_step"] > snap[r["init"]]["boundary"]
        if r["step"] not in checks:
            continue
        base = "R_full" if r["step"] == 0 else "A_nat"
        ref = old_reference[(base, r["goal"], r["init"])]
        checks[r["step"]]["compared"] += 1
        mismatched = [k for k in ("first", "first_step", "ever", "terminal") if r[k] != ref[k]]
        if mismatched:
            checks[r["step"]]["mismatches"].append({"init": r["init"], "goal": r["goal"], "fields": mismatched})
    equality = {}
    for core_path in sorted(RUN.glob("probe_[0-9]*.json")):
        channel_path = RUN / core_path.name.replace("probe_", "probe_channels_", 1)
        a, b = (json.loads(p.read_text()) for p in (core_path, channel_path))
        for record in b:
            for horizon in ("10", "50"):
                record[horizon].pop("contrast_by_channel")
        equality[core_path.stem] = a == b
    print(json.dumps({"shared_snapshot_sha256": digest,
                      "old_native_first_events_after_shared_prefix": True,
                      "endpoint_comparison_to_K14": checks, "original_and_channel_probe_equal": equality},
                     indent=2))


if __name__ == "__main__":
    main()
