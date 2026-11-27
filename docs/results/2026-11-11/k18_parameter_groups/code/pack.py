"""Archive compact K18-C measurements and scripts with a SHA-256 manifest."""

import hashlib
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parent


def main():
    files = {n: ROOT/n for n in ("summary.json", "geometry.json", "protocol.json", "architecture_config.json",
                                  "reference_1000.jsonl", "secondary_protocol.json")}
    for name in ("parameter_groups.py","attention_split.py","rotation_bridge.py","summarize.py","pack.py"):
        files[f"code/{name}"] = ROOT/name
    for pattern in ("probe_*.json","audit_*.json","selection_*.json","native_*_w*.jsonl",
                    "native_*_complete.json","attention_split_*.json","rotation_bridge_*.jsonl"):
        for path in ROOT.glob(pattern):
            files[path.name] = path
    for name in ("probe_mixed_100.log","probe_mixed_1000.log","probe_100.log","probe_fixed_100.log",
                 "rotation_bridge.log","attention_split.log",*(f"native_mixed_{w}.log" for w in range(8)),
                 *(f"native_later_{w}.log" for w in range(4))):
        files[f"logs/{name}"] = ROOT/name
    manifest = ROOT/"evidence_sha256.json"
    manifest.write_text(json.dumps({k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in sorted(files.items())},indent=2))
    files[manifest.name] = manifest
    with tarfile.open(ROOT/"k18c_evidence.tar.gz", "w:gz") as archive:
        for key, path in sorted(files.items()):
            archive.add(path, arcname=key, recursive=False)
    print("PACKED", len(files), flush=True)


if __name__ == "__main__":
    main()
