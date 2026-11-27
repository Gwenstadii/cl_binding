"""Package compact K18 records without copying any model or optimizer checkpoint."""

import hashlib
import json
from pathlib import Path
import tarfile


RUN = Path(__file__).resolve().parent
HISTORICAL = RUN.parent / "r1_then_front_v1/train"


def main():
    evidence = {
        "train/replay_train_steps.jsonl": RUN / "train/train_steps.jsonl",
        "train/historical_train_steps.jsonl": HISTORICAL / "train_steps.jsonl",
    }
    for pattern in ("probe_*.json", "demo_probe_*.json", "native_[0-9]*.jsonl", "attribution_*.jsonl"):
        for path in RUN.glob(pattern):
            evidence[f"records/{path.name}"] = path
    for name in ("front_phases.json", "phase_audit.png", "summary.json", "replay_audit.json", "native_audit.json"):
        evidence[name] = RUN / name
    for name in ("config.json", "measurement.py", "demo_probe.py", "attribute.py", "label_front.py",
                 "replay.py", "audit_replay.py", "audit_native.py", "summarize.py", "pack_evidence.py"):
        evidence[f"code/{name}"] = RUN / name
    for name in ("replay.log", "replay_audit.log", "attribute0.log", "attribute_early.log",
                 "attribute_joint0.log", "probe3.log", "probe_channels.log", "demo_fixed.log",
                 "phase2.log", *(f"native_{i}.log" for i in range(4))):
        evidence[f"logs/{name}"] = RUN / name
    manifest = RUN / "evidence_sha256.json"
    manifest.write_text(json.dumps({k: hashlib.sha256(v.read_bytes()).hexdigest()
                                    for k, v in sorted(evidence.items())}, indent=2))
    evidence["evidence_sha256.json"] = manifest
    with tarfile.open(RUN / "k18_evidence.tar.gz", "w:gz") as archive:
        for name, path in sorted(evidence.items()):
            archive.add(path, arcname=name, recursive=False)
    print(f"Packed {len(evidence)} files")


if __name__ == "__main__":
    main()
