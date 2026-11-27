"""Archive the verified K20 evidence without checkpoint tensors."""
from hashlib import sha256
import json
from pathlib import Path
import tarfile

RUN = Path(__file__).resolve().parent
D = RUN.parent / "k18_training_controls_20261111_01"
K19 = RUN.parent / "k19_retention_20261111_01"
PREFIX = RUN.parent / "stage2m_cross_attention_20261028_02" / "code"
SEEDS = (20261021, 20261121)
STEPS = (30, 100, 300, 1000)


def hash_file(file):
    with file.open("rb") as handle:
        return sha256(handle.read()).hexdigest()


def pack():
    files = {}
    for name in ("config", "protocol", "protocol_initial", "calibration", "summary"):
        files[f"{name}.json"] = RUN / f"{name}.json"
    files["baseline.jsonl"] = RUN / "baseline.jsonl"
    files["code/prior_summary.py"] = D / "summarize.py"
    files["code/native_prefix_reference.py"] = PREFIX / "stage2m_cross_attention.py"
    for name in ("calibrate", "train_route", "train_weak", "eval_k20",
                 "transport_k20", "summary_k20", "pack_k20"):
        files[f"code/{name}.py"] = RUN / f"{name}.py"
    for lane in range(4):
        for name in ("calibration", "action_affine"):
            files[f"{name}_{lane}.json"] = RUN / f"{name}_{lane}.json"
    for suite, seed in zip(("dev", "confirm"), SEEDS, strict=True):
        for lane in range(4):
            stem = f"{suite}_{seed}_w{lane}"
            for suffix in (".jsonl", "_COMPLETE.json"):
                files[stem + suffix] = RUN / (stem + suffix)
        for lane in range(2):
            name = f"transport_{suite}_{lane}.jsonl"
            files[name] = RUN / name
        for arm in ("both", "route"):
            stem = f"runs/{seed}_{arm}"
            for name in ("COMPLETE.json", "train_steps.jsonl",
                         *(f"{step:06d}/READY.json" for step in STEPS)):
                files[f"{stem}/{name}"] = RUN / stem / name
        for arm, folder in (("full", D), ("replay_full", K19)):
            name = f"references/{seed}_{arm}_READY.json"
            files[name] = folder / "runs" / f"{seed}_{arm}" / "001000" / "READY.json"
        name = f"train_ref_{seed}.jsonl"
        files[name] = RUN / name
        for arm in ("weak", "route"):
            for step in STEPS:
                name = f"probe_{seed}_{arm}_{step:06d}.json"
                files[name] = RUN / name
    for file in sorted(RUN.glob("*.log")):
        files[f"logs/{file.name}"] = file
    manifest = {"files": {name: hash_file(file) for name, file in sorted(files.items())},
                "route_bank_sha256": hash_file(RUN / "route_bank.pt"),
                "route_bank_path": str(RUN / "route_bank.pt")}
    (RUN / "evidence_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files["evidence_sha256.json"] = RUN / "evidence_sha256.json"
    with tarfile.open(RUN / "k20_evidence.tar.gz", "w:gz", compresslevel=4) as archive:
        for name, file in sorted(files.items()):
            archive.add(file, arcname=name, recursive=False)


if __name__ == "__main__":
    pack()

