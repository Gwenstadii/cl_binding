"""Archive small bridge artifacts and attest referenced K20 inputs."""
from hashlib import sha256
import json
from pathlib import Path
import tarfile

RUN = Path(__file__).resolve().parent
PRIOR = RUN.parent / "k20_route_margin_20261112_01"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def pack():
    files = {f"code/{name}.py": RUN / f"{name}.py"
             for name in ("train_bridge", "eval_bridge", "probe_bridge", "summary_bridge", "verify_roles", "pack_bridge")}
    for name in ("protocol.json", "summary.json", "probe_dev.json", "probe_confirm.json", "role_verification.json"):
        files[name] = RUN / name
    prior = {"baseline.jsonl", "summary.json", "calibration.json", "action_affine_0.json"}
    for suite, seed in (("dev", 20261021), ("confirm", 20261121)):
        for mode in ("patch", "native"):
            for lane in range(2):
                for suffix in (".jsonl", "_COMPLETE.json"):
                    name = f"{mode}_{suite}_{lane}{suffix}"
                    files[name] = RUN / name
        for lane in range(4):
            prior.add(f"{suite}_{seed}_w{lane}.jsonl")
        for lane in range(2):
            prior.add(f"transport_{suite}_{lane}.jsonl")
        prior.add(f"train_ref_{seed}.jsonl")
        prefix = f"runs/{seed}_d"
        for name in ("COMPLETE.json", "train_steps.jsonl",
                     *(f"{step:06d}/READY.json" for step in (30, 100, 300, 1000))):
            files[f"{prefix}/{name}"] = RUN / prefix / name
    for file in sorted(RUN.glob("*.log")):
        files[f"logs/{file.name}"] = file
    manifest = {
        "files": {name: digest(file) for name, file in sorted(files.items())},
        "prior_k20_files": {name: digest(PRIOR / name) for name in sorted(prior)},
        "route_bank_sha256": digest(PRIOR / "route_bank.pt"),
    }
    (RUN / "evidence_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files["evidence_sha256.json"] = RUN / "evidence_sha256.json"
    with tarfile.open(RUN / "k20_bridge_evidence.tar.gz", "w:gz", compresslevel=4) as out:
        for name, file in sorted(files.items()):
            out.add(file, arcname=name, recursive=False)


if __name__ == "__main__":
    pack()

