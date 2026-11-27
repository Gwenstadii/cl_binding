"""Archive compact K18-D results while leaving weights and optimizers on the server."""
import hashlib
import json
from pathlib import Path
import tarfile

RUN=Path(__file__).resolve().parent


def main():
    names=("summary.json","protocol.json","holdout.json","candidates.json","front_validation_phases.json",
           "historical_1000.jsonl","replay_weight_audit.json","fit_selection.json","fit_summary.json")
    files={name:RUN/name for name in names}
    for pattern in ("probe_*.json","native_w*.jsonl","native_w*_COMPLETE.json","screen_[0-3].json",
                    "fit_w*.jsonl","fit_w*_COMPLETE.json"):
        files.update({p.name:p for p in RUN.glob(pattern)})
    for name in ("train_controls.py","holdout.py","evaluate.py","label_validation.py","summarize.py",
                 "pack.py","queue.sh","config.json","matched_fit.py"):
        files[f"code/{name}"]=RUN/name
    for p in (RUN/"runs").glob("*"):
        for name in ("COMPLETE.json","train_steps.jsonl",*(f"{n:06d}/READY.json" for n in (30,100,300,1000))):
            path=p/name
            files[str(path.relative_to(RUN))]=path
    for pattern in ("train_*.log","native_[0-5].log","probe.log","screen_*.log","holdout.log",
                    "label_validation.log","candidates.log","fit_*.log"):
        files.update({f"logs/{p.name}":p for p in RUN.glob(pattern)})
    manifest=RUN/"evidence_sha256.json"
    manifest.write_text(json.dumps({k:hashlib.sha256(p.read_bytes()).hexdigest()
                                    for k,p in sorted(files.items())},indent=2))
    files[manifest.name]=manifest
    with tarfile.open(RUN/"k18d_evidence.tar.gz","w:gz") as archive:
        for name,p in sorted(files.items()):
            archive.add(p,arcname=name,recursive=False)
    print("PACKED",len(files),flush=True)


if __name__=="__main__":
    main()
