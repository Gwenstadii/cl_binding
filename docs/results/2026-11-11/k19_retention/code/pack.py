"""Audit matched training and archive K19 measurements without model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

RUN=Path(__file__).resolve().parent
D=RUN.parent/"k18_training_controls_20261111_01"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def audit():
    refs={s:jsonl(D/"runs"/f"{s}_full"/"train_steps.jsonl") for s in (20261021,20261121,20261122)}
    info={}
    for folder in sorted((RUN/"runs").glob("*")):
        result=read(folder/"COMPLETE.json")
        logs=jsonl(folder/"train_steps.jsonl")
        assert len(logs)==result["step"]==1000 and result["frozen_unchanged"]
        a=refs[result["seed"]]
        assert len(a)==len(logs)
        for b,c in zip(logs,a,strict=True):
            assert (b["step"],b["indices"],b["lr"])==(c["step"],c["indices"],c["lr"])
        assert abs(logs[0]["front_loss"]-a[0]["loss"])<1e-6
        info[folder.name]={"steps":len(logs),"seconds":result["seconds"],"all_front_batches_lr_match":True,
                           "first_front_loss_matches":True,"frozen_parameters_match":True,
                           "checkpoints":{s:read(folder/f"{s:06d}"/"READY.json") for s in (30,100,300,1000)}}
    return info


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--partial",action="store_true")
    a=p.parse_args()
    info=audit()
    if not a.partial:
        assert len(info)>=9
    (RUN/"training_audit.json").write_text(json.dumps(info,indent=2))
    names=("summary.json","protocol.json","replay_extension.json","soft_extension.json","teacher_READY.json","holdout.json",
           "candidates.json","selection.json","exploratory_confirmation.json","baseline_dev.jsonl","baseline_fields.json",
           "training_audit.json")
    files={n:RUN/n for n in names if (RUN/n).exists()}
    for pat in ("probe_*.json","dev_w*.jsonl","dev_w*_COMPLETE.json","confirm_w*.jsonl",
                "confirm_w*_COMPLETE.json","screen_[0-1].json","gradient_*.jsonl","mu_layers_*.json"):
        files.update({f.name:f for f in RUN.glob(pat)})
    for n in ("train_k19.py","eval_k19.py","holdout.py","summary_k19.py","gradient_diagnosis.py",
              "localize_mu.py","pack.py","queue.sh","confirm.sh","config.json"):
        files[f"code/{n}"]=RUN/n
    files["code/prior_summary.py"]=D/"summarize.py"
    for folder in sorted((RUN/"runs").glob("*")):
        for n in ("COMPLETE.json","train_steps.jsonl",*(f"{s:06d}/READY.json" for s in (30,100,300,1000))):
            f=folder/n
            files[str(f.relative_to(RUN))]=f
    for f in RUN.glob("*.log"):
        files[f"logs/{f.name}"]=f
    manifest=RUN/"evidence_sha256.json"
    manifest.write_text(json.dumps({n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in sorted(files.items())},indent=2))
    files[manifest.name]=manifest
    with tarfile.open(RUN/"k19_evidence.tar.gz","w:gz") as out:
        for n,p in sorted(files.items()):
            out.add(p,arcname=n,recursive=False)
    print("PACKED",len(files),flush=True)


if __name__=="__main__":
    main()
