"""Create fourth-task continuations of an archived three-goal reference."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import sys
from time import monotonic

import torch
from safetensors.torch import load_file, save_file

RUN = Path(__file__).resolve().parent
K19 = RUN.parent / "k19_retention_20261111_01"
sys.path.insert(0, str(K19))
from train_k19 import ROOT, balanced_indices, frozen_hash, open_data, read, setup, write

REFERENCE = K19 / "runs" / "20261121_replay_full" / "001000"
SEED = 20261123
STEPS = 3000
SAVE = (1000, 2000, STEPS)


def sha(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def read_weights(folder):
    path = folder / "trainable.safetensors"
    marker = read(folder / "READY.json")
    assert sha(path) == marker["sha256"]
    return load_file(str(path)), marker


def train(extend):
    begin, end, saves = (STEPS, 9000, (6000, 9000)) if extend else (0, STEPS, SAVE)
    source = RUN / "back" / f"{STEPS:06d}" if extend else REFERENCE
    parent, marker = read_weights(source)
    original_sha = marker["parent_sha256"] if extend else marker["sha256"]
    policy, pre = setup(SEED)
    named = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    assert named.keys() == parent.keys() and len(named) == 155
    with torch.no_grad():
        for n, param in named.items():
            assert param.dtype == parent[n].dtype
            param.copy_(parent[n])
    frozen = frozen_hash(policy)
    ds_path = ROOT / "datasets/stage1_single_back_train_aligned_v1"
    conversion = read(ds_path / "r0_conversion.json")
    spec = conversion["spec"]
    assert spec["goals"] == ["back"] and spec["split"] == "train"
    assert spec["action_alignment"] == "post_action_obs_to_next_action"
    data = open_data(ds_path)
    assert len(data) == sum(r["frames"] for r in conversion["episodes"])
    params = list(named.values())
    c = policy.config
    opt = torch.optim.AdamW(params, lr=1e-4, betas=c.optimizer_betas, eps=c.optimizer_eps,
                            weight_decay=c.optimizer_weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1., (k+1)/100))
    folder = RUN / ("back_long" if extend else "back")
    folder.mkdir()
    write(folder / "manifest.json", {"parent": str(REFERENCE), "parent_sha256": marker["sha256"],
                                     "started_from": str(source), "original_sha256": original_sha,
                                     "training_seed": SEED, "goal": "back", "start_step": begin, "steps": end,
                                     "optimizer": "new AdamW at start of this phase",
                                     "lr": 1e-4, "warmup": 100, "gradient_clip": 10., "batch": 6,
                                     "dataset": str(ds_path), "dataset_split_sha256": spec["split_sha256"],
                                     "frozen_hash": frozen, "normalization": "inherited from R1 via K19"})
    groups = {"back": list(range(len(data)))}
    started = monotonic()
    policy.train()
    with (folder / "train.jsonl").open("x", encoding="utf-8") as log:
        for k in range(begin, end):
            ids = balanced_indices(groups, 6, SEED, k)
            batch = pre(torch.utils.data.default_collate(data.__getitems__(ids)))
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy(batch, reduction="none")
                loss = loss.mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 10., error_if_nonfinite=True)
            opt.step()
            schedule.step()
            row = {"step": k+1, "loss": float(loss.detach()), "grad_norm": float(norm),
                   "lr": schedule.get_last_lr()[0], "indices": ids}
            log.write(json.dumps(row, allow_nan=False)+"\n")
            if k == begin or (k+1) % 50 == 0:
                log.flush()
                print("BACK_TRAIN", row, flush=True)
            if k+1 in saves:
                dest = folder / f"{k+1:06d}"
                dest.mkdir()
                assert frozen_hash(policy) == frozen
                path = dest / "trainable.safetensors"
                save_file({n: p.detach().cpu().contiguous() for n, p in named.items()}, str(path))
                restored = load_file(str(path))
                assert all(torch.equal(restored[n], p.detach().cpu()) for n,p in named.items())
                write(dest / "READY.json", {"step": k+1, "sha256": sha(path), "count": len(named),
                                            "parent_sha256": original_sha, "started_from_sha256": marker["sha256"],
                                            "frozen_unchanged": True})
    write(folder / "COMPLETE.json", {"steps": end, "start_step": begin, "wall_seconds": monotonic()-started,
                                     "frozen_unchanged": True})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--extend", action="store_true")
    train(p.parse_args().extend)
