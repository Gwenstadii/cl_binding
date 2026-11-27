"""Protect the R1-aligned part of the paired K20 transport response."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time

import torch

RUN = Path(__file__).resolve().parent
K20 = RUN.parent / "k20_route_margin_20261112_01"
sys.path.insert(0, str(K20))
from train_route import generated
from train_weak import ORIG, ROOT, balanced_indices, frozen_hash, open_data, read, setup, snapshot, write

PERIOD = 6
FLOOR = .80
UNIT_M = .01


def paired_bank():
    bank = torch.load(K20 / "route_bank.pt", weights_only=True, map_location="cpu")
    by_key = {(r["init"], r["q"]): r for r in bank}
    raw = read(K20 / "calibration.json")["rows"]
    refs = {(r["init"], r["q"]): r["predicted"] for r in raw if r["arm"] == "parent"}
    pairs = []
    for i in sorted({r["init"] for r in bank}):
        a, b = (by_key[i, q] for q in "AB")
        assert torch.equal(a["noise"], b["noise"]) and a["pos"] == b["pos"]
        ya, yb = (torch.tensor(refs[i, q], device="cuda") for q in "AB")
        d = (ya - yb) / 2
        strength = d.norm()
        assert strength > .005
        pairs.append((i, a, b, (ya + yb) / 2, d / strength, strength))
    assert len(pairs) == 20
    return pairs


def displacement(policy, a, b, offset, scale, coef):
    batch = {k: torch.cat([a["batch"][k], b["batch"][k]]).cuda() for k in a["batch"]}
    noise = a["noise"].cuda().expand(2, -1, -1).contiguous()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = generated(policy, batch, noise)
    physical = (x[:, :, :2] * scale + offset).clamp(-1, 1).sum(1) @ coef
    return physical.mean(0), (physical[0] - physical[1]) / 2


def train(seed, steps):
    policy, pre = setup(seed)
    config = read(K20 / "config.json")
    spec = config["training"]
    ds = open_data(ROOT / config["data"]["train_output"])
    groups = {"front": list(range(len(ds)))}
    named = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    assert [n for n, _ in named] == read(ORIG / "trainable_parameters.json")["names"]
    params = [p for _, p in named]
    frozen = frozen_hash(policy)
    optim = torch.optim.AdamW(params, lr=spec["learning_rate"], betas=policy.config.optimizer_betas,
                              eps=policy.config.optimizer_eps, weight_decay=policy.config.optimizer_weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda k: min(1., (k + 1) / spec["warmup_steps"]))
    transform = read(K20 / "action_affine_0.json")
    offset = torch.tensor(transform["offset"][:2], device="cuda")
    scale = torch.tensor(transform["scale"][:2], device="cuda")
    coef = torch.tensor(read(K20 / "calibration.json")["report"]["coef"], device="cuda")
    pairs = paired_bank()
    policy.eval()
    with torch.no_grad():
        for i, a, b, old_mu, u, strength in pairs:
            mu, d = displacement(policy, a, b, offset, scale, coef)
            assert (mu - old_mu).norm() < .002 and (d - strength * u).norm() < .002, (i, mu, d)
    print("PARENT_PAIRED_PROXY_GATE_PASS", len(pairs), flush=True)
    directory = RUN / "runs" / f"{seed}_d"
    directory.mkdir(parents=True)
    started = time.monotonic()
    draw = random.Random(seed + 8352)
    history = []
    with (directory / "train_steps.jsonl").open("x") as handle:
        for k in range(steps):
            ids = balanced_indices(groups, spec["samples_per_goal"], seed, k)
            batch = pre(copy.deepcopy(torch.utils.data.default_collate(ds.__getitems__(ids))))
            optim.zero_grad(set_to_none=True)
            policy.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v, _ = policy(batch, reduction="none")
                front = v.mean()
            front.backward()
            old = None
            if k % PERIOD == 0:
                i, a, b, old_mu, u, strength = draw.choice(pairs)
                policy.eval()
                mu, d = displacement(policy, a, b, offset, scale, coef)
                projection = d @ u
                gap = (FLOOR * strength - projection).relu() / UNIT_M
                if gap.detach().item() > 0:
                    gap.backward()
                old = {"init": i, "loss": float(gap.detach()),
                       "gain": float((projection / strength).detach()),
                       "mu_drift_m": float((mu - old_mu).norm().detach())}
            norm = float(torch.nn.utils.clip_grad_norm_(params, policy.config.optimizer_grad_clip_norm,
                                                        error_if_nonfinite=True))
            lr = optim.param_groups[0]["lr"]
            optim.step()
            sched.step()
            row = {"step": k + 1, "indices": ids, "front_loss": float(front.detach()),
                   "old": old, "lr": lr, "grad_norm": norm, "seconds": time.monotonic() - started}
            history.append(row)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            if k + 1 in (30, 100, 300, 1000) or k + 1 == steps:
                snapshot(directory / f"{k + 1:06d}", named, k + 1, history, optim, sched)
            if k < 3 or (k + 1) % 100 == 0:
                print("BRIDGE_TRAIN", seed, k + 1, row["front_loss"], old, flush=True)
    assert frozen_hash(policy) == frozen
    write(directory / "COMPLETE.json", {"seed": seed, "step": steps, "period": PERIOD,
          "floor": FLOOR, "unit_m": UNIT_M, "frozen_unchanged": True})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("seed", type=int)
    p.add_argument("--steps", type=int, default=1000)
    args = p.parse_args()
    train(args.seed, args.steps)

