"""Same-draft K20 horizontal field swaps and native evaluation of d training."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file

RUN = Path(__file__).resolve().parent
K20 = RUN.parent / "k20_route_margin_20261112_01"
sys.path.insert(0, str(K20))
from eval_k20 import INSTR, K19, Lab5, digest, episode, load_arm, read, rebuild, reset, snapshots, write

# Roles F and R hold matched Front-only and K20 checkpoints in the intervention.
PATCHES = {
    "swap_d": ("F", "F", "R", "AB"),
    "swap_mu": ("F", "R", "F", "AB"),
    "swap_both": ("F", "R", "R", "B"),
    "remove_d": ("R", "R", "F", "B"),
}
WINDOW = 4


def load_patch(lab, seed):
    load_arm(lab, seed, "full", 1000)
    main = lab.rt.models["F"]
    try:
        lab.rt.models["F"] = lab.rt.models["R"]
        load_arm(lab, seed, "route", 1000)
    finally:
        lab.rt.models["F"] = main


def load_bridge(lab, seed):
    path = RUN / "runs" / f"{seed}_d" / "001000"
    deadline = time.monotonic() + 3600
    while not (path / "READY.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(str(path))
        time.sleep(5)
    weights = path / "trainable.safetensors"
    assert digest(weights) == read(path / "READY.json")["sha256"]
    values = load_file(str(weights))
    params = {n:p for n,p in lab.rt.models["F"].named_parameters() if p.requires_grad}
    assert params.keys() == values.keys() and len(values) == 155
    with torch.no_grad():
        for name, param in params.items():
            assert values[name].dtype == param.dtype
            param.copy_(values[name])
    lab.rt.models["F"].eval()


def handoff(lab, record, suite):
    if suite == "dev":
        return rebuild(lab, record)
    obs, rows = reset(lab.ctx, record), []
    assert lab.ctx.signature(obs) == record["signature"]
    for k, control in enumerate(record["controls"]):
        obs, part = lab.execute(obs, np.asarray(control, dtype=np.float32), 10 * k, record["z0"])
        rows.extend(part)
    assert lab.ctx.signature(obs) == record["handoff_signature"]
    return obs, rows


def mixed(lab, obs, q, seed, r, base, mu_source, d_source):
    rt = lab.rt
    roles = set((base, mu_source, d_source))
    pre = {(role, text): rt.prefix(role, obs, INSTR[text])
           for role in sorted(roles) for text in "AB"}
    s = 1 if q == "A" else -1

    def velocity(x, t):
        raw = {key: rt.query(key[0], p, x, t) for key, p in pre.items()}
        a, b = (raw[mu_source, g][:, :10, :2].float() for g in "AB")
        c, d = (raw[d_source, g][:, :10, :2].float() for g in "AB")
        result = raw[base, q].clone()
        result[:, :10, :2] = (a + b) * .5 + s * (c - d) * .5
        return result

    return rt.integrate(velocity, rt.noise(seed, r), 10)[..., :7]


def patch_episode(lab, record, q, suite, condition):
    base, mu, d, _ = PATCHES[condition]
    obs, rows = handoff(lab, record, suite)
    boundary = len(rows)
    assert not any(r["hits"] or any(r["pred"].values()) for r in rows)
    xyz0 = np.asarray(obs[lab.ctx.obj]).copy()
    early = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for r in range(boundary // 10, 40):
            if r < boundary // 10 + WINDOW:
                u = mixed(lab, obs, q, record["seed"], r, base, mu, d)
            else:
                u = lab.native(base, INSTR[q], obs, record["seed"], r)
            obs, part = lab.execute(obs, lab.rt.physical(u), r * 10, record["z0"])
            rows.extend(part)
            if r < boundary // 10 + WINDOW:
                early.append({"step": (r + 1) * 10 - boundary,
                              "displacement": (np.asarray(part[-1]["obj"]) - xyz0).tolist(),
                              "both_contacts": sum(x["c1"] and x["c2"] for x in part)})
    assert len(rows) == 400 and len(early) == WINDOW
    hit = lab.ctx.geo.movement("scene2", lab.ctx.geometry, rows, record["z0"], None)
    return {"init": record["init"], "goal": q, "boundary": boundary, "first": hit["first_targets"],
            "first_step": hit["first_step"], "reached": hit["times"], "terminal": rows[-1]["pred"],
            "early": early}


def evaluate(mode, suite, seed, lane, lanes):
    records = ([snapshots()[i] for i in range(20, 40)] if suite == "dev"
               else read(K19 / "holdout.json")["selected"])
    lab = Lab5()
    try:
        if mode == "patch":
            load_patch(lab, seed)
            first = records[lane]
            obs, rows = handoff(lab, first, suite)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for role in "FR":
                    for q in "AB":
                        r = len(rows) // 10
                        native = lab.native(role, INSTR[q], obs, first["seed"], r)
                        ident = mixed(lab, obs, q, first["seed"], r, role, role, role)
                        assert torch.equal(native, ident), (role, q, (native - ident).abs().max())
            print("SAME_DRAFT_IDENTITY_PASS", suite, lane, flush=True)
            arms = {a: x[-1] for a, x in PATCHES.items()}
        else:
            load_bridge(lab, seed)
            arms = {"d_train": "ABN"}
        with (RUN / f"{mode}_{suite}_{lane}.jsonl").open("x") as handle:
            for arm, qs in arms.items():
                for record in records[lane::lanes]:
                    for q in qs:
                        r = (patch_episode(lab, record, q, suite, arm) if mode == "patch"
                             else episode(lab, record, q, suite))
                        handle.write(json.dumps({"arm": arm, "seed": seed, **r}, allow_nan=False) + "\n")
                        handle.flush()
                        print(mode, suite, arm, r["init"], q, r["first"], flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN / f"{mode}_{suite}_{lane}_COMPLETE.json", {"seed": seed, "lane": lane, "mode": mode})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("patch", "native"))
    p.add_argument("suite", choices=("dev", "confirm"))
    p.add_argument("seed", type=int)
    p.add_argument("lane", type=int)
    p.add_argument("--lanes", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(2)
    evaluate(a.mode, a.suite, a.seed, a.lane, a.lanes)

