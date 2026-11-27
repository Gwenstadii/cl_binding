"""Measure the first four executed blocks after the native evaluation handoff."""
import argparse
import json

import numpy as np
import torch

from calibrate import TARGETS, scores
from eval_k20 import INSTR, K19, Lab5, RUN, load_arm, read, rebuild, reset, snapshots


def run(suite, seed, lane, lanes):
    bank = ([snapshots()[i] for i in range(20, 40)] if suite == "dev"
            else read(K19 / "holdout.json")["selected"])
    coef = np.asarray(read(RUN / "calibration.json")["report"]["coef"])
    lab = Lab5()
    try:
        with (RUN / f"transport_{suite}_{lane}.jsonl").open("x") as handle:
            for arm in ("full", "replay_full", "weak", "route"):
                load_arm(lab, seed, arm, 1000)
                for record in bank[lane::lanes]:
                    for q, goal in (("A", "left"), ("B", "right")):
                        if suite == "dev":
                            obs, prefix = rebuild(lab, record)
                            start = len(prefix)
                        else:
                            obs = reset(lab.ctx, record)
                            assert lab.ctx.signature(obs) == record["signature"]
                            for k, control in enumerate(record["controls"]):
                                obs, _ = lab.execute(obs, np.asarray(control, dtype=np.float32),
                                                     10 * k, record["z0"])
                            assert lab.ctx.signature(obs) == record["handoff_signature"]
                            start = 10 * len(record["controls"])
                        xy = np.asarray(obs[lab.ctx.obj])[:2].copy()
                        centers = [lab.ctx.geometry[k]["centre"][:2] for k in TARGETS]
                        row = {"arm": arm, "init": record["init"], "q": q, "goal": goal,
                               "pos": xy.tolist(), "centers": np.asarray(centers).tolist()}
                        trajectory = []
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            for offset in range(0, 40, 10):
                                u = lab.native("F", INSTR[q], obs, record["seed"], (start + offset) // 10)
                                physical = lab.rt.physical(u)
                                obs, rows = lab.execute(obs, physical, start + offset, record["z0"])
                                observed = np.asarray(rows[9]["obj"])[:2] - xy
                                progress, margin = scores(row, observed)
                                trajectory.append({"steps": offset + 10, "observed": observed.tolist(),
                                                   "actual_progress": progress, "actual_margin": margin})
                                if offset == 0:
                                    predicted = np.clip(physical[:10, :2], -1, 1).sum(0) @ coef
                                    row.update(trajectory[-1], predicted=predicted.tolist())
                                    row["proxy_progress"], row["proxy_margin"] = scores(row, predicted)
                        row["trajectory"] = trajectory
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                        handle.flush()
                print("TRANSPORT", suite, lane, arm, flush=True)
    finally:
        lab.ctx.env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("suite", choices=("dev", "confirm"))
    p.add_argument("seed", type=int)
    p.add_argument("--lane", type=int, required=True)
    p.add_argument("--lanes", type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(2)
    run(a.suite, a.seed, a.lane, a.lanes)

