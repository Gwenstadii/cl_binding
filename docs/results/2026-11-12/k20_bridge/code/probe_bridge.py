"""Link the trained projection to the old fixed-draft fields and real motion."""
import argparse

import numpy as np
import torch

from eval_bridge import INSTR, K19, Lab5, RUN, handoff, load_bridge, read, snapshots, write
from eval_k20 import make_bank, responses
from calibrate import scores
from summary_k20 import fixed_summary


def probe(suite, seed):
    records = ([snapshots()[i] for i in range(20, 40)] if suite == "dev"
               else read(K19 / "holdout.json")["selected"])
    lab = Lab5()
    try:
        load_bridge(lab, seed)
        fixed = responses(lab, make_bank(lab, range(20, 40)), True)
        move = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for record in records:
                for q, goal in (("A", "left"), ("B", "right")):
                    obs, prior = handoff(lab, record, suite)
                    b = len(prior)
                    start = np.asarray(obs[lab.ctx.obj])[:2].copy()
                    centers = [lab.ctx.geometry[g]["centre"][:2] for g in ("left", "right", "front", "back")]
                    row = {"init": record["init"], "q": q, "goal": goal, "pos": start.tolist(),
                           "centers": np.asarray(centers).tolist(), "trajectory": []}
                    for off in range(0, 40, 10):
                        chunk = lab.native("F", INSTR[q], obs, record["seed"], (b + off) // 10)
                        obs, part = lab.execute(obs, lab.rt.physical(chunk), b + off, record["z0"])
                        y = np.asarray(part[-1]["obj"])[:2] - start
                        p, margin = scores(row, y)
                        row["trajectory"].append({"steps": off + 10, "observed": y.tolist(),
                                                  "progress_m": p, "margin_m": margin})
                    move.append(row)
        assert len(move) == 40
        write(RUN / f"probe_{suite}.json", {"seed": seed, "fixed": fixed,
                                          "fixed_summary": fixed_summary(fixed), "motion": move})
        print("BRIDGE_PROBE_COMPLETE", suite, flush=True)
    finally:
        lab.ctx.env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("suite", choices=("dev", "confirm"))
    p.add_argument("seed", type=int)
    a = p.parse_args()
    torch.set_num_threads(2)
    probe(a.suite, a.seed)

