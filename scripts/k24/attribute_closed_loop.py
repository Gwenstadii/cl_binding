"""Evaluate a prespecified single-update K24 fork on common grasp snapshots."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors.torch import load_file

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parent / "k20_route_margin_20261112_01"))
from eval_k20 import K19, Lab5, episode, read, write

ARMS = ("anchor", "full", "scale", "project", "focused")


def run(lane, lanes):
    torch.set_num_threads(2)
    data = read(K19 / "holdout.json")["selected"]
    assert len(data) == 20
    lab = Lab5()
    params = {n:p for n,p in lab.rt.models["F"].named_parameters() if p.requires_grad}
    assert len(params) == 155
    execute_native = lab.execute
    try:
        with (RUN / f"native_{lane}.jsonl").open("x", encoding="utf-8") as file:
            for arm in ARMS:
                v = load_file(str(RUN / f"weights_100_{arm}.safetensors"))
                assert v.keys() == params.keys()
                with torch.no_grad():
                    for k,p in params.items():
                        p.copy_(v[k])
                lab.rt.models["F"].eval()
                for rec in data[lane::lanes]:
                    for q in "ABN":
                        first_block = {}
                        start = 0 if q == "N" else len(rec["controls"]) * 10

                        def track(obs, actions, first, z0):
                            before = np.asarray(obs[lab.ctx.obj])[:2].copy()
                            obs, rows = execute_native(obs, actions, first, z0)
                            if first == start:
                                first_block["displacement_xy_m"] = (np.asarray(rows[-1]["obj"])[:2] - before).tolist()
                            return obs, rows

                        lab.execute = track
                        row = {"arm":arm, **episode(lab, rec, q, "confirm"), **first_block}
                        assert "displacement_xy_m" in row
                        file.write(json.dumps(row, allow_nan=False) + "\n")
                        file.flush()
                        print("K24 native", lane, arm, rec["init"], q, row["first"], flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN / f"native_{lane}_COMPLETE.json", {"lane":lane, "lanes":lanes})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("lane", type=int)
    p.add_argument("--lanes", type=int, default=4)
    a = p.parse_args()
    run(a.lane, a.lanes)
