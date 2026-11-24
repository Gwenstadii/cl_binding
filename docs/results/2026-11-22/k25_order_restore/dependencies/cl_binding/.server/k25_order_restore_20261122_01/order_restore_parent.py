"""Qualify the LF parent on the same fresh states as the curriculum transfer test."""

import argparse
import json
from pathlib import Path

import torch

import order_restore as base

RUN = base.RUN


def worker(lane, lanes):
    base.configure("lf_right_1000")
    protocol = base.read(RUN / "protocol.json")
    bank = base.read(RUN / "cases.json")["test"][lane::lanes]
    assert bank
    expected = protocol["checkpoints"]["lf_right_1000"]["parent_sha256"]
    assert base.sha(base.LF_PARENT / "model.safetensors") == expected
    lab = base.Lab(base.LF_PARENT)
    assert lab.rt.sha == {"F": expected}
    path = RUN / f"parent_{lane}.jsonl"
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            with path.open("x", encoding="utf-8") as out:
                for rec in bank:
                    for goal in "AB":
                        row = base.episode(lab, None, rec, base.sr.INSTR[goal], "native", 0.)
                        row.update(suite="parent", line="lf_parent", variant="canon", goal=goal, target=base.sr.LABEL[goal])
                        out.write(json.dumps(row, allow_nan=False)+"\n")
                        out.flush()
                        print("PARENT", lane, rec["init"], goal, row["first"], flush=True)
    finally:
        lab.ctx.env.close()
    base.write(RUN / f"parent_{lane}_COMPLETE.json", dict(lane=lane, lanes=lanes,
        ids=[r["init"] for r in bank], rows_sha256=base.sha(path), parent_sha256=expected,
        base_protocol_sha256=base.sha(RUN / "protocol.json"), source_sha256=base.sha(Path(__file__)),
        base_source_sha256=base.sha(Path(base.__file__))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=8)
    args = parser.parse_args()
    assert 0 <= args.lane < args.lanes
    torch.set_num_threads(2)
    worker(args.lane, args.lanes)
