"""Pre-test addendum: retain the original 5.294 dose on LF -> Right.

All development doses tied at zero recovery on that curriculum. Test the original
locked dose as well as the smaller dose selected by the predeclared tie breaker.
This is a transfer control, not a new choice fitted to the test states.
"""

import argparse
import json
from pathlib import Path

import torch

import order_restore as base

RUN = base.RUN
LINE = "lf_right_1000"


def prepare():
    assert not (RUN / "standard_protocol.json").exists()
    protocol = base.read(RUN / "protocol.json")
    base.write(RUN / "standard_protocol.json", dict(line=LINE, gain=base.STANDARD_GAIN,
        ids=protocol["test_ids"], base_protocol_sha256=base.sha(RUN / "protocol.json"),
        selection_sha256=base.sha(RUN / "selection.json"), source_sha256=base.sha(Path(__file__)),
        base_source_sha256=base.sha(Path(base.__file__)),
        reason="all three development doses gave zero correct old-goal arrivals; retain original fixed dose as transfer control",
        test_results_used_to_choose_dose=False))


def worker(lane, lanes):
    protocol = base.read(RUN / "standard_protocol.json")
    assert protocol["source_sha256"] == base.sha(Path(__file__))
    assert protocol["base_source_sha256"] == base.sha(Path(base.__file__))
    assert protocol["base_protocol_sha256"] == base.sha(RUN / "protocol.json")
    assert protocol["selection_sha256"] == base.sha(RUN / "selection.json")
    base.configure(LINE)
    parent, full, deltas = base.weights(LINE)
    bank = base.read(RUN / "cases.json")["test"][lane::lanes]
    assert bank
    lab = base.Lab(parent)
    path = RUN / f"standard_{lane}.jsonl"
    try:
        base.sr.assign(lab, "F", full)
        model = base.sr.Restore(lab, full, deltas, protocol["gain"], 0.)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            checks = base.gate(lab, model, full, deltas, bank[0])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            with path.open("x", encoding="utf-8") as out:
                for rec in bank:
                    for goal in "AB":
                        row = base.episode(lab, model, rec, base.sr.INSTR[goal], "forward", protocol["gain"])
                        row.update(suite="standard", line=LINE, variant="canon", goal=goal, target=base.sr.LABEL[goal])
                        out.write(json.dumps(row, allow_nan=False)+"\n")
                        out.flush()
                        print("STANDARD", lane, rec["init"], goal, row["first"], flush=True)
    finally:
        lab.ctx.env.close()
    base.write(RUN / f"standard_{lane}_COMPLETE.json", dict(lane=lane, lanes=lanes,
        ids=[r["init"] for r in bank], gates=checks, rows_sha256=base.sha(path),
        protocol_sha256=base.sha(RUN / "standard_protocol.json"), source_sha256=base.sha(Path(__file__))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "worker"))
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "prepare":
        prepare()
    else:
        assert 0 <= args.lane < args.lanes
        worker(args.lane, args.lanes)
