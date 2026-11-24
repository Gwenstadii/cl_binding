"""Fresh confirmation set for a conservative dose after Left losses on set 800.

The new set 900 is not used to choose the dose. The lowest development dose
improving Right while retaining both Left endpoints is locked before rollouts.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import order_restore as base

RUN = base.RUN
LINE = "lr_front_1000"
IDS = list(range(900, 920))


def prepare():
    assert not (RUN / "reserve_protocol.json").exists()
    selected = base.read(RUN / "selection.json")["selected"][LINE]["dev_stats"]
    native = selected[str(0.)]["per_goal"]
    eligible = []
    for gain in base.DEV_GAINS:
        arm = selected[str(gain)]["per_goal"]
        if (all(arm["A"][k] >= native["A"][k] for k in ("first_correct", "terminal_correct"))
                and arm["B"]["first_correct"] > native["B"]["first_correct"]
                and arm["B"]["terminal_correct"] >= native["B"]["terminal_correct"]):
            eligible.append(gain)
    gain = min(eligible)
    assert gain == 1.
    previous = base.read(RUN / "cases.json")
    refs = [np.asarray(r["state"])[1:] for group in previous.values() if isinstance(group, list) for r in group]
    ctx = base.sr.km.Ctx("scene2")
    cases = []
    try:
        for i, init in enumerate(IDS):
            seed = 996201+i
            np.random.seed(seed)
            ctx.env.seed(seed)
            ctx.env.reset()
            state = np.asarray(ctx.env.get_sim_state(), dtype=np.float64).copy()
            distance = min(float(np.linalg.norm(state[1:]-ref)) for ref in refs)
            assert distance > 1e-6
            rec = dict(init=init, state=state.tolist(), reset_seed=seed, seed=16960041+107*i,
                       min_distance_to_dev_and_primary=distance)
            obs = base.sr.reset(ctx, rec)
            assert not any(ctx.flags().values())
            rec.update(signature=ctx.signature(obs), z0=float(obs[ctx.obj][2]), book_start=obs[ctx.obj].tolist())
            assert rec["signature"]["sim"] not in {r["signature"]["sim"] for r in cases}
            cases.append(rec)
    finally:
        ctx.env.close()
    base.write(RUN / "reserve_cases.json", cases)
    base.write(RUN / "reserve_protocol.json", dict(line=LINE, gain=gain, original_gain=base.STANDARD_GAIN,
        ids=IDS, jobs=[["native", 0.], ["forward", gain], ["reverse", gain], ["forward", base.STANDARD_GAIN]],
        source_sha256=base.sha(Path(__file__)), base_source_sha256=base.sha(Path(base.__file__)),
        base_protocol_sha256=base.sha(RUN / "protocol.json"), cases_sha256=base.sha(RUN / "reserve_cases.json"),
        selection_sha256=base.sha(RUN / "selection.json"), eligible_development_gains=eligible,
        dose_rule="smallest dev dose: Left first and terminal >= native; Right first > native, terminal >= native",
        designed_after_primary_set_left_losses=True, reserve_results_consulted=False,
        state_filter="reset geometry only; no grasp or outcome filtering", window=[80, 180], horizon=400))


def worker(lane, lanes):
    p = base.read(RUN / "reserve_protocol.json")
    assert p["source_sha256"] == base.sha(Path(__file__))
    assert p["base_source_sha256"] == base.sha(Path(base.__file__))
    assert p["base_protocol_sha256"] == base.sha(RUN / "protocol.json")
    assert p["cases_sha256"] == base.sha(RUN / "reserve_cases.json")
    base.configure(LINE)
    parent, full, deltas = base.weights(LINE)
    bank = base.read(RUN / "reserve_cases.json")[lane::lanes]
    assert bank
    path = RUN / f"reserve_{lane}.jsonl"
    lab = base.Lab(parent)
    checks = []
    try:
        base.sr.assign(lab, "F", full)
        with path.open("x", encoding="utf-8") as out:
            for arm, gain in p["jobs"]:
                model = base.sr.Restore(lab, full, deltas, gain or base.STANDARD_GAIN, 0.)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    checks.append(dict(gain=gain, **base.gate(lab, model, full, deltas, bank[0])))
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for rec in bank:
                        for goal in "AB":
                            row = base.episode(lab, model, rec, base.sr.INSTR[goal], arm, gain)
                            row.update(suite="reserve", line=LINE, variant="canon", goal=goal, target=base.sr.LABEL[goal])
                            out.write(json.dumps(row, allow_nan=False)+"\n")
                            out.flush()
                            print("RESERVE", lane, arm, gain, rec["init"], goal, row["first"], flush=True)
                del model
    finally:
        lab.ctx.env.close()
    base.write(RUN / f"reserve_{lane}_COMPLETE.json", dict(lane=lane, lanes=lanes,
        ids=[r["init"] for r in bank], rows_sha256=base.sha(path), gates=checks,
        protocol_sha256=base.sha(RUN / "reserve_protocol.json"), source_sha256=base.sha(Path(__file__))))


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
