"""Repeat the V and query interventions on two separate Front training streams."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import sys

from safetensors.torch import load_file
import torch

RUN = Path(__file__).resolve().parent
V4 = RUN.parent / "k25v4_20261120_01"
V3 = RUN.parent / "k25v3_20261120_01"
HOLDOUT = RUN.parent / "k25v3_v_20261120_01" / "holdout.json"
TRAINING = RUN.parent / "k18_training_controls_20261111_01" / "runs"
sys.path.insert(0, str(V4))
from query_upstream_v4 import Fixed, Queries, group, rolled, same_role_gate
from v_attribution_v3 import (INSTR, WINDOW, Lab5, assign, collect_fields, configuration, fixed_bank, handoff,
                             horizontal, measure_episode, output_chunk, read, write)

SEEDS = (20261121, 20261122)
STEPS = (30, 100)
NATIVE = ("late", "anchor")
RIGHT = (*NATIVE, "v_natural", "v_amplified", "v_reverse", "mlp_window", "mlp_late_query", "mlp_query_only")


def locked_cases():
    rows = read(HOLDOUT)["selected"]
    assert [r["init"] for r in rows] == list(range(600, 620))
    return rows


def states(seed):
    branch = TRAINING / f"{seed}_full"
    complete = read(branch / "COMPLETE.json")
    assert complete["seed"] == seed and complete["arm"] == "full" and complete["frozen_unchanged"]
    checkpoints = {}
    for step in STEPS:
        folder = branch / f"{step:06d}"
        path = folder / "trainable.safetensors"
        marker = read(folder / "READY.json")
        with path.open("rb") as handle:
            assert file_digest(handle, "sha256").hexdigest() == marker["sha256"]
        state = load_file(str(path))
        assert len(state) == marker["count"] == 155
        checkpoints[step] = state, marker["sha256"]
    return checkpoints


def cosine(a, b):
    return float((a*b).sum() / (a.norm()*b.norm()).clamp_min(1e-12))


def probe(seed):
    weights = states(seed)
    anchor, late = (weights[t][0] for t in STEPS)
    selected = read(V4 / "selection.json")
    assert selected["families"] == ["action_time", "mlp"]
    assert late.keys() == anchor.keys()
    for name, family in (("first", "action_time"), ("second", "mlp")):
        assert set(selected[name]) == {k for k in late if group(k) == family}
    lab = Lab5()
    try:
        bank = fixed_bank(lab, locked_cases())
        phases = {}
        for name, w in (("late", late), ("anchor", anchor), ("v", configuration("cross_v", late, anchor))):
            assign(lab, "F", w)
            phases[name] = collect_fields(lab, bank)
        values = []
        for f, a, v in zip(*(phases[n] for n in ("late", "anchor", "v")), strict=True):
            assert f[:3] == a[:3] == v[:3]
            d_late, d_anchor, d_v, axis = f[4], a[4], v[4], f[5]
            values.append({"init": f[0], "route": f[1], "step": f[2],
                           "late_gain": horizontal(d_late, axis), "anchor_gain": horizontal(d_anchor, axis),
                           "v_gain": horizontal(d_v-d_late, axis), "anchor_correction": horizontal(d_anchor-d_late, axis),
                           "v_direction_cos": cosine(d_v-d_late, d_anchor-d_late),
                           "late_direction_cos": cosine(d_late, axis),
                           "mu_drift_rms": float((a[3]-f[3]).square().mean().sqrt())})
        query = Fixed(lab, bank, late, anchor)
        effects = {arm: query.measure(rolled(late, anchor, selected[k])) for arm, k in
                   (("time", "first"), ("mlp", "second"), ("joint", "joint"), ("local", "local"))}
        write(RUN / f"probe_{seed}.json", {"seed": seed, "sha256": {t: weights[t][1] for t in STEPS},
                                           "value": values, "query": effects})
    finally:
        lab.ctx.env.close()
    write(RUN / f"probe_{seed}_COMPLETE.json", {"seed": seed, "states": len(locked_cases()), "drafts": len(values)})


def evaluate(seed, lane, lanes):
    weights = states(seed)
    anchor, late = (weights[t][0] for t in STEPS)
    mlp = rolled(late, anchor, read(V4 / "selection.json")["second"])
    v = configuration("cross_v", late, anchor)
    gain = read(V3 / "v_calibration.json")["scale"]
    cases = locked_cases()[lane::lanes]
    lab = Lab5()
    native = lab.native
    sites = Queries(lab)
    v_doses = {"v_natural": 1., "v_amplified": gain, "v_reverse": -gain}
    shapes = {"mlp_window": (late, mlp), "mlp_query_only": (late, mlp), "mlp_late_query": (mlp, late),
              "anchor": (anchor, late), "late": (late, late), **dict.fromkeys(v_doses, (late, v))}
    try:
        groups = {"action_output": [k for k in late if group(k) == "action_output"]}
        gate = same_role_gate(lab, sites, cases[0], late, anchor, groups)
        assign(lab, "R", v)
        obs = handoff(lab, cases[0])
        noise, start = cases[0]["seed"], len(cases[0]["controls"])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            assert torch.equal(output_chunk(lab, obs, "B", noise, start, "amplified_d", 0.),
                               native("F", INSTR["B"], obs, noise, start))
        gate["zero_v_correction_is_native_bitwise"] = True
        print("REPLICATION_GATE", seed, lane, gate, flush=True)
        with (RUN / f"eval_{seed}_{lane}.jsonl").open("x", encoding="utf-8") as out:
            for arm in RIGHT:
                assign(lab, "F", shapes[arm][0])
                assign(lab, "R", shapes[arm][1])
                goals = "ABN" if arm in NATIVE else "B"
                for rec in cases:
                    start = len(rec["controls"])
                    for q in goals:
                        def sample(role, text, observed, sample_seed, step):
                            assert text == INSTR[q]
                            if arm in NATIVE or not start <= step < start + WINDOW:
                                after = "R" if arm == "mlp_late_query" else role
                                return native(after, text, observed, sample_seed, step)
                            assert q == "B"
                            if arm in v_doses:
                                return output_chunk(lab, observed, q, sample_seed, step, "amplified_d", v_doses[arm])
                            if arm == "mlp_window":
                                return native("R", text, observed, sample_seed, step)
                            return sites.chunk(observed, q, sample_seed, step)

                        lab.native = sample
                        row = {"seed": seed, "arm": arm, **measure_episode(lab, rec, q)}
                        out.write(json.dumps(row, allow_nan=False) + "\n")
                        out.flush()
                        print("REPLICATION", seed, lane, arm, q, rec["init"], row["first"], flush=True)
    finally:
        lab.native = native
        lab.ctx.env.close()
    write(RUN / f"eval_{seed}_{lane}_COMPLETE.json", {"seed": seed, "lane": lane, "lanes": lanes, "gate": gate,
                                                     "sha256": {t: weights[t][1] for t in STEPS}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("probe", "eval"))
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "probe":
        probe(args.seed)
    else:
        evaluate(args.seed, args.lane, args.lanes)
