"""Separate strength, orientation, and source of V-mediated action corrections."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import sys

import torch

RUN = Path(__file__).resolve().parent
V3 = RUN.parent / "k25v3_20261120_01"
sys.path.insert(0, str(V3))
from v_attribution_v3 import (DEV, INSTR, WINDOW, Lab5, Relay, assign, checkpoint_files, configuration, fixed_bank,
                             handoff, horizontal, make_bank, measure_episode, output_chunk, parameters_for, read,
                             records, rms, write)

CORE = ("dose_2", "dose_3", "dose_4", "reverse", "horizontal", "remainder", "rotate_ccw", "rotate_cw")
FRESH = ("baseline", "natural", "amplified", "residual_d", "native_v", "native_extrap", "extrap_d",
         "shared_source", "conditional_source")
LEFT = ("baseline", "amplified", "reverse")
RELAY = {"shared_source": "mean", "conditional_source": "diff"}


def weights(full, anchor, scale):
    _, names = parameters_for("cross_v", full)
    assert len(names) == 8 and all(not torch.equal(full[n], anchor[n]) for n in names)
    extrap = {**full, **{n: (full[n].double() + scale * (anchor[n].double()-full[n].double())).to(full[n].dtype)
                         for n in names}}
    return configuration("cross_v", full, anchor), extrap, names


def contrast(a, b):
    return (a.float() - b.float()) * .5


def direction(arm, raw):
    if arm in ("horizontal", "remainder"):
        result = torch.zeros_like(raw)
        s = slice(0, 2) if arm == "horizontal" else slice(2, None)
        result[..., s] = raw[..., s]
        return result
    if arm in ("rotate_ccw", "rotate_cw"):
        result = raw.clone()
        sign = 1 if arm == "rotate_ccw" else -1
        result[..., 0] = -sign * raw[..., 1]
        result[..., 1] = sign * raw[..., 0]
        return result
    return raw


def compose(base, raw, q, scale):
    return (base.float() + raw * (scale if q == "A" else -scale)).to(base.dtype)


def correction(earlier, later):
    a, b = (d.float() - f.float() for d, f in zip(earlier, later, strict=True))
    return contrast(a, b)


def values(rt, prefixes, role, x, t):
    return [rt.query(role, prefixes[role, q], x, t) for q in "AB"]


def relayed(rt, relay, prefixes, projections, mode, x, t):
    output = []
    for q in "AB":
        with relay.edit(projections, q, mode, False):
            output.append(rt.query("F", prefixes["F", q], x, t))
    return output


def chunk(lab, relay, obs, q, seed, step, arm, scale):
    rt = lab.rt
    roles = "F" if arm in (*RELAY, "residual_d") else "FR"
    prefixes = {(role, g): rt.prefix(role, obs, INSTR[g]) for role in roles for g in "AB"}
    projections = relay.cache({g: prefixes["F", g] for g in "AB"}) if arm in RELAY else None

    def field(x, t):
        base = values(rt, prefixes, "F", x, t)
        if arm == "residual_d":
            raw = contrast(*base)
        else:
            if arm in RELAY:
                donor = relayed(rt, relay, prefixes, projections, RELAY[arm], x, t)
            else:
                donor = values(rt, prefixes, "R", x, t)
            raw = correction(donor, base)
        return compose(base["AB".index(q)], direction(arm, raw), q, scale)

    return rt.integrate(field, rt.noise(seed, step), 10)[..., :7]


def verify(lab, relay, rec, scale):
    obs = handoff(lab, rec)
    seed, step = rec["seed"], len(rec["controls"])
    rt = lab.rt
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        for q in "AB":
            base = lab.native("F", INSTR[q], obs, seed, step)
            assert torch.equal(base, chunk(lab, relay, obs, q, seed, step, "amplified", 0.))
            for s, mode in ((1., "d"), (scale, "amplified_d")):
                expected = output_chunk(lab, obs, q, seed, step, mode, s)
                candidate = chunk(lab, relay, obs, q, seed, step, "amplified", s)
                assert torch.equal(expected, candidate), (q, s, rms(expected-candidate))
        p = {(role, q): rt.prefix(role, obs, INSTR[q]) for role in "FR" for q in "AB"}
        x = rt.noise(seed, step)
        t = torch.ones(1, device=x.device)
        base, donor = (values(rt, p, role, x, t) for role in "FR")
        raw = correction(donor, base)
        xy, rest = (direction(a, raw) for a in ("horizontal", "remainder"))
        assert torch.equal(xy + rest, raw)
        for arm in ("rotate_ccw", "rotate_cw"):
            r = direction(arm, raw)
            assert torch.equal(r[..., 2:], raw[..., 2:])
            assert torch.equal(r[..., :2].square().sum(-1), raw[..., :2].square().sum(-1))
            assert torch.equal((r[..., :2] * raw[..., :2]).sum(-1), torch.zeros_like(r[..., 0]))
        adjusted = [compose(base[i], raw, q, scale) for i, q in enumerate("AB")]
        leak = (adjusted[0].float()+adjusted[1].float()-base[0].float()-base[1].float())*.5
        fp64 = [(b.double()+sign*scale*raw.double()) for b, sign in zip(base, (1, -1), strict=True)]
        torch.testing.assert_close(fp64[0]+fp64[1], base[0].double()+base[1].double(), atol=1e-12, rtol=1e-12)
    return {"init": rec["init"], "identity_and_v3_endpoints_bitwise_for_both_prompts": True,
            "rotated_xy_same_norm_per_slot_and_perpendicular": True, "xy_and_remaining_reconstruct": True,
            "fp64_pair_mean_unchanged": True, "cast_pair_mean_delta_rms": rms(leak)}


def collect(lab, bank, relay, mode=None):
    last = None
    fields = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, _, _, obs, x, t, _, _ in bank:
            if i != last:
                prefixes = {("F", q): lab.rt.prefix("F", obs, INSTR[q]) for q in "AB"}
                if mode:
                    projections = relay.cache({q: prefixes["F", q] for q in "AB"})
                last = i
            result = (relayed(lab.rt, relay, prefixes, projections, mode, x, t) if mode
                      else values(lab.rt, prefixes, "F", x, t))
            fields.append(tuple(v[:, :10].float() for v in result))
    return fields


def cosine(a, b):
    return float((a*b).sum() / (a.norm()*b.norm()).clamp_min(1e-12))


def axis_metrics(bank, measured, scale):
    rows = []
    names = ("rolled", "anchor", "extrap", "shared_source", "conditional_source")
    for j, (i, route, step, _, _, _, a0, b0) in enumerate(bank):
        ref = contrast(a0[:, :10, :2], b0[:, :10, :2])
        base = measured["late"][j]
        delta = {n: correction(measured[n][j], base) for n in names}
        v, all_changes, extrap = (delta[n] for n in ("rolled", "anchor", "extrap"))
        xy = v[..., :2]
        def change(n):
            return horizontal(delta[n][..., :2], ref)

        def mean_shift(n):
            candidate = measured[n][j]
            a, b = (u[..., :2] - old[..., :2] for u, old in zip(candidate, base, strict=True))
            return rms((a+b)*.5)

        rows.append({"init": i, "route": route, "step": step,
                     "base_gain": horizontal(contrast(*base)[..., :2], ref),
                     "v_gain": change("rolled"), "anchor_gain": change("anchor"),
                     "weight_extrap_gain": change("extrap"), "amplified_output_gain": scale*change("rolled"),
                     "v_to_full_xy_cos": cosine(xy, all_changes[..., :2]),
                     "v_to_full_seven_cos": cosine(v[..., :7], all_changes[..., :7]),
                     "v_xy_fraction_of_seven_squared": float(xy.square().sum()/v[..., :7].square().sum().clamp_min(1e-12)),
                     "extrap_to_v_xy_cos": cosine(extrap[..., :2], xy),
                     "v_mu_xy_rms": mean_shift("rolled"), "weight_extrap_mu_xy_rms": mean_shift("extrap"),
                     "shared_gain": change("shared_source"), "conditional_gain": change("conditional_source"),
                     "shared_to_v_xy_cos": cosine(delta["shared_source"][..., :2], xy),
                     "conditional_to_v_xy_cos": cosine(delta["conditional_source"][..., :2], xy),
                     "nonadditivity_gain": horizontal((v-delta["shared_source"]-delta["conditional_source"])[..., :2], ref)})
    return rows


def median_rows(rows, key):
    by = defaultdict(list)
    for r in rows:
        by[r["init"]].append(r[key])
    assert all(len(group) == 6 for group in by.values())
    return median(median(group) for group in by.values())


def probe(suite):
    scale = read(V3 / "v_calibration.json")["scale"]
    full, anchor = checkpoint_files()
    rolled, extrap, names = weights(full, anchor, scale)
    lab = Lab5()
    try:
        bank = make_bank(lab, DEV) if suite == "dev" else fixed_bank(lab, read(RUN / "holdout.json")["selected"])
        relay = Relay(lab, full, anchor)
        measured = {}
        for mode, state in (("rolled", rolled), ("anchor", anchor), ("extrap", extrap), ("late", full)):
            assign(lab, "F", state)
            measured[mode] = collect(lab, bank, relay)
        for mode, old in RELAY.items():
            measured[mode] = collect(lab, bank, relay, old)
        rows = axis_metrics(bank, measured, scale)
        write(RUN / f"fixed_{suite}.json", rows)
        if suite == "dev":
            restored = median_rows(rows, "amplified_output_gain")
            natural = median_rows(rows, "v_gain")
            anchor_gain = median_rows(rows, "anchor_gain")
            assert abs(restored-anchor_gain) < 1e-6 and natural > 0
            residual = restored / median_rows(rows, "base_gain")
            write(RUN / "plan.json", {"dev": list(DEV), "scale": scale, "residual_scale": residual,
                                     "core": list(CORE), "fresh_right": list((*FRESH, *CORE)), "fresh_left": list(LEFT),
                                     "native_v_tensors": names, "duration_chunks": WINDOW,
                                     "gain_target": restored, "native_v_gain": natural})
        print("FIXED", suite, len(rows), flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN / f"fixed_{suite}_COMPLETE.json", {"suite": suite, "drafts": len(rows)})


def strength(arm, gain, residual):
    doses = {"dose_2": 2., "dose_3": 3., "dose_4": 4., "natural": 1., "reverse": -gain,
             "residual_d": residual, "extrap_d": 1.}
    return doses.get(arm, gain)


def evaluate(suite, lane, lanes):
    if suite == "legacy":
        cases = records()[lane::lanes]
        plan = [(a, "B") for a in CORE]
    else:
        cases = read(RUN / "holdout.json")["selected"][lane::lanes]
        plan = [(a, "B") for a in (*FRESH, *CORE)] + [(a, "A") for a in LEFT]
    gain = read(V3 / "v_calibration.json")["scale"]
    residual = read(RUN / "plan.json")["residual_scale"] if suite == "third" else 0.
    full, anchor = checkpoint_files()
    rolled, extrap, _ = weights(full, anchor, gain)
    lab = Lab5()
    native = lab.native
    try:
        relay = Relay(lab, full, anchor)
        assign(lab, "F", full)
        assign(lab, "R", rolled)
        checked = verify(lab, relay, cases[0], gain)
        print("V_DIRECTION_GATE", checked, flush=True)
        with (RUN / f"{suite}_{lane}.jsonl").open("x", encoding="utf-8") as out:
            for arm, q in plan:
                assign(lab, "R", extrap if arm in ("native_extrap", "extrap_d") else rolled)
                scale = strength(arm, gain, residual)
                for rec in cases:
                    start = len(rec["controls"])

                    def sample(role, text, obs, seed, step):
                        assert text == INSTR[q]
                        if arm == "baseline" or not start <= step < start+WINDOW:
                            return native(role, text, obs, seed, step)
                        if arm in ("native_v", "native_extrap"):
                            return native("R", text, obs, seed, step)
                        return chunk(lab, relay, obs, q, seed, step, arm, scale)

                    lab.native = sample
                    row = {"arm": arm, **measure_episode(lab, rec, q)}
                    out.write(json.dumps(row, allow_nan=False) + "\n")
                    out.flush()
                    print(suite.upper(), lane, arm, q, rec["init"], row["first"], flush=True)
    finally:
        lab.native = native
        lab.ctx.env.close()
    write(RUN / f"{suite}_{lane}_COMPLETE.json", {"suite": suite, "lane": lane, "lanes": lanes, "gate": checked})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("probe", "eval"))
    p.add_argument("--suite", choices=("dev", "legacy", "third"), required=True)
    p.add_argument("--lane", type=int, default=0)
    p.add_argument("--lanes", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(2)
    probe(a.suite) if a.mode == "probe" else evaluate(a.suite, a.lane, a.lanes)
