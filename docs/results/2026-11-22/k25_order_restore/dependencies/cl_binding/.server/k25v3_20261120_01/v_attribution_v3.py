"""Separate V's action-output channels and test conditional Front substitution."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import sys

import torch

RUN = Path(__file__).resolve().parent
PATHS = RUN.parent / "k25_paths_20261119_01"
V2 = RUN.parent / "k25v2_20261120_01"
sys.path.insert(0, str(PATHS))
from path_factorization import (DEV, Lab5, assign, cases, checkpoint_files, configuration,
                                fixed_bank, measure_episode, parameters_for, read, write)
from value_relay import Relay, rms
from window_eval import INSTR, WINDOW, make_bank
from deep_selection import handoff

OUTPUT = {
    "v_mu": ("full", "cross_v", "mu"),
    "v_d": ("full", "cross_v", "d"),
    "v_reverse_mu": ("cross_v", "full", "mu"),
    "v_reverse_d": ("cross_v", "full", "d"),
    "v_amplified_d": ("full", "cross_v", "amplified_d"),
    "anchor_d": ("full", "anchor", "d"),
    "anchor_output": ("full", "anchor", "whole"),
}
SPANS = ("v_text", "v_other_positions")
NATIVE_FRESH = ("native_anchor", "native_anchor_add_v")


def records():
    old = cases()
    new = read(V2 / "holdout.json")["selected"]
    assert [r["init"] for r in old] == list(range(200, 220))
    assert [r["init"] for r in new] == list(range(400, 420))
    return old + new


def state(arm, full, anchor):
    if arm != "native_anchor_add_v":
        return configuration("anchor" if arm == "native_anchor" else arm, full, anchor)
    selected = set(parameters_for("cross_v", full)[1])
    return {name: full[name] if name in selected else value for name, value in anchor.items()}


def output_value(rt, prefixes, q, x, t, mode, scale):
    if mode == "whole":
        return rt.query("R", prefixes["R", q], x, t)
    values = {k: rt.query(k[0], p, x, t) for k, p in prefixes.items()}
    a, b = (values["R", g].float() - values["F", g].float() for g in "AB")
    if mode == "mu":
        delta = (a + b) * .5
    else:
        delta = (a - b) * (.5 if q == "A" else -.5)
        if mode == "amplified_d":
            delta *= scale
    base = values["F", q]
    return (base.float() + delta).to(base.dtype)


def output_prefixes(lab, obs, q, mode):
    roles, prompts = ("R", q) if mode == "whole" else ("FR", "AB")
    return {(role, g): lab.rt.prefix(role, obs, INSTR[g]) for role in roles for g in prompts}


def output_chunk(lab, obs, q, seed, chunk, mode, scale=1.):
    rt = lab.rt
    prefixes = output_prefixes(lab, obs, q, mode)
    return rt.integrate(lambda x, t: output_value(rt, prefixes, q, x, t, mode, scale),
                        rt.noise(seed, chunk), 10)[..., :7]


def span_values(relay, pair, arm):
    projections = relay.cache(pair)
    start, stop = pair["B"][2]["text"]
    for record in projections.values():
        for q, (base, donor) in record.items():
            if arm == "v_text":
                substitute = base.clone()
                substitute[:, start:stop] = donor[:, start:stop]
            else:
                substitute = donor.clone()
                substitute[:, start:stop] = base[:, start:stop]
            record[q] = (base, substitute)
    return projections


def span_chunk(lab, relay, obs, q, seed, chunk, arm):
    rt = lab.rt
    pair = relay.prefixes(obs)
    projections = span_values(relay, pair, arm)

    def field(x, t):
        with relay.whole(projections, q, False):
            return rt.query("F", pair[q], x, t)

    return rt.integrate(field, rt.noise(seed, chunk), 10)[..., :7]


def horizontal(d, ref):
    return float((d * ref).sum() / ref.square().sum().clamp_min(1e-12))


def collect_fields(lab, bank):
    grouped = []
    last = None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, route, step, obs, x, t, a0, b0 in bank:
            if i != last:
                prefixes = {q: lab.rt.prefix("F", obs, INSTR[q]) for q in "AB"}
                last = i
            a, b = (lab.rt.query("F", prefixes[q], x, t)[:, :10, :2].float() for q in "AB")
            old_d = (a0[:, :10, :2] - b0[:, :10, :2]).float() * .5
            grouped.append((i, route, step, (a + b) * .5, (a - b) * .5, old_d))
    return grouped


def probe(suite):
    lab = Lab5()
    full, anchor = checkpoint_files()
    try:
        bank = make_bank(lab, DEV) if suite == "dev" else fixed_bank(lab, records()[:20] if suite == "old" else records()[20:])
        data = {}
        for arm in ("full", "cross_v", "anchor", "native_anchor_add_v"):
            assign(lab, "F", state(arm, full, anchor))
            data[arm] = collect_fields(lab, bank)
        rows = []
        for f, v, a, only in zip(data["full"], data["cross_v"], data["anchor"],
                                 data["native_anchor_add_v"], strict=True):
            assert f[:3] == v[:3] == a[:3] == only[:3]
            old = f[-1]
            delta, total = v[4] - f[4], a[4] - f[4]
            alignment = float((delta * total).sum() / (delta.norm() * total.norm()).clamp_min(1e-12))
            rows.append({"init": f[0], "route": f[1], "step": f[2],
                         "native": {a: horizontal(value[4], old) for a, value in
                                    zip(("full", "cross_v", "anchor", "v_only"), (f, v, a, only), strict=True)},
                         "v_d_correction": horizontal(delta, old), "full_d_correction": horizontal(total, old),
                         "v_to_full_d_alignment": alignment, "v_mu_change_rms": rms(v[3] - f[3]),
                         "full_mu_change_rms": rms(a[3] - f[3])})
        write(RUN / f"v_probe_{suite}.json", rows)
        if suite == "dev":
            by = defaultdict(list)
            for r in rows:
                by[r["init"]].append(r)
            vgain, allgain = [median(median(r[k] for r in rr) for rr in by.values())
                              for k in ("v_d_correction", "full_d_correction")]
            assert allgain > vgain > 0
            write(RUN / "v_calibration.json", {"dev_inits": list(DEV), "v_h_gain_change": vgain,
                                              "full_h_gain_change": allgain, "scale": allgain/vgain})
    finally:
        lab.ctx.env.close()
    print("V_PROBE", suite, len(rows), flush=True)


def gate(lab, relay, rec, full, anchor):
    assign(lab, "F", full)
    assign(lab, "R", configuration("cross_v", full, anchor))
    obs = handoff(lab, rec)
    seed, chunk = rec["seed"], len(rec["controls"])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        for q in "AB":
            direct = relay.chunk(obs, q, seed, chunk, "whole", False, verify=True)
            donor = output_chunk(lab, obs, q, seed, chunk, "whole")
            assert torch.equal(direct, donor), (q, rms(direct - donor))
        pair = relay.prefixes(obs)
        all_values = relay.cache(pair)
        text, other = (span_values(relay, pair, s) for s in SPANS)
        for i, per in all_values.items():
            for q, (a, b) in per.items():
                combined = text[i][q][1].float() + other[i][q][1].float() - a.float()
                assert torch.allclose(combined, b.float(), atol=2e-6, rtol=2e-6)
    return {"init": rec["init"], "output_equals_native_v_whole": True, "complementary_v_spans": True}


def evaluate(lane, lanes):
    chosen = records()[lane::lanes]
    full, anchor = checkpoint_files()
    scale = read(RUN / "v_calibration.json")["scale"]
    lab = Lab5()
    native = lab.native
    try:
        relay = Relay(lab, full, anchor)
        checked = gate(lab, relay, chosen[0], full, anchor)
        print("V_GATE", checked, flush=True)
        with (RUN / f"v_eval_{lane}.jsonl").open("x", encoding="utf-8") as out:
            for arm in (*OUTPUT, *SPANS, *NATIVE_FRESH):
                base, donor, mode = OUTPUT.get(arm, (arm if arm in NATIVE_FRESH else "full", None, None))
                assign(lab, "F", state(base, full, anchor))
                if donor is not None:
                    assign(lab, "R", state(donor, full, anchor))
                for rec in chosen:
                    if arm in NATIVE_FRESH and rec["init"] < 400:
                        continue
                    start = len(rec["controls"])

                    def sample(role, text, obs, seed, chunk):
                        if start <= chunk < start + WINDOW:
                            if mode is not None:
                                return output_chunk(lab, obs, "B", seed, chunk, mode, scale)
                            if arm in SPANS:
                                return span_chunk(lab, relay, obs, "B", seed, chunk, arm)
                        return native(role, text, obs, seed, chunk)

                    lab.native = sample
                    row = {"arm": arm, **measure_episode(lab, rec, "B")}
                    out.write(json.dumps(row, allow_nan=False) + "\n")
                    out.flush()
                    print("V_EVAL", lane, arm, rec["init"], row["first"], flush=True)
    finally:
        lab.native = native
        lab.ctx.env.close()
    write(RUN / f"v_eval_{lane}_COMPLETE.json", {"lane": lane, "lanes": lanes, "gates": checked})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("probe", "eval"))
    p.add_argument("--suite", choices=("dev", "old", "fresh"), default="dev")
    p.add_argument("--lane", type=int, default=0)
    p.add_argument("--lanes", type=int, default=6)
    a = p.parse_args()
    torch.set_num_threads(2)
    probe(a.suite) if a.mode == "probe" else evaluate(a.lane, a.lanes)
