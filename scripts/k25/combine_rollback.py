"""Screen grouped K25 rollbacks and test their target-selection mechanism."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import sys

import torch

RUN = Path(__file__).resolve().parent
K24 = RUN.parent / "k24_final_window_20261117_01"
sys.path.insert(0, str(K24))
from window_eval import INSTR, K19, Lab5, WINDOW, assign, group, load_weights, make_bank, measure_episode, mix, read, write
from eval_k20 import validation_bank, validation_fields

GROUPS = ("kv", "cross_other", "self_attention", "mlp", "action_time", "io")
DEV = (20, 23, 26, 29, 32, 35, 38, 39)
ALL = (1 << len(GROUPS)) - 1
PATCHES = {
    "best_mu": ("F", "R", "F", "F"),
    "best_d": ("F", "F", "R", "R"),
    "best_mu_d": ("F", "R", "R", "R"),
    "reverse_mu": ("R", "F", "R", "R"),
    "reverse_d": ("R", "R", "F", "F"),
}


def module(name):
    k = group(name)
    return "io" if k in ("action_output", "state") else k


def includes(mask, key):
    return key in GROUPS and bool(mask & (1 << GROUPS.index(key)))


def weights(mask, full, anchor):
    return {n: anchor[n] if includes(mask, module(n)) else w for n, w in full.items()}


def audit(full, anchor):
    assert full.keys() == anchor.keys() and len(full) == 155
    assert all(torch.equal(v, anchor[n]) for n, v in full.items() if module(n) not in GROUPS)
    counts = defaultdict(int)
    energy = defaultdict(float)
    for n, v in full.items():
        counts[module(n)] += 1
        energy[module(n)] += float((v.float() - anchor[n].float()).square().sum())
    return {"groups": list(GROUPS), "tensors": dict(counts), "delta_squared_norm": dict(energy)}


def fields(lab, bank):
    role = "F"
    rt = lab.rt
    current, values = None, []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, _, _, obs, x, t, old_a, old_b in bank:
            if i != current:
                prefix = {q: rt.prefix(role, obs, INSTR[q]) for q in "AB"}
                current = i
            a, b = (rt.query(role, prefix[q], x, t)[..., :7].float() for q in "AB")
            values.append((i, (a + b) * .5, (a - b) * .5, (old_a - old_b) * .5))
    return values


def measures(result, ref):
    by = defaultdict(list)
    for (i, mu, d, parent), (j, original_mu, original_d, _) in zip(result, ref, strict=True):
        assert i == j
        h, axis = d[:, :10, :2], parent[:, :10, :2]
        target = original_d[:, :10, :2]
        slope = float((h * axis).sum() / axis.square().sum().clamp_min(1e-12))
        mu_error = mu[:, :10] - original_mu[:, :10]
        by[i].append({"d_gain_h": slope,
                      "d_h_to_anchor_rms": float((h - target).square().mean().sqrt()),
                      "mu_h_to_anchor_rms": float(mu_error[..., :2].square().mean().sqrt()),
                      "mu_zr_to_anchor_rms": float(mu_error[..., 2:6].square().mean().sqrt()),
                      "inverse_original_d_norm_ratio": float(parent.norm() / d.norm().clamp_min(1e-12))})
    return {k: median(median(x[k] for x in values) for values in by.values()) for k in next(iter(by.values()))[0]}


def screen(lane, lanes):
    full, anchor = load_weights("full"), load_weights("anchor")
    info = audit(full, anchor)
    if lane == 0:
        write(RUN / "inventory.json", info)
    lab = Lab5()
    try:
        bank = make_bank(lab, DEV)
        assign(lab, "F", anchor)
        ref = fields(lab, bank)
        with (RUN / f"screen_{lane}.jsonl").open("x", encoding="utf-8") as out:
            for mask in range(lane, ALL + 1, lanes):
                assign(lab, "F", weights(mask, full, anchor))
                row = {"mask": mask, **measures(fields(lab, bank), ref)}
                out.write(json.dumps(row, allow_nan=False) + "\n")
                out.flush()
                print("SCREEN", lane, mask, row["d_gain_h"], flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN / f"screen_{lane}_COMPLETE.json", {"lane": lane, "lanes": lanes})


def select(lanes):
    info = read(RUN / "inventory.json")
    rows = {}
    for lane in range(lanes):
        assert read(RUN / f"screen_{lane}_COMPLETE.json") == {"lane": lane, "lanes": lanes}
        for line in (RUN / f"screen_{lane}.jsonl").read_text().splitlines():
            r = json.loads(line)
            assert r["mask"] not in rows
            rows[r["mask"]] = r
    assert set(rows) == set(range(ALL + 1))
    errors = ("d_h_to_anchor_rms", "mu_h_to_anchor_rms", "mu_zr_to_anchor_rms")
    energies = info["delta_squared_norm"]
    for m, r in rows.items():
        r["rolled"] = [k for k in GROUPS if includes(m, k)]
        r["fraction_delta_energy_rolled"] = sum(energies[k] for k in r["rolled"]) / sum(energies.values())
        r["normalized_error"] = {k: r[k] / rows[0][k] for k in errors}
        r["joint_error"] = sum(r["normalized_error"].values()) / len(errors)
    available = [r for m, r in rows.items() if 0 < m < ALL]
    closest = min(available, key=lambda r: (r["joint_error"], r["fraction_delta_energy_rolled"], r["mask"]))
    remainder = [r for r in available if r["mask"] != closest["mask"]]
    eligible = [r for r in remainder if max(r["normalized_error"].values()) <= .35]
    less_rollback = min(eligible, key=lambda r: (r["fraction_delta_energy_rolled"], r["joint_error"], r["mask"])) if eligible else min(
        remainder, key=lambda r: (max(r["normalized_error"].values()), r["joint_error"], r["mask"]))
    candidates = {"closest": closest, "less_rollback": less_rollback}
    remainder = [r for r in available if r["mask"] not in {x["mask"] for x in candidates.values()}]
    candidates["d_only_selection"] = min(remainder, key=lambda r: (abs(r["d_gain_h"] - rows[ALL]["d_gain_h"]),
                                          r["fraction_delta_energy_rolled"], r["mask"]))
    candidates["cross_mlp"] = rows[(1 << GROUPS.index("kv")) | (1 << GROUPS.index("cross_other")) |
                                  (1 << GROUPS.index("mlp"))]
    # Contrast the other hybrids with one that preserves the largest realized update block.
    largest = max(GROUPS, key=energies.get)
    candidates["keep_mlp"] = rows[ALL ^ (1 << GROUPS.index(largest))]
    write(RUN / "selection.json", {"candidates": candidates, "baseline": rows[0], "anchor": rows[ALL],
                                   "gain": rows[0]["inverse_original_d_norm_ratio"], "all_rows": list(rows.values()),
                                   "selection_source": list(DEV), "less_rollback_threshold_pass": bool(eligible)})


def amplified(lab, obs, q, seed, chunk, gain):
    rt = lab.rt
    pref = {role: {g: rt.prefix(role, obs, INSTR[g]) for g in "AB"} for role in "FR"}

    def velocity(x, t):
        old = [rt.query("R", pref["R"][g], x, t).float() for g in "AB"]
        new = [rt.query("F", pref["F"][g], x, t).float() for g in "AB"]
        mu_old = (old[0] + old[1]) * .5
        mu_new = (new[0] + new[1]) * .5
        return (mu_old + gain * (new["AB".index(q)] - mu_new)).to(rt.vdtype)

    return rt.integrate(velocity, rt.noise(seed, chunk), 10)[..., :7]


def run_episode(lab, rec, q, transform, base):
    native = lab.native
    start = len(rec["controls"])

    def patched(role, text, obs, seed, chunk):
        if transform is not None and chunk < start + WINDOW:
            return transform(obs, q, seed, chunk)
        return native(base, text, obs, seed, chunk)

    lab.native = patched
    try:
        return measure_episode(lab, rec, q)
    finally:
        lab.native = native


def evaluate(lane, lanes):
    locked = read(RUN / "selection.json")
    full, anchor = load_weights("full"), load_weights("anchor")
    bank = read(K19 / "holdout.json")["selected"]
    assert [r["init"] for r in bank] == list(range(200, 220))
    lab = Lab5()
    try:
        assign(lab, "F", full)
        chosen = bank[lane]
        check = measure_episode(lab, chosen, "B")
        ref_path = K24 / f"native_{(chosen['init'] - 200) % 4}.jsonl"
        ref = next(r for r in map(json.loads, ref_path.read_text().splitlines())
                   if r["arm"] == "full" and r["init"] == chosen["init"] and r["goal"] == "B")
        assert check == {k: v for k, v in ref.items() if k != "arm"}
        with (RUN / f"eval_{lane}.jsonl").open("x", encoding="utf-8") as out:
            arm_plan = {"k15_calibrated": "AB", "k15_original_gain": "AB",
                        **dict.fromkeys(locked["candidates"], "ABN"),
                        **dict.fromkeys(("best_mu", "best_d", "best_mu_d"), "AB"),
                        **dict.fromkeys(("reverse_mu", "reverse_d"), "B")}
            for arm, qs in arm_plan.items():
                if arm in locked["candidates"]:
                    assign(lab, "F", weights(locked["candidates"][arm]["mask"], full, anchor))
                elif arm in PATCHES:
                    assign(lab, "F", full)
                    assign(lab, "R", weights(locked["candidates"]["closest"]["mask"], full, anchor))
                for rec in bank[lane::lanes]:
                    for q in qs:
                        transform, base = None, "F"
                        if arm.startswith("k15"):
                            gain = locked["gain"] if arm.endswith("calibrated") else 2.959
                            transform = lambda obs, g, seed, chunk: amplified(lab, obs, g, seed, chunk, gain)
                        elif arm in PATCHES:
                            spec = PATCHES[arm]
                            base = spec[0]
                            transform = lambda obs, g, seed, chunk: mix(lab, obs, g, seed, chunk, spec)
                        row = {"arm": arm, **run_episode(lab, rec, q, transform, base)}
                        out.write(json.dumps(row, allow_nan=False) + "\n")
                        out.flush()
                        print("EVAL", lane, arm, rec["init"], q, row["first"], flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN / f"eval_{lane}_COMPLETE.json", {"lane": lane, "lanes": lanes, "baseline_identity": chosen["init"]})


def validation():
    reference_dir = RUN.parent / "k25_deep_d_20261118_01"
    sys.path.insert(0, str(reference_dir))
    from deep_selection import fixed_bank, responses

    locked = read(RUN / "selection.json")
    full, anchor = load_weights("full"), load_weights("anchor")
    lab = Lab5()
    try:
        bank, phases = validation_bank(lab)
        holdout = read(K19 / "holdout.json")["selected"]
        fixed = fixed_bank(lab, holdout)
        choices = {"full": 0, "anchor": ALL, **{k: r["mask"] for k, r in locked["candidates"].items()}}
        for arm, mask in choices.items():
            assign(lab, "F", weights(mask, full, anchor))
            write(RUN / f"demo_{arm}.json", validation_fields(lab, bank, phases))
            result = responses(lab, fixed, True)
            if arm in ("full", "anchor"):
                previous = reference_dir / f"probe_{arm}.json"
                assert result == read(previous), arm
            write(RUN / f"held_probe_{arm}.json", result)
            print("VALIDATION", arm, flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN / "validation_COMPLETE.json", {"arms": list(choices)})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("screen", "select", "eval", "validation"))
    p.add_argument("--lane", type=int, default=0)
    p.add_argument("--lanes", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.mode == "screen":
        screen(a.lane, a.lanes)
    elif a.mode == "select":
        select(a.lanes)
    elif a.mode == "eval":
        evaluate(a.lane, a.lanes)
    else:
        validation()
