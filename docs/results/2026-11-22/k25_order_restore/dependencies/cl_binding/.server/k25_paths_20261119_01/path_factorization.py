"""Factor the trained K/V projections and the gated MLP on the K25 benchmark."""

import argparse
from itertools import combinations
import json
from pathlib import Path
from statistics import median
import sys

import torch

RUN = Path(__file__).resolve().parent
PREVIOUS = RUN.parent / "k25_deep_d_20261118_01"
COMBINED = RUN.parent / "k25_combined_rollback_20261119_01"
sys.path.insert(0, str(PREVIOUS))
from deep_selection import K19, Lab5, assign, check_baseline, fixed_bank, group, load_weights
from deep_selection import measure_episode, read, responses, write
from window_eval import make_bank

CROSS = tuple(range(1, 16, 2))
DEEP = (13, 15)
MLP = ("gate", "up", "down")
DEV = (20, 23, 26, 29, 32, 35, 38, 39)


def variants():
    cross = {"full": (), "cross_k": ("k",), "cross_v": ("v",), "cross_kv": ("k", "v"),
             "deep_k": ("k",), "deep_v": ("v",), "deep_kv": ("k", "v")}
    mlp = {"anchor": (), **{f"mlp_{'_'.join(types)}": types for n in range(1, 4)
                             for types in combinations(MLP, n)}}
    return {"cross": cross, "mlp": mlp}


def checkpoint_files():
    return load_weights("full"), load_weights("anchor")


def parameters_for(arm, full):
    family = "mlp" if arm == "anchor" or arm.startswith("mlp_") else "cross"
    parts = variants()[family][arm]
    if family == "cross":
        layers = DEEP if arm.startswith("deep_") else CROSS
        kinds = {"kv"}
    else:
        layers = range(16)
        kinds = {"mlp"}
    chosen = [n for n in full if group(n) in kinds
              and int(n.split(".layers.")[1].split(".")[0]) in layers
              and n.split(".")[-2].removesuffix("_proj") in parts]
    assert len(chosen) == len(layers) * len(parts), (arm, chosen)
    return family, chosen


def configuration(arm, full, anchor):
    family, chosen = parameters_for(arm, full)
    base, donor = (anchor, full) if family == "mlp" else (full, anchor)
    names = set(chosen)
    return {n: donor[n] if n in names else v for n, v in base.items()}


def run_probes(family, suite):
    arms = variants()[family]
    full, anchor = checkpoint_files()
    assert full.keys() == anchor.keys() and len(full) == 155
    lab = Lab5()
    try:
        bank = make_bank(lab, DEV) if suite == "dev" else fixed_bank(lab, cases())
        for arm in arms:
            assign(lab, "F", configuration(arm, full, anchor))
            rows = responses(lab, bank, True)
            previous = {"full": PREVIOUS / "probe_full.json", "anchor": PREVIOUS / "probe_anchor.json",
                        "deep_kv": PREVIOUS / "probe_deep_kv.json",
                        "mlp_gate_up_down": COMBINED / "held_probe_keep_mlp.json"}
            if suite == "held" and arm in previous:
                assert rows == read(previous[arm]), arm
            write(RUN / f"probe_{suite}_{arm}.json", rows)
            print("PROBE", family, suite, arm, flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN / f"probe_{suite}_{family}_COMPLETE.json", {
        "arms": list(arms), "tensors": {a: parameters_for(a, full)[1] for a in arms}})


def gains(rows):
    by = {}
    for r in rows:
        by.setdefault(r["init"], []).append(r["10"]["contrast_by_channel"]["H"]["gain"])
    return {i: median(values) for i, values in by.items()}


def screen_choice():
    dev = {arm: gains(read(RUN / f"probe_dev_{arm}.json"))
           for family in variants().values() for arm in family}
    assert all(sorted(r) == sorted(DEV) for r in dev.values())
    # A projection's effect is averaged over presence/absence of its counterpart.
    x = {}
    for part, other in (("k", "v"), ("v", "k")):
        x[part] = median(((dev[f"cross_{part}"][i] - dev["full"][i])
                          + (dev["cross_kv"][i] - dev[f"cross_{other}"][i])) / 2 for i in DEV)
    # Average the moved projection's paired effect over the other two settings.
    m = {}
    for part in MLP:
        rest = [n for n in MLP if n != part]
        masks = [()] + [(n,) for n in rest] + [tuple(rest)]

        def name(parts):
            return "mlp_" + "_".join(n for n in MLP if n in parts) if parts else "anchor"

        m[part] = median(sum(dev[name(others)][i] - dev[name((*others, part))][i]
                             for others in masks) / len(masks) for i in DEV)
    best_x = max(x, key=x.get)
    best_m = max(m, key=m.get)
    write(RUN / "selected.json", {"cross": best_x, "mlp": best_m, "cross_effect": x,
                                  "mlp_effect": m, "dev": list(DEV)})
    print("SELECT", best_x, best_m, x, m, flush=True)


def cases():
    records = read(K19 / "holdout.json")["selected"]
    assert [r["init"] for r in records] == list(range(200, 220))
    return records


def native_plan(chosen):
    protected_mlp = "mlp_" + "_".join(x for x in MLP if x != chosen["mlp"])
    return {"cross_k": "AB", "cross_v": "AB", "cross_kv": "AB", "deep_k": "B", "deep_v": "B",
            **{f"mlp_{x}": "AB" if x == chosen["mlp"] else "B" for x in MLP},
            protected_mlp: "AB"}


def evaluate(lane, lanes):
    chosen = read(RUN / "selected.json")
    full, anchor = checkpoint_files()
    records = cases()[lane::lanes]
    lab = Lab5()
    try:
        assign(lab, "F", full)
        gate = check_baseline(lab, records, lane)
        plan = native_plan(chosen)
        with (RUN / f"native_{lane}.jsonl").open("x", encoding="utf-8") as out:
            for arm, goals in plan.items():
                assign(lab, "F", configuration(arm, full, anchor))
                for rec in records:
                    for goal in goals:
                        row = {"arm": arm, **measure_episode(lab, rec, goal)}
                        out.write(json.dumps(row, allow_nan=False) + "\n")
                        out.flush()
                        print("EVAL", lane, arm, rec["init"], goal, row["first"], flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN / f"native_{lane}_COMPLETE.json", {"lane": lane, "lanes": lanes, "gate": gate, "plan": plan})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("probe", "select", "native"))
    parser.add_argument("--family", choices=("cross", "mlp"))
    parser.add_argument("--suite", choices=("dev", "held"))
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "probe":
        run_probes(args.family, args.suite)
    elif args.mode == "select":
        screen_choice()
    else:
        evaluate(args.lane, args.lanes)
