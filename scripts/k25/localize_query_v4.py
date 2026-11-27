"""Refine the upstream query sources and close the intervention chain."""

import argparse
from collections import defaultdict
import json
from statistics import mean, median

import torch

from query_upstream_v4 import (ALL, CROSS, DEV, GROUPS, INSTR, RUN, WINDOW, Fixed, Lab5, Queries,
                               assign, checkpoint_files, d_h, fixed_bank, horizontal,
                               make_bank, measure_episode, ratio, read, records, rolled, same_role_gate, write)
from relay_diagnostics import ObserveAttention, in_gqa

ARM_NAMES = ("first_rollback", "second_rollback", "joint_rollback", "local_rollback",
             "h_from_joint", "h_from_local", "h_reverse_joint", "joint_add", "joint_add_h_fix", "h_anchor")


def input_cases(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["init"]].append(r)
    assert sorted(by) == sorted(DEV) and all(len(g) == 6 for g in by.values())
    keys = [k for k in rows[0] if k not in {"init", "route", "step", "by_layer"}]
    return {i: {k: median(r[k] for r in g) for k in keys} for i, g in by.items()}


def layer_of(name):
    return int(name.split(".layers.")[1].split(".")[0])


def subdivisions(family, names):
    result = {
        f"{family}__{part}": subset
        for part, subset in split_projection(names).items()
    }
    if all(".layers." in n for n in names):
        for start in range(0, 16, 4):
            part = [n for n in names if start <= layer_of(n) < start + 4]
            if part:
                result[f"{family}__layers_{start:02}_{start+3:02}"] = part
    return result


def split_projection(names):
    groups = defaultdict(list)
    for n in names:
        groups[n.split(".")[-2]].append(n)
    return dict(groups)


def choose_coarse(lanes):
    groups = read(RUN / "inventory.json")
    all_masks = {}
    for lane in range(lanes):
        assert read(RUN / f"coarse_{lane}_COMPLETE.json") == {"lane": lane, "lanes": lanes}
        for line in (RUN / f"coarse_{lane}.jsonl").read_text().splitlines():
            r = json.loads(line)
            assert r["mask"] not in all_masks
            all_masks[r["mask"]] = input_cases(r["rows"])
    assert sorted(all_masks) == list(range(ALL+1))
    metrics = {}
    for j, family in enumerate(GROUPS):
        bit = 1 << j
        contexts = [m for m in range(ALL+1) if not m & bit]
        for key in ("query_projection", "query_diff_projection", "query_mean_projection", "q_gain_change"):
            vote = median(mean(all_masks[m | bit][i][key] - all_masks[m][i][key] for m in contexts) for i in DEV)
            alone = median(all_masks[bit][i][key] - all_masks[0][i][key] for i in DEV)
            add = median(all_masks[ALL ^ bit][i][key] - all_masks[ALL][i][key] for i in DEV)
            metrics.setdefault(family, {})[key] = {"average_marginal": vote, "rollback_from_full": alone,
                                                   "add_to_anchor": add}
    order = sorted(GROUPS, key=lambda g: metrics[g]["q_gain_change"]["average_marginal"], reverse=True)
    h_order = sorted(GROUPS, key=lambda g: metrics[g]["query_projection"]["average_marginal"], reverse=True)
    pair = order[:2]
    joint = sorted(n for g in pair for n in groups[g])
    best_mask = sum(1 << GROUPS.index(g) for g in pair)
    subs = {name: subset for g in dict.fromkeys((*pair, h_order[0]))
            for name, subset in subdivisions(g, groups[g]).items()}
    write(RUN / "coarse_selection.json", {"families": pair, "raw_query_order": h_order,
            "functional_order": order, "marginal": metrics, "joint": joint, "fine": subs,
            "q_anchor_change": median(all_masks[ALL][i]["q_gain_change"] for i in DEV),
            "baseline_rows": all_masks[0], "joint_rows": all_masks[best_mask]})
    print("COARSE_SELECTED", pair, "RAW", h_order, flush=True)


def measure_fine(lane, lanes):
    chosen = read(RUN / "coarse_selection.json")
    full, anchor = checkpoint_files()
    lab = Lab5()
    try:
        bank = make_bank(lab, DEV)
        fixed = Fixed(lab, bank, full, anchor)
        groups = read(RUN / "inventory.json")
        tests = list(chosen["fine"].items())
        with (RUN / f"fine_{lane}.jsonl").open("x", encoding="utf-8") as out:
            for name, sub in tests[lane::lanes]:
                family = name.split("__")[0]
                if family in chosen["families"]:
                    background = set(chosen["joint"])
                else:
                    background = set(groups[family])
                assert set(sub) <= background
                alone = fixed.measure(rolled(full, anchor, sub))
                rest = fixed.measure(rolled(full, anchor, background - set(sub)))
                row = {"part": name, "names": sub, "alone": alone, "leave_out": rest}
                out.write(json.dumps(row, allow_nan=False) + "\n")
                out.flush()
                print("FINE", lane, name, median(r["q_gain_change"] for r in alone), flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN / f"fine_{lane}_COMPLETE.json", {"lane": lane, "lanes": lanes})


def lock_fine(lanes):
    coarse = read(RUN / "coarse_selection.json")
    inv = read(RUN / "inventory.json")
    rank = {}
    for lane in range(lanes):
        assert read(RUN / f"fine_{lane}_COMPLETE.json") == {"lane": lane, "lanes": lanes}
        for line in (RUN / f"fine_{lane}.jsonl").read_text().splitlines():
            row = json.loads(line)
            name = row["part"]
            assert row["names"] == coarse["fine"][name] and name not in rank
            alone = input_cases(row["alone"])
            rest = input_cases(row["leave_out"])
            k = "q_gain_change"
            rank[name] = {"alone": median(alone[i][k] for i in DEV),
                          "query_projection_alone": median(alone[i]["query_projection"] for i in DEV),
                          "leave_out_gain": median(rest[i][k] for i in DEV),
                          "tensors": len(row["names"])}
            if name.split("__")[0] in coarse["families"]:
                surrounding = median(coarse["joint_rows"][str(i)][k] - rest[i][k] for i in DEV)
                rank[name].update(removal_from_joint=surrounding, selection_score=(rank[name]["alone"]+surrounding)*.5)
    assert rank.keys() == coarse["fine"].keys()
    chosen = [max((n for n in rank if n.startswith(g+"__")),
                  key=lambda n: (rank[n]["selection_score"], -rank[n]["tensors"])) for g in coarse["families"]]
    exact = sorted(n for sub in chosen for n in coarse["fine"][sub])
    write(RUN / "selection.json", {"families": coarse["families"], "selected_parts": chosen,
                                   "first": inv[coarse["families"][0]], "second": inv[coarse["families"][1]],
                                   "joint": coarse["joint"], "local": exact, "fine_metrics": rank})
    print("FINE_SELECTED", chosen, len(exact), flush=True)


def arm_states(full, anchor):
    sel = read(RUN / "selection.json")
    sources = {"full": full, "anchor": anchor,
               **{g: rolled(full, anchor, sel[g]) for g in ("first", "second", "joint", "local")},
               "late_on_anchor": rolled(anchor, full, sel["joint"])}
    return {
        "first_rollback": (sources["first"], None),
        "second_rollback": (sources["second"], None),
        "joint_rollback": (sources["joint"], None),
        "local_rollback": (sources["local"], None),
        "h_from_joint": (full, sources["joint"]),
        "h_from_local": (full, sources["local"]),
        "h_reverse_joint": (sources["joint"], full),
        "joint_add": (sources["late_on_anchor"], None),
        "joint_add_h_fix": (sources["late_on_anchor"], anchor),
        "h_anchor": (full, anchor),
    }


def window_states(full, anchor):
    selection = read(RUN / "selection.json")
    assert selection["families"] == ["action_time", "mlp"]
    time = rolled(full, anchor, selection["first"])
    mlp = rolled(full, anchor, selection["second"])
    return {
        "mlp_window": (full, mlp, "R", "F"),
        "h_from_mlp": (full, mlp, "query", "F"),
        "mlp_window_h_reverse": (mlp, full, "query", "R"),
        "h_from_time": (full, time, "query", "F"),
    }


def evaluate(lane, lanes, mode):
    rows = records()[lane::lanes]
    full, anchor = checkpoint_files()
    lab = Lab5()
    native = lab.native
    sites = Queries(lab)
    try:
        gate = same_role_gate(lab, sites, rows[0], full, anchor, read(RUN / "inventory.json"))
        print("GATE", gate, flush=True)
        if mode == "window":
            plan = window_states(full, anchor)
        else:
            states = arm_states(full, anchor)
            assert list(states) == list(ARM_NAMES)
            plan = {name: (base, donor, "F" if donor is None else "query", "F")
                    for name, (base, donor) in states.items()}
        with (RUN / f"{mode}_{lane}.jsonl").open("x", encoding="utf-8") as out:
            for arm, (base, donor, inside, after) in plan.items():
                assign(lab, "F", base)
                if donor is not None:
                    assign(lab, "R", donor)
                for rec in rows:
                    start = len(rec["controls"])

                    def sample(role, text, obs, seed, chunk):
                        use = inside if start <= chunk < start + WINDOW else after
                        if use == "query":
                            assert text == INSTR["B"]
                            return sites.chunk(obs, "B", seed, chunk)
                        return native(use, text, obs, seed, chunk)

                    lab.native = sample
                    result = {"arm": arm, **measure_episode(lab, rec, "B")}
                    out.write(json.dumps(result, allow_nan=False) + "\n")
                    out.flush()
                    print(mode.upper(), lane, arm, rec["init"], result["first"], flush=True)
    finally:
        lab.native = native
        lab.ctx.env.close()
    write(RUN / f"{mode}_{lane}_COMPLETE.json", {"lane": lane, "lanes": lanes, "gate": gate})


def attention_coordinates(log):
    return torch.stack([log[i][-1][:, :, :10].float() for i in CROSS])


def fixed_held(suite):
    rows = records()[:20] if suite == "old" else records()[20:]
    full, anchor = checkpoint_files()
    lab = Lab5()
    sites = Queries(lab)
    traces = ObserveAttention(lab.rt.models["F"].model.vlm_with_expert)
    try:
        bank = fixed_bank(lab, rows)
        fixed = Fixed(lab, bank, full, anchor)
        plan = arm_states(full, anchor)
        probes = {}
        leading = {}
        for a, (base, donor) in plan.items():
            if donor is None:
                probes[a] = fixed.measure(base)
            elif a in ("h_from_joint", "h_from_local", "h_anchor"):
                leading[a] = donor
        write(RUN / f"held_{suite}.json", probes)

        # Use the exact native K and V of the receiving model. Each intervention
        # changes only the inputs to its Q projection, at the same draft and time.
        last = None
        pending = {}
        for a in ("h_anchor", "h_from_joint", "h_from_local"):
            assign(lab, "R", leading[a])
            result = []
            last = None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for entry, reference in zip(bank, fixed.references, strict=True):
                    i, route, step, obs, x, t, _, _ = entry
                    f, _, _, axis, _, _ = reference
                    if i != last:
                        p = {q: lab.rt.prefix("R", obs, INSTR[q]) for q in "AB"}
                        last = i
                    before, after, outputs = [], [], []
                    for q in "AB":
                        _, h = sites.capture("R", p[q], x, t)
                        if a == "h_anchor":
                            with traces.record(verify=False) as raw:
                                lab.rt.query("F", f[q], x, t)
                            before.append(attention_coordinates(raw))
                        with traces.record(verify=False) as fixed_log:
                            outputs.append(sites.replace(f[q], x, t, h))
                        after.append(attention_coordinates(fixed_log))
                    token = (i, route, step)
                    if a == "h_anchor":
                        pending[token] = (before[0], before[1], after[0] - before[0], after[1] - before[1])
                    a0, b0, da, db = pending[token]
                    d1, d2 = after[0]-a0, after[1]-b0
                    result.append({"init": i, "route": route, "step": step,
                                   "attention_projection": ratio(torch.stack((d1, d2)), torch.stack((da, db))),
                                   "attention_difference_projection": ratio(d1-d2, da-db),
                                   "h_gain": horizontal(d_h(*outputs), axis)})
            write(RUN / f"reading_{suite}_{a}.json", result)
            print("READING", suite, a, flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN / f"held_{suite}_COMPLETE.json", {"suite": suite, "n": len(rows)})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("choose", "fine", "select", "eval", "window", "held"))
    p.add_argument("--lane", type=int, default=0)
    p.add_argument("--lanes", type=int, default=4)
    p.add_argument("--suite", choices=("old", "fresh"), default="fresh")
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.mode == "choose":
        choose_coarse(a.lanes)
    elif a.mode == "fine":
        measure_fine(a.lane, a.lanes)
    elif a.mode == "select":
        lock_fine(a.lanes)
    elif a.mode == "held":
        fixed_held(a.suite)
    else:
        evaluate(a.lane, a.lanes, a.mode)
