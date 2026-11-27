"""Verify K25V3-V coverage and summarize matched dose and orientation comparisons."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import dist
from pathlib import Path
from statistics import median

from summarize import IDS, TARGET, endpoint, lines, paired, read
from summarize_v3 import FRESH, V_ARMS, WEIGHT_ARMS, add_refs, aggregate, collect, compare, rf_margin

DOSES = ("baseline", "natural", "dose_2", "dose_3", "dose_4", "amplified")


def locked_ids(directory):
    path = directory / "candidates.json"
    candidates = read(path)["records"]
    holdout = read(directory / "holdout.json")
    assert holdout["candidate_sha256"] == sha256(path.read_bytes()).hexdigest()
    screened = sorted((r for i in range(4) for r in read(directory / f"screen_{i}.json")), key=lambda r: r["init"])
    assert [r["init"] for r in screened] == [r["init"] for r in candidates]
    assert all(row["min_distance_to_previous_k25"] > 1e-6 for row in screened)
    qualified = [r for r in screened if r["boundary"]]
    assert holdout["candidates"] == len(candidates) and holdout["eligible"] == len(qualified)
    selected = holdout["selected"]
    assert len(selected) == 20 and selected == qualified[:20]
    assert len({r["handoff_signature"]["sim"] for r in selected}) == 20
    return [r["init"] for r in selected]


def evaluated(directory, suite, ids, lanes, plan):
    trials, gates = {}, []
    for i in range(lanes):
        marker = read(directory / f"{suite}_{i}_COMPLETE.json")
        assert (marker["suite"], marker["lane"], marker["lanes"]) == (suite, i, lanes)
        gate = marker["gate"]
        assert gate["init"] == ids[i]
        assert all(gate[k] for k in ("identity_and_v3_endpoints_bitwise_for_both_prompts",
                                     "rotated_xy_same_norm_per_slot_and_perpendicular",
                                     "xy_and_remaining_reconstruct", "fp64_pair_mean_unchanged"))
        gates.append(gate)
        expected = {(q, arm, j) for q, arms in plan.items() for arm in arms for j in ids[i::lanes]}
        for r in lines(directory / f"{suite}_{i}.jsonl"):
            key = (r["goal"], r["arm"], r["init"])
            assert key not in trials and key in expected
            expected.remove(key)
            trials[key] = r
        assert not expected
    return trials, gates


def historical(parent):
    v3 = parent / "2026-11-20/k25v3"
    rows = collect(v3, "v_eval", 8, (*V_ARMS, *WEIGHT_ARMS))
    add_refs(rows, parent)
    original = {"baseline": "baseline", "natural": "v_d", "amplified": "v_amplified_d",
                "whole_v": "v_whole", "reverse_natural_d": "v_reverse_d", "v_rolled": "v_rolled",
                "anchor_d": "anchor_d", "native_anchor": "native_anchor"}
    return {("B", a, i): rows[b, i] for a, b in original.items() for i in IDS+FRESH}


def doses(rows, ids, scale):
    profiles = []
    for i in ids:
        if rows["baseline", i]["first"] != ["front"]:
            continue
        sequence = [rows[a, i]["first"] for a in DOSES]
        good = [label == ["right"] for label in sequence]
        first = next((j for j, present in enumerate(good) if present), None)
        stays = first is not None and all(good[first:])
        profiles.append({"init": i, "baseline_margin_mm": rf_margin(rows["baseline", i]),
                         "first_tested_dose": None if first is None else (0, 1, 2, 3, 4, scale)[first],
                         "stays_right_at_higher_tested_doses": stays, "first_targets_by_dose": sequence})
    return profiles


def frozen_fields(directory, suite, ids, residual):
    rows = read(directory / f"fixed_{suite}.json")
    assert read(directory / f"fixed_{suite}_COMPLETE.json") == {"suite": suite, "drafts": len(rows)}
    assert len({(r["init"], r["route"], r["step"]) for r in rows}) == len(rows)
    statistic = aggregate(rows, ids)
    metrics = {k: statistic(lambda r: r[k]) for k in rows[0] if k not in {"init", "route", "step"}}
    return {**metrics, "residual_matched_gain": residual * metrics["base_gain"],
            "fraction_of_full_gain": metrics["v_gain"]/metrics["anchor_gain"],
            "share_of_native_gain_from_shared_v": metrics["shared_gain"]/metrics["v_gain"],
            "share_of_native_gain_from_conditional_v": metrics["conditional_gain"]/metrics["v_gain"]}


def goal_progress(row, goal):
    start = row["position_xy_m"]
    end = [x+dx for x, dx in zip(start, row["displacement_xy_m"], strict=True)]
    center = row["centers_xy_m"][goal]
    return 1000 * (dist(start, center) - dist(end, center))


def right_report(rows, ids, scale):
    rows = {(arm, i): r for (q, arm, i), r in rows.items() if q == "B" and i in ids}
    arms = sorted({a for a, _ in rows})
    scores = {}
    for arm in arms:
        selected = [rows[arm, i] for i in ids]
        scores[arm] = {**endpoint(selected, "B"), "rf_median_mm": median(map(rf_margin, selected))}
    requested = {(a, "baseline") for a in arms if a != "baseline"}
    requested.update(("amplified", a) for a in ("natural", "reverse", "horizontal", "remainder", "rotate_ccw",
                                                   "rotate_cw", "residual_d", "native_extrap", "extrap_d",
                                                   "shared_source", "conditional_source") if a in scores)
    if "extrap_d" in scores:
        requested.update({("extrap_d", "native_extrap"), ("native_extrap", "native_v"),
                          ("shared_source", "conditional_source"), ("natural", "native_v")})
    if "reverse_natural_d" in scores:
        requested.add(("reverse_natural_d", "v_rolled"))
    from_front = {a: [i for i in ids if rows["baseline", i]["first"] == ["front"] and rows[a, i]["first"] == ["right"]]
                  for a in arms}
    movement = {a: {g: median(goal_progress(rows[a, i], g) - goal_progress(rows["baseline", i], g) for i in ids)
                    for g in ("right", "front")} for a in arms}
    matched = {}
    for a, b in sorted(requested):
        terminal = Counter(int(rows[a, i]["terminal"]["right"]) - int(rows[b, i]["terminal"]["right"]) for i in ids)
        matched[f"{a}_vs_{b}"] = {**compare(rows, ids, b, a), "terminal_gains": terminal[1],
                                   "terminal_losses": terminal[-1]}
    return {"outcomes": scores, "matched": matched,
            "front_to_right_ids": from_front, "goal_progress_change_mm": movement,
            "original_front_dose_profiles": doses(rows, ids, scale)}


def left_report(trials, ids, arms):
    rows = {a: [trials["A", a, i] for i in ids] for a in arms}
    for trajectories in rows.values():
        for r in trajectories:
            opposite = trials["B", r["arm"], r["init"]]
            assert (r["position_xy_m"], r["centers_xy_m"]) == (opposite["position_xy_m"], opposite["centers_xy_m"])
    tests = {}
    for a, b in (("amplified", "baseline"), ("reverse", "baseline"), ("amplified", "reverse")):
        result = paired(rows[b], rows[a], "left")
        result["left_margin_change_median_mm"] = result.pop("right_margin_change_median_mm")
        tests[f"{a}_vs_{b}"] = result
    return {"outcomes": {a: endpoint(selected, "A") for a, selected in rows.items()}, "matched": tests,
            "both_instructions_first_correct": {a: sum(all(trials[q, a, i]["first"] == [TARGET[q]] for q in "AB")
                                                           for i in ids) for a in arms}}


def summarize(directory, parent):
    plan = read(directory / "plan.json")
    scale = plan["scale"]
    assert scale == read(parent / "2026-11-20/k25v3/v_calibration.json")["scale"]
    third_ids = locked_ids(directory)
    legacy, checks = evaluated(directory, "legacy", IDS+FRESH, 8, {"B": plan["core"]})
    third, third_checks = evaluated(directory, "third", third_ids, 10,
                                    {"B": plan["fresh_right"], "A": plan["fresh_left"]})
    trials = {**historical(parent), **legacy, **third}
    right = {name: right_report(trials, ids, scale) for name, ids in (("original", IDS), ("second", FRESH),
                                                                      ("third", third_ids))}
    assert sum(right[c]["matched"]["amplified_vs_baseline"]["front_to_right"] for c in ("original", "second")) == 27
    pooled = {}
    for a in (*DOSES, "horizontal", "remainder", "reverse", "rotate_ccw", "rotate_cw"):
        counter = Counter()
        for block in right.values():
            counter.update(block["outcomes"][a]["first"])
        matched = [block["matched"][f"{a}_vs_baseline"] for block in right.values()] if a != "baseline" else []
        pooled[a] = {"n": sum(counter.values()), "first": dict(counter),
                     "terminal": sum(block["outcomes"][a]["terminal"] for block in right.values()),
                     "front_to_right": sum(m["front_to_right"] for m in matched),
                     "base_front": counter["front"] if a == "baseline" else sum(m["base_front"] for m in matched),
                     "positive_first_segment": sum(m["right_minus_front_positive"] for m in matched)}
    profiles = [r for block in right.values() for r in block["original_front_dose_profiles"]]
    ordered = sorted(profiles, key=lambda r: r["baseline_margin_mm"])
    groups = (ordered[:len(ordered)//2], ordered[len(ordered)//2:])
    by_margin = {label: {"n": len(group), "baseline_margin_mm": median(r["baseline_margin_mm"] for r in group),
                         "minimum_tested_dose_counts": dict(Counter(str(r["first_tested_dose"]) for r in group))}
                 for label, group in zip(("more_front_leaning", "less_front_leaning"), groups, strict=True)}
    result = {"new_episodes": len(legacy)+len(third), "new_initial_states": third_ids, "plan": plan,
              "max_mean_leak_rms_at_cast_check": max(g["cast_pair_mean_delta_rms"] for g in checks+third_checks),
              "right": right, "left_third": left_report(trials, third_ids, plan["fresh_left"]), "pooled_right": pooled,
              "original_front_first_tested_recovery": dict(Counter(str(r["first_tested_dose"]) for r in profiles)),
              "original_front_regressed_at_higher_tested_dose": [r["init"] for r in profiles if
                  r["first_tested_dose"] is not None and not r["stays_right_at_higher_tested_doses"]],
              "original_front_by_baseline_margin": by_margin,
              "fixed": {s: frozen_fields(directory, s, ids, plan["residual_scale"]) for s, ids in
                        (("dev", plan["dev"]), ("third", third_ids))}}
    (directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("previous_results", type=Path)
    args = p.parse_args()
    summarize(args.directory, args.previous_results)
