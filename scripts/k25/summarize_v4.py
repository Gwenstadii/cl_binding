"""Verify the upstream query experiment and summarize matched Right trajectories."""

import argparse
import json
from pathlib import Path
from statistics import median

from summarize_v3 import FRESH, IDS, add_refs, aggregate, compare, endpoint, lines, read, rf_margin

NATIVE = ("first_rollback", "second_rollback", "joint_rollback", "local_rollback", "joint_add")
SWAPS = ("h_from_joint", "h_from_local", "h_reverse_joint", "joint_add_h_fix", "h_anchor")
PARTS = ("query_mu", "query_d")
WINDOW = ("mlp_window", "h_from_mlp", "mlp_window_h_reverse", "h_from_time")
DEV = sorted([20, 23, 26, 29, 32, 35, 38, 39])


def trials(directory, tag, ids, lanes, plan):
    rows = {}
    for lane in range(lanes):
        marker = read(directory / f"{tag}_{lane}_COMPLETE.json")
        assert marker["lane"] == lane and marker["lanes"] == lanes
        gate = marker["gate"]
        assert gate["init"] == ids[lane] and gate["same_query_is_native_bitwise"]
        assert gate["action_output_cannot_change_fixed_query"]
        if tag == "output":
            assert gate["two_output_parts_reconstruct_query_transfer"]
        expected = {(arm, i) for i in ids[lane::lanes] for arm in plan}
        found = lines(directory / f"{tag}_{lane}.jsonl")
        assert len(found) == len(expected)
        for r in found:
            key = r["arm"], r["init"]
            assert r["goal"] == "B" and key in expected and key not in rows
            rows[key] = r
    return rows


def measures(data, ids):
    a = aggregate(data, ids)
    keys = [k for k in data[0] if k not in ("init", "route", "step", "by_layer", "h_rms_both_prompts",
                                            "entry_equals_weight_swap_bitwise")]
    result = {k: a(lambda r: r[k]) for k in keys}
    if "by_layer" in data[0]:
        result["by_layer"] = [a(lambda r: r["by_layer"][i]) for i in range(8)]
    if "h_rms_both_prompts" in data[0]:
        result["query_input_first_effect_on_layer"] = 2 * next(i for i in range(8) if any(
            val[i] != 0 for r in data for val in r["h_rms_both_prompts"])) + 1
    return result


def summarize_held(directory, suite, ids):
    assert read(directory / f"held_{suite}_COMPLETE.json") == {"suite": suite, "n": len(ids)}
    data = read(directory / f"held_{suite}.json")
    assert set(data) == set(NATIVE)
    held = {arm: measures(rows, ids) for arm, rows in data.items()}
    for arm, rows in data.items():
        a = aggregate(rows, ids)
        held[arm]["native_gain_change"] = a(lambda r: r["native_gain"] - r["q_gain"] + r["q_gain_change"])
        held[arm]["native_minus_query_only"] = a(lambda r: r["native_gain"] - r["q_gain"])
    reading = {a: measures(read(directory / f"reading_{suite}_{a}.json"), ids)
               for a in ("h_anchor", "h_from_joint", "h_from_local")}
    for key, arm in (("joint_rollback", "h_from_joint"), ("local_rollback", "h_from_local")):
        assert reading[arm]["h_gain"] == held[key]["q_gain"]
    return {"fixed_query": held, "attention": reading}


def summarize(directory, parent):
    main = trials(directory, "eval", IDS+FRESH, 8, (*NATIVE, *SWAPS))
    output = trials(directory, "output", FRESH, 2, PARTS)
    windows = trials(directory, "window", IDS+FRESH, 8, WINDOW)
    rows = {**main, **output, **windows}
    result = {"new_episodes": len(rows), "coarse": read(directory / "coarse_selection.json"),
              "selection": read(directory / "selection.json"), "cohorts": {}}
    add_refs(rows, parent)
    v3 = parent / "2026-11-20/k25v3"
    prev_h_anchor = {}
    for lane in range(2):
        for r in lines(v3 / f"q_eval_{lane}.jsonl"):
            if r["arm"] == "p_query_hidden":
                prev_h_anchor[r["init"]] = r
    for lane in range(8):
        for r in lines(v3 / f"v_eval_{lane}.jsonl"):
            if r["arm"] == "native_anchor":
                rows["native_anchor", r["init"]] = r
    result["new_h_anchor_matches_previous_trajectory"] = sum(
        rows["h_anchor", i] == {**prev_h_anchor[i], "arm": "h_anchor"} for i in FRESH)
    parts = read(directory / "local_entry.json")
    bands = read(directory / "down_depth.json")
    marker = read(directory / "location_COMPLETE.json")
    assert marker["parts"] == result["selection"]["selected_parts"]
    assert marker["bands"] == len(bands) and marker["drafts"] == len(parts["time_out"]) == len(parts["down_proj"])
    assert all(r["entry_equals_weight_swap_bitwise"] for group in parts.values() for r in group)
    result["location"] = {name: measures(data, DEV) for name, data in parts.items()}
    result["down_depth"] = {name: measures(r["rows"], DEV) for name, r in bands.items()}
    last_layer = bands["layer_15"]["rows"]
    result["down_depth"]["layer_15_native_only"] = median(
        median(r["native_gain"] for r in last_layer if r["init"] == i)
        - result["coarse"]["baseline_rows"][str(i)]["native_gain"] for i in DEV)
    comparisons = {a: "baseline" for a in (*NATIVE, *SWAPS, *PARTS, *WINDOW)}
    comparisons.update(h_reverse_joint="joint_rollback", joint_add="native_anchor", joint_add_h_fix="joint_add",
                       mlp_window_h_reverse="mlp_window")
    for suite, ids in (("old", IDS), ("fresh", FRESH)):
        arms = (*NATIVE, *SWAPS, *WINDOW)
        if suite == "fresh":
            arms = (*arms, *PARTS)
        rel = {(a, comparisons[a]) for a in arms}
        rel.update({("h_from_joint", "joint_rollback"), ("h_from_local", "local_rollback"),
                    ("h_from_joint", "h_anchor"), ("h_from_mlp", "mlp_window"),
                    ("mlp_window", "second_rollback"), ("mlp_window_h_reverse", "baseline")})
        if suite == "fresh":
            rel.update({("query_mu", "h_from_joint"), ("query_d", "h_from_joint")})
        scores = summarize_held(directory, suite, ids)
        scores["mlp_window_and_full_identical_first_displacement"] = sum(
            rows["mlp_window", i]["displacement_xy_m"] == rows["second_rollback", i]["displacement_xy_m"] for i in ids)
        scores["outcomes"] = {a: {**endpoint([rows[a, i] for i in ids], "B"),
                                    "initial_right_minus_front_median_mm": median(rf_margin(rows[a, i]) for i in ids)}
                              for a in (*arms, "baseline", "native_anchor")}
        scores["matched"] = {f"{a}_vs_{b}": compare(rows, ids, b, a) for a, b in sorted(rel)}
        result["cohorts"][suite] = scores
    (directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("previous_results", type=Path)
    a = p.parse_args()
    summarize(a.directory, a.previous_results)
