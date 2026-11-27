"""Verify K25V3 and count matched Front substitutions, not only correct arrivals."""

import argparse
from collections import defaultdict
import json
from math import dist
from pathlib import Path
from statistics import median

from summarize import IDS, endpoint, lines, paired, paired_interval, read

FRESH = list(range(400, 420))
V_ARMS = ("v_mu", "v_d", "v_reverse_mu", "v_reverse_d", "v_amplified_d", "anchor_d", "anchor_output",
          "v_text", "v_other_positions")
WEIGHT_ARMS = ("native_anchor", "native_anchor_add_v")
P_ARMS = ("p_restore", "p_query", "p_key", "pv_restore", "p_reverse", "pv_reverse")
Q_ARMS = ("p_query_hidden", "q_weights")


def collect(directory, tag, lanes, arms):
    ids = FRESH if tag == "q_eval" else IDS + FRESH
    rows = {}
    for lane in range(lanes):
        marker = read(directory / f"{tag}_{lane}_COMPLETE.json")
        assert marker["lane"] == lane and marker["lanes"] == lanes
        if tag == "v_eval":
            gate = marker["gates"]
            assert all(gate[t] for t in ("output_equals_native_v_whole", "complementary_v_spans"))
        else:
            gate = marker["gate"]
            assert gate["same_weights_native_bitwise_for"] == ["p", "q", "k", "pv", "h"]
        assert gate["init"] == ids[lane]
        expected = {(a, i) for a in arms for i in ids[lane::lanes] if a not in WEIGHT_ARMS or i in FRESH}
        actual = lines(directory / f"{tag}_{lane}.jsonl")
        assert len(actual) == len(expected)
        for r in actual:
            key = r["arm"], r["init"]
            assert r["goal"] == "B" and key in expected and key not in rows
            rows[key] = r
    return rows


def add_refs(rows, parent):
    k24 = parent / "2026-11-17/k24_final_window"
    old = parent / "2026-11-19/k25_paths"
    v2 = parent / "2026-11-20/k25v2"
    sources = (
        (k24, "native", 4, {"full": "baseline", "anchor": "native_anchor"}),
        (old, "native", 6, {"cross_v": "v_rolled"}),
        (old, "relay_path", 6, {"repair_whole": "v_whole", "repair_mean": "v_mean", "reverse_mean": "v_mean_reverse"}),
        (old, "combo", 4, {"v_on_anchor": "native_anchor_add_v"}),
        (v2, "fresh_v", 4, {"v_adapted": "baseline", "v_protected": "v_rolled", "v_restore_whole": "v_whole",
                            "v_restore_mean": "v_mean", "v_reverse_mean": "v_mean_reverse"}),
    )
    for directory, prefix, lanes, aliases in sources:
        for lane in range(lanes):
            for row in lines(directory / f"{prefix}_{lane}.jsonl"):
                if row["goal"] == "B" and row["arm"] in aliases:
                    key = aliases[row["arm"]], row["init"]
                    assert key not in rows
                    rows[key] = row


def rf_margin(r):
    start = r["position_xy_m"]
    end = [x + dx for x, dx in zip(start, r["displacement_xy_m"], strict=True)]
    centers = r["centers_xy_m"]
    right, front = (dist(start, centers[q]) - dist(end, centers[q]) for q in ("right", "front"))
    return 1000 * (right - front)


def compare(rows, ids, base, changed):
    x, y = ([rows[arm, i] for i in ids] for arm in (base, changed))
    assert all(a["position_xy_m"] == b["position_xy_m"] and a["centers_xy_m"] == b["centers_xy_m"]
               for a, b in zip(x, y, strict=True))
    exact = paired(x, y, "right")
    delta = [rf_margin(b) - rf_margin(a) for a, b in zip(x, y, strict=True)]
    return {**exact, "base_front": sum(a["first"] == ["front"] for a in x),
            "base_right": sum(a["first"] == ["right"] for a in x),
            "front_to_right": sum(a["first"] == ["front"] and b["first"] == ["right"]
                                  for a, b in zip(x, y, strict=True)),
            "right_to_front": sum(a["first"] == ["right"] and b["first"] == ["front"]
                                  for a, b in zip(x, y, strict=True)),
            "front_net": sum(int(b["first"] == ["front"]) - int(a["first"] == ["front"])
                             for a, b in zip(x, y, strict=True)),
            "right_minus_front_change_median_mm": median(delta),
            "right_minus_front_mean_ci_mm": paired_interval(delta),
            "right_minus_front_positive": sum(v > 0 for v in delta)}


def aggregate(rows, ids):
    by = defaultdict(list)
    for r in rows:
        by[r["init"]].append(r)
    assert sorted(by) == ids and all(len(group) == 6 for group in by.values())
    return lambda f: median(median(f(row) for row in group) for group in by.values())


def summarize_probes(directory, suite, ids):
    v = aggregate(read(directory / f"v_probe_{suite}.json"), ids)
    p = read(directory / f"p_probe_{suite}.json")
    assert p["gate"]["init"] == ids[0]
    output = {"v_native_gains": {x: v(lambda r: r["native"][x]) for x in ("full", "cross_v", "anchor", "v_only")},
              "v_effect": {x: v(lambda r: r[x]) for x in ("v_d_correction", "full_d_correction",
                                                              "v_to_full_d_alignment", "v_mu_change_rms",
                                                              "full_mu_change_rms")}, "p": {}}
    for a, rows in p["arms"].items():
        grouped = aggregate(rows, ids)
        output["p"][a] = {k: grouped(lambda r: r[k]) for k in ("h_gain", "mu_h_to_r1")}
    return output


def trace_p(directory):
    rows = read(directory / "p_trace.json")
    assert len(rows) == 20 * 6 * 8
    assert len({(r["init"], r["route"], r["step"], r["layer"]) for r in rows}) == len(rows)
    by_init = defaultdict(list)
    for r in rows:
        by_init[r["init"]].append(r)
    assert sorted(by_init) == FRESH
    result = {}
    for k in ("query_share", "key_share", "common_probability_share", "differential_probability_share"):
        results = []
        for group in by_init.values():
            energy = [r["direct_change_d_rms"] ** 2 for r in group]
            results.append(sum(e * r[k] for e, r in zip(energy, group, strict=True)) / sum(energy))
        result[k] = median(results)
    return result


def summarize(directory, parents, v_lanes, p_lanes, q_lanes):
    arms = {*V_ARMS, *P_ARMS, *WEIGHT_ARMS, "baseline", "v_whole", "v_mean", "v_rolled", "v_mean_reverse"}
    trials = {**collect(directory, "v_eval", v_lanes, (*V_ARMS, *WEIGHT_ARMS)),
              **collect(directory, "p_eval", p_lanes, P_ARMS)}
    if q_lanes:
        trials.update(collect(directory, "q_eval", q_lanes, Q_ARMS))
        arms.update(Q_ARMS)
    total = len(trials)
    add_refs(trials, parents)
    comparisons = {}
    for a in sorted(arms - {"baseline", "native_anchor"}):
        base = "baseline"
        if a in ("p_reverse", "pv_reverse", "native_anchor_add_v"):
            base = "native_anchor"
        elif a in ("v_reverse_mu", "v_reverse_d", "v_mean_reverse"):
            base = "v_rolled"
        comparisons[a, base] = True
    comparisons["v_amplified_d", "v_d"] = True
    comparisons["v_d", "v_whole"] = True
    comparisons["pv_restore", "p_restore"] = True
    comparisons["pv_restore", "v_whole"] = True
    comparisons["pv_reverse", "p_reverse"] = True
    comparisons["p_query", "p_key"] = True
    if q_lanes:
        comparisons["p_query_hidden", "p_query"] = True
        comparisons["q_weights", "p_query"] = True
    result = {"new_episodes": total, "calibration": read(directory / "v_calibration.json"),
              "p_trace": trace_p(directory), "cohorts": {}}
    for suite, ids in (("old", IDS), ("fresh", FRESH)):
        observed = arms - set(Q_ARMS) if suite == "old" else arms
        probed = summarize_probes(directory, suite, ids)
        v = aggregate(read(directory / f"v_probe_{suite}.json"), ids)
        probed["v_amplified_gain_at_fixed_drafts"] = v(
            lambda r: r["native"]["full"] + result["calibration"]["scale"] * r["v_d_correction"])
        stats = {"probe": probed,
                 "outcomes": {a: {**endpoint([trials[a, i] for i in ids], "B"),
                                   "right_minus_front_initial_median_mm": median(rf_margin(trials[a, i]) for i in ids)}
                              for a in sorted(observed)},
                 "paired": {f"{a}_vs_{b}": compare(trials, ids, b, a) for a, b in comparisons if a in observed}}
        result["cohorts"][suite] = stats
    (directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("previous_results", type=Path)
    p.add_argument("--v-lanes", type=int, default=8)
    p.add_argument("--p-lanes", type=int, default=6)
    p.add_argument("--q-lanes", type=int, default=0)
    a = p.parse_args()
    summarize(a.directory, a.previous_results, a.v_lanes, a.p_lanes, a.q_lanes)
