"""Verify combined K25 rollbacks and aggregate held-out comparisons."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median

from summarize import IDS, METRICS, TARGET, endpoint, label, lines, margin, paired, paired_interval, probe, read


def score(rows, arm):
    old = {q: [rows[arm, i, q] for i in IDS] for q in "AB"}
    result = {q: endpoint(old[q], q) for q in old}
    result["paired_correct"] = sum(
        a["first"] == [TARGET["A"]] and b["first"] == [TARGET["B"]]
        for a, b in zip(old["A"], old["B"], strict=True)
    )
    front = [rows[arm, i, "N"] for i in IDS if (arm, i, "N") in rows]
    if front:
        assert len(front) == len(IDS)
        result["N"] = {"first": dict(Counter(map(label, front))),
                       "terminal": sum(row["terminal"]["front"] for row in front),
                       "ever": sum(row["ever"]["front"] for row in front),
                       "stable_grasps": sum(row["first_stable_grasp"] is not None for row in front)}
    return result


def demo_scores(rows):
    totals = defaultdict(lambda: defaultdict(lambda: [0., 0]))
    for row in rows:
        acc = totals[row["demo"]]
        for phase, channels in row["sq_error"].items():
            for channel, value in channels.items():
                for key in ("overall", phase, f"{phase}_{channel}"):
                    acc[key][0] += value["sum"]
                    acc[key][1] += value["count"]
    assert len(rows) == 90 and len(totals) == 10
    values = defaultdict(list)
    for acc in totals.values():
        for key, (amount, count) in acc.items():
            if count:
                values[key].append(amount / count)
    return {key: median(items) for key, items in values.items()}


def summary(directory, k24, previous, lanes, graded_lanes):
    selected = read(directory / "selection.json")
    inv = read(directory / "inventory.json")
    plan = {"k15_calibrated": "AB", "k15_original_gain": "AB",
            **dict.fromkeys(selected["candidates"], "ABN"),
            **dict.fromkeys(("best_mu", "best_d", "best_mu_d"), "AB"),
            **dict.fromkeys(("reverse_mu", "reverse_d"), "B")}
    graded = read(directory / "graded_selection.json")
    graded_arms = dict.fromkeys(graded["candidates"], "ABN")
    rows = {}
    for lane in range(lanes):
        marker = read(directory / f"eval_{lane}_COMPLETE.json")
        assert marker == {"lane": lane, "lanes": lanes, "baseline_identity": IDS[lane]}
        actual = lines(directory / f"eval_{lane}.jsonl")
        expected = {(a, i, q) for a, qs in plan.items() for i in IDS[lane::lanes] for q in qs}
        assert len(actual) == len(expected)
        for row in actual:
            key = row["arm"], row["init"], row["goal"]
            assert key in expected and key not in rows
            rows[key] = row
    for lane in range(graded_lanes):
        marker = read(directory / f"graded_eval_{lane}_COMPLETE.json")
        assert marker == {"lane": lane, "lanes": graded_lanes}
        actual = lines(directory / f"graded_eval_{lane}.jsonl")
        expected = {(a, i, q) for a, qs in graded_arms.items() for i in IDS[lane::graded_lanes] for q in qs}
        assert len(actual) == len(expected)
        for row in actual:
            key = row["arm"], row["init"], row["goal"]
            assert key in expected and key not in rows
            rows[key] = row
    plan.update(graded_arms)
    evaluated = len(rows)
    for lane in range(4):
        for row in lines(k24 / f"native_{lane}.jsonl"):
            if row["arm"] in ("full", "anchor"):
                key = row["arm"], row["init"], row["goal"]
                assert key not in rows
                rows[key] = row

    outcomes = {arm: score(rows, arm) if "A" in qs else endpoint(
        [rows[arm, i, "B"] for i in IDS], "B")
        for arm, qs in {"full": "ABN", "anchor": "ABN", **plan}.items()}
    fixed = {}
    probe_outcomes = {}
    demo = {}
    for arm in (*read(directory / "validation_COMPLETE.json")["arms"], *graded_arms):
        fixed[arm] = probe(directory / f"held_probe_{arm}.json")
        if arm in ("full", "anchor"):
            assert fixed[arm] == probe(previous / f"probe_{arm}.json")
        probe_outcomes[arm] = {k: median(r[k] for r in fixed[arm].values()) for k in METRICS}
        demos = read(directory / f"demo_{arm}.json")
        if arm in ("full", "anchor"):
            assert demos == read(k24 / f"demo_{arm}.json")
        demo[arm] = demo_scores(demos)
    assert set(fixed) == {"full", "anchor", *selected["candidates"], *graded_arms}
    right_effect = {arm: paired([rows["closest" if arm.startswith("reverse") else "full", i, "B"] for i in IDS],
                                [rows[arm, i, "B"] for i in IDS], "right")
                    for arm in plan}
    for arm in ("best_mu", "best_d", "best_mu_d"):
        for comparator in ("best_mu", "best_d"):
            if comparator != arm:
                right_effect[f"{arm}_vs_{comparator}"] = paired(
                    [rows[comparator, i, "B"] for i in IDS], [rows[arm, i, "B"] for i in IDS], "right")
    for arm, baseline in (("d_only_selection", "anchor"), ("closest", "anchor"),
                          ("less_rollback", "cross_mlp"), ("best_mu_d", "closest"),
                          ("keep_mlp", "graded_tight"), ("graded_tight", "anchor")):
        right_effect[f"{arm}_vs_{baseline}"] = paired(
            [rows[baseline, i, "B"] for i in IDS], [rows[arm, i, "B"] for i in IDS], "right")
    front_effect = {}
    for arm in ("anchor", *selected["candidates"], *graded_arms):
        changes = [int(rows[arm, i, "N"]["terminal"]["front"]) - int(rows["full", i, "N"]["terminal"]["front"])
                   for i in IDS]
        front_effect[arm] = {"wins": changes.count(1), "losses": changes.count(-1),
                             "risk_difference": mean(changes), "paired_bootstrap_95": paired_interval(changes)}
    factors = {}
    dev = {r["mask"]: r for r in selected["all_rows"]}
    all_mask = max(dev)
    for j, name in enumerate(inv["groups"]):
        bit = 1 << j
        keys = ("d_gain_h", "d_h_to_anchor_rms", "mu_h_to_anchor_rms", "mu_zr_to_anchor_rms")
        factors[name] = {k: {"added_to_full": dev[bit][k] - dev[0][k],
                             "left_updated_near_anchor": dev[all_mask ^ bit][k] - dev[all_mask][k],
                             "median_rollback_effect_across_contexts": median(
                                 dev[m | bit][k] - r[k] for m, r in dev.items() if not m & bit)}
                         for k in keys}
    terms = (("best_mu_d", 1), ("best_mu", -1), ("best_d", -1), ("full", 1))
    interaction = [sum(mult * (rows[arm, i, "B"]["first"] == ["right"]) for arm, mult in terms) for i in IDS]
    margin_interaction = [sum(mult * margin(rows[arm, i, "B"]) for arm, mult in terms) for i in IDS]
    result = {"episodes": evaluated, "selection": {k: v for k, v in selected.items() if k != "all_rows"},
              "graded_selection": {k: v for k, v in graded.items() if k != "all_rows"},
              "outcomes": outcomes, "fixed_probes": probe_outcomes, "front_validation": demo,
              "right_effect": right_effect, "front_terminal_effect": front_effect, "dev_factors": factors,
              "paired_probe_gain_change": {arm: median(fixed[arm][i]["horizontal_gain"]
                                                       - fixed["full"][i]["horizontal_gain"] for i in IDS)
                                           for arm in fixed},
              "paired_mu_d_interaction": {"risk": mean(interaction),
                                          "paired_bootstrap_95": paired_interval(interaction),
                                          "mean_margin_mm": mean(margin_interaction),
                                          "margin_mm_bootstrap_95": paired_interval(margin_interaction)}}
    (directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("k24", type=Path)
    parser.add_argument("previous", type=Path)
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--graded-lanes", type=int, default=4)
    args = parser.parse_args()
    summary(args.directory, args.k24, args.previous, args.lanes, args.graded_lanes)
