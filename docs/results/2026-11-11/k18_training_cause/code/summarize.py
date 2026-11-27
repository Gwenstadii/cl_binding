"""Summarize K18 evidence without treating repeated probes as independent scenes."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median, quantiles


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def describe(values):
    values = list(values)
    low, _, high = quantiles(values, method="inclusive") if len(values) > 1 else values * 3
    return {"median": median(values), "q1": low, "q3": high, "n": len(values),
            "negative": sum(v < 0 for v in values), "positive": sum(v > 0 for v in values)}


def summarize_probe(rows):
    assert len(rows) == len({(r["init"], r["route"], r["denoise_index"]) for r in rows}) == 120
    by_init = defaultdict(list)
    for row in rows:
        by_init[row["init"]].append(row)
    assert set(by_init) == set(range(20, 40)) and {len(g) for g in by_init.values()} == {6}
    result = {}
    for horizon in ("10", "50"):
        by_scene = [[r[horizon] for r in g] for g in by_init.values()]
        result[horizon] = {
            key: describe(median(r[key] for r in g) for g in by_scene)
            for key in ("gain", "norm_ratio", "cosine", "noncollinear_ratio", "parent_contrast_rms")
        }
        result[horizon]["mu_drift_rms"] = {
            k: describe(median(r["mu_drift_rms"][k] for r in g) for g in by_scene)
            for k in "HZRGP"
        }
        result[horizon]["minimum_parent_contrast_rms"] = min(
            r["parent_contrast_rms"] for g in by_scene for r in g
        )
        if "contrast_by_channel" in by_scene[0][0]:
            result[horizon]["contrast_by_channel"] = {
                k: {key: describe(median(r["contrast_by_channel"][k][key] for r in g) for g in by_scene)
                    for key in ("gain", "energy_fraction", "parent_contrast_rms")}
                for k in "HZRG"
            }
            result[horizon]["minimum_parent_contrast_rms_by_channel"] = {
                k: min(r["contrast_by_channel"][k]["parent_contrast_rms"] for g in by_scene for r in g)
                for k in "HZRG"
            }
    return result


def summarize_demo(rows):
    by_goal = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_goal[row["goal"]][row["demo"]].append(row)
    assert len(rows) == 108 and set(by_goal) == {"left", "right", "front"}
    assert {len(d) for d in by_goal.values()} == {4}
    result = {}
    for goal, demos in by_goal.items():
        assert {len(g) for g in demos.values()} == {9}
        result[goal] = {
            key: describe(mean(r[key] for r in g) for g in demos.values())
            for key in ("gain", "mu_front_distance", "matched_instruction_mse")
        }
        result[goal]["mu_drift"] = {
            k: describe(mean(r["mu_drift"][k] for r in g) for g in demos.values())
            for k in "HZRGP"
        }
        result[goal]["minimum_parent_contrast_rms"] = min(
            r["parent_contrast_rms"] for g in demos.values() for r in g
        )
    return result


def summarize_native(rows):
    assert len(rows) == len({(r["goal"], r["init"]) for r in rows})
    outcomes = {}
    by_goal = {"A": "left", "B": "right", "N": "front"}
    both = defaultdict(set)
    for prompt, target in by_goal.items():
        group = [r for r in rows if r["goal"] == prompt]
        outcomes[target] = {
            "n": len(group),
            "first": dict(Counter("/".join(r["first"]) or "none" for r in group)),
            "ever": {k: sum(r["ever"][k] for r in group) for k in by_goal.values()},
            "terminal": {k: sum(r["terminal"][k] for r in group) for k in by_goal.values()},
        }
        for r in group:
            if r["first"] == [target]:
                both[r["init"]].add(target)
    outcomes["old_paired_first_correct"] = sum({"left", "right"} <= correct for correct in both.values())
    return outcomes


def summarize_attribution(rows):
    assert len(rows) == len({r["offset"] for r in rows}) and rows[0]["offset"] == 0
    all_branches = sorted(set().union(*(r["branches"] for r in rows)))
    branches = {}
    for name in all_branches:
        group = [r for r in rows if name in r["branches"]]
        b = [r["branches"][name] for r in group]
        branches[name] = {
            "gain_delta": describe(mean(v["gain_delta"]) for v in b),
            "update_l2": describe(v["actual_update_l2"] for v in b),
            "full_loss_delta": describe(v["full_batch_loss_after"] - r["branches"]["full"]["loss_before_on_mask"]
                                        for v, r in zip(b, group, strict=True)),
        }
        for key in ("rms_change", "projection_to_remaining_front12000_drift"):
            branches[name][key] = {k: describe(mean(v["mu"][k][key]) for v in b) for k in "HZRG"}
        if name == "full" or name.startswith(("without:", "only:")):
            reference = "full" if name.startswith("without:") else "zero_gradient"
            # Full minus Without is the conditional marginal contribution.
            multiplier = -1 if reference == "full" else 1
            branches[name]["paired_gain_delta"] = describe(
                multiplier * (mean(v["gain_delta"]) - mean(r["branches"][reference]["gain_delta"]))
                for v, r in zip(b, group, strict=True)
            )
            branches[name]["paired_mu_projection_delta"] = {
                k: describe(multiplier * (mean(v["mu"][k]["projection_to_remaining_front12000_drift"]) -
                                          mean(r["branches"][reference]["mu"][k]["projection_to_remaining_front12000_drift"]))
                            for v, r in zip(b, group, strict=True)) for k in "HZRG"
            }
    return {"batches": len(rows), "historical_step": rows[0]["historical_full_step_comparison"],
            "max_gradient_reconstruction_error": max(
                v for r in rows for v in r["preclip_gradient_reconstruction_rel_l2"].values()),
            "max_loss_reconstruction_error": max(
                v for r in rows for v in r["loss_reconstruction_abs"].values()),
            "branches": branches}


def main(path, complete):
    summary = {k: {} for k in ("probe", "probe_channels", "demo", "native", "attribution")}
    for prefix, key, fn in (("probe_", "probe", summarize_probe),
                            ("probe_channels_", "probe_channels", summarize_probe),
                            ("demo_probe_", "demo", summarize_demo)):
        for file in sorted(path.glob(prefix + "[0-9]*.json")):
            summary[key][int(file.stem.split("_")[-1])] = fn(json.loads(file.read_text(encoding="utf-8")))
    native = defaultdict(list)
    for file in sorted(path.glob("native_[0-9]*_w*.jsonl")):
        native[int(file.stem.split("_")[1])].extend(read_jsonl(file))
    for step, rows in native.items():
        summary["native"][step] = summarize_native(rows)
    for file in sorted(path.glob("attribution_*_[0-9]*.jsonl")):
        rows = read_jsonl(file)
        if rows:
            summary["attribution"][file.stem.removeprefix("attribution_")] = summarize_attribution(rows)
    if complete:
        assert set(summary["probe"]) == set(summary["demo"]) == {0, 10, 30, 100, 300, 1000, 3000, 6000, 12000}
        assert set(summary["probe_channels"]) == set(summary["probe"])
        assert {0, 1000, 3000, 6000, 12000} <= set(summary["native"])
        assert all(g[k]["n"] == 20 for g in summary["native"].values() for k in ("left", "right", "front"))
        assert {"joint_000000", *(f"marginal_{step:06d}" for step in (0, 30, 100, 300, 3000))} <= set(summary["attribution"])
        assert all(a["batches"] == 8 for a in summary["attribution"].values())
        assert all(a["historical_step"] == {"loss_delta": 0, "preclip_norm_delta": 0,
                                           "ids_match": True, "lr_match": True}
                   for a in summary["attribution"].values())
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    main(args.path, args.complete)
