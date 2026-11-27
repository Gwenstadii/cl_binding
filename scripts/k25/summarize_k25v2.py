"""Verify K25V2 coverage and reconstruct the paired results."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median

from summarize import IDS, TARGET, endpoint, lines, margin, paired, paired_interval, read

OLD_UP = {"up_restore_whole": "AB", "up_restore_mean": "AB", "up_restore_diff": "AB",
          "up_reverse_whole": "B", "up_reverse_mean": "B", "up_reverse_diff": "B"}


def fresh_plan(site):
    return {f"{site}_{arm}": goals for arm, goals in {
        "adapted": "AB", "protected": "AB", "restore_whole": "B", "restore_mean": "AB",
        "restore_diff": "B", "reverse_mean": "B", "reverse_diff": "B"}.items()}


def lock_ids(directory):
    candidates = read(directory / "candidates.json")
    holdout = read(directory / "holdout.json")
    screened = sorted((r for lane in range(4) for r in read(directory / f"screen_{lane}.json")),
                      key=lambda r: r["init"])
    fresh = [r for r in screened if r["boundary"] is not None]
    assert holdout["selected"] == fresh[:20] and holdout["eligible"] == len(fresh)
    assert holdout["candidates"] == len(candidates["records"]) == len(screened)
    assert [r["init"] for r in candidates["records"]] == [r["init"] for r in screened]
    assert holdout["candidate_sha256"] == sha256((directory / "candidates.json").read_bytes()).hexdigest()
    chosen = holdout["selected"]
    assert len(chosen) == 20 and all(r["min_distance_to_preceding_k25"] > 1e-6 for r in chosen)
    return [r["init"] for r in chosen], {
        "inits": [r["init"] for r in chosen], "qualified": len(fresh), "candidates": len(screened),
        "preceding_resets_compared": candidates["preceding_k25_resets_compared"],
        "min_initial_state_distance": min(r["min_distance_to_preceding_k25"] for r in chosen),
        "handoff_step_range": [min(r["boundary"] for r in chosen), max(r["boundary"] for r in chosen)],
        "reset_seed_base": candidates["reset_seed_base"], "noise_seed_base": candidates["noise_seed_base"]}


def load_suite(directory, suite, plan, ids, lanes):
    rows = {}
    for lane in range(lanes):
        done = read(directory / f"{suite}_{lane}_COMPLETE.json")
        assert (done["suite"], done["lane"], done["lanes"]) == (suite, lane, lanes)
        gate = done["gate"]
        assert gate["init"] == ids[lane]
        if suite == "fresh_v":
            assert all(gate[k] for k in ("reverse_True_whole_chunk_bitwise", "reverse_False_whole_chunk_bitwise",
                                         "value_decomposition_bitwise", "fp32_split_readout_verified"))
        else:
            assert gate["identity_bitwise_both_bases"] and gate["whole_bitwise_both_directions"]
        expected = {(a, i, q) for a, qs in plan.items() for i in ids[lane::lanes] for q in qs}
        actual = lines(directory / f"{suite}_{lane}.jsonl")
        assert len(actual) == len(expected)
        for r in actual:
            key = r["arm"], r["init"], r["goal"]
            assert key in expected and key not in rows
            rows[key] = r
    return rows


def old_refs(rows, prior, combined):
    for location, count, prefix, arm, output in (
        (prior, 6, "native", "mlp_gate_down", "up_protected"),
        (combined, 8, "eval", "keep_mlp", "up_adapted"),
    ):
        for lane in range(count):
            for r in lines(location / f"{prefix}_{lane}.jsonl"):
                if r["arm"] == arm and r["goal"] in "AB":
                    rows[output, r["init"], r["goal"]] = r


def score_suite(rows, plan, ids, site):
    plan = {f"{site}_adapted": "AB", f"{site}_protected": "AB", **plan}
    result = {}
    for arm, qs in plan.items():
        score = {q: endpoint([rows[arm, i, q] for i in ids], q) for q in qs}
        if qs == "AB":
            score["paired_correct"] = sum(all(rows[arm, i, q]["first"] == [TARGET[q]] for q in qs) for i in ids)
        result[arm] = score
    pairs = {}
    for arm, ctrl in (
        ("restore_whole", "adapted"), ("restore_mean", "adapted"), ("restore_diff", "adapted"),
        ("restore_mean", "restore_diff"), ("restore_mean", "restore_whole"),
        ("reverse_whole", "protected"), ("reverse_mean", "protected"), ("reverse_diff", "protected"),
        ("reverse_mean", "reverse_diff"),
    ):
        arm, ctrl = f"{site}_{arm}", f"{site}_{ctrl}"
        if arm not in plan:
            continue
        base = [rows[ctrl, i, "B"] for i in ids]
        new = [rows[arm, i, "B"] for i in ids]
        diff = [margin(b) - margin(a) for a, b in zip(base, new, strict=True)]
        pairs[f"{arm}_vs_{ctrl}"] = {**paired(base, new, "right"), "margin_mean_mm": mean(diff),
                                    "margin_ci_mm": paired_interval(diff),
                                    "margin_positive": sum(d > 0 for d in diff)}
    return {"outcomes": result, "right_paired": pairs}


def probes(directory, ids, plan, suite):
    assert read(directory / f"probe_{suite}_COMPLETE.json")["arms"] == list(plan)
    result = {}
    for a in plan:
        by = defaultdict(list)
        for r in read(directory / f"probe_{a}.json"):
            by[r["init"]].append(r)
        assert sorted(by) == ids and all(len(group) == 6 for group in by.values())
        result[a] = {k: median(median(r[k] for r in group) for group in by.values())
                     for k in ("h_gain", "mu_h")}
    return result


def training(directory, step):
    done = read(directory / f"training_{step:06d}_COMPLETE.json")
    rows = lines(directory / f"training_{step:06d}.jsonl")
    assert done["step"] == step and [r["offset"] for r in rows] == done["offsets"]
    assert all(r["step"] == step and r["historical_training_step_reproduced"] for r in rows)
    assert len(done["v_tensors"]) == 8 and len(done["up_tensors"]) == 16
    so = "without:channel:R"
    comparisons = {"rotation_present_minus_omitted": ("full", so),
                   "rotation_removed_minus_random_1": (so, "random_1"),
                   "rotation_removed_minus_random_2": (so, "random_2"),
                   "only_v": ("only_v", so), "only_up": ("only_up", so),
                   "only_v_up": ("only_v_up", so), "remove_v_up": ("remove_v_up", "full"),
                   **{name: (name, so) for name in ("v_mean", "v_diff", "up_mean", "up_diff")}}
    result = {}
    for label, (a, b) in comparisons.items():
        per_batch = [1000 * median(x - y for x, y in zip(r["gain_h"][a], r["gain_h"][b], strict=True))
                     for r in rows]
        result[label] = {"gain_delta_per_1000": median(per_batch), "by_batch": per_batch,
                         "negative_batches": sum(x < 0 for x in per_batch)}
    return {"batches": len(rows), "contrasts": result}


def summarize(args):
    d = args.directory
    ids, selection = lock_ids(d)
    records = {"fresh_selection": selection, "results": {}}
    added = 0
    for suite, cases, lanes, site in (("old_up", IDS, 6, "up"), ("fresh_up", ids, 4, "up"), ("fresh_v", ids, 4, "v")):
        plan = OLD_UP if suite == "old_up" else fresh_plan(site)
        rows = load_suite(d, suite, plan, cases, lanes)
        added += len(rows)
        if suite == "old_up":
            old_refs(rows, args.prior, args.combined)
        records["results"][suite] = score_suite(rows, plan, cases, site)
        if suite != "old_up":
            records["results"][suite]["fixed"] = probes(d, cases, plan, suite)
    records["new_episodes"] = added
    records["training"] = {step: training(d, step) for step in (30, 100)}
    (d / "summary.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("combined", type=Path)
    summarize(parser.parse_args())
