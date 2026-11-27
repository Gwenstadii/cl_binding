"""K16 locked eight-cell channel-source audit and paired statistics.

The physical snapshot is the statistical cluster. Its Left/Right episodes and
all eight interventions are resampled jointly. This is development screening;
the intervals are unadjusted and do not establish equivalence or necessity.
"""
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
from statistics import median


INITS = tuple(range(20, 40))
GROUPS = ("N", "H", "S", "HS", "P", "HP", "SP", "HSP")
FACTORS = ("H", "S", "P")
BITS = {"H": 1, "S": 2, "P": 4}
BLOCKS = {"H": (0, 1), "S": tuple(range(2, 7)), "P": tuple(range(7, 32))}
GROUP_BITS = {g: sum(BITS[f] for f in FACTORS if f in g) for g in GROUPS}
BY_BITS = {b: g for g, b in GROUP_BITS.items()}
LABEL = {"A": "left", "B": "right"}
EXPECTED_KEYS = {(i, c, g) for i in INITS for c in GROUPS for g in LABEL}
MODEL_SHA = {"R": "225322ba28d0ad152fcced19950c43e3e40a435f1835cf64a664226da7830003",
             "F": "8ff1bf8f372d728a3218101a3cc93fb6fb9d484e8fefb39c48d81a0c8ab2c515"}
BOOTSTRAPS, BOOT_SEED = 10000, 20261110
SINGLE_METRICS = {"dir", "L_first", "R_first", "L_ever", "R_ever", "L_term", "R_term", "both_term"}
MAIN_METRICS = ("dir", "total_first", "place", "term", "L_first", "R_first", "L_ever", "R_ever", "L_term", "R_term", "none_first", "front_first")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tag(goals):
    return "/".join(goals) or "none"


def indices(group):
    return sorted(j for factor in FACTORS if GROUP_BITS[group] & BITS[factor] for j in BLOCKS[factor])


def metrics(left, right):
    result = {
        "dir": [int(a["first"] == ["left"] and b["first"] == ["right"]) for a, b in zip(left, right)],
        "total_first": [int(a["first"] == ["left"]) + int(b["first"] == ["right"]) for a, b in zip(left, right)],
        "place": [int(a["ever"]["left"]) + int(b["ever"]["right"]) for a, b in zip(left, right)],
        "term": [int(a["terminal"]["left"]) + int(b["terminal"]["right"]) for a, b in zip(left, right)],
        "front_first": [int("front" in a["first"]) + int("front" in b["first"]) for a, b in zip(left, right)],
        "none_first": [int(not a["first"]) + int(not b["first"]) for a, b in zip(left, right)],
        "front_ever": [int(a["ever"]["front"]) + int(b["ever"]["front"]) for a, b in zip(left, right)],
        "front_term": [int(a["terminal"]["front"]) + int(b["terminal"]["front"]) for a, b in zip(left, right)],
        "both_term": [int(a["terminal"]["left"] and b["terminal"]["right"]) for a, b in zip(left, right)],
    }
    for side, rows, target in (("L", left, "left"), ("R", right, "right")):
        result[f"{side}_first"] = [int(a["first"] == [target]) for a in rows]
        result[f"{side}_ever"] = [int(a["ever"][target]) for a in rows]
        result[f"{side}_term"] = [int(a["terminal"][target]) for a in rows]
    return result


class PairedStatistics:
    def __init__(self, per):
        self.per = per
        rng = random.Random(BOOT_SEED)
        self.boot_indices = [[rng.randrange(len(INITS)) for _ in INITS] for _ in range(BOOTSTRAPS)]
        self.cache = {}

    def contrast(self, coefficients, metric):
        """Linear contrast over policy regimes; never treats episodes as iid."""
        coefficients = {g: v for g, v in coefficients.items() if v}
        cache_key = tuple(sorted(coefficients.items())), metric
        if cache_key in self.cache:
            return self.cache[cache_key]
        assert abs(sum(coefficients.values())) < 1e-12
        assert set(coefficients).issubset(GROUPS)
        n_per_cluster = 1 if metric in SINGLE_METRICS else 2
        denominator = len(INITS) * n_per_cluster
        differences = [sum(v * self.per[g][metric][k] for g, v in coefficients.items()) for k in range(len(INITS))]
        samples = sorted(sum(differences[k] for k in idx) / denominator for idx in self.boot_indices)
        point = sum(differences) / denominator
        lo, hi = samples[int(.025 * BOOTSTRAPS)], samples[int(.975 * BOOTSTRAPS) - 1]
        result = {
            "coefficients": coefficients, "metric": metric, "point": point,
            "ci95_cluster_bootstrap": [lo, hi], "n_snapshot_clusters": len(INITS),
            "denominator": denominator, "snapshot_contributions": dict(zip(map(str, INITS), differences)),
            "snapshot_positive": [i for i, d in zip(INITS, differences) if d > 0],
            "snapshot_negative": [i for i, d in zip(INITS, differences) if d < 0],
            "clearly_positive_descriptive_rule": point >= .15 - 1e-12 and lo > 0,
            "clearly_negative_descriptive_rule": point <= -.15 + 1e-12 and hi < 0,
            "point_gap_within_0_10_only": abs(point) <= .10 + 1e-12,
            "not_an_equivalence_test": True, "unadjusted_development_screening": True,
        }
        self.cache[cache_key] = result
        return result

    def compare(self, after, before, metric):
        return {"after": after, "before": before, **self.contrast({after: 1, before: -1}, metric)}


def episode_transitions(records, before, after, metric):
    changed, gained, lost = [], [], []
    for init in INITS:
        for goal, label in LABEL.items():
            a, b = records[init, before, goal], records[init, after, goal]
            if metric == "first":
                old, new = a["first"] == [label], b["first"] == [label]
            else:
                field = "ever" if metric == "ever" else "terminal"
                old, new = bool(a[field][label]), bool(b[field][label])
            key = f"{init}{goal}"
            if old != new:
                changed.append(key)
                (gained if new else lost).append(key)
    return {"gained": gained, "lost": lost, "net": len(gained) - len(lost), "changed": changed}


def first_transition_table(records, before, after):
    result = {}
    for goal in LABEL:
        counts = Counter((tag(records[i, before, goal]["first"]), tag(records[i, after, goal]["first"])) for i in INITS)
        result[goal] = [{"before": a, "after": b, "count": n} for (a, b), n in sorted(counts.items())]
    return result


def paired_comparison(statistics, records, after, before):
    return {
        "after": after, "before": before,
        "metrics": {m: statistics.compare(after, before, m) for m in MAIN_METRICS},
        "episode_transitions": {m: episode_transitions(records, before, after, m) for m in ("first", "ever", "terminal")},
        "first_transition_table_by_goal": first_transition_table(records, before, after),
    }


def side_summary(rows):
    requested = rows[0]["requested"]
    return {
        "n": len(rows), "requested": requested,
        "first_distribution": dict(sorted(Counter(tag(r["first"]) for r in rows).items())),
        "first_correct": sum(r["first"] == [requested] for r in rows),
        "ever_correct": sum(bool(r["ever"][requested]) for r in rows),
        "terminal_correct": sum(bool(r["terminal"][requested]) for r in rows),
        "front_ever": sum(bool(r["ever"]["front"]) for r in rows),
        "terminal_distribution": dict(sorted(Counter(tag([g for g, hit in r["terminal"].items() if hit]) for r in rows).items())),
    }


def statistics_report(records):
    per = {c: metrics([records[i, c, "A"] for i in INITS], [records[i, c, "B"] for i in INITS]) for c in GROUPS}
    statistics = PairedStatistics(per)
    edges = {}
    for factor, bit in BITS.items():
        for before in GROUPS:
            if GROUP_BITS[before] & bit:
                continue
            after = BY_BITS[GROUP_BITS[before] | bit]
            edges[f"add_{factor}:{before}_to_{after}"] = {
                "factor": factor, "other_factors_held_fixed": {f: bool(GROUP_BITS[before] & BITS[f]) for f in FACTORS if f != factor},
                "added": paired_comparison(statistics, records, after, before),
                "removed": paired_comparison(statistics, records, before, after),
            }
    assert len(edges) == 12
    comparisons = {}
    for reference in ("N", "HSP"):
        for group in GROUPS:
            if group != reference:
                comparisons[f"{group}_minus_{reference}"] = paired_comparison(statistics, records, group, reference)
    interaction_coefficients = {
        "H_x_S_given_P_new": {"HS": 1, "H": -1, "S": -1, "N": 1},
        "H_x_S_given_P_old": {"HSP": 1, "HP": -1, "SP": -1, "P": 1},
        "H_x_P_given_S_new": {"HP": 1, "H": -1, "P": -1, "N": 1},
        "H_x_P_given_S_old": {"HSP": 1, "HS": -1, "SP": -1, "S": 1},
        "S_x_P_given_H_new": {"SP": 1, "S": -1, "P": -1, "N": 1},
        "S_x_P_given_H_old": {"HSP": 1, "HS": -1, "HP": -1, "H": 1},
    }
    interactions = {name: {m: statistics.contrast(coef, m) for m in MAIN_METRICS}
                    for name, coef in interaction_coefficients.items()}
    three_way = {"HSP": 1, "HS": -1, "HP": -1, "SP": -1, "H": 1, "S": 1, "P": 1, "N": -1}
    interactions["H_x_S_x_P_exploratory"] = {m: statistics.contrast(three_way, m) for m in MAIN_METRICS}
    marginal = {}
    for factor, bit in BITS.items():
        coefficients = {g: (.25 if GROUP_BITS[g] & bit else -.25) for g in GROUPS}
        marginal[f"{factor}_averaged_over_other_factor_levels_exploratory"] = {
            m: statistics.contrast(coefficients, m) for m in MAIN_METRICS}
    summary = {c: {"n_snapshots": 20, "n_episodes": 40, "old_blocks": [f for f in FACTORS if GROUP_BITS[c] & BITS[f]],
                   "old_indices": indices(c), "totals": {m: sum(v) for m, v in per[c].items()},
                   "per_goal": {g: side_summary([records[i, c, g] for i in INITS]) for g in LABEL}}
               for c in GROUPS}
    return {"summary": summary, "per_snapshot": per, "factor_added_removed_edges": edges,
            "comparisons_to_references": comparisons, "interactions": interactions, "marginal_factor_effects": marginal,
            "statistics": {"unit": "physical snapshot; Left/Right and all eight regimes resampled together",
                "n_clusters": 20, "bootstrap_replicates": BOOTSTRAPS, "seed": BOOT_SEED,
                "unadjusted_intervals": True, "development_screening": True,
                "no_familywise_significance_claim": True, "point_closeness_is_not_equivalence": True,
                "paired_split_is_a_product_endpoint_not_evidence_of_internal_multiplication": True,
                "three_factor_interaction_and_marginal_effects_are_exploratory": True}}


def discover_records(archive):
    records, traces, paths = {}, {}, []
    for init in INITS:
        for suffix, target in (("", records), ("_traces", traces)):
            path = archive / f"i{init}{suffix}.jsonl"
            paths.append(path)
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row["init"], row["condition"], row["goal"]
                assert row["init"] == init and key not in target, (path, key)
                target[key] = row["rows"] if suffix else row
    assert set(records) == set(traces) == EXPECTED_KEYS
    return records, traces, paths


def load_references(archive):
    records, traces, paths = {}, {}, []
    for name, target in (("reference_episodes.jsonl", records), ("reference_traces.jsonl", traces)):
        path = archive / name
        paths.append(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["init"], row["condition"], row["goal"]
            assert key not in target, (path, key)
            assert row["source_condition"] == {"N": "new_gain", "HSP": "old_gain"}[row["condition"]]
            target[key] = row["rows"] if target is traces else row
    expected = {(i, c, g) for i in INITS for c in ("N", "HSP") for g in LABEL}
    assert set(records) == set(traces) == expected
    return records, traces, paths


def audit_records(archive, protocol):
    records, traces, paths = discover_records(archive)
    reference_records, reference_traces, reference_paths = load_references(archive)
    paths.extend(reference_paths)
    script_path = archive / "k16.py"
    script_sha = digest(script_path)
    paths.append(script_path)
    assert protocol["inits"] == list(INITS) and protocol["alpha"] == 2.959
    assert protocol["blocks"] == {f: list(BLOCKS[f]) for f in FACTORS}
    assert protocol["conditions"] == {g: [f for f in FACTORS if GROUP_BITS[g] & BITS[f]] for g in GROUPS}
    assert protocol["episodes"]["total"] == 320 and protocol["timing"]["window"] == 60
    assert protocol["timing"]["horizon"] == 400
    assert digest(archive / "reference_k15.py") == protocol["k15_script_sha256"]
    paths.append(archive / "reference_k15.py")
    for filename, field in (("source_code/k14.py", "code/k14.py"),
                            ("source_code/k14s5.py", "code/k14s5.py"),
                            ("snapshots.json", "snapshots/snapshots.json")):
        assert digest(archive / filename) == protocol["source_sha256"][field], filename
        paths.append(archive / filename)
    geometry = read(archive / "geometry.json")["sites"]
    geo_path = archive / "dependencies/k1_geometry.py"
    spec = importlib.util.spec_from_file_location("k16_audited_geometry", geo_path)
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    assert tuple(geo.GOALS) == ("left", "front", "right", "back")
    paths.extend([archive / "geometry.json", geo_path])
    for init in INITS:
        path = archive / f"i{init}_COMPLETE.json"
        marker = read(path)
        assert marker["init"] == init and marker["n_episodes"] == 16, path
        assert marker["model_sha256"] == MODEL_SHA and marker["script_sha256"] == script_sha, path
        assert marker["source_hashes"] == protocol["source_sha256"], path
        assert marker["k15_script_sha256"] == protocol["k15_script_sha256"], path
        assert marker["alpha"] == 2.959
        assert marker["condition_old_indices"] == {g: indices(g) for g in GROUPS}, path
        for processor in ("policy_preprocessor_step_5_normalizer_processor.safetensors",
                          "policy_postprocessor_step_0_unnormalizer_processor.safetensors"):
            checked = marker["processor_check"][processor]
            assert checked["compared_keys"] == 40 and checked["identical_across"] == ["F", "R"], path
            assert checked["bookkeeping_keys_differing_ignored"] == [], path
        paths.append(path)
    exact, prefixes = [], {}
    for init in INITS:
        baseline = records[init, "N", "A"]
        boundary = baseline["boundary"]
        assert boundary % 10 == 0 and 70 <= boundary <= 110
        common = traces[init, "N", "A"][:boundary]
        assert not any(r["hits"] or any(r["pred"].values()) for r in common)
        prefixes[str(init)] = {"boundary": boundary, "signature": baseline["boundary_signature"]}
        for condition in GROUPS:
            for goal, label in LABEL.items():
                key = init, condition, goal
                row, trace = records[key], traces[key]
                assert row["requested"] == label and row["alpha"] == 2.959 and row["window"] == 60, key
                assert row["prefix_model"] == "Front-12000" and row["prefix_goal"] == "Front", key
                assert row["mu_old_blocks"] == [f for f in FACTORS if GROUP_BITS[condition] & BITS[f]], key
                assert row["mu_old_indices"] == indices(condition), key
                assert row["n_window_queries"] == 60, key
                for field in ("seed", "boundary", "z0", "boundary_signature"):
                    assert row[field] == baseline[field], (key, field)
                assert len(trace) == 400 and [r["step"] for r in trace] == list(range(1, 401)), key
                assert trace[:boundary] == common, ("shared prefix", key)
                movement = geo.movement("scene2", geometry, trace, row["z0"], None)
                assert row["first"] == movement["first_targets"] and row["first_step"] == movement["first_step"], key
                assert row["first_times"] == movement["times"], key
                assert row["terminal"] == trace[-1]["pred"], key
                assert row["ever"] == {g: any(r["pred"][g] for r in trace) for g in geo.GOALS}, key
                for field, part in (("clip_rate_window", trace[boundary:boundary + 60]), ("clip_rate_all", trace[boundary:])):
                    assert abs(row[field] - sum(r["clip"] for r in part) / len(part)) < 1e-12, (key, field)
                if key in reference_records:
                    old = reference_records[key]
                    for field in ("requested", "seed", "boundary", "window", "alpha", "prefix_model", "prefix_goal", "z0",
                                  "boundary_signature", "first", "first_step", "first_times", "ever", "terminal",
                                  "clip_rate_window", "clip_rate_all", "n_window_queries"):
                        assert row[field] == old[field], ("reference field", key, field)
                    assert trace == reference_traces[key], ("reference full trace", key)
                    exact.append({"init": init, "condition": condition, "goal": goal})
    assert len(exact) == 80
    reference_summary = read(archive / "reference_k15_summary.json")
    for condition, old in (("N", "new_gain"), ("HSP", "old_gain")):
        per = metrics([records[i, condition, "A"] for i in INITS], [records[i, condition, "B"] for i in INITS])
        for metric, values in reference_summary["per_snapshot"][old].items():
            assert per[metric] == values, ("K15 reference summary", condition, metric)
    paths.append(archive / "reference_k15_summary.json")
    return records, traces, paths, {
        "script_sha256": script_sha, "complete_markers": 20, "episodes_checked": 320,
        "full_400_step_traces_checked": 320, "geometry_first_goals_recomputed": 320,
        "all_16_branches_share_prefix_per_init": True, "exact_reference_trace_count": 80,
        "exact_reference_episodes": exact, "shared_prefixes": prefixes,
    }


def observational_diagnostics(records, traces):
    """Post-treatment descriptors, never interpreted as causal mediators."""
    report = {}
    for condition in GROUPS:
        per_episode = {}
        for init in INITS:
            for goal in LABEL:
                row, trace = records[init, condition, goal], traces[init, condition, goal]
                boundary = row["boundary"]
                window = trace[boundary:boundary + 60]
                initial, final = trace[boundary - 1], window[-1]
                actions = [r["a"] for r in window]
                assert all(len(a) == 7 and all(math.isfinite(v) for v in a) for a in actions)
                per_episode[f"{init}{goal}"] = {
                    "window_end_lift_m": final["obj"][2] - row["z0"],
                    "window_end_both_contacts": bool(final["c1"] and final["c2"]),
                    "window_end_object_displacement_m": [b - a for a, b in zip(initial["obj"], final["obj"])],
                    "arm_any_out_of_bounds_rate": sum(any(abs(v) > 1 for v in a[:6]) for a in actions) / 60,
                    "gripper_out_of_bounds_rate": sum(abs(a[6]) > 1 for a in actions) / 60,
                    "out_of_bounds_rate_by_coordinate": [sum(abs(a[j]) > 1 for a in actions) / 60 for j in range(7)],
                    "mean_clipped_absolute_amount_by_coordinate": [sum(max(abs(a[j]) - 1, 0) for a in actions) / 60 for j in range(7)],
                }
        values = list(per_episode.values())
        report[condition] = {
            "n_episodes": 40,
            "window_end_lift_m_median": median(r["window_end_lift_m"] for r in values),
            "window_end_both_contacts_count": sum(r["window_end_both_contacts"] for r in values),
            "arm_any_out_of_bounds_rate_mean": sum(r["arm_any_out_of_bounds_rate"] for r in values) / 40,
            "gripper_out_of_bounds_rate_mean": sum(r["gripper_out_of_bounds_rate"] for r in values) / 40,
            "out_of_bounds_rate_by_coordinate_mean": [sum(r["out_of_bounds_rate_by_coordinate"][j] for r in values) / 40 for j in range(7)],
            "per_episode": per_episode,
            "post_treatment_descriptors_not_mediation_proof": True,
        }
        selected = [records[i, condition, goal] for i in INITS for goal in LABEL]
        diagnostic_keys = set.intersection(*(set(r["diagnostics"]) for r in selected))
        report[condition]["query_diagnostics_median_of_episode_means"] = {
            key: median(r["diagnostics"][key][0] for r in selected) for key in sorted(diagnostic_keys)}
    return report


def audit_gates(archive):
    path = archive / "gates.json"
    gates = read(path)
    assert gates["pass"] is True and gates["all32_dimensions_partitioned"] is True
    assert gates["blocks"] == {f: list(BLOCKS[f]) for f in FACTORS}
    assert gates["condition_old_indices"] == {g: indices(g) for g in GROUPS}
    assert [r["init"] for r in gates["checks"]] == [20, 21, 22]
    n_same_input_checks = 0
    for check in gates["checks"]:
        assert check["same_snapshot_signature"] is True
        assert check["alpha0_instruction_identity_all_conditions"] is True
        roundoff = check["amplified_mean_roundoff_rms_real"]
        assert set(roundoff) == set(GROUPS)
        assert all(math.isfinite(v) and 0 <= v < .03 for v in roundoff.values())
        assert [q["goal"] for q in check["goal_checks"]] == ["A", "B"]
        for goal_check in check["goal_checks"]:
            assert goal_check["alpha1_N_native_bitwise"] is True
            assert [c["condition"] for c in goal_check["conditions"]] == list(GROUPS)
            for condition_check in goal_check["conditions"]:
                assert condition_check["selected_indices"] == indices(condition_check["condition"])
                for flag in ("selected_equals_old_bitwise", "unselected_equals_new_bitwise",
                             "dispatch_matches_formula_bitwise", "all32_finite"):
                    assert condition_check[flag] is True
                for field in ("cast_roundoff_rms_real", "cast_roundoff_max_all32"):
                    assert math.isfinite(condition_check[field]) and condition_check[field] >= 0
                n_same_input_checks += 1
    assert n_same_input_checks == 48
    return gates, {"engineering_gates_passed": True, "same_input_3_snapshots_x_2_goals_x_8_masks": 48,
                   "selected_old_and_unselected_new_channels_bitwise_checked": True,
                   "float32_expression_and_native_dtype_roundoff_separately_documented": True}


def verify_reference_manifest(archive):
    manifest_path = archive / "reference_source_hashes.json"
    manifest = read(manifest_path)
    checked = []
    names = ("reference_episodes.jsonl", "reference_traces.jsonl", "reference_k15_summary.json", "reference_k15.py",
             "snapshots.json", "source_code/k14.py", "source_code/k14s5.py",
             "dependencies/k1_main.py", "dependencies/k1_runtime.py", "dependencies/k1_geometry.py")
    for name in names:
        path = archive / name
        assert digest(path) == manifest[name], ("reference/source manifest", name)
        checked.append(path)
    return [manifest_path, *checked]


def format_effect(value):
    lo, hi = value["ci95_cluster_bootstrap"]
    return f"{value['point']:+.3f} [{lo:+.3f},{hi:+.3f}]"


def write_markdown(archive, report):
    lines = ["# K16 共同响应通道来源：八格验收与配对统计", "",
             "320回合、128000轨迹步；80条N/HSP参照的完整400步轨迹精确复现K15。",
             "N全新，H替换0–1，S替换2–6，P替换7–31。字母表示共同平均的旧来源；条件区别固定来自续训模型并放大2.959倍。", "",
             "| 组 | 左先左/20 | 右先右/20 | 成对/20 | 总首次正确/40 | 左曾成功/20 | 右曾成功/20 | 左终态/20 | 右终态/20 | 先前/40 | 无首次区域/40 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for condition in GROUPS:
        s = report["summary"][condition]["totals"]
        columns = ("L_first", "R_first", "dir", "total_first", "L_ever", "R_ever", "L_term", "R_term", "front_first", "none_first")
        lines.append("| " + condition + " | " + " | ".join(str(s[k]) for k in columns) + " |")
    lines.extend(["", "## 12条来源增加边", "",
                  "下列差值为增加该旧来源之后减去之前；移除方向及每指令统计在JSON中完整列出。", "",
                  "| 增加因素：比较 | 成对分流 | 总首次正确 | 曾成功 | 终态 |",
                  "|---|---|---|---|---|"])
    for edge in report["factor_added_removed_edges"].values():
        value = edge["added"]
        effects = " | ".join(format_effect(value["metrics"][m]) for m in ("dir", "total_first", "place", "term"))
        lines.append(f"| {edge['factor']}: {value['after']}−{value['before']} | {effects} |")
    lines.extend(["", "## 条件交互", "",
                  "每项均是同一快照上的差上差；三因素交互为探索性结果。", "",
                  "| 交互 | 成对分流 | 总首次正确 | 曾成功 | 终态 |", "|---|---|---|---|---|"])
    for name, value in report["interactions"].items():
        effects = " | ".join(format_effect(value[m]) for m in ("dir", "total_first", "place", "term"))
        lines.append(f"| {name} | {effects} |")
    lines.extend(["", "所有区间按20个物理快照聚类，左右指令与全部组别一起重采样10000次。区间未做多重比较校正，供开发性机制筛查。",
                  "成对分流本身是左右两项的乘积型终点，其交互不证明模型内部存在乘法机制。点估计接近或区间跨零不证明等效或没有作用。",
                  "首次区域判据要求书抬高超过3厘米且连续5步处于目标水平区域；未满足此判据不等于从未短暂经过。曾成功和终态采用原生目标成功判据。",
                  "限幅、窗口末抓持和位移是干预后的观察描述，不是机制中介的因果证明。S是异质通道组；P仅指未直接执行的去噪维度。",
                  "本次固定增益、窗口与开发状态上的来源移植效果，不能推广为普遍必要性、语义存储位置或训练更新原因。"])
    (archive / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(archive):
    protocol_path = archive / "protocol.json"
    protocol = read(protocol_path)
    gates, gate_audits = audit_gates(archive)
    paths = verify_reference_manifest(archive)
    records, traces, source_paths, audits = audit_records(archive, protocol)
    paths.extend([protocol_path, archive / "gates.json", *source_paths, Path(__file__).resolve()])
    report = {
        "experiment": "K16 common-response source full H x S x P factorial",
        "episodes": 320, "trace_steps": 128000, "inits": list(INITS), "alpha": 2.959,
        "model_sha256": MODEL_SHA, "blocks": {f: list(BLOCKS[f]) for f in FACTORS},
        "audits": {**audits, **gate_audits}, "protocol": protocol, "engineering_gates": gates,
        **statistics_report(records), "observational_diagnostics": observational_diagnostics(records, traces),
    }
    source_hashes = {}
    for path in paths:
        resolved = path.resolve()
        try:
            name = str(resolved.relative_to(archive)).replace("\\", "/")
        except ValueError:
            name = "executed_summarizer:" + str(resolved)
        source_hashes[name] = digest(resolved)
    report["source_sha256"] = source_hashes
    (archive / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_markdown(archive, report)
    print("AUDITED 320 episodes / 128000 steps; 80 exact K15 reference traces; 48 same-input mask gates")
    print("condition L_first R_first paired total_first L_ever R_ever L_term R_term front_first none_first")
    for condition in GROUPS:
        s = report["summary"][condition]["totals"]
        print(condition, *(s[k] for k in ("L_first", "R_first", "dir", "total_first", "L_ever", "R_ever", "L_term", "R_term", "front_first", "none_first")))
    for name, values in report["interactions"].items():
        print(name, json.dumps({m: {k: v for k, v in values[m].items() if k in ("point", "ci95_cluster_bootstrap")}
                               for m in ("dir", "total_first", "place", "term")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    main(args.archive.resolve())
