"""Audit K15 records and compare paired background/gain effects.

Statistical unit: a physical snapshot (20), with its two requested goals resampled
together. Small point gaps are not treated as proof of equivalence.
"""
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
from statistics import median


INITS = tuple(range(20, 40))
GROUPS = ("native", "new_gain", "old_gain")
LABEL = {"A": "left", "B": "right"}
EXPECTED_KEYS = {(i, c, g) for i in INITS for c in GROUPS for g in LABEL}
MODEL_SHA = {"R": "225322ba28d0ad152fcced19950c43e3e40a435f1835cf64a664226da7830003",
             "F": "8ff1bf8f372d728a3218101a3cc93fb6fb9d484e8fefb39c48d81a0c8ab2c515"}
BOOTSTRAPS, BOOT_SEED = 10000, 20261109


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tag(goals):
    return "/".join(goals) or "none"


def metrics(a, b):
    return {
        "dir": [int(x["first"] == ["left"] and y["first"] == ["right"]) for x, y in zip(a, b)],
        "place": [int(x["ever"]["left"]) + int(y["ever"]["right"]) for x, y in zip(a, b)],
        "term": [int(x["terminal"]["left"]) + int(y["terminal"]["right"]) for x, y in zip(a, b)],
        "L_first": [int(x["first"] == ["left"]) for x in a],
        "R_first": [int(y["first"] == ["right"]) for y in b],
        "front_first": [int("front" in x["first"]) + int("front" in y["first"]) for x, y in zip(a, b)],
        "none_first": [int(not x["first"]) + int(not y["first"]) for x, y in zip(a, b)],
        "L_ever": [int(x["ever"]["left"]) for x in a],
        "R_ever": [int(y["ever"]["right"]) for y in b],
        "L_term": [int(x["terminal"]["left"]) for x in a],
        "R_term": [int(y["terminal"]["right"]) for y in b],
        "front_ever": [int(x["ever"]["front"]) + int(y["ever"]["front"]) for x, y in zip(a, b)],
        "front_term": [int(x["terminal"]["front"]) + int(y["terminal"]["front"]) for x, y in zip(a, b)],
        "both_term": [int(x["terminal"]["left"] and y["terminal"]["right"]) for x, y in zip(a, b)],
    }


def compare(per, x, y, metric, boot_indices):
    count_per_snapshot = 1 if metric in ("dir", "L_first", "R_first", "L_ever", "R_ever", "L_term", "R_term", "both_term") else 2
    a, b = per[x][metric], per[y][metric]
    delta = [v - w for v, w in zip(a, b)]
    denominator = len(INITS) * count_per_snapshot
    samples = sorted(sum(delta[k] for k in idx) / denominator for idx in boot_indices)
    point = sum(delta) / denominator
    ci = [samples[int(.025 * len(samples))], samples[int(.975 * len(samples)) - 1]]
    return {
        "x": x, "y": y, "metric": metric, "n_snapshot_clusters": len(INITS),
        "denominator": denominator, "point": point,
        "ci95_cluster_bootstrap": ci,
        "clearly_better_descriptive_rule": point >= .15 - 1e-12 and ci[0] > 0,
        "snapshot_gained": [i for i, d in zip(INITS, delta) if d > 0],
        "snapshot_lost": [i for i, d in zip(INITS, delta) if d < 0],
        "point_gap_within_0_10_only": abs(point) <= .10 + 1e-12,
        "not_an_equivalence_test": True,
    }


def episode_transitions(rows, before, after, metric):
    outcomes = {}
    for i in INITS:
        for g, label in LABEL.items():
            key = i, g
            a, b = rows[i, before, g], rows[i, after, g]
            if metric == "first":
                old, new = a["first"] == [label], b["first"] == [label]
            elif metric == "ever":
                old, new = a["ever"][label], b["ever"][label]
            else:
                old, new = a["terminal"][label], b["terminal"][label]
            outcomes[key] = bool(old), bool(new)
    gained = [f"{i}{g}" for (i, g), (a, b) in outcomes.items() if not a and b]
    lost = [f"{i}{g}" for (i, g), (a, b) in outcomes.items() if a and not b]
    return {"gained": gained, "lost": lost, "net": len(gained) - len(lost),
            "changed": len(gained) + len(lost)}


def side_summary(part):
    requested = part[0]["requested"]
    return {
        "n": len(part), "requested": requested,
        "first": dict(sorted(Counter(tag(x["first"]) for x in part).items())),
        "first_correct": sum(x["first"] == [requested] for x in part),
        "ever_correct": sum(x["ever"][requested] for x in part),
        "terminal_correct": sum(x["terminal"][requested] for x in part),
        "ever_front": sum(x["ever"]["front"] for x in part),
        "terminal_distribution": dict(sorted(Counter(tag([g for g, hit in x["terminal"].items() if hit]) for x in part).items())),
    }


def discover_records(archive):
    folder = archive / "episodes" if (archive / "episodes").is_dir() else archive
    records, traces, source_files = {}, {}, []
    for path in sorted(folder.glob("*.jsonl")):
        if path.name.startswith("reference_"):
            continue
        source_files.append(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            group = row.get("group", row.get("condition"))
            if group not in GROUPS:
                continue
            key = row["init"], group, row["goal"]
            target = traces if "rows" in row else records
            assert key not in target, f"duplicate key {key} in {path}"
            target[key] = row["rows"] if target is traces else row
    assert set(records) == EXPECTED_KEYS, ("record keys", EXPECTED_KEYS - set(records), set(records) - EXPECTED_KEYS)
    assert set(traces) == EXPECTED_KEYS, ("trace keys", EXPECTED_KEYS - set(traces), set(traces) - EXPECTED_KEYS)
    return records, traces, source_files


def main(archive):
    records, traces, source_files = discover_records(archive)
    script_path = archive / "k15.py"
    script_sha = digest(script_path)
    protocol = read(archive / "protocol.json")
    assert protocol["inits"] == list(INITS) and protocol["alpha"] == 2.959
    assert set(protocol["conditions"]) == set(GROUPS)
    source_files.extend([script_path, archive / "protocol.json", archive / "geometry.json"])
    gates = read(archive / "gates.json")
    assert gates["pass"] is True and len(gates["checks"]) == 3
    assert [c["init"] for c in gates["checks"]] == [20, 21, 22]
    for check in gates["checks"]:
        assert check["alpha1_native_bitwise"] and check["alpha0_instruction_identity"] and check["same_snapshot_signature"]
        assert check["amplified_mean_roundoff_rms"] < .03
    source_files.append(archive / "gates.json")
    geometry = read(archive / "geometry.json")["sites"]
    geo_path = archive / "dependencies/k1_geometry.py"
    spec = importlib.util.spec_from_file_location("k15_audited_geometry", geo_path)
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    source_files.append(geo_path)
    assert tuple(geo.GOALS) == ("left", "front", "right", "back")

    marker_files = []
    marker_folder = archive / "episodes" if (archive / "episodes").is_dir() else archive
    for init in INITS:
        path = marker_folder / f"i{init}_COMPLETE.json"
        marker = read(path)
        assert marker["init"] == init and marker["n_episodes"] == 6, path
        assert marker["model_sha256"] == MODEL_SHA, path
        assert marker["script_sha256"] == script_sha, path
        assert marker["source_hashes"] == protocol["source_sha256"], path
        marker_files.append(path)

    reference_rows, reference_traces = {}, {}
    for filename, target in (("reference_episodes.jsonl", reference_rows), ("reference_traces.jsonl", reference_traces)):
        path = archive / filename
        source_files.append(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            condition = row.get("condition", row.get("group"))
            key = row["init"], condition, row["goal"]
            assert key not in target, (filename, key)
            target[key] = row["rows"] if target is reference_traces else row
    expected_reference = {(i, c, g) for i in INITS for c in ("native", "old_gain") for g in LABEL}
    assert set(reference_rows) == set(reference_traces) == expected_reference

    exact_reproductions = []
    shared_prefixes = {}
    for init in INITS:
        baseline = records[init, "native", "A"]
        boundary = baseline["boundary"]
        assert boundary % 10 == 0 and 30 <= boundary <= 150
        common_prefix = traces[init, "native", "A"][:boundary]
        assert not any(r["hits"] or any(r["pred"].values()) for r in common_prefix)
        shared_prefixes[str(init)] = {"boundary": boundary, "signature": baseline["boundary_signature"]}
        for condition in GROUPS:
            for goal, label in LABEL.items():
                key = init, condition, goal
                row, trace = records[key], traces[key]
                assert row["requested"] == label
                assert row["window"] == 60 and row["alpha"] == 2.959
                assert row["prefix_model"] == "Front-12000" and row["prefix_goal"] == "Front"
                for field in ("seed", "boundary", "z0", "boundary_signature"):
                    assert row[field] == baseline[field], (key, field)
                assert row["n_window_queries"] == (0 if condition == "native" else 60)
                assert len(trace) == 400 and [r["step"] for r in trace] == list(range(1, 401)), key
                assert trace[:boundary] == common_prefix, ("shared prefix", key)
                movement = geo.movement("scene2", geometry, trace, row["z0"], None)
                assert row["first"] == movement["first_targets"] and row["first_step"] == movement["first_step"], key
                assert row["first_times"] == movement["times"], key
                assert row["terminal"] == trace[-1]["pred"], key
                assert row["ever"] == {g: any(r["pred"][g] for r in trace) for g in geo.GOALS}, key
                expected_clip_all = sum(r["clip"] for r in trace[boundary:]) / (400 - boundary)
                expected_clip_window = sum(r["clip"] for r in trace[boundary:boundary + 60]) / 60
                assert abs(row["clip_rate_all"] - expected_clip_all) < 1e-12, key
                assert abs(row["clip_rate_window"] - expected_clip_window) < 1e-12, key
                if key in expected_reference:
                    old = reference_rows[key]
                    for field in ("first", "first_step", "first_times", "ever", "terminal", "clip_rate_all"):
                        if field in old:
                            assert row[field] == old[field], ("reference field", key, field)
                    assert trace == reference_traces[key], ("reference full trace", key)
                    exact_reproductions.append({"init": init, "condition": condition, "goal": goal})
    assert len(exact_reproductions) == 80

    per = {c: metrics([records[i, c, "A"] for i in INITS], [records[i, c, "B"] for i in INITS]) for c in GROUPS}
    old_gain_path = archive / "reference_k14_s5_judgement.json"
    if old_gain_path.exists():
        old_gain_reference = read(old_gain_path)
        for key, values in old_gain_reference["per_snapshot"]["S_canon"].items():
            assert per["old_gain"][key] == values, ("K14 S_canon summary", key)
        source_files.append(old_gain_path)
    baseline_path = archive / "reference_k14_judgement.json"
    has_old_native = baseline_path.exists()
    if has_old_native:
        original = read(baseline_path)
        assert original["inits"] == list(INITS)
        for key, values in original["per_snapshot"]["A_nat"].items():
            assert per["native"][key] == values, ("K14 original native summary", key)
        per["old_background_native_gain_reference"] = original["per_snapshot"]["B_mean"]
        source_files.append(baseline_path)
    rng = random.Random(BOOT_SEED)
    boot_indices = [[rng.randrange(len(INITS)) for _ in INITS] for _ in range(BOOTSTRAPS)]
    comparisons = {}
    for x, y in (("new_gain", "native"), ("new_gain", "old_gain"), ("old_gain", "native")):
        comparisons[f"{x}_minus_{y}"] = {
            "metrics": {m: compare(per, x, y, m, boot_indices) for m in ("dir", "place", "term", "L_first", "R_first", "L_term", "R_term")},
            "episode_transitions": {m: episode_transitions(records, y, x, m) for m in ("first", "ever", "terminal")},
        }
    factorial = {"available": has_old_native, "reference_is_reused_k14_B_mean": has_old_native}
    if has_old_native:
        factorial["cells"] = {
            "new_background_native_gain": "native", "new_background_fixed_gain": "new_gain",
            "old_background_native_gain": "old_background_native_gain_reference", "old_background_fixed_gain": "old_gain"}
        factorial["gain_effect_old_background"] = {
            m: compare(per, "old_gain", "old_background_native_gain_reference", m, boot_indices) for m in ("dir", "place", "term")}
        factorial["interaction_gain_effect_old_minus_new"] = {}
        for metric in ("dir", "place", "term"):
            n = 20 if metric == "dir" else 40
            delta = [a - b - c + d for a, b, c, d in zip(per["old_gain"][metric], per["old_background_native_gain_reference"][metric], per["new_gain"][metric], per["native"][metric])]
            values = sorted(sum(delta[k] for k in idx) / n for idx in boot_indices)
            factorial["interaction_gain_effect_old_minus_new"][metric] = {
                "point": sum(delta) / n,
                "ci95_cluster_bootstrap": [values[int(.025 * BOOTSTRAPS)], values[int(.975 * BOOTSTRAPS) - 1]],
                "n_snapshot_clusters":20,
            }

    summary = {}
    diagnostics = {}
    for condition in GROUPS:
        groups = {g: [records[i, condition, g] for i in INITS] for g in LABEL}
        summary[condition] = {"n_snapshots":20, "n_episodes":40,
            "totals":{k:sum(v) for k,v in per[condition].items()},
            "per_goal":{g:side_summary(part) for g,part in groups.items()}}
        if condition == "native":
            diagnostics[condition] = {"window_queries":0}
        else:
            all_rows = groups["A"] + groups["B"]
            diag_keys = set.intersection(*(set(r["diagnostics"]) for r in all_rows))
            diagnostics[condition] = {"window_queries":sum(r["n_window_queries"] for r in all_rows),
                "median_of_episode_means":{k:median(r["diagnostics"][k][0] for r in all_rows) for k in sorted(diag_keys)}}
        diagnostics[condition]["clip_rate_window_mean"] = sum(r["clip_rate_window"] for part in groups.values() for r in part) / 40
        diagnostics[condition]["clip_rate_all_mean"] = sum(r["clip_rate_all"] for part in groups.values() for r in part) / 40
    report = {"experiment":"K15 background x fixed gain missing cell", "episodes":120, "trace_steps":48000,
        "alpha":2.959, "inits":list(INITS), "model_sha256":MODEL_SHA,
        "audits":{"script_sha256":script_sha, "complete_markers":len(marker_files),
                  "engineering_gates_passed":True,
                  "geometry_first_goals_recomputed":120, "all_400_step_traces_checked":120,
                  "all_six_branches_share_prefix_per_init":True,
                  "native_and_old_gain_exact_reproductions":exact_reproductions,
                  "exact_reference_trace_count":80},
        "summary":summary, "per_snapshot":per, "comparisons":comparisons, "factorial":factorial,
        "diagnostics":diagnostics, "shared_prefixes":shared_prefixes,
        "statistics":{"unit":"physical snapshot; both goals resampled together", "n_clusters":20,
                      "bootstrap_replicates":BOOTSTRAPS, "seed":BOOT_SEED,
                      "point_gap_within_0_10_is_not_equivalence":True,
                      "development_data_only":True},
        "protocol":protocol,
        "source_sha256":{str(p.relative_to(archive)).replace('\\','/'):digest(p) for p in (*source_files,*marker_files)}}
    (archive / "summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    lines=["# K15 新共同背景＋固定增益验收", "", "120 回合、48000 轨迹步；80 条原生／旧背景固定增益基线逐步精确复现。", "",
           "| 条件 | 左先到左 /20 | 右先到右 /20 | 成对分流 /20 | 曾成功 /40 | 左终态 /20 | 右终态 /20 | 终态 /40 | 先到前 /40 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for condition in GROUPS:
        s=summary[condition]["totals"]
        lines.append(f"| {condition} | {s['L_first']} | {s['R_first']} | {s['dir']} | {s['place']} | {s['L_term']} | {s['R_term']} | {s['term']} | {s['front_first']} |")
    lines += ["", "| 配对比较 | 指标 | 差值 | 95% 快照聚类 bootstrap 区间 |", "|---|---|---:|---|"]
    for comparison in comparisons.values():
        for metric in ("dir","place","term"):
            c=comparison["metrics"][metric]
            lo,hi=c["ci95_cluster_bootstrap"]
            lines.append(f"| {c['x']} − {c['y']} | {metric} | {c['point']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", "dir 为成对首次水平分流；place 为曾满足原生请求目标成功；term 为第400步原生成功。", "",
              "本批是同20个开发快照上的机制补格，不是独立确认。未显著或点估计接近不代表等效；新背景下效果只适用于此固定增益与窗口。"]
    (archive / "summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("AUDITED 120 episodes / 48000 trace steps; 80 exact reference trace reproductions")
    for line in lines[4:10]:
        print(line)
    for key, value in comparisons.items():
        print(key, json.dumps({m:value["metrics"][m] for m in ("dir","place","term")},ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    main(args.archive.resolve())
