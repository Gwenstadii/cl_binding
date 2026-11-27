"""Offline mechanism-chain analysis of saved training probes and K13 trajectories.

No model inference, training, or new robot episodes are run by this script.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import median


INITS = tuple(range(20, 40))
GOALS = {"A": "left", "B": "right"}
CONDITIONS = ("ft4", "mean", "sites", "joint", "anti", "random", "oracle", "r1")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def extents(region):
    return [sum(abs(region["rotation"][i][j]) * region["half_size"][j] for j in range(3)) for i in range(3)]


def xy_inside(point, region):
    extent = extents(region)
    return all(abs(point[i] - region["centre"][i]) <= extent[i] for i in range(2))


def heights(point, region):
    return (point[2] - region["centre"][2] - extents(region)[2]) * 100


def quantile(values, probability):
    a = sorted(values)
    index = (len(a) - 1) * probability
    low = int(index)
    return a[low] + (a[min(low + 1, len(a) - 1)] - a[low]) * (index - low)


def main(repo, output, evidence):
    output.mkdir(parents=True, exist_ok=True)
    archive = repo / ".tmp/k13_r1_front_20_20261107_01/archive/k13_r1_front_20_20261107_01"
    k13 = load_json(repo / "docs/results/2026-11-07/k13_r1_front_20/summary.json")
    geometry_path = repo / "docs/results/2026-11-07/k13_r1_front_20/geometry.json"
    geometry = load_json(geometry_path)["sites"]
    sources = [geometry_path, repo / "docs/results/2026-11-07/k13_r1_front_20/summary.json"]
    rows, traces = {}, {}
    for init in INITS:
        for folder in (archive, archive / "oracle"):
            for suffix, destination in ((".jsonl", rows), ("_traces.jsonl", traces)):
                path = folder / f"i{init}{suffix}"
                sources.append(path)
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    if row["goal"] not in "AB":
                        continue
                    key = init, row["condition"], row["goal"]
                    assert key not in destination
                    destination[key] = row if destination is rows else row["rows"]
    assert len(rows) == len(traces) == len(INITS) * len(CONDITIONS) * 2
    for key, trace in traces.items():
        assert len(trace) == 400 and [r["step"] for r in trace] == list(range(1, 401))
        init, condition, goal = key
        if condition != "r1":
            onset = rows[key]["boundary"]
            assert trace[:onset] == traces[init, "ft4", goal][:onset]

    per_episode = []
    for (init, condition, goal), row in rows.items():
        trace = traces[init, condition, goal]
        region = geometry[GOALS[goal]]
        end = row["boundary"] + row["window"]
        first_native = next((r["step"] for r in trace if r["pred"][GOALS[goal]]), None)
        native_times = [r["step"] for r in trace if r["pred"][GOALS[goal]]]
        correct_xy = row["first"] == [GOALS[goal]]
        details = {"init": init, "condition": condition, "goal": goal,
                   "window_end": end, "first_correct_xy": row["first_step"] if correct_xy else None,
                   "first_native_success": first_native,
                   "first_correct_xy_confirmed_in_window": correct_xy and row["first_step"] + 4 <= end,
                   "ever_native_success": bool(native_times),
                   "native_success_in_window": any(row["boundary"] < t <= end for t in native_times),
                   "native_success_first_after_window": first_native is not None and first_native > end,
                   "terminal_native_success": trace[-1]["pred"][GOALS[goal]],
                   "terminal_front_success": trace[-1]["pred"]["front"],
                   "any_front_success": any(r["pred"]["front"] for r in trace),
                   "terminal_xy_correct": xy_inside(trace[-1]["obj"], region),
                   "terminal_height_above_region_top_cm": heights(trace[-1]["obj"], region),
                   "window_end_height_above_region_top_cm": heights(trace[end - 1]["obj"], region),
                   "window_end_xy_correct": xy_inside(trace[end - 1]["obj"], region)}
        details["arrival_aligned"] = {}
        if correct_xy:
            for offset in (4, 20, 60, 100):
                step = row["first_step"] + offset
                if step > 400:
                    continue
                r = trace[step - 1]
                details["arrival_aligned"][str(offset)] = {"step": step,
                    "height_above_region_top_cm": heights(r["obj"], region),
                    "xy_correct": xy_inside(r["obj"], region), "native_success": r["pred"][GOALS[goal]]}
            after = trace[row["first_step"] - 1:min(row["first_step"] + 60, 400)]
            details["native_success_within_60_after_xy_arrival"] = any(r["pred"][GOALS[goal]] for r in after)
        per_episode.append(details)

    grouped = {}
    for condition in CONDITIONS:
        grouped[condition] = {}
        for goal in "AB":
            group = [r for r in per_episode if r["condition"] == condition and r["goal"] == goal]
            outcome = k13["conditions"][condition]["goals"][goal]
            assert sum(r["ever_native_success"] for r in group) == outcome["ever_correct"]
            assert sum(r["terminal_native_success"] for r in group) == outcome["terminal_correct"]
            grouped[condition][goal] = {"n": len(group),
                **{key: sum(bool(r[key]) for r in group) for key in (
                    "first_correct_xy_confirmed_in_window", "ever_native_success", "native_success_in_window",
                    "native_success_first_after_window", "terminal_native_success", "terminal_front_success", "any_front_success", "terminal_xy_correct")},
                "terminal_xy_correct_and_above_top": sum(r["terminal_xy_correct"] and r["terminal_height_above_region_top_cm"] > 0 for r in group),
                "median_window_end_height_above_top_cm": median(r["window_end_height_above_region_top_cm"] for r in group),
                "median_terminal_height_above_top_cm": median(r["terminal_height_above_region_top_cm"] for r in group),
                "first_native_success_steps": [r["first_native_success"] for r in group if r["first_native_success"] is not None]}

    paired = {}
    for goal in "AB":
        paired[goal] = {}
        for offset in (0, 10, 20, 40, 60, 100, 200, 300):
            z_diff, xy_diff, xyz_diff = [], [], []
            for init in INITS:
                step = rows[init, "joint", goal]["boundary"] + offset
                a, b = (traces[init, c, goal][step - 1]["obj"] for c in ("joint", "oracle"))
                z_diff.append((a[2] - b[2]) * 100)
                xy_diff.append(math.dist(a[:2], b[:2]) * 100)
                xyz_diff.append(math.dist(a, b) * 100)
            paired[goal][str(offset)] = {"n": 20, "median_joint_minus_oracle_z_cm": median(z_diff),
                "n_joint_higher_by_over_0_5cm": sum(x > .5 for x in z_diff),
                "n_joint_lower_by_over_0_5cm": sum(x < -.5 for x in z_diff),
                "median_xy_separation_cm": median(xy_diff), "median_xyz_separation_cm": median(xyz_diff)}

    matched = [i for i in INITS if all(rows[i, c, "B"]["first"] == ["right"] for c in ("joint", "oracle"))]
    arrival = {"matched_inits": matched, "n": len(matched)}
    for condition in ("joint", "oracle"):
        group = [r for r in per_episode if r["init"] in matched and r["condition"] == condition and r["goal"] == "B"]
        arrival[condition] = {"ever_success": sum(r["ever_native_success"] for r in group),
            "terminal_success": sum(r["terminal_native_success"] for r in group),
            "success_within_60_after_xy_arrival": sum(r["native_success_within_60_after_xy_arrival"] for r in group),
            "by_offset": {str(offset): {"median_height_above_top_cm": median(r["arrival_aligned"][str(offset)]["height_above_region_top_cm"] for r in group),
                "xy_correct": sum(r["arrival_aligned"][str(offset)]["xy_correct"] for r in group),
                "native_success": sum(r["arrival_aligned"][str(offset)]["native_success"] for r in group)} for offset in (4, 20, 60, 100)}}

    fixed_path = repo / ".tmp/supervision_probe_20261105_01/evaluations/fixed_probe.json"
    fixed = load_json(fixed_path)
    sources.append(fixed_path)
    training = {}
    for step in (3000, 6000, 9000, 12000):
        records = [r for r in fixed["summary"] if r["step"] == step]
        assert len(records) == 5
        training[str(step)] = {"branches": {r["branch"]: {k: r[k] for k in ("old_diff_alignment", "shared_shift_toward_ft4_right", "shared_distance_to_own_right")} for r in records},
            "old_diff_projection_range": [min(r["old_diff_alignment"] for r in records), max(r["old_diff_alignment"] for r in records)]}
    k11_path = repo / ".tmp/k11_closed_loop_20261105_01/summary.json"
    sources.append(k11_path)
    assert next(r for r in fixed["summary"] if r["branch"] == "a_original" and r["step"] == 12000)["sha256"] == load_json(k11_path)["models"]["F"]
    trainables_path = evidence / "r1_front_trainable_parameters.json"
    trainables = load_json(trainables_path)
    sources.extend([trainables_path, evidence / "k1_existing_evidence.md"])
    audit_path = repo / ".tmp/supervision_probe_20261105_01/evaluations/training_audit.json"
    evaluation_path = repo / ".tmp/supervision_probe_20261105_01/evaluation_summary.json"
    manifest_path = repo / ".tmp/r1_then_front_review/training/manifest.json"
    sources.extend([audit_path, evaluation_path, manifest_path])
    supervision = load_json(audit_path)
    masking = {}
    for name in ("c_phase", "d_random"):
        counts = supervision["branches"][name]["mask_totals"]
        masking[name] = {**counts, "phase_fraction_masked": counts["phase_masked"] / counts["phase_valid"],
                        "all_fraction_masked": counts["masked"] / counts["valid"]}
    outcome_profile = {}
    for condition in CONDITIONS:
        groups = [r for r in per_episode if r["condition"] == condition]
        outcome_profile[condition] = {"n_old_commands": len(groups),
            "first_horizontal_new_front": sum(k13["conditions"][condition]["goals"][g]["first"].get("front", 0) for g in "AB"),
            "first_native_new_front": sum(k13["conditions"][condition]["goals"][g]["native_first"].get("front", 0) for g in "AB"),
            "ever_native_new_front": sum(r["any_front_success"] for r in groups),
            "terminal_correct_requested": sum(r["terminal_native_success"] for r in groups),
            "terminal_new_front": sum(r["terminal_front_success"] for r in groups),
            "terminal_other_or_none": sum(not r["terminal_front_success"] and not r["terminal_native_success"] for r in groups)}
        assert sum(outcome_profile[condition][k] for k in ("terminal_correct_requested", "terminal_new_front", "terminal_other_or_none")) == 40
    report = {"analysis_type": "offline reanalysis; no new training or rollouts", "k13_episodes_in_source": 340,
        "old_instruction_episodes_reanalyzed": len(rows), "old_instruction_trace_steps": len(rows) * 400,
        "new_probe_inputs": 0, "training_probe": {"line": "R0 left+front -> right ONLY; not an R1 time series",
            "snapshots": fixed["snapshot_count"], "anchors_per_checkpoint": fixed["anchors_per_checkpoint"],
            "checkpoint_results": training},
        "r1_trainables": {"parameter_count": trainables["count"], "tensor_count": len(trainables["names"]),
            "outside_expert": [n for n in trainables["names"] if ".lm_expert." not in n]},
        "r1_strict_outcome_profile": outcome_profile, "training_phase_mask_audit": masking,
        "trajectory_groups": grouped, "paired_joint_vs_oracle_by_offset_after_start": paired,
        "right_arrival_matched_subset": arrival, "per_episode": per_episode,
        "source_sha256": {str(p.relative_to(repo)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (output / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.6), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    time = list(range(0, 341))
    for condition, label, color in (("joint", "两层补丁＋旧平均", "#D87932"), ("oracle", "完整旧响应（同一短窗口）", "#24748B")):
        values = [[heights(traces[i, condition, "B"][rows[i, condition, "B"]["boundary"] + t - 1]["obj"], geometry["right"]) for i in INITS] for t in time]
        middle = [median(v) for v in values]
        axes[0].plot(time, middle, color=color, label=label, linewidth=2.5)
        axes[0].fill_between(time, [quantile(v, .25) for v in values], [quantile(v, .75) for v in values], color=color, alpha=.12)
        cumulative = [sum(any(r["pred"]["right"] for r in traces[i, condition, "B"][:rows[i, condition, "B"]["boundary"] + t]) for i in INITS) for t in time]
        axes[1].step(time, cumulative, color=color, linewidth=2.5, where="post", label=label)
    axes[0].axhline(0, color="#777777", linewidth=1, linestyle="--")
    axes[0].text(180, .8, "右格上边界", color="#555555")
    axes[0].set_ylabel("书中心高于右格上边界（厘米）")
    axes[0].set_title("R1 学前：朝右搬运与真正放入右格，仍有明显差距", loc="left", fontsize=14, pad=12)
    axes[0].legend(frameon=False, loc="best")
    axes[1].set_ylabel("曾满足右格原生成功的回合数 /20")
    axes[1].set_xlabel("从干预开始算起的控制步数（各初态按窗口起点对齐）")
    axes[1].set_ylim(-.5, 20.5)
    axes[1].set_yticks((0, 5, 10, 15, 20))
    for ax in axes:
        ax.axvspan(0, 60, color="#555555", alpha=.06)
        ax.axvline(60, color="#777777", linestyle=":", linewidth=1.3)
        ax.grid(axis="y", color="#dddddd", alpha=.6)
        ax.spines[["right", "top"]].set_visible(False)
    fig.text(.13, .015, "色带：中间 50% 范围（非置信区间）；灰区：干预 60 步。高度曲线不表示水平位置始终正确。", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(output / "right_placement_trajectory.png", dpi=180)
    fig.savefig(output / "right_placement_trajectory.svg")
    plt.close(fig)
    print("Analyzed", len(rows), "old-command episodes and", len(rows) * 400, "trace steps")
    for goal in "AB":
        print("paired", goal, json.dumps(paired[goal], ensure_ascii=False))
    print("matched right arrivals", json.dumps(arrival, ensure_ascii=False))
    print("groups", json.dumps({c: grouped[c]["B"] for c in ("joint", "oracle", "r1")}, ensure_ascii=False))
    print("trainables", json.dumps(report["r1_trainables"], ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    main(args.repo.resolve(), args.output.resolve(), args.evidence.resolve())
