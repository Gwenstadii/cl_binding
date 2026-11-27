"""Audit all 300 R1 transfer rollouts and summarize paired causal contrasts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
import importlib.util

ROOT = Path(__file__).resolve().parent
CONDITIONS = ("ft4", "mean", "sites", "joint", "anti", "random")
INITS = tuple(range(20, 40))
LABEL = {"A": "left", "B": "right", "N": "front"}
KEYS = {(c, g) for c in CONDITIONS for g in ("ABN" if c == "ft4" else "AB")} | {("r1", g) for g in "AB"}
MODELS = {"R": "225322ba28d0ad152fcced19950c43e3e40a435f1835cf64a664226da7830003",
          "F": "8ff1bf8f372d728a3218101a3cc93fb6fb9d484e8fefb39c48d81a0c8ab2c515"}


def load(path, expected=KEYS):
    raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows = {(r["condition"], r["goal"]): r for r in raw}
    assert set(rows) == expected and len(raw) == len(expected), path
    return rows


def tag(first):
    return "/".join(first) or "none"


def group_stats(part):
    return {"n": len(part),
            "first": dict(sorted(Counter(tag(r["first"]) for r in part).items())),
            "first_correct": sum(r["first"] == [r["requested"]] for r in part),
            "first_new_front": sum(r["first"] == ["front"] for r in part),
            "ever_correct": sum(r["ever"][r["requested"]] for r in part),
            "terminal_correct": sum(r["terminal"][r["requested"]] for r in part),
            "native_first": dict(sorted(Counter(tag(r["audit_native_first"]) for r in part).items())),
            "first_correct_after_window": sum(r["first"] == [r["requested"]] and r["first_step"] > r["boundary"] + r["window"] for r in part),
            "first_correct_then_terminal_front": sum(r["first"] == [r["requested"]] and r["terminal"]["front"] for r in part),
            "first_correct_above_region_top_for_5_steps": sum(r["audit_first_correct_above_top"] for r in part),
            "terminal_xy_correct_but_native_false": sum(r["audit_terminal_xy_correct"] and not r["terminal"][r["requested"]] for r in part),
            "terminal_above_region_top_with_correct_xy": sum(r["audit_terminal_above_top"] for r in part),
            "first_to_terminal": dict(sorted(Counter(f"{tag(r['first'])} -> {tag([g for g, hit in r['terminal'].items() if hit])}" for r in part).items())),
            "terminal_distribution": dict(sorted(Counter(tag([g for g, hit in r["terminal"].items() if hit]) for r in part).items()))}


def main(folder, require_oracle=True):
    dependency_hashes = json.loads((folder / "dependency_sha256.json").read_text())
    for name, expected in dependency_hashes.items():
        assert hashlib.sha256((folder / "dependencies" / name).read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location("audited_geometry", folder / "dependencies/k1_geometry.py")
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    geometry = json.loads((folder / "geometry.json").read_text())["sites"]

    def spatial_audit(row, trace):
        region = geometry[row["requested"]]
        upper = region["centre"][2] + geo.extents(region, (2,))[0]
        row["audit_terminal_xy_correct"] = geo.inside(trace[-1]["obj"], region)
        row["audit_terminal_above_top"] = row["audit_terminal_xy_correct"] and trace[-1]["obj"][2] > upper
        row["audit_first_correct_above_top"] = (row["first"] == [row["requested"]]
            and all(r["obj"][2] > upper for r in trace[row["first_step"] - 1:row["first_step"] + 4]))
    digests = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
               for name in ("r1_front_start.py", "k11_reference.py")}
    # Ensure that the mathematical intervention helper was transferred byte for byte.
    assert digests["k11_reference.py"] == "7add12361efdceb577744fbe68b3fd5696c59e33884084d4ce1ce5e80865f308"
    rows, anchors, boundary, processors = {}, [], {}, None
    for init in INITS:
        marker = json.loads((folder / f"i{init}_COMPLETE.json").read_text())
        onset = 50 if init == 22 else 60
        assert marker["init"] == init and marker["boundary"] == onset and marker["window"] == 60
        assert marker["episode_count"] == 15 and tuple(marker["conditions"]) == CONDITIONS
        assert marker["script_sha256"] == digests["r1_front_start.py"]
        assert marker["intervention_sha256"] == digests["k11_reference.py"]
        assert marker["model_sha256"] == MODELS
        assert marker["instructions"] == {g: f"pick up the book and place it in the {name} compartment of the caddy"
                                             for g, name in LABEL.items()}
        assert marker["sites"] == ["CA-7", "MLP-9"]
        if processors is None:
            processors = marker["processor_check"]
        assert processors == marker["processor_check"]
        part, traces = load(folder / f"i{init}.jsonl"), load(folder / f"i{init}_traces.jsonl")
        for (cond, goal), row in part.items():
            rows[init, cond, goal] = row
            assert row["init"] == init and row["requested"] == LABEL[goal]
            assert row["prefix_model"] == ("R1" if cond == "r1" else "Front-12000")
            assert row["prefix_goal"] == goal
            assert row["reset_signature"] == part["ft4", "A"]["reset_signature"]
            assert row["boundary"] == onset and row["window"] == 60
            trace = traces[cond, goal]["rows"]
            assert [r["step"] for r in trace] == list(range(1, 401))
            if cond != "r1":
                base = part["ft4", goal]
                for field in ("seed", "z0", "boundary_signature", "held_at_boundary", "pretarget_at_boundary"):
                    assert row[field] == base[field]
                assert trace[:onset] == traces["ft4", goal]["rows"][:onset]
            assert row["terminal"] == trace[-1]["pred"]
            recomputed = geo.movement("scene2", geometry, trace, row["z0"], None)
            assert row["first"] == recomputed["first_targets"]
            assert row["first_step"] == recomputed["first_step"] and row["first_times"] == recomputed["times"]
            assert row["ever"] == {g: any(r["pred"][g] for r in trace) for g in row["terminal"]}
            row["audit_native_first"] = next(([g for g, hit in r["pred"].items() if hit] for r in trace if any(r["pred"].values())), [])
            spatial_audit(row, trace)
            first = min((t for t in row["first_times"].values() if t is not None), default=None)
            assert first == row["first_step"]
            assert set(row["first"]) == {g for g, t in row["first_times"].items() if t is not None and t == first}
            if cond == "ft4" and goal in "AB":
                assert row["fixed_anchor"]["prefix_goal"] == goal
                anchors.append({"init": init, "goal": goal, **row["fixed_anchor"]})
        boundary[init] = {g: {k: part["ft4", g][k] for k in ("held_at_boundary", "pretarget_at_boundary")}
                          for g in "AB"}
        boundary[init]["same_ab_boundary"] = part["ft4", "A"]["boundary_signature"] == part["ft4", "B"]["boundary_signature"]
    assert len(rows) == 300
    if not require_oracle:
        print("AUDITED 300 main episodes / 120000 trace steps")
        return rows
    oracle_digest = hashlib.sha256((folder / "oracle_start.py").read_bytes()).hexdigest()
    digests["oracle_start.py"] = oracle_digest
    for init in INITS:
        sub = folder / "oracle"
        marker = json.loads((sub / f"i{init}_COMPLETE.json").read_text())
        onset = 50 if init == 22 else 60
        assert marker["model_sha256"] == MODELS and marker["episode_count"] == 2
        assert marker["conditions"] == ["oracle"] and marker["boundary"] == onset and marker["window"] == 60
        assert marker["script_sha256"] == oracle_digest and marker["intervention_sha256"] == digests["k11_reference.py"]
        assert marker["processor_check"] == processors
        part = load(sub / f"i{init}.jsonl", {("oracle", g) for g in "AB"})
        traces = load(sub / f"i{init}_traces.jsonl", {("oracle", g) for g in "AB"})
        native_traces = load(folder / f"i{init}_traces.jsonl")
        for (cond, goal), row in part.items():
            base = rows[init, "ft4", goal]
            for field in ("seed", "z0", "reset_signature", "boundary_signature", "prefix_model", "prefix_goal", "held_at_boundary", "pretarget_at_boundary"):
                assert row[field] == base[field]
            assert row["requested"] == LABEL[goal] and row["boundary"] == onset and row["window"] == 60
            trace = traces[cond, goal]["rows"]
            assert [r["step"] for r in trace] == list(range(1, 401))
            assert trace[:onset] == native_traces["ft4", goal]["rows"][:onset]
            assert row["terminal"] == trace[-1]["pred"]
            assert row["ever"] == {g: any(r["pred"][g] for r in trace) for g in row["terminal"]}
            row["audit_native_first"] = next(([g for g, hit in r["pred"].items() if hit] for r in trace if any(r["pred"].values())), [])
            spatial_audit(row, trace)
            movement = geo.movement("scene2", geometry, trace, row["z0"], None)
            assert row["first"] == movement["first_targets"] and row["first_times"] == movement["times"]
            assert row["first_step"] == movement["first_step"]
            assert abs(row["diagnostics"]["old_direction_projection"] - 1) < .02
            rows[init, cond, goal] = row
    assert len(rows) == 340
    conditions = {}
    for cond in (*CONDITIONS, "oracle", "r1"):
        groups = {g: [rows[i, cond, g] for i in INITS] for g in "AB"}
        mixed = groups["A"] + groups["B"]
        pairs = {i: [rows[i, cond, g]["first"] for g in "AB"] for i in INITS}
        conditions[cond] = {"goals": {g: group_stats(part) for g, part in groups.items()},
                            "combined": group_stats(mixed),
                            "pairs_correct": [i for i, p in pairs.items() if p == [["left"], ["right"]]],
                            "pairs_terminal_correct": [i for i in INITS if all(rows[i, cond, g]["terminal"][LABEL[g]] for g in "AB")],
                            "pairs_both_front": [i for i, p in pairs.items() if p == [["front"], ["front"]]],
                            "pairs": pairs}
    comparisons = {}
    for old, new in (("ft4", "mean"), ("mean", "joint"), ("joint", "anti"), ("joint", "random"), ("r1", "joint"), ("joint", "oracle"), ("r1", "oracle")):
        pairsets = {c: set(conditions[c]["pairs_correct"]) for c in (old, new)}
        comparisons[f"{old}_to_{new}"] = {
            "paired_added": sorted(pairsets[new] - pairsets[old]),
            "paired_lost": sorted(pairsets[old] - pairsets[new]),
            "first_transitions": {g: dict(sorted(Counter(
                f"{tag(rows[i, old, g]['first'])} -> {tag(rows[i, new, g]['first'])}" for i in INITS).items())) for g in "AB"},
            "terminal_transitions": {g: dict(sorted(Counter(
                f"{int(rows[i, old, g]['terminal'][LABEL[g]])} -> {int(rows[i, new, g]['terminal'][LABEL[g]])}" for i in INITS).items())) for g in "AB"}}
    added = set(conditions["joint"]["pairs_correct"]) - set(conditions["mean"]["pairs_correct"])
    comparisons["joint_added_unique_vs_controls"] = sorted(added - set(conditions["anti"]["pairs_correct"]) - set(conditions["random"]["pairs_correct"]))
    fixed_fields = ("adapted_mean_distance_to_front_over_r1", "adapted_diff_norm_over_r1",
                    "adapted_diff_cosine_to_r1", "adapted_diff_projection_over_r1")
    fixed = {"n": len(anchors), **{key: median(a[key] for a in anchors) for key in fixed_fields},
             "mean_closer_to_adapted_front": sum(a[fixed_fields[0]] < 1 for a in anchors),
             "edit_projection": {c: median(a["edits"][c]["old_direction_projection"] for a in anchors) for c in CONDITIONS[1:]}}
    fixed["by_prefix_goal"] = {g: {**{key: median(a[key] for a in anchors if a["goal"] == g) for key in fixed_fields},
                                    "edit_projection": {c: median(a["edits"][c]["old_direction_projection"] for a in anchors if a["goal"] == g)
                                                        for c in CONDITIONS[1:]}} for g in "AB"}
    bcounts = {g: {field: sum(x[g][field] for x in boundary.values())
                   for field in ("held_at_boundary", "pretarget_at_boundary")} for g in "AB"}
    bcounts["same_ab_boundary"] = sum(x["same_ab_boundary"] for x in boundary.values())
    report = {"experiment": "k13_r1_front_20_20261107_01", "episodes": len(rows), "model_sha256": MODELS,
              "script_sha256": digests, "normalization_audit": processors, "conditions": conditions,
              "comparisons": comparisons, "boundary_counts": bcounts, "boundary_checks": boundary,
              "fixed_anchor_summary": fixed, "fixed_anchors": anchors,
              "native_front": group_stats([rows[i, "ft4", "N"] for i in INITS])}
    (ROOT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("AUDITED", len(rows), "episodes / 136000 trace steps")
    for c, data in conditions.items():
        print(c, json.dumps({"goals": data["goals"], "first_front": data["combined"]["first_new_front"],
                             "pairs_correct": data["pairs_correct"], "pairs_terminal_correct": data["pairs_terminal_correct"]}))
    print("comparisons", json.dumps(comparisons))
    print("fixed", json.dumps(fixed))
    print("boundary", json.dumps(bcounts))
    print("native_front", json.dumps(report["native_front"]))


if __name__ == "__main__":
    main(Path(sys.argv[1]), require_oracle="--main-only" not in sys.argv[2:])
