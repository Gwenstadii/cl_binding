"""Exploratory, descriptive K15 trajectory audit; no causal attribution.

Reads raw i20..i39 episode/trace files. Writes only the new diagnostic JSON.
The formal preregistered audit is summarize_k15.py, maintained separately.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
CONDITIONS = ("native", "new_gain", "old_gain")
GOALS = ("left", "front", "right", "back")
REQUESTED = {"A": "left", "B": "right"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stat(values):
    values = list(values)
    return {"n": len(values), "minimum": min(values), "median": median(values),
            "maximum": max(values)} if values else {"n": 0}


def tag(first):
    return "/".join(first) or "none"


def xy_gap(position, site):
    extent = [sum(abs(site["rotation"][i][j]) * site["half_size"][j]
                  for j in range(3)) for i in range(2)]
    return math.hypot(*(max(0.0, abs(position[i] - site["centre"][i]) - extent[i])
                        for i in range(2)))


def quaternion_change(a, b):
    dot = abs(sum(x * y for x, y in zip(a, b)))
    denominator = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return 2 * math.degrees(math.acos(min(1.0, dot / denominator)))


def independently_first(rows, z0, geometry):
    runs = Counter()
    times = {g: None for g in GOALS}
    all_hits = []
    for row in rows:
        hits = [g for g in GOALS if row["obj"][2] - z0 > .03
                and xy_gap(row["obj"], geometry[g]) == 0]
        assert hits == row["hits"], (row["step"], hits, row["hits"])
        all_hits.extend(hits)
        for g in GOALS:
            runs[g] = runs[g] + 1 if g in hits else 0
            if runs[g] == 5 and times[g] is None:
                times[g] = row["step"] - 4
    earliest = min((x for x in times.values() if x is not None), default=None)
    first = [g for g in GOALS if earliest is not None and times[g] == earliest]
    return first, earliest, times, all_hits


def endpoint(row, boundary, z0, requested, geometry):
    target = geometry[requested]["centre"]
    toward = [target[i] - boundary["obj"][i] for i in range(2)]
    distance = math.hypot(*toward)
    delta = [row["obj"][i] - boundary["obj"][i] for i in range(2)]
    return {
        "step": row["step"], "book_xyz_m": row["obj"], "eef_xyz_m": row["eef"],
        "both_finger_contacts": row["c1"] and row["c2"],
        "book_lift_above_initial_m": row["obj"][2] - z0,
        "lifted_over_3cm": row["obj"][2] - z0 > .03,
        "book_xy_displacement_from_boundary_m": math.hypot(*delta),
        "book_y_displacement_from_boundary_m": delta[1],
        "toward_requested_center_xy_projection_m": sum(x * y for x, y in zip(delta, toward)) / distance,
        "distance_to_requested_center_xy_m": math.dist(row["obj"][:2], target[:2]),
        "requested_box_xy_gap_m": xy_gap(row["obj"], geometry[requested]),
        "nearest_any_box_xy_gap_m": min(xy_gap(row["obj"], geometry[g]) for g in GOALS),
        "book_relative_to_eef_position_change_m": math.dist(row["relpos"], boundary["relpos"]),
        "book_relative_to_eef_orientation_change_degrees": quaternion_change(row["relquat"], boundary["relquat"]),
    }


def action_clip(rows):
    arm = [any(abs(v) > 1 for v in r["a"][:6]) for r in rows]
    grip = [abs(r["a"][6]) > 1 for r in rows]
    assert all(r["clip"] == (a or g) for r, a, g in zip(rows, arm, grip))
    return {
        "n_rows": len(rows), "arm_first6_clipped_rows": sum(arm),
        "gripper_clipped_rows": sum(grip),
        "gripper_only_clipped_rows": sum(g and not a for a, g in zip(arm, grip)),
        "any_clipped_rows": sum(a or g for a, g in zip(arm, grip)),
        "maximum_absolute_arm_component": max(abs(v) for r in rows for v in r["a"][:6]),
        "maximum_absolute_gripper_component": max(abs(r["a"][6]) for r in rows),
        "maximum_arm_component_clip_difference": max(max(0.0, abs(v) - 1) for r in rows for v in r["a"][:6]),
        "maximum_gripper_clip_difference": max(max(0.0, abs(r["a"][6]) - 1) for r in rows),
    }


def aggregate(part):
    if not part:
        return {"n": 0}
    out = {
        "n": len(part), "first_distribution": dict(sorted(Counter(x["first"] for x in part).items())),
        "first_correct": sum(x["first"] == x["requested"] for x in part),
        "ever_requested_success": sum(x["ever_requested_success"] for x in part),
        "terminal_requested_success": sum(x["terminal_requested_success"] for x in part),
        "any_single_step_target_region_hit": sum(x["any_single_step_target_region_hit"] for x in part),
        "window_both_contacts_fraction": stat(x["window_both_contacts_fraction"] for x in part),
    }
    for end in ("window_end", "final"):
        out[end] = {
            "both_finger_contacts_count": sum(x[end]["both_finger_contacts"] for x in part),
            "lifted_over_3cm_count": sum(x[end]["lifted_over_3cm"] for x in part),
            "both_contacts_and_lifted_count": sum(x[end]["both_finger_contacts"] and x[end]["lifted_over_3cm"] for x in part),
        }
        for field in ("book_lift_above_initial_m", "book_xy_displacement_from_boundary_m",
                      "book_y_displacement_from_boundary_m", "toward_requested_center_xy_projection_m",
                      "distance_to_requested_center_xy_m", "requested_box_xy_gap_m", "nearest_any_box_xy_gap_m",
                      "book_relative_to_eef_position_change_m", "book_relative_to_eef_orientation_change_degrees"):
            out[end][field] = stat(x[end][field] for x in part)
    return out


def main():
    records, traces, paths = {}, {}, []
    for init in range(20, 40):
        for suffix, target in ((".jsonl", records), ("_traces.jsonl", traces)):
            path = HERE / f"i{init}{suffix}"
            paths.append(path)
            for line in path.read_text(encoding="utf-8").splitlines():
                x = json.loads(line)
                key = x["init"], x["condition"], x["goal"]
                assert key not in target
                target[key] = x["rows"] if target is traces else x
    expected = {(i, c, g) for i in range(20, 40) for c in CONDITIONS for g in REQUESTED}
    assert set(records) == set(traces) == expected
    geometry_path = HERE / "geometry.json"
    geometry = read(geometry_path)["sites"]
    paths.append(geometry_path)
    episodes = []
    window_rows = {c: [] for c in CONDITIONS}
    for key in sorted(records):
        r, rows = records[key], traces[key]
        assert len(rows) == 400 and [x["step"] for x in rows] == list(range(1, 401))
        first, first_step, times, hits = independently_first(rows, r["z0"], geometry)
        assert (first, first_step, times) == (r["first"], r["first_step"], r["first_times"])
        b = r["boundary"]
        boundary, win, post = rows[b - 1], rows[b:b + 60], rows[b:]
        assert boundary["c1"] and boundary["c2"] and boundary["obj"][2] - r["z0"] > .03
        window_rows[r["condition"]].extend(win)
        episodes.append({
            "init": r["init"], "condition": r["condition"], "goal": r["goal"], "requested": r["requested"],
            "boundary_step": b, "first": tag(first), "first_step": first_step,
            "ever_requested_success": r["ever"][r["requested"]],
            "terminal_requested_success": r["terminal"][r["requested"]],
            "any_single_step_target_region_hit": bool(hits),
            "post_window_any_native_success": any(any(x["pred"].values()) for x in post),
            "window_both_contacts_fraction": sum(x["c1"] and x["c2"] for x in win) / 60,
            "window_lifted_fraction": sum(x["obj"][2] - r["z0"] > .03 for x in win) / 60,
            "window_end": endpoint(win[-1], boundary, r["z0"], r["requested"], geometry),
            "final": endpoint(rows[-1], boundary, r["z0"], r["requested"], geometry),
            "window_action_clip": action_clip(win),
        })
    groups = {}
    for c in CONDITIONS:
        part = [x for x in episodes if x["condition"] == c]
        groups[c] = {"all": aggregate(part), "none": aggregate([x for x in part if x["first"] == "none"]),
                     "by_goal": {g: aggregate([x for x in part if x["goal"] == g]) for g in REQUESTED},
                     "none_by_goal": {g: aggregate([x for x in part if x["goal"] == g and x["first"] == "none"]) for g in REQUESTED},
                     "window_action_clip": action_clip(window_rows[c])}
    transitions = {}
    for before, after in (("native", "new_gain"), ("old_gain", "new_gain")):
        for goal in (None, "A", "B"):
            counts = Counter((tag(records[i, before, g]["first"]), tag(records[i, after, g]["first"]))
                             for i in range(20, 40) for g in REQUESTED if goal is None or g == goal)
            transitions[f"{before}_to_{after}_{goal or 'all'}"] = [
                {"before": a, "after": b, "count": n} for (a, b), n in sorted(counts.items())]
    result = {
        "status": "exploratory descriptive audit; formal acceptance belongs to summarize_k15.py",
        "definitions": {
            "none": "No 5-consecutive-step goal XY entry while book is >3cm above step-1 height, by absolute step400; does not mean no movement.",
            "window_end": "At boundary+60 after the patch window; displacement relative to common boundary, lift relative to step1.",
            "contacts": "Simultaneous geometric contacts with the two fingers; not a force/secure-grasp measurement.",
            "requested_box_xy_gap_m": "Euclidean XY distance to the requested compartment rectangle; zero is inside its horizontal projection.",
            "orientation": "Book orientation relative to gripper compared with boundary, not world orientation.",
            "clip": "First6 arm versus seventh gripper pre-clamp components outside [-1,1]. Executed values inferred from archived source's exact clamp, not a separate recorded executed vector.",
            "inference_limit": "These trajectory associations describe failure forms; they do not locate training-update causes or establish mediators as causal.",
        },
        "n_episodes": len(episodes), "n_steps": sum(len(x) for x in traces.values()),
        "independent_geometry_recomputation_matches_all_120": True,
        "groups": groups, "paired_first_transitions": transitions, "episodes": episodes,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    out = HERE / "exploratory_trajectory_diagnostics.json"
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(out), "episodes": len(episodes),
                      "first": {c: {g: groups[c]["by_goal"][g]["first_distribution"] for g in REQUESTED} for c in CONDITIONS},
                      "none": {c: groups[c]["none"]["n"] for c in CONDITIONS},
                      "arm_clip_rows": {c: groups[c]["window_action_clip"]["arm_first6_clipped_rows"] for c in CONDITIONS}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
