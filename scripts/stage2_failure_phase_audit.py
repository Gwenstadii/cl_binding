"""Where do episodes that never reach any compartment fail? Offline, from saved per-step evaluation traces.

Compartment centres are the mean book position at the first step each goal predicate holds, pooled over every
successful episode in the given evaluation folders. Failures are split into: never lifted the book, lifted then
dropped away from all compartments, carried near the requested compartment without satisfying it, and carried
near a different compartment. Thresholds are fixed below and reported with the results.
"""
import argparse
import json
from pathlib import Path

import numpy as np

GOALS = ("left", "front", "right", "back")
LIFT_M = 0.05      # book raised at least 5 cm above its initial height
NEAR_M = 0.06      # within 6 cm (xy) of a compartment centre
HELD_M = 0.06      # book centre within 6 cm of the end effector


def load(folder):
    rows = {(r["init_index"], r["requested_goal"]): r for r in map(json.loads, open(folder / "episodes.jsonl"))}
    traces = {}
    for path in sorted(folder.glob("*_trace.jsonl")):
        goal, index = path.stem.replace("_trace", "").rsplit("_", 1)
        traces[int(index), goal] = [json.loads(line) for line in open(path)]
    return rows, traces


OBJECT = "black_book_1"


def book(trace):
    """Trajectory of the manipulated object (the book by default; --object overrides, e.g. akita_black_bowl_1)."""
    position = np.array([s["object_positions_after"][f"{OBJECT}_pos"] for s in trace])
    to_eef = np.array([s["object_positions_after"][f"{OBJECT}_to_robot0_eef_pos"] for s in trace])
    return position, np.linalg.norm(to_eef, axis=1)


def centres(sources):
    found = {g: [] for g in GOALS}
    for rows, traces in sources:
        for key, trace in traces.items():
            for g in GOALS:
                step = next((i for i, s in enumerate(trace) if s["predicates_after"][g]), None)
                if step is not None:
                    found[g].append(book(trace)[0][step])
    return {g: np.mean(v, axis=0) for g, v in found.items() if v}, {g: len(v) for g, v in found.items()}


def classify(trace, requested, centre):
    position, gap = book(trace)
    lift = position[:, 2] - position[0, 2]
    lifted = bool(lift.max() > LIFT_M)
    distance = {g: np.linalg.norm(position[:, :2] - c[:2], axis=1) for g, c in centre.items()}
    closest = {g: float(d.min()) for g, d in distance.items()}
    nearest = min(closest, key=closest.get)
    held_end = bool(gap[-1] < HELD_M)
    reached = any(s["predicates_after"][g] for s in trace for g in GOALS)
    if reached:
        category = "reached_some_goal"
    elif not lifted:
        category = "never_lifted"
    elif closest.get(requested, 9) < NEAR_M:
        category = "near_requested_not_satisfied"
    elif closest[nearest] < NEAR_M:
        category = f"near_other_{nearest}"
    else:
        category = "lifted_never_near_any"
    return {"category": category, "lifted": lifted, "max_lift_m": float(lift.max()), "held_at_end": held_end,
            "closest_xy_m": closest, "final_book_position": position[-1].tolist()}


def main(args):
    global OBJECT
    OBJECT = args.object
    sources = {label: load(Path(folder)) for label, folder in (item.split("=", 1) for item in args.eval)}
    centre, support = centres(sources.values())
    report = {"thresholds_m": {"lift": LIFT_M, "near": NEAR_M, "held": HELD_M},
              "compartment_centres": {g: c.tolist() for g, c in centre.items()}, "centre_support": support, "models": {}}
    for label, (rows, traces) in sources.items():
        per = {}
        for (index, goal), trace in traces.items():
            row = rows[index, goal]
            per.setdefault(goal, []).append({"init_index": index, "correct_ever": row["correct_ever"],
                                             "first_goal": row["first_goal"], **classify(trace, goal, centre)})
        summary = {}
        for goal, items in per.items():
            fails = [x for x in items if not x["correct_ever"]]
            counts = {}
            for x in fails:
                counts[x["category"]] = counts.get(x["category"], 0) + 1
            summary[goal] = {"episodes": len(items), "never_requested_success": len(fails), "failure_categories": counts,
                             "failures_lifted": sum(x["lifted"] for x in fails),
                             "failures_held_at_end": sum(x["held_at_end"] for x in fails)}
        report["models"][label] = {"summary": summary, "episodes": per}
        print(label, json.dumps({g: summary[g]["failure_categories"] for g in GOALS if g in summary}, ensure_ascii=False))
    args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval", nargs="+", required=True, help="label=evaluation folder holding episodes.jsonl and *_trace.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--object", default="black_book_1")
    main(p.parse_args())
