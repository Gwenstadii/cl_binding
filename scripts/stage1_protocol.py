"""Pure, testable data-access and behavioral-label rules for stage one."""

import random

GOALS = ("front", "left", "right", "back")


def split_trajectories(names, train_count, seed):
    if len(set(names)) != len(names) or not 0 < train_count < len(names):
        raise ValueError("Expected unique trajectories and nonempty train/validation splits")
    shuffled = sorted(names, key=lambda x: int(x.split("_")[-1]))
    random.Random(seed).shuffle(shuffled)
    return {"train": shuffled[:train_count], "validation": shuffled[train_count:]}


def summarize_events(requested, events):
    if requested not in GOALS or not events:
        raise ValueError("Expected a known request and initial/terminal events")
    if any(events[0].get(g, False) for g in GOALS):
        raise ValueError("Initial state already satisfies a goal")
    first_times = {g: next((i for i, e in enumerate(events) if e.get(g, False)), None) for g in GOALS}
    active = [g for g in GOALS if events[-1].get(g, False)]
    earliest = min((t for t in first_times.values() if t is not None), default=None)
    first = [g for g, t in first_times.items() if t is not None and t == earliest]
    terminal = active[0] if len(active) == 1 else None
    category = "ambiguous" if len(active) > 1 else "unfinished" if not active else "correct" if terminal == requested else "wrong_goal"
    return {"requested_goal": requested, "first_goal": first[0] if len(first) == 1 else None,
            "first_goal_times": first_times, "terminal_goal": terminal, "terminal_category": category,
            "correct_ever": first_times[requested] is not None,
            "correct_then_destroyed": first_times[requested] is not None and terminal != requested}


def qualification(rows, goals, expected_count, required_correct):
    counts = {}
    for goal in goals:
        selected = [r for r in rows if r["requested_goal"] == goal]
        counts[goal] = {"total": len(selected), "correct": sum(r["terminal_category"] == "correct" for r in selected)}
    return {"qualified": all(c["total"] == expected_count and c["correct"] >= required_correct for c in counts.values()), "counts": counts}
