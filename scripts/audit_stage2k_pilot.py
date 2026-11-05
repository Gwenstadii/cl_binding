"""Independently recompute stage2k counts and verify saved trace/restore evidence.

Standard library only; no policy imports, GPU calls, training, or source edits.
Run on the server with the two completed pilot directories, then archive the
small JSON output together with the source episode rows and entry manifest.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


GOALS = ("left", "front", "right", "back")
MODES = ("new", "old_mean", "old_difference", "old")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(rows):
    return {
        "total": len(rows), "correct_ever": sum(r["correct_ever"] for r in rows),
        "terminal_correct": sum(r["terminal_category"] == "correct" for r in rows),
        "correct_then_lost": sum(r["correct_then_destroyed"] for r in rows),
        "first_goal": dict(Counter(r["first_goal"] or "none_or_tied" for r in rows)),
        "terminal_category": dict(Counter(r["terminal_category"] for r in rows)),
    }


def audit(response, entry):
    protocol = json.loads((response / "protocol.json").read_text())
    finished = json.loads((response / "COMPLETE.json").read_text())
    manifest = json.loads((entry / "manifest.json").read_text())
    assert finished["status"] == manifest["status"] == "complete"
    assert finished["teacher_sha256"] == manifest["model_sha256"]["teacher"]
    indices = protocol["indices"]
    expected = {(g, i) for g in ("left", "front") for i in indices}
    reference_resets = {}
    result = {"scope": "development pilot only; strict native-goal events, not near-goal routing proxy",
              "indices": indices, "response": {}, "entry": {}, "source_files": {}}

    def record(path):
        result["source_files"][str(path)] = {"bytes": path.stat().st_size, "sha256": digest(path)}

    for path in (response / "protocol.json", response / "COMPLETE.json", entry / "manifest.json"):
        record(path)
    identities = json.loads((response / "sampling_identity.json").read_text())
    assert len(identities) == 4 and all(x["max_abs"] == x["rms"] == 0 for x in identities)
    result["sampling_identity_max_abs"] = max(x["max_abs"] for x in identities)
    for mode in MODES:
        episodes = response / mode / "episodes.jsonl"
        rows = [json.loads(line) for line in episodes.open()]
        assert len(rows) == len(expected)
        assert {(r["requested_goal"], r["init_index"]) for r in rows} == expected
        record(episodes)
        per_case = []
        clip_dimensions = [0] * 7
        for row in rows:
            goal, index = row["requested_goal"], row["init_index"]
            reset = tuple(row[key] for key in ("initial_state_sha256", "scene_xml_sha256", "seed"))
            key = (goal, index)
            assert reference_resets.setdefault(key, reset) == reset
            assert reset[1] == manifest["scene_xml_sha256"]
            trace_path = response / mode / f"{goal}_{index:02d}_trace.jsonl"
            trace = [json.loads(line) for line in trace_path.open()]
            assert len(trace) == row["steps"] == protocol["horizon"] == 400
            assert [r["step"] for r in trace] == list(range(1, 401))
            times = {g: next((r["step"] for r in trace if r["predicates_after"][g]), None) for g in GOALS}
            active = [g for g in GOALS if trace[-1]["predicates_after"][g]]
            earliest = min((t for t in times.values() if t is not None), default=None)
            first = [g for g, t in times.items() if t is not None and t == earliest]
            first_goal = first[0] if len(first) == 1 else None
            category = ("ambiguous" if len(active) > 1 else "unfinished" if not active else
                        "correct" if active[0] == goal else "wrong_goal")
            assert row["first_goal_times"] == times
            assert row["first_goal"] == first_goal
            assert row["terminal_category"] == category
            assert row["correct_ever"] == (times[goal] is not None)
            assert row["correct_then_destroyed"] == (times[goal] is not None and goal not in active)
            assert row["clipped_steps"] == sum(any(r["clipped_dimensions"]) for r in trace)
            for r in trace:
                assert len(r["action_executed"]) == 7
                assert all(math.isfinite(v) and -1 <= v <= 1 for v in r["action_executed"])
                for j in range(7):
                    clip_dimensions[j] += int(r["clipped_dimensions"][j])
            record(trace_path)
            per_case.append({key: row[key] for key in ("requested_goal", "init_index", "correct_ever",
                "first_goal", "terminal_goal", "terminal_category", "correct_then_destroyed", "clipped_steps")})
        result["response"][mode] = {**counts(rows), "by_goal": {
            goal: counts([r for r in rows if r["requested_goal"] == goal]) for goal in ("left", "front")},
            "clipped_dimensions_steps": clip_dimensions, "cases": per_case}

    prefixes = manifest["prefix_rows"]
    continuations = manifest["continuation_rows"]
    expected_prefix = {(s, g, i) for s in ("teacher", "student") for g, i in expected}
    assert len(prefixes) == len(expected_prefix) == 16
    assert {(r["source"], r["goal"], r["init_index"]) for r in prefixes} == expected_prefix
    snapshots = {(r["source"], r["goal"], r["init_index"]) for r in prefixes if r["entry_step"] is not None}
    assert len(continuations) == 2 * len(snapshots)
    assert {(r["source"], r["goal"], r["init_index"], r["continuation_policy"]) for r in continuations} == {
        (*key, policy) for key in snapshots for policy in ("teacher", "student")}
    self_checks = 0
    for row in continuations:
        assert row["steps"] == 120 and row["self_first_step_passed"]
        assert row["restore_check"]["sim_state_max_abs"] == 0
        assert all(v == 0 for v in row["restore_check"]["observation_max_abs"].values())
        if row["source"] == row["continuation_policy"]:
            assert row["self_first_action_max_abs"] == row["self_first_next_state_max_abs"] == 0
            self_checks += 1
    assert self_checks == len(snapshots)
    for source in ("teacher", "student"):
        for goal in ("left", "front"):
            subset = [r for r in prefixes if r["source"] == source and r["goal"] == goal]
            cell = {"prefix_counts": counts(subset), "entry_count": sum(r["entry_step"] is not None for r in subset),
                    "missing_entry_indices": [r["init_index"] for r in subset if r["entry_step"] is None],
                    "continuations": {}}
            for policy in ("teacher", "student"):
                rs = [r for r in continuations if r["source"] == source and r["goal"] == goal
                      and r["continuation_policy"] == policy]
                cell["continuations"][policy] = counts(rs)
                original = manifest["summary"][f"{source}:{goal}"]["continuations"][policy]
                assert original["total"] == len(rs) and original["correct_ever"] == sum(r["correct_ever"] for r in rs)
            result["entry"][f"{source}:{goal}"] = cell
    result["entry_cases"] = [{k: r[k] for k in ("source", "goal", "init_index", "entry_step",
        "continuation_policy", "correct_ever", "first_goal", "terminal_category")} for r in continuations]
    result["validation"] = {"response_traces_recomputed": 32, "response_steps_recomputed": 12800,
        "prefixes": len(prefixes), "entry_snapshots": len(snapshots), "continuations": len(continuations),
        "exact_restores": len(continuations), "exact_self_first_steps": self_checks,
        "trace_limit": "entry continuations have saved event summaries, not per-step event traces; no independent 120-step event recount",
        "same_reset_across_response_conditions": True, "same_teacher_between_experiments": True}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output already exists")
    report = audit(args.response, args.entry)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation": report["validation"], "response": {
        mode: {k: report["response"][mode][k] for k in ("total", "correct_ever", "terminal_correct", "first_goal")}
        for mode in MODES}}, ensure_ascii=False))
