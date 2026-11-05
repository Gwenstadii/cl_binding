"""Independent, CPU-only audit of a completed 12-state common-noise P1 run.

No policy/worker imports. Recounts both event protocols, verifies every trace,
recorded Gaussian schedule, restore check, file digest, and RGB-array header.
"""

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile


GOALS = ("left", "front", "right", "back")
POLICIES = ("teacher", "student")
SUMMARY_KEYS = ("requested_goal", "first_goal", "first_goal_times", "terminal_goal",
                "terminal_category", "correct_ever", "correct_then_destroyed")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def events_summary(goal, events):
    assert events and all(set(event) == set(GOALS) for event in events)
    assert all(type(value) is bool for event in events for value in event.values())
    assert not any(events[0].values()), "Initial snapshot already satisfies a goal"
    times = {g: next((step for step, event in enumerate(events) if event[g]), None) for g in GOALS}
    earliest = min((step for step in times.values() if step is not None), default=None)
    first = [g for g, step in times.items() if step is not None and step == earliest]
    active = [g for g in GOALS if events[-1][g]]
    terminal = active[0] if len(active) == 1 else None
    category = ("ambiguous" if len(active) > 1 else "unfinished" if not active else
                "correct" if terminal == goal else "wrong_goal")
    return {"requested_goal": goal, "first_goal": first[0] if len(first) == 1 else None,
            "first_goal_times": times, "terminal_goal": terminal, "terminal_category": category,
            "correct_ever": times[goal] is not None,
            "correct_then_destroyed": times[goal] is not None and not events[-1][goal]}


def counts(rows, native=False):
    summaries = [r["native_post_step_summary"] if native else r for r in rows]
    return {"continuations": len(rows), "distinct_snapshots": len({r["snapshot_id"] for r in rows}),
            "correct_ever": sum(r["correct_ever"] for r in summaries),
            "terminal_correct": sum(r["terminal_category"] == "correct" for r in summaries),
            "correct_then_lost": sum(r["correct_then_destroyed"] for r in summaries),
            "first_goal_counts": dict(Counter(r["first_goal"] or "none_or_tied" for r in summaries))}


def cells(rows, native=False):
    return {f"{source}:{goal}": {policy: counts([r for r in rows if r["source"] == source
            and r["goal"] == goal and r["continuation_policy"] == policy], native)
            for policy in POLICIES} for source in POLICIES for goal in ("left", "front")}


def rgb_headers(path):
    headers = {}
    with zipfile.ZipFile(path) as archive:
        expected = {"agentview_image.npy", "robot0_eye_in_hand_image.npy"}
        assert set(archive.namelist()) == expected, path
        for name in sorted(expected):
            with archive.open(name) as handle:
                assert handle.read(6) == b"\x93NUMPY", (path, name)
                version = tuple(handle.read(2))
                assert version in ((1, 0), (2, 0), (3, 0)), version
                width = 2 if version == (1, 0) else 4
                size = struct.unpack("<H" if width == 2 else "<I", handle.read(width))[0]
                header = ast.literal_eval(handle.read(size).decode("utf-8" if version == (3, 0) else "latin1").strip())
            assert tuple(header["shape"]) == (121, 128, 128, 3), (path, name, header)
            assert header["descr"] in ("|u1", "<u1", ">u1"), header
            assert header["fortran_order"] is False, header
            headers[name] = {"shape": list(header["shape"]), "dtype": header["descr"], "npy_version": list(version)}
    return headers


def audit(run):
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    complete_path = run / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    assert manifest["status"] == complete["status"] == "complete"
    ids, seeds, rows = manifest["selected_snapshot_ids"], manifest["continuation_seeds"], manifest["rows"]
    assert len(ids) == len(set(ids)) == 12
    assert len(seeds) == len(set(seeds)) == 3
    assert manifest["arguments"]["seeds"] == seeds
    assert manifest["primary_horizon"] == 120
    assert manifest["planned_continuations"] == complete["continuations"] == len(rows) == 72
    expected = {(ident, policy, seed) for ident in ids for policy in POLICIES for seed in seeds}
    keyed = {(r["snapshot_id"], r["continuation_policy"], r["noise_seed"]): r for r in rows}
    assert len(keyed) == len(rows) and set(keyed) == expected
    report = {"scope": "72 continuations of 12 selected existing states, not 72 independent states",
              "distinct_snapshots": 12, "seeds": seeds, "continuations": 72, "primary_horizon": 120,
              "source_files": {}, "per_rollout": [], "noise_schedules": {}}

    def record(path):
        item = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        report["source_files"][str(path)] = item
        return item

    record(manifest_path); record(complete_path)
    source_checks = manifest["source_native_self_checks"]
    assert len(source_checks) == 12 and {r["snapshot_id"] for r in source_checks} == set(ids)
    assert len({r["snapshot_id"] for r in source_checks}) == 12
    for check in source_checks:
        assert check["passed"] and 0 <= check["first_action_max_abs"] <= 1e-5
        assert 0 <= check["first_next_state_max_abs"] <= 1e-5
        assert check["restore_check"]["sim_state_max_abs"] == 0
        assert all(v == 0 for v in check["restore_check"]["observation_max_abs"].values())
    contact_total = contact_force = clip_steps = 0
    clip_dimensions = [0] * 7
    primary_native_disagreements = Counter()
    identities = {}
    for row in rows:
        assert row["status"] == "complete" and row["steps"] == 120
        source, goal, index = row["snapshot_id"].split("_")
        assert source == row["source"] and goal == row["goal"] and int(index) == row["init_index"]
        identities[row["snapshot_id"]] = (source, goal, int(index))
        restore = row["restore_check"]
        assert restore["sim_state_max_abs"] == 0
        assert all(v == 0 for v in restore["observation_max_abs"].values())
        folder = run / row["rollout"]
        assert folder.resolve().parent == run.resolve(), folder
        result = json.loads((folder / "result.json").read_text())
        assert result == row, folder
        record(folder / "result.json")
        for filename in ("initial.json", "trace.jsonl", "observations.npz"):
            assert record(folder / filename) == row["files"][filename], (folder, filename)
        initial = json.loads((folder / "initial.json").read_text())
        assert initial["rgb_frame"] == 0 and initial["restore_check"] == restore
        trace = [json.loads(line) for line in (folder / "trace.jsonl").open()]
        assert len(trace) == 120 and [r["step"] for r in trace] == list(range(1, 121))
        assert [r["input_obs_frame_index"] for r in trace] == list(range(120))
        assert trace[0]["observation_before"] == initial["observations"]
        assert all(a["observation_after"] == b["observation_before"] for a, b in zip(trace, trace[1:]))
        noise_records = []
        rollout_clip = 0
        for step in trace:
            n = step["step"]
            assert step["replanning_chunk"] == (n - 1) // 10
            raw, executed, clipped = step["action_unclipped"], step["action_executed"], step["clipped_dimensions"]
            assert len(raw) == len(executed) == len(clipped) == 7
            assert all(math.isfinite(v) for v in raw)
            assert all(math.isfinite(v) and -1 <= v <= 1 for v in executed)
            assert executed == [max(-1, min(1, v)) for v in raw]
            assert clipped == [a != b for a, b in zip(raw, executed)]
            rollout_clip += int(any(clipped))
            for j, value in enumerate(clipped): clip_dimensions[j] += int(value)
            noise = step["new_noise_chunk"]
            if (n - 1) % 10 == 0:
                assert noise is not None and noise["chunk"] == (n - 1) // 10
                assert noise["shape"] == [1, 50, 32]
                assert len(noise["sha256"]) == 64 and all(c in "0123456789abcdef" for c in noise["sha256"])
                noise_records.append(noise)
            else:
                assert noise is None
            native, refreshed = step["native_post_step_predicates"], step["boundary_refresh_predicates"]
            assert (refreshed is not None) == (n % 10 == 0 and n < 120)
            assert step["primary_predicates_after"] == (refreshed if refreshed is not None else native)
            assert set(step["target_xy_distances_m"]) == set(GOALS)
            assert all(math.isfinite(v) and v >= 0 for v in step["target_xy_distances_m"].values())
            assert len(step["book_relative_to_eef"]["position_in_eef_frame"]) == 3
            assert len(step["book_relative_to_eef"]["quaternion_xyzw"]) == 4
            for contact in step["contacts_after"]:
                contact_total += 1
                if contact.get("force") is not None:
                    assert len(contact["force"]) == 6 and all(math.isfinite(v) for v in contact["force"])
                    contact_force += 1
        assert len(noise_records) == 12 and noise_records == row["noise_chunks"]
        schedule = [r["sha256"] for r in noise_records]
        key = str(row["noise_seed"])
        assert report["noise_schedules"].setdefault(key, schedule) == schedule
        assert manifest["noise_schedules"][key] == schedule
        primary = events_summary(goal, [initial["events"]] + [r["primary_predicates_after"] for r in trace])
        native = events_summary(goal, [initial["events"]] + [r["native_post_step_predicates"] for r in trace])
        assert {key: row[key] for key in SUMMARY_KEYS} == primary, folder
        assert row["native_post_step_summary"] == native, folder
        for key in SUMMARY_KEYS:
            if primary[key] != native[key]: primary_native_disagreements[key] += 1
        clip_steps += rollout_clip
        headers = rgb_headers(folder / "observations.npz")
        report["per_rollout"].append({"rollout": row["rollout"], "snapshot_id": row["snapshot_id"],
            "policy": row["continuation_policy"], "seed": row["noise_seed"], "clipped_steps": rollout_clip,
            "primary_summary": primary, "native_summary": native, "rgb_headers": headers})
    assert Counter((s, g) for s, g, i in identities.values()) == {
        ("teacher", "left"): 3, ("teacher", "front"): 4, ("student", "left"): 1, ("student", "front"): 4}
    report["cells"] = cells(rows)
    report["native_cells"] = cells(rows, native=True)
    report["cells_by_seed"] = {str(seed): cells([r for r in rows if r["noise_seed"] == seed]) for seed in seeds}
    report["per_snapshot"] = {}
    for ident in ids:
        part = [keyed[ident, policy, seed] for policy in POLICIES for seed in seeds]
        pair_counts = Counter()
        for seed in seeds:
            t, s = [keyed[ident, policy, seed]["correct_ever"] for policy in POLICIES]
            pair_counts["both" if t and s else "teacher_only" if t else "student_only" if s else "neither"] += 1
        report["per_snapshot"][ident] = {"source": identities[ident][0], "goal": identities[ident][1],
            "init_index": identities[ident][2], "seed_count": 3,
            "policies": {policy: counts([r for r in part if r["continuation_policy"] == policy]) for policy in POLICIES},
            "paired_outcomes": dict(pair_counts)}
    report["seed_source_pair_front"] = []
    source_pairs = {policy: Counter() for policy in POLICIES}
    for seed in seeds:
        for index in range(4):
            item = {"init_index": index, "noise_seed": seed, "outcomes": {}}
            for policy in POLICIES:
                t = keyed[f"teacher_front_{index:02d}", policy, seed]["correct_ever"]
                s = keyed[f"student_front_{index:02d}", policy, seed]["correct_ever"]
                item["outcomes"][policy] = {"teacher_source": t, "student_source": s}
                source_pairs[policy]["both" if t and s else "teacher_source_only" if t else
                                     "student_source_only" if s else "neither"] += 1
            report["seed_source_pair_front"].append(item)
    report["front_source_pair_counts"] = {key: dict(value) for key, value in source_pairs.items()}
    report["validation"] = {"complete_matrix": True, "traces": 72, "steps": 8640,
        "rgb_arrays_headers_verified": 144, "exact_initial_restores": 72,
        "native_self_checks": 12, "self_action_max_abs": max(r["first_action_max_abs"] for r in source_checks),
        "self_next_state_max_abs": max(r["first_next_state_max_abs"] for r in source_checks),
        "noise_chunks_per_rollout": 12, "common_schedules_verified_across_every_state_source_policy": True,
        "primary_native_summary_disagreements": dict(primary_native_disagreements),
        "clipped_steps": clip_steps, "clipped_dimensions_steps": clip_dimensions,
        "contacts_recorded": contact_total, "contacts_with_native_force": contact_force,
        "limits": "no model/GPU rerun; RGB headers/digests checked without decoding all pixels; 12 selected states only"}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = audit(args.run)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"validation": result["validation"], "cells": result["cells"]}, ensure_ascii=False))
