"""Independent standard-library audit of fixed closed-loop CA-message rescue.

Task success is recomputed from all four native predicates at every step.
No model, simulator, worker or training module is imported.
"""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path


GOALS = ("front", "left", "right", "back")
OLD_GOALS = ("left", "front")
CONDITIONS = {
    "FT4": ("native", "cross_same", "cross_wrong", "random_delta"),
    "R0": ("native", "cross_same", "same_model_wrong", "random_delta"),
}
INDICES = (0, 1)
SEEDS = (2026102801, 2026102802)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line, parse_constant=reject_constant) for line in stream]


def file_info(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return {"bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def expected_episode_keys():
    keys = {(model, condition, goal, index, seed)
            for model, conditions in CONDITIONS.items() for condition in conditions
            for goal in OLD_GOALS for index in INDICES for seed in SEEDS}
    keys.update(("FT4", "native", "right", index, seed) for index in INDICES for seed in SEEDS)
    return keys


def index_unique(rows, fields, expected=None):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        require(key not in result, f"Duplicate evidence cell: {key}")
        result[key] = row
    if expected is not None:
        require(set(result) == set(expected),
                f"Evidence matrix differs: expected {len(expected)}, actual {len(result)}")
    return result


def summarize_predicates(requested_goal, flags, horizon=400, window=120):
    """Distinguish ever, first, terminal, and post-withdrawal success explicitly."""
    require(requested_goal in GOALS, "Unknown requested goal")
    require(len(flags) == horizon+1, "Trace must contain reset and every executed step")
    for state in flags:
        require(set(state) == set(GOALS) and all(type(state[g]) is bool for g in GOALS),
                "Four strict Boolean native predicates required")
    require(not any(flags[0].values()), "Reset already satisfies a goal")
    require(0 < window < horizon, "Invalid intervention window")
    first_times = {goal: next((i for i, state in enumerate(flags) if state[goal]), None)
                   for goal in GOALS}
    earliest = min((value for value in first_times.values() if value is not None), default=None)
    first_goals = [goal for goal, step in first_times.items() if step == earliest and step is not None]
    terminal = [goal for goal in GOALS if flags[-1][goal]]
    terminal_goal = terminal[0] if len(terminal) == 1 else None
    terminal_category = ("ambiguous" if len(terminal) > 1 else "unfinished" if not terminal else
                         "correct" if terminal_goal == requested_goal else "wrong_goal")
    requested_first = first_times[requested_goal]
    correct_ever = requested_first is not None
    at_boundary = flags[window][requested_goal]
    after_boundary = any(state[requested_goal] for state in flags[window+1:])
    return {
        "requested_goal": requested_goal,
        "first_goal_times": first_times,
        "first_goal": first_goals[0] if len(first_goals) == 1 else None,
        "first_goals": first_goals,
        "first_goal_step": earliest,
        "first_goal_ambiguous": len(first_goals) > 1,
        "terminal_goal": terminal_goal,
        "terminal_category": terminal_category,
        "terminal_correct": terminal_category == "correct",
        "correct_ever": correct_ever,
        "correct_then_destroyed": correct_ever and not flags[-1][requested_goal],
        "correct_lost_at_any_later_step": (correct_ever and
                any(not state[requested_goal] for state in flags[requested_first+1:])),
        "first_goal_within_window": earliest is not None and earliest <= window,
        "first_goal_after_window": earliest is not None and earliest > window,
        "correct_first_within_window": requested_first is not None and requested_first <= window,
        "correct_first_after_window": requested_first is not None and requested_first > window,
        "correct_at_window_end": at_boundary,
        "correct_present_after_window": after_boundary,
        "correct_at_window_end_and_terminal_correct": at_boundary and terminal_category == "correct",
        "correct_at_window_end_but_terminal_lost": at_boundary and not flags[-1][requested_goal],
        "correct_after_window_but_terminal_lost": after_boundary and not flags[-1][requested_goal],
        "new_right_ever": requested_goal != "right" and first_times["right"] is not None,
        "new_right_terminal": requested_goal != "right" and terminal_category == "wrong_goal" and terminal_goal == "right",
        "first_new_right": requested_goal != "right" and first_goals == ["right"],
        "wrong_goal_ever": any(first_times[g] is not None for g in GOALS if g != requested_goal),
    }


def expected_donor(model, condition, goal):
    if condition == "native":
        return None, None
    if condition in ("cross_same", "random_delta"):
        return ("R0" if model == "FT4" else "FT4"), goal
    other = "front" if goal == "left" else "left"
    if condition == "cross_wrong" and model == "FT4":
        return "R0", other
    if condition == "same_model_wrong" and model == "R0":
        return "R0", other
    raise ValueError("Unregistered condition")


def validate_replans(model, condition, goal, rows, horizon=400, window=120, action_steps=10, flow_steps=10):
    require(horizon % action_steps == 0 and window % action_steps == 0,
            "Window must align with action-block boundary")
    keyed = index_unique(rows, ("replan_index",), {(i,) for i in range(horizon//action_steps)})
    total = 0
    for (index,), row in keyed.items():
        start = index*action_steps
        require(row["start_step"] == start, "Replan starts at wrong executed-step boundary")
        expected = flow_steps if condition != "native" and start < window else 0
        require(row["patch_flow_count"] == expected, "Intervention window or flow coverage mismatch")
        if expected:
            require((row["donor_model"], row["donor_goal"]) == expected_donor(model, condition, goal),
                    "Wrong donor model or instruction")
        else:
            require(row.get("donor_model") is None and row.get("donor_goal") is None,
                    "A donor remains active outside registered intervention window")
        total += expected
    require(total == (120 if condition != "native" else 0), "Unexpected per-episode intervention count")
    return total


def group_summary(rows):
    """Failures remain in denominators; terminal outcomes form a full partition."""
    result = {"episodes": len(rows)}
    booleans = ("terminal_correct", "correct_ever", "correct_then_destroyed",
                "correct_lost_at_any_later_step", "new_right_ever", "new_right_terminal",
                "first_new_right", "wrong_goal_ever", "correct_first_within_window",
                "correct_first_after_window", "correct_at_window_end", "correct_present_after_window",
                "correct_at_window_end_and_terminal_correct", "correct_at_window_end_but_terminal_lost",
                "correct_after_window_but_terminal_lost")
    result.update({name: sum(row[name] for row in rows) for name in booleans})
    result["terminal_counts"] = {name: sum(row["terminal_category"] == name for row in rows)
                                 for name in ("correct", "wrong_goal", "unfinished", "ambiguous")}
    require(sum(result["terminal_counts"].values()) == len(rows), "Terminal partition mismatch")
    result["terminal_goal_counts"] = {goal: sum(row["terminal_goal"] == goal for row in rows) for goal in GOALS}
    result["first_goal_counts"] = {goal: sum(row["first_goal"] == goal for row in rows) for goal in GOALS}
    result["terminal_correct_rate"] = result["terminal_correct"]/len(rows) if rows else None
    result["correct_ever_rate"] = result["correct_ever"]/len(rows) if rows else None
    return result


def paired_retention(lookup, model, condition, goal, reference_model="R0"):
    """Only originally terminal-correct reference episodes enter retention."""
    eligible = []
    for index, seed in itertools.product(INDICES, SEEDS):
        old = lookup[reference_model, "native", goal, index, seed]
        if old["terminal_correct"]:
            eligible.append(lookup[model, condition, goal, index, seed])
    return {"reference_model": reference_model, "reference_condition": "native",
            "model": model, "condition": condition, "requested_goal": goal,
            "original_terminal_correct_denominator": len(eligible), **group_summary(eligible)}


def paired_condition_changes(lookup, model, condition, goal):
    result = {"model":model, "condition":condition, "requested_goal":goal,
              "paired_episodes":4, "native_correct_kept":0, "native_incorrect_to_correct":0,
              "native_correct_lost":0, "both_incorrect":0, "native_terminal_right_to_correct":0}
    for index, seed in itertools.product(INDICES, SEEDS):
        before = lookup[model, "native", goal, index, seed]
        after = lookup[model, condition, goal, index, seed]
        label = ("native_correct_kept" if before["terminal_correct"] and after["terminal_correct"] else
                 "native_incorrect_to_correct" if after["terminal_correct"] else
                 "native_correct_lost" if before["terminal_correct"] else "both_incorrect")
        result[label] += 1
        result["native_terminal_right_to_correct"] += int(before["new_right_terminal"] and after["terminal_correct"])
    require(sum(result[k] for k in ("native_correct_kept", "native_incorrect_to_correct",
                                   "native_correct_lost", "both_incorrect")) == 4,
            "Paired terminal transition counts do not partition all conditions")
    return result


def validate_actions(event):
    """Recompute the environment command, including componentwise clipping."""
    before, after, clipped = (event[name] for name in
                              ("action_unclipped", "action_executed", "clipped_dimensions"))
    require(len(before) == len(after) == len(clipped) == 7, "Expected seven physical action dimensions")
    for original, executed, changed in zip(before, after, clipped):
        require(type(original) in (int, float) and type(executed) in (int, float)
                and math.isfinite(original) and math.isfinite(executed), "Nonfinite action")
        expected = min(1., max(-1., original))
        require(abs(executed-expected) <= 1e-12, "Executed action is not the fixed clipped command")
        require(type(changed) is bool and changed == (executed != original),
                "Action clipping mask differs from actual command")


def validate_engineering(gate):
    require(gate.get("passed") is True, "Engineering gate did not pass")
    require(gate["tolerance"] == 1e-5 and gate["site"] == 9, "Engineering tolerance/site changed")
    require(gate["cache_isolation"] == {"deep_copy_per_query":True,"key_value_storage_unique":True},
            "Engineering cache-isolation proof missing")
    require(gate["no_physical_actions_beyond_reset_settle"] is True,
            "Engineering consumed unregistered physical actions")
    indexed = index_unique(gate["checks"], ("model", "goal"),
                           {(model,goal) for model in ("R0","FT4") for goal in OLD_GOALS})
    for row in indexed.values():
        fields = ("max_abs", "native_chunk_max_abs", "self_chunk_max_abs", "post_native_max_abs",
                  "post_self_max_abs", "clip_max_abs", "post_per_action_max_abs")
        errors = []
        for name in fields:
            value = row[name]
            require(type(value) in (float,int) and math.isfinite(value) and 0 <= value <= 1e-5,
                    f"Engineering official/self/postprocess/clip gate failed: {name}")
            errors.append(value)
        flows = index_unique(row["flows"], ("flow_index",), {(i,) for i in range(10)})
        for flow in flows.values():
            for name in ("velocity_max_abs", "witness_max_abs"):
                value = flow[name]
                require(type(value) in (float,int) and math.isfinite(value) and 0 <= value <= 1e-5,
                        f"Engineering self-flow/witness gate failed: {name}")
                errors.append(value)
        require(row["max_abs"] == max(errors), "Engineering aggregate hides a failed subcheck")
    return gate


def validate_configuration(cfg):
    constants = {"site":9, "indices":[0,1], "seeds":[2026102801,2026102802],
                 "goals":["left","front"], "horizon":400, "intervention_steps":120,
                 "n_action_steps":10, "flow_steps":10, "noise_replan_stride":1000003,
                 "expected_episodes":68, "expected_replans":2720,
                 "expected_recipient_denoise_calls":29120, "expected_patch_calls":5760,
                 "expected_random_norm_records":1920, "identity_tolerance":1e-5}
    for field,value in constants.items():
        require(cfg[field] == value, f"Sealed closed-loop configuration differs: {field}")
    require({role:tuple(modes) for role,modes in cfg["roles_and_modes"].items()} == CONDITIONS,
            "Role/condition matrix changed")
    require(cfg["new_goal_check"] == {"model":"FT4","mode":"native","goal":"right",
                                      "indices":[0,1],"seeds":[2026102801,2026102802]},
            "New-goal native control changed")
    require(cfg["teacher_model_sha256"] == "15df8ed2839c281ba5d55b411b43a15177013b4d12b0e5cb98baefdcd60313f7"
            and cfg["student_model_sha256"] == "c67c3fc45763dd4ebd63d24354bd9b32aae830edb67892d44f199d1dd08ca54f",
            "Registered teacher/student checkpoint differs")
    return cfg


def audit(run):
    manifest = read_json(run/"manifest.json")
    require(manifest == read_json(run/"COMPLETE.json") and manifest["status"] == "complete",
            "Incomplete registered run")
    require(manifest.get("no_followup_launched") is True, "Unexpected followup scope")
    cfg = validate_configuration(manifest["configuration"])
    require(manifest["site"] == 9 and manifest["horizon"] == 400 and manifest["intervention_steps"] == 120
            and manifest["no_early_stop_on_success"] is True, "Actual run scope differs")
    require(manifest["script_sha256"] == file_info(Path(__file__).with_name("stage2p_ca_closedloop.py"))["sha256"],
            "Worker source differs from audited version")
    require(manifest["model_sha256"] == {"R0":cfg["teacher_model_sha256"],"FT4":cfg["student_model_sha256"]},
            "Actual model hashes do not match sealed sources")
    for path, expected in cfg["expected_model_sources"].items():
        require(any(row["path"] == path and row["sha256"] == expected for row in manifest["sources"].values()),
                "Actual model source differs from pinned implementation")
    files = {}
    for name, expected in manifest["files"].items():
        actual = file_info(run/name)
        require(actual == expected, f"Raw evidence hash mismatch: {name}")
        files[name] = actual
    gate = validate_engineering(read_json(run/"engineering.json"))
    for field in ("script_sha256","config_sha256","model_sha256","sources"):
        require(gate[field] == manifest[field], f"Engineering/actual-run provenance differs: {field}")
    episodes = read_rows(run/"episodes.jsonl")
    lookup = index_unique(episodes, ("model", "condition", "requested_goal", "init_index", "noise_seed"),
                          expected_episode_keys())
    by_id = index_unique(episodes, ("episode_id",))
    require(len(by_id) == 68, "Episode IDs do not uniquely cover matrix")
    events = read_rows(run/"events.jsonl.gz")
    event_lookup = index_unique(events, ("episode_id", "step"),
                                {(episode_id, step) for (episode_id,) in by_id for step in range(401)})
    replans = read_rows(run/"replans.jsonl.gz")
    replan_lookup = index_unique(replans, ("episode_id", "replan_index"),
                                 {(episode_id, index) for (episode_id,) in by_id for index in range(40)})
    reset_hashes, xml_hashes, noise_hashes = defaultdict(set), set(), defaultdict(set)
    patch_flows, recipient_calls, donor_calls = 0, 0, 0
    recomputed, output_lookup = [], {}
    max_witness = 0.
    for key, row in lookup.items():
        model, condition, goal, initial, seed = key
        episode_id = row["episode_id"]
        reset_hashes[initial].add(row["initial_state_sha256"])
        xml_hashes.add(row["scene_xml_sha256"])
        steps = [event_lookup[episode_id, step] for step in range(401)]
        flags = [step["predicates"] for step in steps]
        actual = summarize_predicates(goal, flags)
        # Legacy worker labels are checked but never trusted as the raw source.
        for field in ("requested_goal", "first_goal", "first_goal_times", "terminal_goal",
                      "terminal_category", "correct_ever", "correct_then_destroyed"):
            require(row[field] == actual[field], f"Recorded behavioral summary differs: {episode_id}/{field}")
        aliases = {
            "first_requested_before_or_at_window_end":"correct_first_within_window",
            "requested_ever_after_window":"correct_present_after_window",
            "newly_reached_requested_after_window":"correct_first_after_window",
        }
        for recorded, field in aliases.items():
            if recorded in row:
                require(row[recorded] == actual[field], f"Window summary differs: {episode_id}/{recorded}")
        if "active_goals_at_window_end" in row:
            require(set(row["active_goals_at_window_end"]) == {g for g in GOALS if flags[120][g]},
                    "Recorded window-boundary predicate set differs")
        for event in steps[1:]:
            validate_actions(event)
        episode_replans = [replan_lookup[episode_id, i] for i in range(40)]
        patch_count = validate_replans(model, condition, goal, episode_replans)
        patch_flows += patch_count
        if "patch_flows" in row:
            require(row["patch_flows"] == patch_count, "Episode patch count differs")
        for replan in episode_replans:
            index = replan["replan_index"]
            noise_hashes[seed, index].add(replan["noise_sha256"])
            expected_calls = 20 if condition == "random_delta" and index < 12 else 10
            require(replan["recipient_denoise_calls"] == expected_calls,
                    "Recipient integration does not cover all ten flow steps")
            require(replan["noise_generator_seed"] == seed+1000003*index,
                    "Noise replanning seed schedule changed")
            require(replan["donor_denoise_calls"] == replan["patch_flow_count"],
                    "Donor query count does not match current-state patch computation")
            recipient_calls += replan["recipient_denoise_calls"]
            donor_calls += replan["donor_denoise_calls"]
            witness = replan["actual_message_witness_max_abs"]
            require(type(witness) in (float,int) and math.isfinite(witness) and witness >= 0,
                    "Missing/invalid actual message witness")
            max_witness = max(max_witness, witness)
        result = {"episode_id":episode_id, "model":model, "condition":condition,
                  "init_index":initial, "noise_seed":seed, **actual}
        recomputed.append(result)
        output_lookup[key] = result
    require(all(len(values) == 1 for values in reset_hashes.values()),
            "Physical reset differs across paired conditions/models/instructions/noise seeds")
    require(len(xml_hashes) == 1, "Scene XML differs across episode matrix")
    require(all(len(values) == 1 for values in noise_hashes.values()),
            "Initial replan noise differs across paired conditions/models/instructions")
    require(patch_flows == 5760 and recipient_calls == 29120 and donor_calls == 5760,
            "Registered inference budget does not match actual completed matrix")
    require(max_witness <= 1e-5, "Actual message injection witness failed")
    require(manifest["actual_message_witness_max_abs"] == max_witness,
            "Manifest witness aggregate differs from recorded replans")
    expected_counts = {"episodes":68,"steps":27200,"replans":2720,
                       "patch_flows":5760,"recipient_denoise_calls":29120,
                       "donor_denoise_calls":5760,"prefix_builds":3296,
                       "integrator_velocity_calls":27200,"extra_random_recipient_calibration_calls":1920,
                       "official_engineering_flow_calls":0,"engineering_self_replacements":0,
                       "denoise_calls_by_model":{"R0":17600,"FT4":17280}}
    require(manifest["counts"] == expected_counts, "Manifest actual query and execution counts differ")
    require(manifest["cache_isolation_checks"] == 34880, "Cache-isolation check coverage differs")
    norms = read_rows(run/"norms.jsonl.gz")
    random_ids = {row["episode_id"] for row in episodes if row["condition"] == "random_delta"}
    norm_lookup = index_unique(norms, ("episode_id", "replan_index", "flow_index"),
                              {(episode_id, index, flow) for episode_id in random_ids
                               for index in range(12) for flow in range(10)})
    require(len(norm_lookup) == 1920, "Random amplitude-control matrix differs")
    from audit_stage2n_cross_checkpoint import audit_random_norms
    norm_results = []
    for record in norms:
        norm_results.append(audit_random_norms(record,
            {"effective_block_l2":record["correct_delta_block_l2"]}))
    per_condition = []
    for model, conditions in CONDITIONS.items():
        for condition in conditions:
            for goal in OLD_GOALS:
                selected = [row for row in recomputed if (row["model"],row["condition"],row["requested_goal"])
                            == (model, condition, goal)]
                require(len(selected) == 4, "Per-instruction condition denominator changed")
                per_condition.append({"model":model, "condition":condition, "requested_goal":goal,
                                      **group_summary(selected)})
    right = [row for row in recomputed if row["requested_goal"] == "right"]
    per_condition.append({"model":"FT4", "condition":"native", "requested_goal":"right",
                          **group_summary(right)})
    retention = [paired_retention(output_lookup, model, condition, goal)
                 for model, conditions in CONDITIONS.items() for condition in conditions for goal in OLD_GOALS]
    changes = [paired_condition_changes(output_lookup, model, condition, goal)
               for model, conditions in CONDITIONS.items() for condition in conditions
               if condition != "native" for goal in OLD_GOALS]
    return {
        "status":"passed", "audit_source_sha256":file_info(__file__)["sha256"],
        "source_files":files, "counts":{"physical_states":2, "inference_seeds":2, "episodes":68,
            "events_including_reset":len(events), "replans":len(replans), "recipient_denoise_calls":recipient_calls,
            "recipient_integration_calls":27200,"random_extra_calibration_calls":1920,
            "donor_denoise_calls":donor_calls, "patched_flow_queries":patch_flows, "random_norm_records":len(norms)},
        "actual_message_witness_max_abs":max_witness,
        "random_norm_max_relative_actual_difference":max(row["max_relative_actual_norm_difference"] for row in norm_results),
        "reset_hashes":{str(k):next(iter(v)) for k,v in reset_hashes.items()},
        "scene_xml_sha256":next(iter(xml_hashes)),
        "per_condition":per_condition, "paired_original_success":retention,
        "paired_condition_changes":changes, "episodes":recomputed,
        "limits":[
            "Only two reused development initial states and two noise seeds; four episodes per instruction/condition.",
            "No population probability or independent training replication claim.",
            "Teacher-assisted message rescue is not proof of retained endogenous student knowledge or a learned repair method.",
            "Terminal success, ever success, and loss after success are separate outcomes.",
            "Success within the active patch window cannot establish autonomous completion after withdrawal.",
            "Post-W_O norm blocks are bookkeeping coordinates, not original attention heads.",
            "Logged query/witness/norm claims do not allow independent replay of hidden model tensors.",
        ],
    }


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError("Existing audit must be retained")
    result=audit(args.run)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"counts":result["counts"]}))

