"""Step 3 only: controlled cross-checkpoint CA message interventions.

No model update, P/C intervention, output-projection intervention, or policy-
generated simulator action. The existing reset includes fixed zero-action settling.
"""

import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

from stage2m_cross_attention import (
    Observer, append, denoise, make_inputs, make_prefix, parameter_audit,
    rms, sha, stats, tensor_sha, transfer_metrics, velocity_tensor, write,
)


ROLES = ("R0", "FT4")
GOALS = ("left", "front")
MODES = ("cross_same", "cross_wrong", "random_delta", "self")


def opposite(value):
    return {"R0": "FT4", "FT4": "R0", "left": "front", "front": "left"}[value]


def expected_jobs(indices, seeds, flows, sites):
    """Pure protocol enumeration, independent of model implementation."""
    import itertools
    return list(itertools.product(indices, seeds, flows, ROLES, GOALS, sites, MODES))


def random_seed_for(anchor_id, role, goal, layer, seed):
    payload = f"{seed}|{anchor_id}|{role}|{goal}|{layer}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def record_id(anchor_id, role, goal, layer, mode):
    return f"{anchor_id}|{role}|{goal}|j{layer:02d}|{mode}"


def load_prior(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    result = {}
    for row in rows:
        if row["goal"] not in GOALS:
            continue
        key = (row["state_index"], row["noise_seed"], row["flow_index"], row["model"], row["goal"])
        if key in result:
            raise ValueError(f"Duplicate prior baseline: {key}")
        result[key] = row
    return result


def norm_blocks(value, block_count):
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != 50:
        raise ValueError(f"Unexpected message shape: {tuple(value.shape)}")
    if value.shape[-1] % block_count:
        raise ValueError("Output message width is not divisible by bookkeeping block count")
    return value.double().reshape(50, block_count, -1).square().sum(-1).sqrt()


def norm_list(value):
    return value.detach().cpu().tolist()


def matched_random_message(base, correct, block_count, seed, dtype, torch):
    """Match intended delta norms, and explicitly audit the native BF16 cast.

    Blocks are contiguous output-coordinate bookkeeping groups after W_O.
    They are not the original attention heads.
    """
    base, correct = base.detach().float().cpu(), correct.detach().float().cpu()
    target = norm_blocks(correct - base, block_count)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(base.shape, generator=generator, dtype=torch.float32, device="cpu")
    raw_blocks = raw.double().reshape(50, block_count, -1)
    raw_norms = raw_blocks.square().sum(-1).sqrt()
    if bool(((raw_norms == 0) & (target > 0)).any()):
        raise ValueError("Zero random direction for a nonzero target; no redraw allowed")
    factor = torch.where(raw_norms > 0, target / raw_norms, torch.zeros_like(target))
    desired_delta = (raw_blocks * factor[..., None]).reshape_as(base).float()
    planned_message = base + desired_delta
    # Observer inserts this tensor using precisely the recipient's native dtype.
    effective_message = planned_message.to(dtype=dtype).float()
    desired = norm_blocks(desired_delta, block_count)
    effective = norm_blocks(effective_message - base, block_count)
    cast_error = norm_blocks(effective_message - planned_message, block_count)
    cast_input = norm_blocks(planned_message, block_count)
    construction_error = (desired - target).abs()
    # Addition in float32 creates another small, explicitly measured residual.
    addition_error = norm_blocks((planned_message - base) - desired_delta, block_count)
    if float(construction_error.max()) > 1e-5:
        raise AssertionError("Intended random delta did not match registered block norms")
    triangle_excess = (effective - target).abs() - cast_error - construction_error - addition_error
    if float(triangle_excess.max()) > 1e-6:
        raise AssertionError("Actual random norm mismatch exceeds measured arithmetic triangle bound")
    if dtype == torch.bfloat16:
        unit_roundoff = 2**-8
    elif dtype == torch.float16:
        unit_roundoff = 2**-11
    elif dtype == torch.float32:
        unit_roundoff = 2**-24
    else:
        raise ValueError(f"Unsupported native message dtype: {dtype}")
    if float((cast_error - unit_roundoff * cast_input).max()) > 1e-6:
        raise AssertionError("Actual message cast error exceeds dtype roundoff bound")
    ratio = torch.where(target > 1e-12, effective / target, torch.ones_like(target))
    evidence = {
        "block_count": block_count, "block_width": base.shape[-1] // block_count,
        "block_semantics": "contiguous post-output-projection coordinates; not original attention heads",
        "message_dtype": str(dtype), "unit_roundoff": unit_roundoff, "random_seed": seed,
        "target_block_l2": norm_list(target), "desired_block_l2": norm_list(desired),
        "effective_block_l2": norm_list(effective), "cast_error_block_l2": norm_list(cast_error),
        "cast_input_block_l2": norm_list(cast_input), "addition_error_block_l2": norm_list(addition_error),
        "desired_norm_max_abs_error": float(construction_error.max()),
        "effective_norm_max_abs_error": float((effective - target).abs().max()),
        "effective_target_ratio_min_nonzero": float(ratio[target > 1e-12].min())
            if bool((target > 1e-12).any()) else None,
        "effective_target_ratio_max_nonzero": float(ratio[target > 1e-12].max())
            if bool((target > 1e-12).any()) else None,
        "triangle_excess_max": float(triangle_excess.max()),
        "guarantees_actual_exact_match": False,
    }
    return effective_message, evidence


class MessageObserver(Observer):
    """Validated recording path, plus actual output-message dtype."""

    def __init__(self, policy, torch):
        self.message_dtypes = {}
        self.applied_messages = {}
        super().__init__(policy, torch)
        for row in self.layer_map:
            if row["kind"] == "CA":
                module = self.expert.lm_expert.layers[row["expert_layer"]].self_attn.o_proj
                self.handles.append(module.register_forward_hook(self._witness_hook(row["vlm_layer"])))

    def _witness_hook(self, index):
        def hook(module, inputs, output):
            if self.active:
                self.applied_messages[index] = output.detach().clone().float().cpu()
            return None
        return hook

    def begin(self, patch=None):
        self.applied_messages = {}
        super().begin(patch)

    def _message_hook(self, index):
        original = super()._message_hook(index)

        def hook(module, inputs, output):
            if self.active:
                self.message_dtypes[index] = output.dtype
            return original(module, inputs, output)
        return hook


def input_hashes(inputs):
    images, masks, tokens, language_mask, state = inputs
    return {
        "images": [tensor_sha(x) for x in images],
        "image_masks": [tensor_sha(x) for x in masks],
        "language_tokens": tensor_sha(tokens), "language_mask": tensor_sha(language_mask),
        "state_before_projection": tensor_sha(state),
        "image_shapes": [list(x.shape) for x in images],
        "language_shape": list(tokens.shape), "state_shape": list(state.shape),
    }


def prefix_diagnostics(prepared, report, tolerance, torch):
    rows = []
    report.setdefault("prefix_audit", []).append(rows)
    for goal in GOALS:
        a, b = [prepared[role][goal][1] for role in ROLES]
        if a[2] != b[2] or not torch.equal(a[0], b[0]):
            raise AssertionError("Prefix coordinates or mask changed")
        for index, (old, new) in enumerate(zip(a[1].layers, b[1].layers)):
            for kind in ("keys", "values"):
                for group, (lo, hi) in a[2].items():
                    delta = (getattr(old, kind)[:, :, lo:hi].float()
                             - getattr(new, kind)[:, :, lo:hi].float())
                    position_mask = a[0][:, lo:hi].bool()
                    by_position = delta.permute(0, 2, 1, 3)
                    subsets = {}
                    for label, mask in (("active", position_mask), ("masked", ~position_mask)):
                        value = by_position[mask]
                        subsets[label] = {
                            "count": int(mask.sum()), "element_count": value.numel(),
                            "max_abs": float(value.abs().max()) if value.numel() else 0.0,
                            "rms": rms(value) if value.numel() else 0.0,
                        }
                    row = {"goal": goal, "layer": index, "tensor": kind, "group": group,
                           "max_abs": float(delta.abs().max()), "rms": rms(delta), **subsets}
                    rows.append(row)
                    if group in ("image", "text") and row["active"]["max_abs"] > tolerance:
                        raise AssertionError(f"Frozen valid prefix changed: {row}")
    caches = [prepared[role][goal][1][1] for role in ROLES for goal in GOALS]
    pointers = [layer.keys.data_ptr() for cache in caches for layer in cache.layers]
    if len({id(x) for x in caches}) != len(caches) or len(set(pointers)) != len(pointers):
        raise AssertionError("Independent prefix caches alias")


def check_prior(common, role, goal, velocity, prior, tolerance):
    key = (common["state_index"], common["noise_seed"], common["flow_index"], role, goal)
    old = prior[key]
    if tuple(velocity.shape) != (50, 7) or len(old["velocity"]) != 50 or any(
            len(row) != 7 for row in old["velocity"]):
        raise ValueError("Prior/current effective velocity must both be exactly 50 by 7")
    for name in ("anchor_id", "x_sha256", "state_sha256"):
        if common[name] != old[name]:
            raise AssertionError(f"Prior input identity changed: {key}/{name}")
    if abs(common["time"] - old["time"]) > 1e-7:
        raise AssertionError(f"Prior flow time changed: {key}")
    error = max(abs(a - b) for row_a, row_b in zip(velocity.tolist(), old["velocity"])
                for a, b in zip(row_a, row_b))
    if error > tolerance:
        raise AssertionError(f"Prior effective velocity changed: {key}/{error}")
    return error


def json_line(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def run_analysis(args, cfg, report, models, observers, initial, prefixes, noise,
                 prior, euler_integrate, guard, torch):
    engineering_path = args.output.parent / "engineering" / "engineering.json"
    engineering = json.loads(engineering_path.read_text(encoding="utf-8"))
    if engineering["passed"] is not True or engineering["layer_map"] != report["layer_map"]:
        raise AssertionError("Engineering gate missing or changed layer layout")
    if (engineering["source_hashes"] != report["sources"]
            or engineering["prior_baseline_sha256"] != report["prior_baseline_sha256"]
            or engineering["script_sha256"] != report["script_sha256"]
            or engineering["config_sha256"] != report["config_sha256"]):
        raise AssertionError("Engineering provenance changed before analysis")
    write(args.output / "engineering.json", engineering)
    report["engineering_source"] = {"path": str(engineering_path), "sha256": sha(engineering_path)}
    report["repeat_v_rms"] = engineering["repeat_v_rms"]
    report["metric_epsilon_v"] = max(10 * report["repeat_v_rms"], 1e-6)
    epsilon = report["metric_epsilon_v"]
    salt = cfg.get("random_seed_salt", cfg.get("random_seed", 2026102903))
    block_count = report["block_count"]
    tolerance = report["identity_tolerance"]
    max_prior = 0.0
    random_errors = []
    with gzip.open(args.output / "norms.jsonl.gz", "wt", encoding="utf-8") as norms_handle:
        for state_index in report["indices"]:
            observation = initial(state_index)
            prepared = prefixes(observation)
            for seed_position, seed in enumerate(report["seeds"]):
                reference_goal = GOALS[(state_index + seed_position) % 2]
                ref_prefix = prepared["R0"][reference_goal][1]
                references, counter = {}, [0]

                def reference_velocity(action_x, timestep):
                    index = counter[0]
                    if index in report["flow_indices"]:
                        references[index] = (action_x.detach().clone(), timestep.detach().clone())
                    value, _ = denoise(models["R0"], observers["R0"], ref_prefix, action_x, timestep)
                    counter[0] += 1
                    report["counts"]["reference_denoise"] += 1
                    return value

                euler_integrate(reference_velocity, noise(seed), 10)
                if set(references) != set(report["flow_indices"]):
                    raise AssertionError("Flow reference grid mismatch")
                for flow_index in report["flow_indices"]:
                    x_t, timestep = references[flow_index]
                    anchor_id = f"i{state_index:02d}_s{seed}_f{flow_index:02d}"
                    common = {
                        "anchor_id": anchor_id, "state_index": state_index, "noise_seed": seed,
                        "flow_index": flow_index, "time": float(timestep[0]),
                        "reference_goal": reference_goal, "x_sha256": tensor_sha(x_t),
                        "state_sha256": report["physical_state_sha256"][str(state_index)],
                    }
                    captures, velocities = {}, {}
                    for role in ROLES:
                        captures[role], velocities[role] = {}, {}
                        for goal in GOALS:
                            value, capture = denoise(models[role], observers[role],
                                                      prepared[role][goal][1], x_t, timestep, record=True)
                            velocity = velocity_tensor(value)
                            prior_error = check_prior(common, role, goal, velocity, prior, tolerance)
                            max_prior = max(max_prior, prior_error)
                            captures[role][goal], velocities[role][goal] = capture, velocity
                            append(args.output / "baseline.jsonl", {
                                **common, "model": role, "goal": goal, "velocity": velocity.tolist(),
                                "prior_velocity_max_abs": prior_error,
                                "message_stats": {str(layer): stats(capture[layer]["M"])
                                                  for layer in report["layers"]},
                                "message_dtypes": {str(layer): str(observers[role].message_dtypes[layer])
                                                   for layer in report["layers"]},
                            })
                            report["counts"]["baseline"] += 1
                    for layer in report["layers"]:
                        for role in ROLES:
                            donor_role = opposite(role)
                            for goal in GOALS:
                                base_message = captures[role][goal][layer]["M"]
                                dtype = observers[role].message_dtypes[layer]
                                correct_message = captures[donor_role][goal][layer]["M"].to(dtype=dtype).float()
                                wrong_message = captures[donor_role][opposite(goal)][layer]["M"]
                                seed_random = random_seed_for(anchor_id, role, goal, layer, salt)
                                random_message, norm_evidence = matched_random_message(
                                    base_message, correct_message, block_count, seed_random, dtype, torch)
                                random_errors.append({
                                    key: norm_evidence[key] for key in
                                    ("desired_norm_max_abs_error", "effective_norm_max_abs_error",
                                     "effective_target_ratio_min_nonzero",
                                     "effective_target_ratio_max_nonzero", "triangle_excess_max")
                                })
                                source_messages = {
                                    "cross_same": correct_message, "cross_wrong": wrong_message,
                                    "random_delta": random_message, "self": base_message,
                                }
                                for mode in MODES:
                                    source = source_messages[mode]
                                    patched, _ = denoise(
                                        models[role], observers[role], prepared[role][goal][1],
                                        x_t, timestep, record=True, patch=(layer, source))
                                    velocity = velocity_tensor(patched)
                                    effective_message = observers[role].applied_messages[layer]
                                    intended_native = source.to(dtype=dtype).float()
                                    witness_error = float((effective_message - intended_native).abs().max())
                                    if witness_error > tolerance:
                                        raise AssertionError("Actual o_proj message does not match registered patch")
                                    effective_delta = effective_message - base_message
                                    rid = record_id(anchor_id, role, goal, layer, mode)
                                    identity = {
                                        **common, "model": role, "recipient": goal, "layer": layer,
                                        "mode": mode, "norm_record_id": rid,
                                    }
                                    if mode in ("cross_same", "random_delta"):
                                        norm_row = {**identity, "block_count": block_count,
                                                    "block_width": report["block_width"],
                                                    "block_semantics": report["block_semantics"]}
                                        if mode == "cross_same":
                                            norm_row["effective_block_l2"] = norm_list(
                                                norm_blocks(effective_delta, block_count))
                                        else:
                                            norm_row.update(norm_evidence)
                                            actual_norms = norm_blocks(effective_delta, block_count)
                                            recorded_norms = torch.tensor(
                                                norm_evidence["effective_block_l2"], dtype=torch.float64)
                                            if float((actual_norms - recorded_norms).abs().max()) > tolerance:
                                                raise AssertionError("Recorded random norms disagree with actual injection")
                                            norm_row["effective_block_l2"] = norm_list(actual_norms)
                                        norm_row["actual_patch_witness_error_max_abs"] = witness_error
                                        norm_row["effective_message_sha256"] = tensor_sha(effective_message)
                                        json_line(norms_handle, norm_row)
                                        report["counts"]["norm_records"] += 1
                                    metrics = {
                                        "cross_same": transfer_metrics(
                                            velocities[role][goal], velocities[donor_role][goal],
                                            velocity, epsilon),
                                        "healthy_same": transfer_metrics(
                                            velocities[role][goal], velocities["R0"][goal],
                                            velocity, epsilon),
                                        "same_model_other": transfer_metrics(
                                            velocities[role][goal], velocities[role][opposite(goal)],
                                            velocity, epsilon),
                                    }
                                    row = {
                                        **identity,
                                        "donor_model": role if mode == "self" else donor_role,
                                        "donor": opposite(goal) if mode == "cross_wrong" else goal,
                                        "source_kind": "registered_random_delta" if mode == "random_delta"
                                            else "native_message",
                                        "random_seed": seed_random if mode == "random_delta" else None,
                                        "message_dtype": str(dtype),
                                        "actual_patch_witness_error_max_abs": witness_error,
                                        "base_message_sha256": tensor_sha(base_message),
                                        "effective_message_sha256": tensor_sha(effective_message),
                                        "effective_delta_sha256": tensor_sha(effective_delta),
                                        "effective_delta_stats": stats(effective_delta),
                                        "velocity": velocity.tolist(), "metrics": metrics,
                                        "norm_evidence_saved": mode in ("cross_same", "random_delta"),
                                    }
                                    if mode == "self":
                                        identity_error = float(
                                            (velocity - velocities[role][goal]).abs().max())
                                        row["self_max_abs"] = identity_error
                                        if identity_error > tolerance:
                                            raise AssertionError("Self message changed effective velocity")
                                    append(args.output / "patches.jsonl", row)
                                    report["counts"]["patches"] += 1
                    del captures, velocities
                    norms_handle.flush()
                    report["prior_velocity_max_abs"] = max_prior
                    guard()
                    print(json.dumps({"completed_anchor": anchor_id,
                                      "counts": report["counts"]}), flush=True)
    expected = len(expected_jobs(report["indices"], report["seeds"],
                                 report["flow_indices"], report["layers"]))
    if report["counts"]["patches"] != expected:
        raise AssertionError("Incomplete registered patch matrix")
    expected_baseline = len(report["indices"]) * len(report["seeds"]) * len(report["flow_indices"]) * 4
    if report["counts"]["baseline"] != expected_baseline or report["counts"]["norm_records"] != expected // 2:
        raise AssertionError("Incomplete baseline or norm evidence matrix")
    minimum_ratios = [x["effective_target_ratio_min_nonzero"] for x in random_errors
                      if x["effective_target_ratio_min_nonzero"] is not None]
    maximum_ratios = [x["effective_target_ratio_max_nonzero"] for x in random_errors
                      if x["effective_target_ratio_max_nonzero"] is not None]
    report["random_norm_summary"] = {
        "cases": len(random_errors),
        "desired_norm_max_abs_error": max(x["desired_norm_max_abs_error"] for x in random_errors),
        "effective_norm_max_abs_error": max(x["effective_norm_max_abs_error"] for x in random_errors),
        "effective_target_ratio_min_nonzero": min(minimum_ratios) if minimum_ratios else None,
        "effective_target_ratio_max_nonzero": max(maximum_ratios) if maximum_ratios else None,
        "triangle_excess_max": max(x["triangle_excess_max"] for x in random_errors),
        "interpretation": "intended deltas match; native cast permits only approximately matched applied deltas",
    }


def run_engineering(args, cfg, report, models, observers, initial, prefixes, noise,
                    prior, guard, torch):
    index, seed, flow = report["indices"][0], report["seeds"][0], report["flow_indices"][0]
    if flow != 0:
        raise ValueError("Engineering anchor must be the original t=1 input")
    observation = initial(index)
    prepared = prefixes(observation)
    x_t = noise(seed)
    timestep = torch.ones(1, device=x_t.device, dtype=torch.float32)
    common = {
        "anchor_id": f"i{index:02d}_s{seed}_f00", "state_index": index,
        "noise_seed": seed, "flow_index": 0, "time": 1.0,
        "reference_goal": GOALS[index % 2], "x_sha256": tensor_sha(x_t),
        "state_sha256": report["physical_state_sha256"][str(index)],
    }
    captures, velocities, checks = {}, {}, []
    repeated = []
    for role in ROLES:
        captures[role], velocities[role] = {}, {}
        for goal in GOALS:
            native, _ = denoise(models[role], observers[role], prepared[role][goal][1], x_t, timestep)
            repeat, _ = denoise(models[role], observers[role], prepared[role][goal][1], x_t, timestep)
            recorded, capture = denoise(models[role], observers[role], prepared[role][goal][1],
                                         x_t, timestep, record=True)
            velocity = velocity_tensor(recorded)
            prior_error = check_prior(common, role, goal, velocity, prior, report["identity_tolerance"])
            record_error = float((recorded[..., :7] - native[..., :7]).abs().max())
            repeat_error = rms(repeat[..., :7] - native[..., :7])
            repeated.append(repeat_error)
            row = {
                "model": role, "goal": goal, "record_v_max_abs": record_error,
                "prior_velocity_max_abs": prior_error, "repeat_v_rms": repeat_error,
                "readout_reconstruction_max_abs": observers[role].readout_error_max,
                "site_self_max_abs": {},
            }
            captures[role][goal], velocities[role][goal] = capture, velocity
            for layer in report["layers"]:
                patched, _ = denoise(models[role], observers[role], prepared[role][goal][1],
                                     x_t, timestep, record=True, patch=(layer, capture[layer]["M"]))
                error = float((velocity_tensor(patched) - velocity).abs().max())
                witness = observers[role].applied_messages[layer]
                witness_error = float((witness - capture[layer]["M"]).abs().max())
                if max(error, witness_error) > report["identity_tolerance"]:
                    raise AssertionError("Engineering self-message/witness identity failed")
                row["site_self_max_abs"][str(layer)] = error
                report["counts"]["engineering_self_queries"] += 1
            row["self_max_abs"] = max(row["site_self_max_abs"].values())
            if max(record_error, prior_error, row["self_max_abs"],
                   row["readout_reconstruction_max_abs"]) > report["identity_tolerance"]:
                raise AssertionError(f"Engineering gate failed: {row}")
            checks.append(row)
            report["counts"]["baseline"] += 1
            write(args.output / "engineering_checks_progress.json", checks)
            guard()
    norm_checks = []
    salt = cfg.get("random_seed_salt", cfg.get("random_seed", 2026102903))
    for layer in report["layers"]:
        for role in ROLES:
            for goal in GOALS:
                base = captures[role][goal][layer]["M"]
                dtype = observers[role].message_dtypes[layer]
                correct = captures[opposite(role)][goal][layer]["M"].to(dtype=dtype).float()
                seed_random = random_seed_for(common["anchor_id"], role, goal, layer, salt)
                _, evidence = matched_random_message(
                    base, correct, report["block_count"], seed_random, dtype, torch)
                norm_checks.append({
                    "model": role, "recipient": goal, "layer": layer,
                    **{key: evidence[key] for key in (
                        "desired_norm_max_abs_error", "effective_norm_max_abs_error",
                        "effective_target_ratio_min_nonzero", "effective_target_ratio_max_nonzero",
                        "triangle_excess_max", "random_seed")},
                })
    report["repeat_v_rms"] = max(repeated)
    report["metric_epsilon_v"] = max(10 * report["repeat_v_rms"], 1e-6)
    engineering = {
        "passed": True, "checks": checks, "norm_engineering_checks": norm_checks,
        "identity_tolerance": report["identity_tolerance"], "layer_map": report["layer_map"],
        "repeat_v_rms": report["repeat_v_rms"], "metric_epsilon_v": report["metric_epsilon_v"],
        "block_count": report["block_count"], "block_width": report["block_width"],
        "block_semantics": report["block_semantics"],
        "cache_isolation": {"object_unique": True, "tensor_storage_unique": True,
                            "deep_copy_per_query": True},
        "source_hashes": report["sources"], "prior_baseline_sha256": report["prior_baseline_sha256"],
        "script_sha256": report["script_sha256"], "config_sha256": report["config_sha256"],
        "no_policy_generated_actions": True,
    }
    write(args.output / "engineering.json", engineering)
    report["engineering_checks"] = checks


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    indices = cfg.get("indices", cfg.get("development_indices"))
    seeds = cfg.get("seeds", cfg.get("noise_seeds"))
    flows = cfg.get("flow_indices", cfg.get("flow_step_indices"))
    sites = cfg["sites"]
    if indices != [0, 1, 2, 3] or seeds != [2026102801, 2026102802, 2026102803] or flows != [0, 4, 8]:
        raise ValueError("Step 3 accepts only the original sealed input matrix")
    if sorted(sites) != [1, 3, 9] or len(sites) != 3:
        raise ValueError("Only the two selected sites and the predeclared control site are allowed")
    if cfg.get("candidate_sites", [3, 9]) != [3, 9] or cfg.get("control_site", 1) != 1:
        raise ValueError("Unexpected selected/control site definition")
    if tuple(cfg.get("goals", GOALS)) != GOALS or tuple(cfg.get("modes", MODES)) != MODES:
        raise ValueError("The registered goals or intervention modes changed")
    tolerance = cfg.get("identity_tolerance", 1e-5)
    if tolerance != 1e-5:
        raise ValueError("The registered engineering tolerance cannot be changed")
    report = {
        "status": "initializing", "phase": args.phase,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "step 3 core only; no parameter/P-C/W-O intervention or policy-generated actions",
        "indices": indices, "seeds": seeds, "flow_indices": flows, "layers": sites, "sites": sites,
        "candidate_sites": [3, 9], "control_site": 1, "roles": list(ROLES),
        "goals": list(GOALS), "modes": list(MODES), "identity_tolerance": tolerance,
        "counts": {"baseline": 0, "patches": 0, "norm_records": 0,
                   "reference_denoise": 0, "engineering_self_queries": 0},
        "config_sha256": sha(args.config), "script_sha256": sha(__file__),
        "configuration": cfg,
        "random_seed_salt": cfg.get("random_seed_salt", cfg.get("random_seed", 2026102903)),
        "effective_phase_inputs": {"indices": [indices[0]], "seeds": [seeds[0]], "flow_indices": [0]}
            if args.phase == "engineering" else {"indices": indices, "seeds": seeds, "flow_indices": flows},
    }
    observers, env = {}, None
    try:
        sys.path.insert(0, str(args.runtime_scripts.resolve()))
        import numpy as np
        import torch
        import lerobot.policies as policies
        from lerobot.utils import constants
        from lerobot.policies.common.vla_utils import make_att_2d_masks
        from lerobot.policies.common.flow_matching import euler_integrate
        from robosuite.utils.transform_utils import quat2axisangle
        import stage1_r0_eval as evaluation
        import stage2m_cross_attention as prior_helper
        from stage1_smoke import DEFAULT_SCENE
        from stage2k_entry_handoff import _fresh_observation
        from stage2k_response_swap import verify_processors

        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError("The registered inference runtime requires CUDA")
        root_cfg = json.loads(Path(cfg["base_config"]).read_text(encoding="utf-8"))
        if not set(indices) <= set(root_cfg["evaluation"]["development_indices"]):
            raise ValueError("These states are not registered development states")
        prior_path = Path(cfg["prior_baseline_path"])
        if sha(prior_path) != cfg["prior_baseline_sha256"]:
            raise ValueError("Prior baseline evidence bytes changed")
        prior = load_prior(prior_path)
        report["prior_baseline_path"] = str(prior_path)
        report["prior_baseline_sha256"] = sha(prior_path)
        expected_prior = {(i, s, f, role, goal) for i in indices for s in seeds for f in flows
                          for role in ROLES for goal in GOALS}
        if set(prior) != expected_prior:
            raise ValueError("Prior baseline does not contain exactly the registered old-condition matrix")
        candidates_path = Path(cfg["prior_candidates_path"])
        if sha(candidates_path) != cfg["prior_candidates_sha256"]:
            raise ValueError("Prior candidate-selection evidence changed")
        if json.loads(candidates_path.read_text(encoding="utf-8"))["selected"] != [3, 9]:
            raise ValueError("The two fixed sites do not match the prior selection")
        report["prior_candidates_sha256"] = sha(candidates_path)
        model_paths = {"R0": Path(cfg["teacher"]), "FT4": Path(cfg["student"])}
        report["processor_hashes"] = verify_processors(model_paths["R0"], model_paths["FT4"])
        report["model_sha256"] = {role: sha(path / "model.safetensors") for role, path in model_paths.items()}
        for role, key in (("R0", "teacher"), ("FT4", "student")):
            if report["model_sha256"][role] != cfg[key + "_model_sha256"]:
                raise ValueError(f"Model identity changed: {role}")
        models = {role: evaluation.load_policy(path) for role, path in model_paths.items()}
        preprocessors = {}
        for role, policy in models.items():
            policy.eval()
            policy.config.n_action_steps = 10
            policy.config.num_steps = 10
            if (policy.config.adapt_to_pi_aloha or policy.model._rtc_enabled()
                    or policy.config.n_obs_steps != 1 or not policy.model.config.use_cache):
                raise ValueError("Unexpected runtime action/observation/cache settings")
            if policy.config.chunk_size != 50 or policy.config.action_feature.shape[0] != 7:
                raise ValueError("Unexpected action shape")
            observers[role] = MessageObserver(policy, torch)
            preprocessors[role], _ = policies.make_pre_post_processors(
                policy.config, pretrained_path=model_paths[role])
        report["layer_map"] = {role: observer.layer_map for role, observer in observers.items()}
        report["ca_layers"] = {role: observer.ca_layers for role, observer in observers.items()}
        if any(not set(sites) <= set(observer.ca_layers) for observer in observers.values()):
            raise ValueError("Registered intervention site is not an actual CA block")
        actual_control = min(x for x in observers["R0"].ca_layers if x not in (3, 9))
        if actual_control != cfg.get("control_site", 1):
            raise ValueError("The registered smallest noncandidate CA control changed")
        report["parameter_audit"] = parameter_audit(models["R0"], models["FT4"], torch)
        heads = [observer.expert.num_attention_heads for observer in observers.values()]
        widths = [policy.model.vlm_with_expert.expert_hidden_size for policy in models.values()]
        if len(set(heads)) != 1 or len(set(widths)) != 1 or widths[0] % heads[0]:
            raise ValueError("Post-output-projection bookkeeping coordinates are incompatible")
        report["block_count"], report["block_width"] = heads[0], widths[0] // heads[0]
        report["block_semantics"] = "contiguous post-output-projection coordinates; not original attention heads"
        report["model_configuration"] = {
            role: {name: getattr(policy.config, name, None) for name in
                   ("attention_mode", "num_vlm_layers", "num_expert_layers", "self_attn_every_n_layers",
                    "train_expert_only", "train_state_proj", "prefix_length", "pad_language_to",
                    "num_steps", "n_action_steps", "chunk_size", "max_action_dim", "use_cache")}
            for role, policy in models.items()
        }
        report["sources"] = {}
        for name, source in (
                ("evaluation", evaluation.__file__), ("stage2m_helper", prior_helper.__file__),
                ("model", sys.modules[type(models["R0"].model).__module__].__file__),
                ("expert", sys.modules[type(models["R0"].model.vlm_with_expert).__module__].__file__)):
            report["sources"][name] = {"path": source, "sha256": sha(source)}
        for source, expected in cfg.get("expected_model_sources", {}).items():
            if sha(source) != expected:
                raise ValueError(f"Model source changed: {source}")
        tasks, paths, _, physical_states = evaluation.task_resources(root_cfg.get("scene", DEFAULT_SCENE))
        instructions = {goal: tasks[goal].language for goal in GOALS}
        report["instructions"] = instructions
        env = evaluation.make_env(paths["front"])
        scene_xml = evaluation.canonical_scene(paths["front"], root_cfg["seed"])
        report["scene_xml_sha256"] = hashlib.sha256(scene_xml.encode()).hexdigest()

        def guard():
            if time.monotonic() - started > args.max_minutes * 60:
                raise TimeoutError("Registered walltime budget")
            size = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
            if size > args.max_output_bytes or shutil.disk_usage(args.output).free < 2 * 1024**3:
                raise RuntimeError("Registered disk/output budget; no data are deleted")
            write(args.output / "progress.json",
                  {**report, "elapsed_seconds": time.monotonic() - started})

        def initial(index):
            evaluation.reset_common(env, physical_states["front"][index], root_cfg["seed"] + index,
                                    root_cfg["evaluation"]["settle_steps"], scene_xml)
            observation = _fresh_observation(env)
            report.setdefault("physical_state_sha256", {})[str(index)] = hashlib.sha256(
                np.asarray(env.get_sim_state()).tobytes()).hexdigest()
            return observation

        def prefixes(observation):
            prepared, hashes = {}, {}
            for role in ROLES:
                prepared[role], hashes[role] = {}, {}
                for goal in GOALS:
                    inputs = make_inputs(models[role], preprocessors[role], observation,
                                         instructions[goal], torch, np, quat2axisangle, constants)
                    hashes[role][goal] = input_hashes(inputs)
                    prepared[role][goal] = (inputs, make_prefix(models[role], inputs, make_att_2d_masks))
            report.setdefault("prefix_input_hashes", []).append(hashes)
            for goal in GOALS:
                if hashes["R0"][goal] != hashes["FT4"][goal]:
                    raise AssertionError(f"Cross-checkpoint preprocessed input differs: {goal}")
            for name in ("images", "image_masks", "state_before_projection"):
                if hashes["R0"]["left"][name] != hashes["R0"]["front"][name]:
                    raise AssertionError(f"Physical input changed across instructions: {name}")
            prefix_diagnostics(prepared, report, tolerance, torch)
            return prepared

        def noise(seed):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            return torch.randn((1, 50, models["R0"].config.max_action_dim), generator=generator,
                               dtype=torch.float32, device="cpu").cuda()

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if args.phase == "engineering":
                run_engineering(args, cfg, report, models, observers, initial, prefixes,
                                noise, prior, guard, torch)
            else:
                run_analysis(args, cfg, report, models, observers, initial, prefixes,
                             noise, prior, euler_integrate, guard, torch)
        report.update(
            status="complete", elapsed_seconds=time.monotonic() - started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            cache_isolation={"object_unique": True, "tensor_storage_unique": True,
                             "deep_copy_per_query": True},
            attention_readout_reconstruction_max_abs={
                role: observer.readout_error_max for role, observer in observers.items()},
            no_followup_launched=True,
            scope_exclusions=["no P/C intervention", "no output-projection intervention",
                              "no closed loop", "no parameter update", "no conditional extra arms"],
        )
        report["files"] = {
            p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in args.output.iterdir() if p.is_file()
            and p.name not in ("progress.json", "manifest.json", "COMPLETE.json")
        }
        write(args.output / "manifest.json", report)
        write(args.output / "COMPLETE.json", report)
        write(args.output / "progress.json", report)
    except Exception as error:
        report.update(status="failed", error=repr(error),
                      elapsed_seconds=time.monotonic() - started,
                      finished_at=datetime.now(timezone.utc).isoformat())
        write(args.output / "FAILED.json", report)
        write(args.output / "progress.json", report)
        raise
    finally:
        for observer in observers.values():
            observer.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-scripts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("engineering", "analysis"), required=True)
    parser.add_argument("--max-minutes", type=float, default=45)
    parser.add_argument("--max-output-bytes", type=int, default=536870912)
    main(parser.parse_args())
