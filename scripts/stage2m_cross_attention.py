"""Read-only open-loop Cross-Attention diagnostics: plan steps 0--2.

No training, policy-generated control or cross-checkpoint intervention is
performed. Environment reset includes the existing fixed zero-action settling.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time


GOALS = ("left", "front", "right")
FLOW_INDICES = (0, 4, 8)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def tensor_sha(value):
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def rms(value):
    return float(value.double().square().mean().sqrt())


def stats(value):
    value = value.detach().float()
    return {"rms": rms(value), "norm": float(value.norm()), "max_abs": float(value.abs().max()),
            "shape": list(value.shape)}


def pair_stats(a, b):
    a, b = a.float(), b.float()
    if a.shape != b.shape:
        return {"comparable": False, "left_shape": list(a.shape), "front_shape": list(b.shape)}
    denominator = float(a.norm() * b.norm())
    return {"comparable": True, "rms_diff": rms(a - b), "rms_common": rms((a + b) / 2),
            "rms_left": rms(a), "rms_front": rms(b),
            "cosine": float((a * b).sum()) / denominator if denominator > 1e-20 else None}


def transfer_metrics(base, donor, patched, epsilon):
    selections = {"primary": (slice(0, 10), slice(0, 7)),
                  "motion": (slice(0, 10), slice(0, 6)),
                  "gripper": (slice(0, 10), slice(6, 7)),
                  "full50": (slice(0, 50), slice(0, 7))}
    result = {}
    for name, selection in selections.items():
        a, b, c = (value[selection].double() for value in (base, donor, patched))
        delta, contrast = c - a, b - a
        d = rms(contrast)
        valid = d > epsilon
        coefficient = float((delta * contrast).sum() / contrast.square().sum()) if valid else None
        result[name] = {"rms_delta": rms(delta), "rms_native_contrast": d, "T": coefficient,
                        "aligned_rms": abs(coefficient) * d if valid else None,
                        "perp_rms": rms(delta - coefficient * contrast) if valid else None,
                        "epsilon": epsilon, "valid_T": valid}
    return result


class Observer:
    """Read actual CA tensors; return the original eager result unchanged."""

    def __init__(self, policy, torch):
        self.torch = torch
        self.model = policy.model
        self.expert = self.model.vlm_with_expert
        self.active = False
        self.current_layer = None
        self.record = {}
        self.patch = None
        self.readout_error_max = 0.0
        self.handles = []
        self.original_cross = self.expert.forward_cross_attn_layer
        self.original_eager = self.expert.eager_attention_forward
        self.layer_map = []
        layers = self.expert.get_model_layers(
            [self.expert.get_vlm_model().text_model, self.expert.lm_expert])
        for index, layer in enumerate(layers[1]):
            if layer is None:
                continue
            expert_index = next(i for i, candidate in enumerate(self.expert.lm_expert.layers)
                                if candidate is layer)
            kind = ("SA" if "cross" not in self.expert.attention_mode
                    or (self.expert.self_attn_every_n_layers > 0
                        and index % self.expert.self_attn_every_n_layers == 0) else "CA")
            self.layer_map.append({"vlm_layer": index, "expert_layer": expert_index, "kind": kind})
            if kind == "CA":
                self.handles.append(layer.self_attn.o_proj.register_forward_hook(
                    self._message_hook(index)))
                self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
                    self._residual_hook(index)))
                self.handles.append(layer.mlp.register_forward_hook(self._mlp_hook(index)))
        self.handles.append(self.expert.lm_expert.norm.register_forward_hook(self._final_hook))
        self.expert.forward_cross_attn_layer = self._cross
        self.expert.eager_attention_forward = self._eager

    @property
    def ca_layers(self):
        return [row["vlm_layer"] for row in self.layer_map if row["kind"] == "CA"]

    def _cross(self, *args, **kwargs):
        previous = self.current_layer
        self.current_layer = kwargs.get("layer_idx", args[2] if len(args) > 2 else None)
        try:
            return self.original_cross(*args, **kwargs)
        finally:
            self.current_layer = previous

    def _eager(self, mask, batch_size, head_dim, query, key, value):
        output = self.original_eager(mask, batch_size, head_dim, query, key, value)
        if self.active and self.current_layer is not None:
            torch = self.torch
            groups = self.expert.num_attention_heads // self.expert.num_key_value_heads
            expanded = key[:, :, :, None, :].expand(
                batch_size, key.shape[1], self.expert.num_key_value_heads, groups, head_dim)
            expanded = expanded.reshape(batch_size, key.shape[1], -1, head_dim)
            scores = torch.matmul(query.float().transpose(1, 2),
                                  expanded.float().transpose(1, 2).transpose(2, 3))
            scores *= head_dim ** -0.5
            # Native eager explicitly upcasts AFTER the possibly autocast matmul/scale.
            scores = scores.to(dtype=torch.float32)
            scores = torch.where(mask[:, None, :, :], scores, torch.finfo(scores.dtype).min)
            probabilities = scores.softmax(-1).to(value.dtype)
            expanded_value = value[:, :, :, None, :].expand(
                batch_size, value.shape[1], self.expert.num_key_value_heads, groups, head_dim)
            expanded_value = expanded_value.reshape(batch_size, value.shape[1], -1, head_dim)
            reconstructed = torch.matmul(probabilities, expanded_value.permute(0, 2, 1, 3))
            reconstructed = reconstructed.permute(0, 2, 1, 3).reshape_as(output)
            difference = float((reconstructed.float() - output.float()).abs().max())
            self.readout_error_max = max(self.readout_error_max, difference)
            if not math.isfinite(difference) or difference > 1e-5:
                raise AssertionError(f"Recorded P/C do not reconstruct native readout: {difference}")
            self.record.setdefault(self.current_layer, {}).update(
                P=probabilities.detach().float().cpu(), C=value.detach().float().cpu(),
                PC=output.detach().float().cpu())
        return output

    def _message_hook(self, index):
        def hook(module, inputs, output):
            if not self.active:
                return None
            self.record.setdefault(index, {})["M"] = output.detach().clone().float().cpu()
            if self.patch is not None and self.patch[0] == index:
                source = self.patch[1]
                if source.shape != output.shape:
                    raise ValueError("Message shape mismatch")
                return source.to(device=output.device, dtype=output.dtype).clone()
            return None
        return hook

    def _residual_hook(self, index):
        def hook(module, inputs):
            if self.active:
                self.record.setdefault(index, {})["attention_residual"] = (
                    inputs[0].detach().clone().float().cpu())
        return hook

    def _mlp_hook(self, index):
        def hook(module, inputs, output):
            if self.active:
                # The native forward then adds the residual in-place to this object.
                self.record.setdefault(index, {})["_post_mlp_reference"] = output.detach()
        return hook

    def _final_hook(self, module, inputs, output):
        if self.active:
            self.record["final_norm"] = output.detach().clone().float().cpu()

    def begin(self, patch=None):
        self.record = {}
        self.patch = patch
        self.active = True

    def finish(self):
        self.active = False
        self.patch = None
        for layer in self.ca_layers:
            if layer not in self.record:
                raise AssertionError(f"CA layer {layer} did not execute")
            reference = self.record[layer].pop("_post_mlp_reference")
            self.record[layer]["post_mlp"] = reference.float().cpu().clone()
        return self.record

    def close(self):
        self.expert.forward_cross_attn_layer = self.original_cross
        self.expert.eager_attention_forward = self.original_eager
        for handle in self.handles:
            handle.remove()


def make_inputs(policy, preprocessor, observation, instruction, torch, np, quat2axisangle, constants):
    raw = {"task": instruction, "observation.state": torch.from_numpy(np.concatenate((
        observation["robot0_eef_pos"],
        quat2axisangle(observation["robot0_eef_quat"].copy()),
        observation["robot0_gripper_qpos"])).astype(np.float32))}
    for key, image in (("observation.images.image", "agentview_image"),
                       ("observation.images.image2", "robot0_eye_in_hand_image")):
        raw[key] = torch.from_numpy(np.ascontiguousarray(
            observation[image][::-1, ::-1])).permute(2, 0, 1).float() / 255
    batch = preprocessor(copy.deepcopy(raw))
    images, masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    return (images, masks, batch[constants.OBS_LANGUAGE_TOKENS],
            batch[constants.OBS_LANGUAGE_ATTENTION_MASK], state)


def make_prefix(policy, inputs, make_att_2d_masks):
    images, masks, tokens, language_mask, state = inputs
    model = policy.model
    embedded, pad, attention = model.embed_prefix(images, masks, tokens, language_mask, state=state)
    _, cache = model.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(pad, attention),
        position_ids=pad.long().cumsum(1) - 1, past_key_values=None,
        inputs_embeds=[embedded, None], use_cache=True)
    state_length = 1 if state.ndim == 2 else state.shape[1]
    image_length = embedded.shape[1] - state_length - tokens.shape[1]
    if image_length < 0 or model.config.prefix_length > 0:
        raise ValueError("Expected no extra prefix padding")
    groups = {"image": [0, image_length],
              "text": [image_length, image_length + tokens.shape[1]],
              "state": [image_length + tokens.shape[1], embedded.shape[1]]}
    return pad, cache, groups


def denoise(policy, observer, prefix, x_t, timestep, record=False, patch=None):
    pad, source_cache, _ = prefix
    cache = copy.deepcopy(source_cache)
    if cache is source_cache or any(a.keys.data_ptr() == b.keys.data_ptr() for a, b in
                                   zip(cache.layers, source_cache.layers)):
        raise AssertionError("Cache object/storage alias")
    observer.active = False
    if record:
        observer.begin(patch)
    try:
        velocity = policy.model.denoise_step(pad, cache, x_t.clone(), timestep.clone())
        if not velocity.isfinite().all():
            raise ValueError("Nonfinite velocity")
        captured = observer.finish() if record else None
    finally:
        observer.active = False
        observer.patch = None
    if any(layer.keys.shape[-2] != pad.shape[1] for layer in cache.layers):
        raise AssertionError("Denoise failed to crop suffix cache")
    return velocity.detach().clone(), captured


def velocity_tensor(velocity):
    return velocity[0, :50, :7].detach().float().cpu()


def summarize_record(record, groups):
    result = {}
    for index, features in record.items():
        if index == "final_norm":
            continue
        row = {name: stats(value) for name, value in features.items()}
        probabilities = features["P"][:, :, :10]
        row["P"]["mass_by_source"] = {
            name: probabilities[..., start:end].sum(-1).mean((0, 2)).tolist()
            for name, (start, end) in groups.items()}
        result[str(index)] = row
    return result


def parameter_audit(teacher, student, torch):
    result = {}
    other = dict(student.named_parameters())
    for name, a in teacher.named_parameters():
        if name not in other or a.shape != other[name].shape:
            raise ValueError(f"Parameter coordinate mismatch: {name}")
        b = other[name]
        group = ("VLM" if ".vlm." in name else
                 "state_proj" if "state_proj" in name else
                 "action_out_proj" if "action_out_proj" in name else
                 "expert" if "lm_expert" in name else "other_action_projections")
        row = result.setdefault(group, {"count": 0, "different_elements": 0,
                                        "sum_squared_difference": 0.0, "max_abs": 0.0,
                                        "teacher_requires_grad": 0, "student_requires_grad": 0})
        row["count"] += a.numel()
        row["teacher_requires_grad"] += a.numel() if a.requires_grad else 0
        row["student_requires_grad"] += b.numel() if b.requires_grad else 0
        diff = a.detach().float() - b.detach().float()
        row["different_elements"] += int(torch.count_nonzero(diff))
        row["sum_squared_difference"] += float(diff.double().square().sum())
        row["max_abs"] = max(row["max_abs"], float(diff.abs().max()))
    for row in result.values():
        row["rms_difference"] = math.sqrt(row["sum_squared_difference"] / row["count"])
    if result.get("VLM", {}).get("different_elements", 1) != 0:
        raise AssertionError("Frozen VLM weights differ")
    return result


def run_analysis(args, report, models, observers, initial, prefixes, noise,
                 euler_integrate, guard, torch, np):
    source = args.output.parent / "engineering/engineering.json"
    engineering = json.loads(source.read_text(encoding="utf-8"))
    if engineering["passed"] is not True or engineering["layer_map"] != report["layer_map"]:
        raise AssertionError("Engineering gate missing or changed layout")
    write(args.output / "engineering.json", engineering)
    report["engineering_source"] = {"path": str(source), "sha256": sha(source)}
    report["identity_epsilon_v"] = engineering["repeat_v_rms_max"]
    epsilon = max(10 * report["identity_epsilon_v"], 1e-6)
    patch_summaries = []
    for state_index in args.indices:
        observation = initial(state_index)
        prepared = prefixes(observation)
        for seed_index, seed in enumerate(args.seeds):
            reference_goal = ("left", "front")[(state_index + seed_index) % 2]
            reference_policy = models["R0"]
            reference_prefix = prepared["R0"][reference_goal][1]
            references = {}
            counter = [0]
            def reference_velocity(action_x, timestep):
                flow_index = counter[0]
                if flow_index in FLOW_INDICES:
                    references[flow_index] = (action_x.detach().clone(), timestep.detach().clone())
                value, _ = denoise(reference_policy, observers["R0"], reference_prefix,
                                   action_x, timestep)
                counter[0] += 1
                report["counts"]["reference_denoise"] += 1
                return value
            euler_integrate(reference_velocity, noise(seed), 10)
            if set(references) != set(FLOW_INDICES):
                raise AssertionError("Flow grid mismatch")
            for flow_index in FLOW_INDICES:
                x_t, timestep = references[flow_index]
                if abs(float(timestep[0]) - (1 - flow_index / 10)) > 1e-5:
                    raise AssertionError("Unexpected flow time convention")
                anchor_id = f"i{state_index:02d}_s{seed}_f{flow_index:02d}"
                common = {"anchor_id": anchor_id, "state_index": state_index,
                          "noise_seed": seed, "flow_index": flow_index,
                          "time": float(timestep[0]), "reference_goal": reference_goal,
                          "x_sha256": tensor_sha(x_t),
                          "state_sha256": report["physical_state_sha256"][str(state_index)]}
                captures, velocities = {}, {}
                for role, policy in models.items():
                    captures[role], velocities[role] = {}, {}
                    for goal in GOALS:
                        prefix = prepared[role][goal][1]
                        value, capture = denoise(policy, observers[role], prefix, x_t,
                                                timestep, record=True)
                        active_v = velocity_tensor(value)
                        captures[role][goal] = capture
                        velocities[role][goal] = active_v
                        append(args.output / "baseline.jsonl", {
                            **common, "model": role, "goal": goal, "velocity": active_v.tolist(),
                            "layers": summarize_record(capture, prefix[2]),
                            "final_norm": stats(capture["final_norm"])})
                        report["counts"]["baseline"] += 1
                    for ga, gb in (("left", "front"), ("left", "right"), ("front", "right")):
                        for layer in observers[role].ca_layers:
                            features = {name: pair_stats(captures[role][ga][layer][name],
                                                         captures[role][gb][layer][name])
                                        for name in ("P", "C", "PC", "M", "attention_residual", "post_mlp")}
                            primary_features = {}
                            for name in ("P", "C", "PC", "M", "attention_residual", "post_mlp"):
                                a, b = captures[role][ga][layer][name], captures[role][gb][layer][name]
                                if name == "P":
                                    a, b = a[:, :, :10], b[:, :, :10]
                                elif name != "C":
                                    a, b = a[:, :10], b[:, :10]
                                primary_features[name] = pair_stats(a, b)
                            append(args.output / "paired.jsonl", {
                                **common, "model": role, "goal_pair": [ga, gb], "layer": layer,
                                "features": features, "features_primary": primary_features,
                                "velocity": pair_stats(velocities[role][ga], velocities[role][gb]),
                                "velocity_primary": pair_stats(velocities[role][ga][:10],
                                                               velocities[role][gb][:10]),
                                "velocity_motion_primary": pair_stats(velocities[role][ga][:10, :6],
                                                                      velocities[role][gb][:10, :6]),
                                "velocity_gripper_primary": pair_stats(velocities[role][ga][:10, 6:7],
                                                                       velocities[role][gb][:10, 6:7]),
                                "final_norm": pair_stats(captures[role][ga]["final_norm"],
                                                         captures[role][gb]["final_norm"]),
                                "final_norm_primary": pair_stats(captures[role][ga]["final_norm"][:, :10],
                                                                 captures[role][gb]["final_norm"][:, :10])})
                            report["counts"]["paired"] += 1
                if seed == args.seeds[0]:
                    for role, policy in models.items():
                        for recipient, donor in (("left", "front"), ("front", "left")):
                            for layer in observers[role].ca_layers:
                                patched, _ = denoise(
                                    policy, observers[role], prepared[role][recipient][1],
                                    x_t, timestep, record=True,
                                    patch=(layer, captures[role][donor][layer]["M"]))
                                patched_v = velocity_tensor(patched)
                                metrics = transfer_metrics(velocities[role][recipient],
                                                           velocities[role][donor], patched_v, epsilon)
                                row = {**common, "model": role, "recipient": recipient,
                                       "donor": donor, "layer": layer,
                                       "velocity": patched_v.tolist(), "metrics": metrics}
                                append(args.output / "patches.jsonl", row)
                                patch_summaries.append({key: row[key] for key in
                                                        ("state_index", "flow_index", "model",
                                                         "recipient", "donor", "layer", "metrics")})
                                report["counts"]["patches"] += 1
                del captures, velocities
                guard()
                print(json.dumps({"completed_anchor": anchor_id, "counts": report["counts"]}), flush=True)
    expected_baseline = len(args.indices) * len(args.seeds) * 3 * 2 * 3
    expected_patch = len(args.indices) * 3 * 2 * sum(
        len(observer.ca_layers) for observer in observers.values())
    if report["counts"]["baseline"] != expected_baseline or report["counts"]["patches"] != expected_patch:
        raise AssertionError("Incomplete matrix")
    candidates = []
    for layer in observers["R0"].ca_layers:
        rows = [row for row in patch_summaries if row["model"] == "R0" and row["layer"] == layer]
        directions = {}
        for recipient in ("left", "front"):
            selected = [row["metrics"]["primary"] for row in rows if row["recipient"] == recipient]
            valid = [row["T"] for row in selected if row["valid_T"]]
            directions[recipient] = {"valid": len(valid), "total": len(selected),
                                     "median_T": float(np.median(valid)) if valid else None,
                                     "median_rms_delta": float(np.median(
                                         [row["rms_delta"] for row in selected]))}
        eligible = all(row["valid"] > 0 and row["median_T"] > 0
                       and row["median_rms_delta"] > epsilon for row in directions.values())
        score = min(row["median_T"] for row in directions.values()) if eligible else None
        candidates.append({"layer": layer, "directions": directions, "eligible": eligible, "score": score})
    candidates.sort(key=lambda row: row["score"] if row["score"] is not None else -math.inf, reverse=True)
    report["candidates"] = [row["layer"] for row in candidates if row["eligible"]][:2]
    write(args.output / "candidates.json", {
        "ranking": candidates, "selected": report["candidates"],
        "interpretation": "development ranking of velocity transfer, not causal localization/significance",
        "no_followup_launched": True})


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    start = time.monotonic()
    report = {"status": "initializing", "phase": args.phase,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "indices": args.indices, "seeds": args.seeds, "flow_indices": list(FLOW_INDICES),
              "roles": ["R0", "FT4"], "goals": list(GOALS),
              "scope": "open-loop steps0-2; fixed reset settling only, no policy-generated actions or parameter update",
              "counts": {"baseline": 0, "patches": 0, "paired": 0, "reference_denoise": 0},
              "script_sha256": sha(__file__), "identity_tolerance": args.identity_tolerance}
    observers = {}
    env = None
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
        from stage1_smoke import DEFAULT_SCENE
        from stage2k_entry_handoff import _fresh_observation
        from stage2k_response_swap import verify_processors

        torch.set_num_threads(4)
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        if not args.indices or len(set(args.indices)) != len(args.indices):
            raise ValueError("Physical indices must be nonempty and unique")
        if not args.seeds or len(set(args.seeds)) != len(args.seeds):
            raise ValueError("Noise seeds must be nonempty and unique")
        if not set(args.indices) <= set(cfg["evaluation"]["development_indices"]):
            raise ValueError("Only sealed development indices are allowed")
        report["processor_hashes"] = verify_processors(args.teacher, args.student)
        report["model_sha256"] = {role: sha(path / "model.safetensors") for role, path in
                                 (("R0", args.teacher), ("FT4", args.student))}
        models = {"R0": evaluation.load_policy(args.teacher),
                  "FT4": evaluation.load_policy(args.student)}
        preprocessors = {}
        for role, policy in models.items():
            policy.eval()
            policy.config.n_action_steps = 10
            policy.config.num_steps = 10
            if policy.config.adapt_to_pi_aloha or policy.model._rtc_enabled():
                raise ValueError("Unsupported action coordinate or RTC mode")
            if policy.config.n_obs_steps != 1 or not policy.model.config.use_cache:
                raise ValueError("Expected single observation and KV cache")
            if policy.config.action_feature.shape[0] != 7:
                raise ValueError("Expected exactly seven active action dimensions")
            observers[role] = Observer(policy, torch)
            preprocessors[role], _ = policies.make_pre_post_processors(
                policy.config, pretrained_path=args.teacher if role == "R0" else args.student)
        report["layer_map"] = {role: observer.layer_map for role, observer in observers.items()}
        report["ca_layers"] = {role: observer.ca_layers for role, observer in observers.items()}
        if not all(report["ca_layers"].values()):
            raise ValueError("No Cross-Attention layers in actual runtime")
        report["parameter_audit"] = parameter_audit(models["R0"], models["FT4"], torch)
        fields = ("attention_mode", "num_vlm_layers", "num_expert_layers", "self_attn_every_n_layers",
                  "train_expert_only", "train_state_proj", "freeze_vision_encoder", "prefix_length",
                  "pad_language_to", "chunk_size", "n_action_steps", "num_steps", "n_obs_steps",
                  "max_action_dim", "max_state_dim", "expert_width_multiplier", "use_cache")
        report["configuration"] = {role: {name: getattr(policy.config, name, None) for name in fields}
                                   for role, policy in models.items()}
        report["checkpoint_config"] = {role: json.loads((path / "config.json").read_text())
                                        for role, path in (("R0", args.teacher), ("FT4", args.student))}
        report["sources"] = {}
        for name, obj in (("evaluation", evaluation),
                          ("model", sys.modules[type(models["R0"].model).__module__]),
                          ("expert", sys.modules[type(models["R0"].model.vlm_with_expert).__module__])):
            report["sources"][name] = {"path": obj.__file__, "sha256": sha(obj.__file__)}

        tasks, paths, _, physical_states = evaluation.task_resources(cfg.get("scene", DEFAULT_SCENE))
        instructions = {goal: tasks[goal].language for goal in GOALS}
        report["instructions"] = instructions
        env = evaluation.make_env(paths["front"])
        scene_xml = evaluation.canonical_scene(paths["front"], cfg["seed"])
        report["scene_xml_sha256"] = hashlib.sha256(scene_xml.encode()).hexdigest()

        def guard():
            if time.monotonic() - start > args.max_minutes * 60:
                raise TimeoutError("Predeclared worker time budget")
            size = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
            if size > args.max_output_bytes or shutil.disk_usage(args.output).free < 2 * 1024**3:
                raise RuntimeError("Output/disk budget; nothing deleted")
            write(args.output / "progress.json",
                  {**report, "elapsed_seconds": time.monotonic() - start})

        def initial(index):
            evaluation.reset_common(env, physical_states["front"][index], cfg["seed"] + index,
                                    cfg["evaluation"]["settle_steps"], scene_xml)
            observation = _fresh_observation(env)
            report.setdefault("physical_state_sha256", {})[str(index)] = hashlib.sha256(
                np.asarray(env.get_sim_state()).tobytes()).hexdigest()
            return observation

        def prefixes(obs):
            result = {}
            input_hashes = {}
            for role, policy in models.items():
                result[role] = {}
                input_hashes[role] = {}
                for goal in GOALS:
                    inputs = make_inputs(policy, preprocessors[role], obs, instructions[goal],
                                         torch, np, quat2axisangle, constants)
                    images, image_masks, tokens, language_mask, state = inputs
                    input_hashes[role][goal] = {
                        "images": [tensor_sha(value) for value in images],
                        "image_masks": [tensor_sha(value) for value in image_masks],
                        "language_tokens": tensor_sha(tokens),
                        "language_mask": tensor_sha(language_mask),
                        "state_before_projection": tensor_sha(state),
                        "image_shapes": [list(value.shape) for value in images],
                        "language_shape": list(tokens.shape), "state_shape": list(state.shape)}
                    result[role][goal] = (inputs, make_prefix(policy, inputs, make_att_2d_masks))
            report.setdefault("prefix_input_hashes", []).append(input_hashes)
            for goal in GOALS:
                if input_hashes["R0"][goal] != input_hashes["FT4"][goal]:
                    raise AssertionError(f"Preprocessed image/language/state inputs differ: {goal}")
            caches = [result[role][goal][1][1] for role in models for goal in GOALS]
            if len(set(map(id, caches))) != len(caches):
                raise AssertionError("Independent prefix objects alias")
            pointers = [layer.keys.data_ptr() for cache in caches for layer in cache.layers]
            if len(set(pointers)) != len(pointers):
                raise AssertionError("Independent prefix storage aliases")
            prefix_rows = []
            # Append before checks so any failure preserves the diagnostic rows.
            report.setdefault("prefix_audit", []).append(prefix_rows)
            for goal in GOALS:
                a, b = result["R0"][goal][1], result["FT4"][goal][1]
                if a[2] != b[2] or not torch.equal(a[0], b[0]):
                    raise AssertionError("Prefix group/mask coordinates differ")
                for layer_index, (old_layer, new_layer) in enumerate(zip(a[1].layers, b[1].layers)):
                    for kind in ("keys", "values"):
                        old, new = getattr(old_layer, kind), getattr(new_layer, kind)
                        for group, (lo, hi) in a[2].items():
                            difference = old[:, :, lo:hi].float() - new[:, :, lo:hi].float()
                            max_abs = float(difference.abs().max())
                            position_mask = a[0][:, lo:hi].bool()
                            by_position = difference.permute(0, 2, 1, 3)
                            subsets = {}
                            for label, selector in (("active", position_mask),
                                                    ("masked", ~position_mask)):
                                values = by_position[selector]
                                subsets[label] = {
                                    "count": int(selector.sum()),
                                    "element_count": values.numel(),
                                    "max_abs": float(values.abs().max()) if values.numel() else 0.0,
                                    "rms": rms(values) if values.numel() else 0.0}
                            row = {"goal": goal, "layer": layer_index, "tensor": kind,
                                   "group": group, "max_abs": max_abs, "rms": rms(difference),
                                   **subsets}
                            prefix_rows.append(row)
                            # PAD query states are not valid conditioning material. They
                            # remain reported, but only valid image/text positions gate.
                            if (group in ("image", "text")
                                    and subsets["active"]["max_abs"] > args.identity_tolerance):
                                raise AssertionError(
                                    f"Frozen valid prefix differs: goal={goal} layer={layer_index} "
                                    f"tensor={kind} group={group} "
                                    f"active_max={subsets['active']['max_abs']} "
                                    f"masked_max={subsets['masked']['max_abs']}")
            return result

        def noise(seed):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            return torch.randn((1, 50, models["R0"].config.max_action_dim),
                               generator=generator, dtype=torch.float32).cuda()

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if args.phase == "engineering":
                obs = initial(args.indices[0])
                prepared = prefixes(obs)
                x = noise(args.seeds[0])
                t = torch.ones(1, device=x.device)
                checks = []
                repeated = []
                for role, policy in models.items():
                    for goal in GOALS:
                        inputs, prefix = prepared[role][goal]
                        observer = observers[role]
                        native, _ = denoise(policy, observer, prefix, x, t)
                        repeated_v, _ = denoise(policy, observer, prefix, x, t)
                        recorded, trace = denoise(policy, observer, prefix, x, t, record=True)
                        self_error = 0.0
                        for layer in observer.ca_layers:
                            patched, _ = denoise(policy, observer, prefix, x, t, record=True,
                                                 patch=(layer, trace[layer]["M"]))
                            self_error = max(self_error,
                                             float((patched[..., :7] - native[..., :7]).abs().max()))
                        observer.active = False
                        native_action = policy.model.sample_actions(*inputs, noise=x).float()
                        observer.active = True
                        observer.record = {}
                        recorded_action = policy.model.sample_actions(*inputs, noise=x).float()
                        observer.active = False
                        first_layer = observer.ca_layers[0]
                        def sampled(action_x, timestep):
                            _, capture = denoise(policy, observer, prefix, action_x, timestep, record=True)
                            patched, _ = denoise(policy, observer, prefix, action_x, timestep,
                                                 record=True, patch=(first_layer, capture[first_layer]["M"]))
                            return patched
                        self_action = euler_integrate(sampled, x, 10).float()
                        row = {"model": role, "goal": goal,
                               "record_v_max_abs": float((recorded[..., :7] - native[..., :7]).abs().max()),
                               "self_v_max_abs": self_error,
                               "record_action_max_abs": float((recorded_action[..., :7] - native_action[..., :7]).abs().max()),
                               "self_action_max_abs": float((self_action[..., :7] - native_action[..., :7]).abs().max()),
                               "repeat_v_rms": rms(repeated_v[..., :7] - native[..., :7]),
                               "native_velocity_rms": rms(native[..., :7]),
                               "attention_readout_reconstruction_max_abs": observer.readout_error_max}
                        checks.append(row)
                        repeated.append(row["repeat_v_rms"])
                        write(args.output / "engineering_checks_progress.json", checks)
                        if max(row[key] for key in ("record_v_max_abs", "self_v_max_abs",
                                                   "record_action_max_abs", "self_action_max_abs",
                                                   "attention_readout_reconstruction_max_abs")) > args.identity_tolerance:
                            raise AssertionError(f"Native/noop/self-replacement mismatch: {row}")
                        guard()
                engineering = {"passed": True, "checks": checks, "layer_map": report["layer_map"],
                               "identity_tolerance": args.identity_tolerance,
                               "cache_isolation": {"object_unique": True, "tensor_storage_unique": True,
                                                   "deep_copy_per_query": True},
                               "repeat_v_rms_max": max(repeated)}
                write(args.output / "engineering.json", engineering)
                report["identity_epsilon_v"] = max(repeated)
            else:
                run_analysis(args, report, models, observers, initial, prefixes, noise,
                             euler_integrate, guard, torch, np)
        report.update(status="complete", elapsed_seconds=time.monotonic() - start,
                      finished_at=datetime.now(timezone.utc).isoformat(),
                      cache_isolation={"object_unique": True, "tensor_storage_unique": True,
                                       "deep_copy_per_query": True},
                      scope_exclusions=["no policy-generated control; fixed reset settling only", "no cross-checkpoint patches",
                                        "no parameter update", "no P/C intervention"])
        report["attention_readout_reconstruction_max_abs"] = {
            role: observer.readout_error_max for role, observer in observers.items()}
        report["files"] = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)}
                           for path in args.output.iterdir() if path.is_file()
                           and path.name not in ("progress.json", "manifest.json", "COMPLETE.json")}
        write(args.output / "manifest.json", report)
        write(args.output / "COMPLETE.json", report)
        write(args.output / "progress.json", report)
    except Exception as error:
        report.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic() - start,
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
    parser.add_argument("--runtime-scripts", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("engineering", "analysis"), required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026102801, 2026102802, 2026102803])
    parser.add_argument("--identity-tolerance", type=float, default=1e-5)
    parser.add_argument("--max-minutes", type=float, default=45)
    parser.add_argument("--max-output-bytes", type=int, default=536870912)
    main(parser.parse_args())
