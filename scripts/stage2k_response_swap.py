"""Development-only old/new common-response versus condition-difference swaps.

Run against the stage2a runtime using --runtime-scripts. The source worktree is
never changed. Every Euler step queries both checkpoints on the SAME complete
observation, language pair, noisy action and time. Prefix caches remain separate.
No gradients, teacher training, checkpoint edits, or implicit retries are used.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time


MODES = ("new", "old_mean", "old_difference", "old")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_processors(teacher, student):
    names = (
        "policy_preprocessor.json", "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    result = {}
    for name in names:
        a, b = teacher / name, student / name
        if not a.is_file() or not b.is_file():
            raise ValueError(f"Missing processor file: {name}")
        ha, hb = sha256(a), sha256(b)
        if ha != hb:
            raise ValueError(f"Different preprocessing/action coordinates: {name}")
        result[name] = ha
    return result


def main(args):
    sys.path.insert(0, str(args.runtime_scripts.resolve()))
    import torch
    import lerobot.policies as policies
    from lerobot.policies.common import flow_matching
    from lerobot.policies.common.flow_matching import euler_integrate
    from lerobot.policies.common.vla_utils import make_att_2d_masks
    import stage1_r0_eval as evaluation
    from stage1_smoke import DEFAULT_SCENE

    if args.output.exists():
        raise FileExistsError(args.output)
    if len(set(args.indices)) != len(args.indices):
        raise ValueError("Duplicate physical state index")
    config = json.loads(args.config.read_text())
    if not set(args.indices) <= set(config["evaluation"]["development_indices"]):
        raise ValueError("This pilot accepts DEVELOPMENT states only")
    processor_hashes = verify_processors(args.teacher, args.student)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    config = copy.deepcopy(config)
    config["evaluation"]["development_indices"] = args.indices
    config["evaluation"]["horizon"] = args.horizon
    config["old_goals"] = ["left", "front"]
    tasks, paths, _, states = evaluation.task_resources(config.get("scene", DEFAULT_SCENE))
    instructions = [tasks[goal].language for goal in ("left", "front")]
    report = {
        "status": "initializing", "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "exploratory functional intervention, not neural-module localization",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": sha256(__file__), "runtime_scripts": str(args.runtime_scripts.resolve()),
        "runtime_source_hashes": {name: sha256(args.runtime_scripts / name)
                                  for name in ("stage1_r0_eval.py", "stage1_smoke.py")},
        "runtime_dirty": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
        "actual_sources": {"evaluation": {"path": evaluation.__file__, "sha256": sha256(evaluation.__file__)},
                           "flow_matching": {"path": flow_matching.__file__, "sha256": sha256(flow_matching.__file__)}},
        "teacher": str(args.teacher), "student": str(args.student),
        "teacher_sha256": sha256(args.teacher / "model.safetensors"),
        "student_sha256": sha256(args.student / "model.safetensors"),
        "processor_hashes": processor_hashes, "instructions": instructions,
        "indices": args.indices, "horizon": args.horizon, "modes": args.modes,
        "parent_qualification": config.get("parent", {}).get("status", "not verified"),
        "intervention_window": "all replanning/denoising steps; normal task starts",
        "denominators": "all development conditions; no selection by post-test success",
        "runs": [],
    }
    save(args.output / "protocol.json", {**report, "config": config})
    save(args.output / "progress.json", report)
    original_load = evaluation.load_policy
    teacher = original_load(args.teacher)
    student = original_load(args.student)
    for policy in (teacher, student):
        policy.config.n_action_steps = config["model"]["n_action_steps"]
        policy.config.num_steps = config["model"]["flow_steps"]
        policy.eval()
        if policy.model._rtc_enabled():
            raise ValueError("RTC is not supported in this diagnostic")
    for name in ("chunk_size", "max_action_dim", "n_obs_steps", "adapt_to_pi_aloha"):
        if getattr(teacher.config, name, None) != getattr(student.config, name, None):
            raise ValueError(f"Incompatible model configuration: {name}")
    model_file = inspect.getfile(type(student.model))
    expert_file = inspect.getfile(type(student.model.vlm_with_expert))
    report["actual_sources"].update({"model": {"path": model_file, "sha256": sha256(model_file)},
                                     "expert": {"path": expert_file, "sha256": sha256(expert_file)}})
    original_factory = policies.make_pre_post_processors
    original_sample = student.model.sample_actions
    selection = {"requested": 0, "mode": "new", "identity_error_max": 0.0, "calls": 0}

    def factory(*factory_args, **kwargs):
        pre, post = original_factory(*factory_args, **kwargs)

        def paired_pre(batch):
            if batch["task"] not in instructions:
                raise ValueError("Swap is defined only for the two acquired old conditions")
            selection["requested"] = instructions.index(batch["task"])
            expanded = {key: (value.unsqueeze(0).repeat(2, *([1] * value.dim()))
                              if torch.is_tensor(value) else value)
                        for key, value in batch.items() if key != "task"}
            expanded["task"] = instructions
            return pre(expanded)

        def first_post(action):
            return post(action)[0:1]

        return paired_pre, first_post

    def prefix_cache(model, images, img_masks, lang_tokens, lang_masks, state):
        emb, pad, att = model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        _, cache = model.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(pad, att),
            position_ids=torch.cumsum(pad, dim=1) - 1,
            past_key_values=None, inputs_embeds=[emb, None], use_cache=model.config.use_cache,
        )
        return pad, cache

    def swapped_sample(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        if state.shape[0] != 2:
            raise ValueError("Expected the fixed left/front language pair")
        if noise is None:
            shared_noise = student.model.sample_noise(
                (1, student.config.chunk_size, student.config.max_action_dim), state.device)
        else:
            shared_noise = noise[0:1].to(device=state.device, dtype=torch.float32)
        initial = shared_noise.expand(2, -1, -1).contiguous()
        # Singleton forwards avoid BF16 batch-size-dependent changes relative to
        # the ordinary policy. A previous batch-2 smoke failed the locked gate.
        prefixes0, prefixes1 = [], []
        for i in range(2):
            inputs = ([value[i:i + 1] for value in images], [value[i:i + 1] for value in img_masks],
                      lang_tokens[i:i + 1], lang_masks[i:i + 1], state[i:i + 1])
            prefixes0.append(prefix_cache(teacher.model, *inputs))
            prefixes1.append(prefix_cache(student.model, *inputs))
        caches = [cache for _, cache in (*prefixes0, *prefixes1) if cache is not None]
        if len({id(cache) for cache in caches}) != len(caches):
            raise AssertionError("Condition/checkpoint prefix cache alias")

        def velocity(x_t, timestep):
            # Return identical rows, so both conditions always share ONE action draft.
            shared = x_t[0:1].expand_as(x_t).contiguous()
            v0 = torch.cat([teacher.model.denoise_step(pad, cache, shared[0:1], timestep[0:1])
                            for pad, cache in prefixes0])
            v1 = torch.cat([student.model.denoise_step(pad, cache, shared[0:1], timestep[0:1])
                            for pad, cache in prefixes1])
            f0, f1 = v0.float(), v1.float()
            mu0, mu1 = f0.mean(dim=0, keepdim=True), f1.mean(dim=0, keepdim=True)
            r0, r1 = f0 - mu0, f1 - mu1
            identity = max(float((mu0 + r0 - f0).abs().max()), float((mu1 + r1 - f1).abs().max()))
            if identity > 1e-5 or not torch.isfinite(v0).all() or not torch.isfinite(v1).all():
                raise AssertionError("Invalid decomposition or nonfinite response")
            selection["identity_error_max"] = max(selection["identity_error_max"], identity)
            k = selection["requested"]
            mixed = {
                "new": v1[k:k + 1],
                "old_mean": mu0 + r1[k:k + 1],
                "old_difference": mu1 + r0[k:k + 1],
                "old": v0[k:k + 1],
            }[selection["mode"]]
            selection["calls"] += 1
            return mixed.to(v1.dtype).expand_as(v1).contiguous()

        return euler_integrate(velocity, initial, student.config.num_steps)

    # Runtime preprocessor factory is imported inside evaluate, as in its original CLI.
    policies.make_pre_post_processors = factory
    evaluation.load_policy = lambda path: student
    student.model.sample_actions = swapped_sample
    start = time.monotonic()
    try:
        # Compare ordinary batch-1 sampling against each requested row of the
        # batch-2 complete-model controls, using identical explicit noise.
        import numpy as np
        from robosuite.utils.transform_utils import quat2axisangle
        env = evaluation.make_env(paths["front"])
        identity_checks = []
        try:
            scene_xml = evaluation.canonical_scene(paths["front"], config["seed"])
            obs, _ = evaluation.reset_common(env, states["front"][args.indices[0]],
                config["seed"] + args.indices[0], config["evaluation"]["settle_steps"], scene_xml)
            generator = torch.Generator().manual_seed(20261028)
            noise = torch.randn(1, student.config.chunk_size, student.config.max_action_dim,
                                generator=generator).cuda()
            single_pre, _ = original_factory(student.config, pretrained_path=args.student)
            paired_pre, _ = factory(student.config, pretrained_path=args.student)
            for goal in ("left", "front"):
                raw = {"task": tasks[goal].language, "observation.state": torch.from_numpy(
                    np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"].copy()),
                                    obs["robot0_gripper_qpos"])).astype(np.float32))}
                for key, image_name in (("observation.images.image", "agentview_image"),
                                         ("observation.images.image2", "robot0_eye_in_hand_image")):
                    raw[key] = torch.from_numpy(np.ascontiguousarray(obs[image_name][::-1, ::-1])).permute(2, 0, 1).float() / 255
                for mode, baseline in (("new", student), ("old", teacher)):
                    selection["mode"] = mode
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        student.model.sample_actions = original_sample
                        ordinary = baseline.predict_action_chunk(single_pre(copy.deepcopy(raw)), noise=noise).float()
                        student.model.sample_actions = swapped_sample
                        paired = student.predict_action_chunk(paired_pre(copy.deepcopy(raw)),
                                                              noise=noise.expand(2, -1, -1))[0:1].float()
                    difference = paired - ordinary
                    row = {"goal": goal, "mode": mode, "max_abs": float(difference.abs().max()),
                           "rms": float(difference.square().mean().sqrt()),
                           "tolerance": args.identity_tolerance, "coordinate_system": "normalized actions"}
                    identity_checks.append(row)
                    save(args.output / "sampling_identity.json", identity_checks)
                    if row["max_abs"] > args.identity_tolerance or not torch.isfinite(paired).all():
                        raise AssertionError(f"Ordinary-versus-paired sampler mismatch: {row}")
        finally:
            env.close()
            student.model.sample_actions = swapped_sample
        report["sampling_identity"] = identity_checks
        report["status"] = "running"
        for mode in args.modes:
            selection.update(mode=mode, calls=0, identity_error_max=0.0)
            output = args.output / mode
            ns = argparse.Namespace(output=output, checkpoint=args.student, partition="development", limit=None,
                                    trace=True, instruction_goals=["left", "front"], base_policy=False, audit=None)
            evaluation.evaluate(ns, config)
            row = {"mode": mode, "seconds_since_start": time.monotonic() - start,
                   "velocity_calls": selection["calls"], "identity_error_max": selection["identity_error_max"],
                   "evaluation": json.loads((output / "evaluation.json").read_text())}
            report["runs"].append(row)
            report["seconds"] = time.monotonic() - start
            save(args.output / "progress.json", report)
            print(json.dumps({"completed_mode": mode, "seconds": report["seconds"]}), flush=True)
        report.update(status="complete", seconds=time.monotonic() - start,
                      finished_at=datetime.now(timezone.utc).isoformat(),
                      new_episodes=len(args.indices) * 2 * len(args.modes))
        save(args.output / "COMPLETE.json", report)
    except Exception as error:
        report.update(status="failed", error=repr(error), seconds=time.monotonic() - start)
        raise
    finally:
        save(args.output / "progress.json", report)
        policies.make_pre_post_processors = original_factory
        evaluation.load_policy = original_load
        student.model.sample_actions = original_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-scripts", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--identity-tolerance", type=float, default=0.03,
                        help="Predeclared BF16 batch-1/batch-2 maximum normalized-action error")
    main(parser.parse_args())
