"""Stage 2A training from the sealed dual-goal parent: matched FT4, ordinary ER, or the full-replay ceiling.

All branches share, step by step, the same four right-goal records, their flow noise and their flow time.
Each record's loss is divided by the same fixed denominator (6). ER adds one left and one front record from
the self-contained memory; full replay draws them from the complete old training split instead.
"""

import argparse
import copy
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

from stage1_r0_train import frozen_hash
from stage1_smoke import ROOT, dump, load_policy
from stage2_memory_export import load_memory, memory_sample, old_groups, sha256
from stage2_protocol import (BRANCHES, LOSS_DENOMINATOR, OLD_GOALS, exposures, flow_time_from_uniform, loss_weights,
                             step_indices, step_plan, stream_rng, stream_seed)

KEYS = ("observation.images.image", "observation.images.image2", "observation.state", "action", "action_is_pad", "task")


def flow_inputs(count, seed, step, stream, chunk, max_dim):
    """Noise and time for `count` records from a step-keyed generator, independent of any other draw."""
    import torch
    generator = torch.Generator().manual_seed(stream_seed(stream, seed, step))
    noise = torch.randn(count, chunk, max_dim, generator=generator)
    times = flow_time_from_uniform(torch.rand(count, generator=generator, dtype=torch.float64)).float()
    return noise, times


def git_state():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True, cwd=ROOT).strip()
    return commit, bool(dirty)


def validate(config, branch):
    if branch not in BRANCHES:
        raise ValueError(f"Unknown branch {branch}")
    t = config["training"]
    if (t["new_per_step"], t["old_per_goal_per_step"], t["loss_denominator"]) != (4, 1, LOSS_DENOMINATOR):
        raise ValueError("Config composition differs from the registered 4 new + 1 left + 1 front, sum/6 protocol")
    seeds = config["seeds"]
    if len({seeds["new_samples"], seeds["old_samples"]}) != 2:
        raise ValueError("New and old sampling need distinct seeds")
    if config["data"]["normalization_source"] != config["memory"]["source_dataset"]:
        raise ValueError("Normalization must stay the joint left+front training statistics")


def train(args, config):
    import numpy as np
    import torch
    from lerobot.datasets import LeRobotDataset
    from lerobot.policies import make_pre_post_processors

    validate(config, args.branch)
    spec, seeds, t = BRANCHES[args.branch], config["seeds"], config["training"]
    torch.set_num_threads(4)
    random.seed(seeds["new_samples"]); np.random.seed(seeds["new_samples"]); torch.manual_seed(seeds["new_samples"])
    run = args.run_dir
    if run.exists() and not args.resume:
        raise FileExistsError(run)
    if args.resume and args.resume.resolve().parent.parent != run.resolve():
        raise ValueError("Resume must belong to this run")
    parent = ROOT / config["parent"]["checkpoint"]
    marker = json.loads((parent / "CHECKPOINT_COMPLETE.json").read_text())
    parent_manifest = json.loads((parent.parent.parent / "manifest.json").read_text())
    if marker["status"] != "pass" or not marker["frozen_hash_unchanged"]:
        raise ValueError("Parent checkpoint is not verified")
    parent_sha = sha256(parent / "model.safetensors")
    if parent_sha != config["parent"]["model_sha256"]:
        raise ValueError(f"Parent weights changed: {parent_sha}")
    if Path(parent_manifest["normalization_source"]).resolve() != (ROOT / config["data"]["normalization_source"]).resolve():
        raise ValueError("Parent normalization source differs from the configured joint old statistics")
    run.mkdir(parents=True, exist_ok=bool(args.resume))

    new_root = ROOT / config["data"]["train_output"]
    new_conversion = json.loads((new_root / "r0_conversion.json").read_text())
    assert new_conversion["spec"]["goals"] == ["right"] and new_conversion["spec"]["split"] == "train"
    assert new_conversion["spec"]["action_alignment"] == config["data"]["action_alignment"]
    chunk = config["model"]["chunk_size"]
    delta = {"action": [i / config["data"]["fps"] for i in range(chunk)]}
    new_data = LeRobotDataset("local/cl_binding_r0_train", root=new_root, video_backend="pyav", delta_timestamps=delta)
    new_frames = range(len(new_data))
    memory = old_data = None
    memory_budget = None
    old_pools = {goal: [] for goal in OLD_GOALS}
    if spec["old_source"] == "memory":
        arrays, memory_manifest, memory_budget = load_memory(ROOT / config["memory"]["output"], config["memory"]["budget_bytes"])
        memory = (arrays, memory_manifest)
        for goal_index, goal in enumerate(OLD_GOALS):
            old_pools[goal] = [int(i) for i in np.flatnonzero(arrays["goal_index"] == goal_index)]
    pseudo = mem_pools = None
    if spec["old_source"] in ("pseudo", "pseudo_or_memory"):
        labels = np.load(ROOT / config["pseudo_labels"]["directory"] / "pseudo_labels.npz")
        label_manifest = json.loads((ROOT / config["pseudo_labels"]["directory"] / "manifest.json").read_text())
        if label_manifest["parent_model_sha256"] != config["parent"]["model_sha256"] or label_manifest["frames"] != len(new_data):
            raise ValueError("Pseudo-labels do not come from this parent or do not cover the new split")
        pseudo = ({goal: torch.from_numpy(labels[goal]) for goal in OLD_GOALS}, label_manifest["instructions"])
        old_pools = {goal: list(range(len(new_data))) for goal in OLD_GOALS}
    if spec["old_source"] == "pseudo_or_memory":
        arrays, memory_manifest, memory_budget = load_memory(ROOT / config["memory"]["output"], config["memory"]["budget_bytes"])
        memory = (arrays, memory_manifest)
        mem_pools = {goal: [int(i) for i in np.flatnonzero(arrays["goal_index"] == g)] for g, goal in enumerate(OLD_GOALS)}
    elif spec["old_source"] == "full_old_train_split":
        old_root = ROOT / config["memory"]["source_dataset"]
        old_conversion = json.loads((old_root / "r0_conversion.json").read_text())
        old_pools, total = old_groups(old_conversion)
        old_data = LeRobotDataset("local/cl_binding_r0_train", root=old_root, video_backend="pyav", delta_timestamps=delta)
        assert total == len(old_data)

    initial = args.resume or parent
    policy = load_policy(initial)
    policy.config.n_action_steps = config["model"]["n_action_steps"]
    policy.config.num_steps = config["model"]["flow_steps"]
    pre, post = make_pre_post_processors(policy.config, pretrained_path=parent)
    names = [name for name, p in policy.named_parameters() if p.requires_grad]
    assert not any(".vlm." in name for name in names), "VLM must be frozen"
    dump(run / "trainable_parameters.json", {"names": names, "count": sum(p.numel() for p in policy.parameters() if p.requires_grad)})
    original_frozen = frozen_hash(policy)
    assert original_frozen == parent_manifest["frozen_hash"], "Initialization changed frozen parameters"
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=t["learning_rate"], betas=policy.config.optimizer_betas,
                                  eps=policy.config.optimizer_eps, weight_decay=policy.config.optimizer_weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / t["warmup_steps"]))
    start_step = 0
    commit, dirty = git_state()
    if args.resume:
        saved = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=False)
        assert saved["config"] == config and saved["branch"] == args.branch and saved["frozen_hash"] == original_frozen
        optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"]); random.setstate(saved["python_rng"])
        start_step = saved["step"]
        dump(run / f"resume_{start_step:06d}.json", {"status": "pass", "restored_step": start_step, "git_commit": commit})
    else:
        dump(run / "manifest.json", {
            "config": config, "branch": args.branch, "composition": spec, "loss_denominator": LOSS_DENOMINATOR,
            "git_commit": commit, "git_tracked_changes": dirty, "frozen_hash": original_frozen,
            "initialization_checkpoint": str(parent.resolve()), "initialization_model_sha256": parent_sha,
            "initialization_step": marker["step"], "optimizer_reset": True, "scheduler_reset": True,
            "normalization_source": parent_manifest["normalization_source"],
            "normalization_policy": "inherit_parent_pre_and_post_processors",
            "parent_processor_sha256": {p.name: sha256(p) for p in parent.glob("*processor*.safetensors")},
            "new_split_sha256": new_conversion["spec"]["split_sha256"],
            "memory": None if memory is None else {"directory": config["memory"]["output"], "budget": memory_budget,
                                                    "memory_npz_sha256": memory[1]["memory_npz_sha256"]},
            "old_data_access": spec["old_source"], "teacher_calls": 0,
            "pseudo_labels": config.get("pseudo_labels") if pseudo is not None else None,
            "matching": "same new indices, noise and time per step across branches; FLOPs and post-AdamW/clipping "
                        "new-gradient contribution are not matched"})
    maximum = args.max_steps or t["max_steps"]
    if maximum <= start_step:
        raise ValueError("Target step must exceed resume step")
    weights = torch.tensor(loss_weights(args.branch), device="cuda")
    max_dim = policy.config.max_action_dim
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    last_step = start_step

    def checkpoint(step, raw_batch):
        policy.eval(); policy.reset()
        frozen_now = frozen_hash(policy)
        assert frozen_now == original_frozen, "Frozen parameter changed"
        cpu_rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state_all()
        np_rng = np.random.get_state(); python_rng = random.getstate()
        folder = run / "checkpoints" / f"{step:06d}"
        if folder.exists():
            raise FileExistsError(folder)
        noise = torch.randn(len(raw_batch["task"]), chunk, max_dim,
                            generator=torch.Generator(device="cuda").manual_seed(seeds["new_samples"] + step), device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            reference = policy.predict_action_chunk(pre(copy.deepcopy(raw_batch)), noise=noise).float().cpu()
        policy.save_pretrained(folder); pre.save_pretrained(folder); post.save_pretrained(folder)
        torch.save({"step": step, "config": config, "branch": args.branch, "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "torch_rng": cpu_rng, "cuda_rng": cuda_rng,
                    "numpy_rng": np_rng, "python_rng": python_rng, "frozen_hash": frozen_now}, folder / "training_state.pt")
        restored = load_policy(folder); restored.eval()
        pre2, post2 = make_pre_post_processors(restored.config, pretrained_path=folder)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            actual = restored.predict_action_chunk(pre2(copy.deepcopy(raw_batch)), noise=noise).float().cpu()
            physical = post2(actual.to("cuda")).float().cpu()
        assert torch.allclose(reference, actual, atol=1e-3, rtol=1e-3), "Checkpoint reload differs"
        assert physical.shape[-1] == 7 and torch.isfinite(physical).all()
        difference = (reference - actual).abs().max().item()
        del restored, pre2, post2
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(np_rng); random.setstate(python_rng)
        dump(folder / "CHECKPOINT_COMPLETE.json", {"status": "pass", "step": step, "branch": args.branch,
             "reload_max_difference": difference, "frozen_hash_unchanged": True, "optimizer_and_rng_saved": True,
             "model_sha256": sha256(folder / "model.safetensors")})
        policy.train(); torch.cuda.empty_cache()
        print(json.dumps({"checkpoint_complete": str(folder), "step": step, "reload_difference": difference}), flush=True)

    for step in range(start_step, maximum):
        plan = step_plan(args.branch, step, seeds, new_frames, old_pools)
        samples = [{k: s[k] for k in KEYS} for s in new_data.__getitems__(plan["new"])]
        noise, times = flow_inputs(spec["new"], seeds["new_samples"], step, "new-flow", chunk, max_dim)
        for goal in OLD_GOALS:
            if not plan[goal]:
                continue
            if pseudo is not None:
                use_memory = mem_pools is not None and stream_rng(f"mix-{goal}", seeds["old_samples"], step).random() < 0.5
                if use_memory:
                    samples += [memory_sample(*memory, i) for i in step_indices(mem_pools[goal], len(plan[goal]), seeds["old_samples"], step, f"mem-{goal}")]
                else:
                    for s_, i in zip(new_data.__getitems__(plan[goal]), plan[goal], strict=True):
                        record = {k: s_[k] for k in KEYS}
                        record.update(task=pseudo[1][goal], action=pseudo[0][goal][i].clone(),
                                      action_is_pad=torch.zeros_like(record["action_is_pad"]))
                        samples.append(record)
            elif memory is not None:
                samples += [memory_sample(*memory, i) for i in plan[goal]]
            else:
                samples += [{k: s[k] for k in KEYS} for s in old_data.__getitems__(plan[goal])]
            old_noise, old_times = flow_inputs(len(plan[goal]), seeds["old_samples"], step, f"{goal}-flow", chunk, max_dim)
            noise, times = torch.cat([noise, old_noise]), torch.cat([times, old_times])
        raw = torch.utils.data.default_collate(samples)
        batch = pre(copy.deepcopy(raw)); policy.train(); optimizer.zero_grad(set_to_none=True)
        tick = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            per_record, _ = policy(batch, noise=noise.cuda(), time=times.cuda(), reduction="none")
            loss = (per_record * weights).sum()
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, policy.config.optimizer_grad_clip_norm, error_if_nonfinite=True)
        assert norm.item() > 0
        used_lr = optimizer.param_groups[0]["lr"]
        optimizer.step(); scheduler.step(); last_step = step + 1
        losses = per_record.detach().float().cpu().tolist()
        row = {"step": last_step, "loss": loss.item(), "new_record_losses": losses[:spec["new"]],
               "old_record_losses": losses[spec["new"]:], "gradient_norm": norm.item(), "lr": used_lr,
               "step_seconds": time.monotonic() - tick, "indices": plan,
               "new_flow_checksum": [float(noise[:spec["new"]].double().sum()), [round(float(x), 7) for x in times[:spec["new"]]]],
               "exposures": exposures(args.branch, last_step)}
        with (run / "train_steps.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if last_step <= 3 or last_step % t["log_every"] == 0:
            print(json.dumps({k: row[k] for k in ("step", "loss", "new_record_losses", "old_record_losses", "gradient_norm", "lr")}), flush=True)
        stop_file = args.stop_file or run / "STOP_REQUESTED.json"
        stop = stop_file.exists() or time.monotonic() - started > t["walltime_hours"] * 3600
        if last_step % t["save_every"] == 0 or last_step == maximum or stop:
            checkpoint(last_step, raw)
        if stop:
            break
    dump(run / f"training_finished_{last_step:06d}.json", {
        "status": "complete", "branch": args.branch, "step": last_step, "target": maximum,
        "seconds": time.monotonic() - started, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "frozen_parameters_unchanged": frozen_hash(policy) == original_frozen,
        "exposures": exposures(args.branch, last_step)})
    print("TRAINING_PROCESS_COMPLETE", last_step, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/stage2a_replay_v1.json")
    p.add_argument("--branch", choices=sorted(BRANCHES), required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path, help="Resume this run including optimizer, scheduler and RNG")
    p.add_argument("--max-steps", type=int, help="Engineering smoke only; formal runs use the config endpoint")
    p.add_argument("--stop-file", type=Path)
    a = p.parse_args()
    train(a, json.loads(a.config.read_text()))
