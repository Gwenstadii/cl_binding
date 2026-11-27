"""Fork real AdamW steps from shared weights, optimizer state, and Front input."""
import argparse
import copy
import json
from pathlib import Path
import random

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset
from stage1_protocol import balanced_indices
from stage1_smoke import ROOT, load_policy

from measurement import RUN, ORIGINAL, Lab5, BLOCKS, INSTR, checkpoint, load, make_bank

OFFSETS = (0, 13, 41, 79, 127, 193, 271, 389)
PROBE_INITS = (20, 23, 26, 29, 32, 35)


def read(path):
    return json.loads(path.read_text())


def fields(lab, bank):
    result = []
    policy = lab.rt.models["F"]
    policy.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for init, route, index, obs, x, t, old_a, old_b in bank:
            a, b = [lab.rt.query("F", lab.rt.prefix("F", obs, INSTR[g]), x, t).float() for g in "AB"]
            result.append(torch.stack(((a + b) / 2, (a - b) / 2))[..., :7])
    return torch.cat(result, dim=1)


def restore_rng(state):
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state_all(state[1])


def capture_rng():
    return torch.get_rng_state(), torch.cuda.get_rng_state_all()


def make_optimizer(params, policy, config):
    spec = policy.config
    optimizer = torch.optim.AdamW(params, lr=config["training"]["learning_rate"], betas=spec.optimizer_betas,
                                  eps=spec.optimizer_eps, weight_decay=spec.optimizer_weight_decay)
    torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., (step + 1) / config["training"]["warmup_steps"]))
    return optimizer


def forward(policy, batch, rng, keep):
    restore_rng(rng)
    policy.train()
    original = policy.model.forward
    if keep is not None:
        def masked(*args, **kwargs):
            losses = original(*args, **kwargs)
            assert losses.shape[:2] == keep.shape[:2] and losses.shape[-1] == 32
            return torch.cat((losses[..., :7] * keep, losses[..., 7:]), dim=-1)
        policy.model.forward = masked
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return policy(copy.deepcopy(batch), reduction="none")[0].mean()
    finally:
        if keep is not None:
            policy.model.forward = original


def metric_delta(before, after, parent_d, late):
    gain = ((after[1] - before[1]) * parent_d).flatten(1).sum(1) / parent_d.square().flatten(1).sum(1).clamp_min(1e-12)
    result = {"gain_delta": gain.tolist(), "mu": {}}
    for block in "HZRG":
        ix = BLOCKS[block]
        delta = (after[0] - before[0])[..., ix].flatten(1)
        target = (late[0] - before[0])[..., ix].flatten(1)
        result["mu"][block] = {
            "rms_change": delta.square().mean(1).sqrt().tolist(),
            "projection_to_remaining_front12000_drift": ((delta * target).sum(1) / target.square().sum(1).clamp_min(1e-12)).tolist(),
        }
    return result


def masks_for(batch, indices, labels, episode_end, mode):
    valid = ~batch["action_is_pad"]
    times = torch.arange(valid.shape[1], device=valid.device)
    absolute = torch.tensor(indices, device=valid.device)[:, None] + times
    ends = torch.tensor([episode_end[i] for i in indices], device=valid.device)[:, None]
    assert bool((~valid | (absolute < ends)).all())
    phase = labels[absolute.clamp(max=len(labels) - 1)]
    masks = {}
    for block in "HZRG":
        channel = torch.zeros_like(valid[..., None].expand(-1, -1, 7))
        channel[..., BLOCKS[block]] = True
        if mode != "joint":
            masks["channel:" + block] = channel
        else:
            for pi, name in enumerate(("grasp_lift", "transport", "approach_place", "unknown")):
                masks[name + ":" + block] = channel & (phase == pi)[..., None]
    if mode == "marginal":
        for pi, name in enumerate(("grasp_lift", "transport", "approach_place", "unknown")):
            masks["phase:" + name] = (phase == pi)[..., None].expand(-1, -1, 7)
    return valid, masks


def main(steps, mode, offsets):
    config = read(RUN / "config.json")
    path = ROOT / config["data"]["train_output"]
    conversion = read(path / "r0_conversion.json")
    phase = read(RUN / "front_phases.json")
    assert phase["split_sha256"] == conversion["spec"]["split_sha256"] and phase["action_offset"] == 1
    dataset = LeRobotDataset("local/cl_binding_r0_train", root=path, video_backend="pyav",
                             delta_timestamps={"action": [i / config["data"]["fps"] for i in range(50)]})
    episode_end = []
    for demo in conversion["episodes"]:
        episode_end.extend([len(episode_end) + demo["frames"]] * demo["frames"])
    assert len(dataset) == len(episode_end) == len(phase["labels"])
    labels = torch.tensor(phase["labels"], device="cuda")
    groups = {"front": list(range(len(dataset)))}
    originals = {r["step"]: r for r in map(json.loads, (ORIGINAL / "train_steps.jsonl").read_text().splitlines())}
    lab = Lab5()
    policy = lab.rt.models["F"]
    named = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    assert [n for n, _ in named] == read(ORIGINAL / "trainable_parameters.json")["names"]
    bank = [x for x in make_bank(lab, PROBE_INITS) if x[1] == "A" and x[2] == 4]
    parent_d = torch.cat([(x[-2] - x[-1]) / 2 for x in bank], dim=0)[..., :7]
    late = fields(lab, bank)
    for step in steps:
        ckpt = checkpoint(lab, step)
        load(lab, ckpt)
        reference = [p.detach().clone() for p in params]
        base = fields(lab, bank)
        if step:
            saved = torch.load(ckpt / "training_state.pt", map_location="cpu", weights_only=False)
            assert saved["step"] == step and saved["config"] == config
            optimizer_state = saved["optimizer"]
            original_rng = saved["torch_rng"], saved["cuda_rng"]
        else:
            random.seed(config["seed"])
            np.random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            # Recreate the historical model-loading RNG consumption before the first batch.
            initial = load_policy(ckpt)
            del initial
            original_rng = capture_rng()
            optimizer_state = make_optimizer(params, policy, config).state_dict()
        with (RUN / f"attribution_{mode}_{step:06d}.jsonl").open("x") as handle:
            for offset in offsets:
                if offset == 0:
                    restore_rng(original_rng)
                else:
                    torch.manual_seed(config["seed"] + 1000 * step + offset)
                indices = balanced_indices(groups, 6, config["seed"], step + offset)
                raw = torch.utils.data.default_collate(dataset.__getitems__(indices))
                batch = lab.rt.pre["F"](raw)
                rng = capture_rng()
                valid, masks = masks_for(batch, indices, labels, episode_end, mode)
                branches = [("full", None), ("zero_gradient", None)]
                counts = {}
                for source, keep in masks.items():
                    counts[source] = int((keep & valid[..., None]).sum())
                    if counts[source]:
                        branches.extend([("only:" + source, keep), ("without:" + source, ~keep)])
                full_grad = None
                sums = {}
                loss_parts = {}
                rows = {}
                for branch, keep in branches:
                    with torch.no_grad():
                        for p, w in zip(params, reference, strict=True):
                            p.copy_(w)
                    optimizer = make_optimizer(params, policy, config)
                    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
                    optimizer.zero_grad(set_to_none=True)
                    if branch == "zero_gradient":
                        for p in params:
                            p.grad = torch.zeros_like(p)
                        loss_value = None
                    else:
                        loss = forward(policy, batch, rng, keep)
                        loss_value = float(loss.detach())
                        loss.backward()
                        del loss
                    if branch == "full":
                        assert all(p.grad is not None for p in params)
                        full_grad = [p.grad.detach().float().clone() for p in params]
                    elif branch.startswith("only:"):
                        family = branch.split(":")[1] if mode == "marginal" else "joint"
                        if family not in sums:
                            sums[family] = [torch.zeros_like(g) for g in full_grad]
                            loss_parts[family] = 0.
                        for target, p in zip(sums[family], params, strict=True):
                            if p.grad is not None:
                                target.add_(p.grad)
                        loss_parts[family] += loss_value
                    norm = torch.nn.utils.clip_grad_norm_(params, policy.config.optimizer_grad_clip_norm, error_if_nonfinite=True)
                    lr = optimizer.param_groups[0]["lr"]
                    optimizer.step()
                    with torch.no_grad():
                        update_norm = float(torch.stack([(p.float() - w.float()).square().sum() for p, w in zip(params, reference, strict=True)]).sum().sqrt())
                        after_loss = float(forward(policy, batch, rng, None))
                    rows[branch] = {"loss_before_on_mask": loss_value, "full_batch_loss_after": after_loss, "preclip_norm": float(norm),
                                    "lr": lr, "actual_update_l2": update_norm,
                                    **metric_delta(base, fields(lab, bank), parent_d, late)}
                    del optimizer
                total = torch.stack([g.square().sum() for g in full_grad]).sum()
                reconstruction = {family: float((torch.stack([(x-y).square().sum() for x, y in zip(full_grad, acc, strict=True)]).sum() / total).sqrt())
                                  for family, acc in sums.items()}
                errors = {k: abs(v - rows["full"]["loss_before_on_mask"]) for k, v in loss_parts.items()}
                assert all(e < 1e-5 for e in errors.values()), errors
                expected = originals[step + 1] if offset == 0 else None
                match = None if expected is None else {
                    "loss_delta": rows["full"]["loss_before_on_mask"] - expected["loss"],
                    "preclip_norm_delta": rows["full"]["preclip_norm"] - expected["gradient_norm"],
                    "ids_match": expected["batch_indices"] == indices,
                    "lr_match": rows["full"]["lr"] == expected["lr"]}
                result = {"step": step, "offset": offset, "indices": indices, "counts": counts, "mask_families": mode,
                          "loss_reconstruction_abs": errors, "preclip_gradient_reconstruction_rel_l2": reconstruction,
                          "historical_full_step_comparison": match, "branches": rows}
                handle.write(json.dumps(result, allow_nan=False) + "\n")
                handle.flush()
                print("ATTRIBUTION", mode, step, offset, reconstruction, match, flush=True)
                del full_grad, sums
        del reference
    lab.ctx.env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, nargs="+", required=True)
    p.add_argument("--mode", choices=("marginal", "joint"), default="marginal")
    p.add_argument("--offsets", nargs="+", type=int, default=OFFSETS)
    args = p.parse_args()
    main(args.steps, args.mode, args.offsets)
