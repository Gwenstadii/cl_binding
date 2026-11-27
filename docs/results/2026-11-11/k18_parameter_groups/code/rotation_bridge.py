"""Trace the local marginal effect of Front rotation loss into a K18-C group."""

import argparse
import copy
import json
from pathlib import Path
import random

from parameter_groups import K18, RUN, ORIGINAL, Lab5, checkpoint, group_of, load, make_bank, read

import numpy as np
import torch
from attribute import (
    OFFSETS, PROBE_INITS, ROOT, balanced_indices, capture_rng, fields, forward,
    LeRobotDataset, load_policy, make_optimizer, masks_for, restore_rng,
)


def real_step(policy, params, initial, state, config, batch, rng, mask):
    with torch.no_grad():
        for p, w in zip(params, initial, strict=True):
            p.copy_(w)
    optimizer = make_optimizer(params, policy, config)
    optimizer.load_state_dict(copy.deepcopy(state))
    optimizer.zero_grad(set_to_none=True)
    loss = forward(policy, batch, rng, mask)
    value = float(loss.detach())
    loss.backward()
    del loss
    norm = torch.nn.utils.clip_grad_norm_(params, policy.config.optimizer_grad_clip_norm, error_if_nonfinite=True)
    lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    result = {"loss_before_on_mask": value, "preclip_norm": float(norm), "lr": lr}
    return [p.detach().clone() for p in params], result


def compare(old, base, value, n, cols):
    d0 = old[:, :n, cols].flatten(1)
    change = (value[1] - base[1])[:, :n, cols].flatten(1)
    return ((change * d0).sum(1) / d0.square().sum(1).clamp_min(1e-12)).tolist()


def run(group, steps):
    config = read(K18 / "config.json")
    path = ROOT / config["data"]["train_output"]
    conversion = read(path / "r0_conversion.json")
    phase = read(K18 / "front_phases.json")
    assert phase["split_sha256"] == conversion["spec"]["split_sha256"]
    dataset = LeRobotDataset("local/cl_binding_r0_train", root=path, video_backend="pyav",
                             delta_timestamps={"action": [i/config["data"]["fps"] for i in range(50)]})
    ends = []
    for item in conversion["episodes"]:
        ends.extend([len(ends) + item["frames"]] * item["frames"])
    labels = torch.tensor(phase["labels"], device="cuda")
    assert len(ends) == len(labels) == len(dataset)
    groups = {"front": list(range(len(dataset)))}
    lab = Lab5()
    policy = lab.rt.models["F"]
    named = [(n,p) for n,p in policy.named_parameters() if p.requires_grad]
    assert [n for n,_ in named] == read(ORIGINAL / "trainable_parameters.json")["names"]
    params = [p for _,p in named]
    selected = [group_of(n) == group for n,_ in named]
    bank = [x for x in make_bank(lab, PROBE_INITS) if x[1] == "A" and x[2] == 4]
    parent_d = torch.cat([(x[-2]-x[-1])/2 for x in bank],dim=0)[..., :7]
    try:
        for step in steps:
            reference = {r["offset"]: r for r in map(json.loads, (K18 / f"attribution_marginal_{step:06d}.jsonl").read_text().splitlines())}
            ckpt = checkpoint(lab, step)
            load(lab, ckpt)
            initial = [p.detach().clone() for p in params]
            base = fields(lab, bank)
            if step:
                saved = torch.load(ckpt / "training_state.pt", map_location="cpu", weights_only=False)
                assert saved["step"] == step and saved["config"] == config
                state, original_rng = saved["optimizer"], (saved["torch_rng"],saved["cuda_rng"])
            else:
                random.seed(config["seed"])
                np.random.seed(config["seed"])
                torch.manual_seed(config["seed"])
                temporary_policy = load_policy(ckpt)
                del temporary_policy
                original_rng = capture_rng()
                state = make_optimizer(params, policy, config).state_dict()
            with (RUN / f"rotation_bridge_{step:06d}.jsonl").open("x") as handle:
                for offset in OFFSETS:
                    if offset == 0:
                        restore_rng(original_rng)
                    else:
                        torch.manual_seed(config["seed"] + 1000*step + offset)
                    indices = balanced_indices(groups, 6, config["seed"], step+offset)
                    assert indices == reference[offset]["indices"]
                    batch = lab.rt.pre["F"](torch.utils.data.default_collate(dataset.__getitems__(indices)))
                    rng = capture_rng()
                    _, masks = masks_for(batch, indices, labels, ends, "marginal")
                    full, full_diag = real_step(policy, params, initial, state, config, batch, rng, None)
                    y_full = fields(lab, bank)
                    without, no_diag = real_step(policy, params, initial, state, config, batch, rng, ~masks["channel:R"])
                    y_without = fields(lab, bank)
                    for name, diag, y in (("full",full_diag,y_full), ("without:channel:R",no_diag,y_without)):
                        ref = reference[offset]["branches"][name]
                        assert all(abs(ref[k]-v)<1e-6 for k,v in diag.items()), (step,offset,name,diag)
                        delta = compare(parent_d,base,y,50,slice(0,7))
                        assert max(abs(a-b) for a,b in zip(delta,ref["gain_delta"],strict=True)) < 2e-6
                    response = {"full": y_full, "without_rotation": y_without}
                    for name, keep in (("add_group_to_without",True), ("remove_group_from_full",False)):
                        with torch.no_grad():
                            for p,f,w,g in zip(params,full,without,selected,strict=True):
                                p.copy_(f if g == keep else w)
                        response[name] = fields(lab,bank)
                    result = {"step":step,"offset":offset,"group":group,"indices":indices,
                              "gain_H":{k:compare(parent_d,base,v,10,slice(0,2)) for k,v in response.items()},
                              "gain_all":{k:compare(parent_d,base,v,50,slice(0,7)) for k,v in response.items()},
                              "reproduced_B_losses_norms_and_probe":True}
                    handle.write(json.dumps(result,allow_nan=False)+"\n")
                    handle.flush()
                    print("ROTATION_BRIDGE",step,offset,group,flush=True)
                    del full, without
            del initial, state
    finally:
        lab.ctx.env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", nargs="+", type=int, default=(0,30,300))
    a = p.parse_args()
    group = read(RUN / "selection_000030_000100.json")["top_group"]
    torch.set_num_threads(2)
    run(group,a.steps)
