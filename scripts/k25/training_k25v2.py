"""Test whether one real Front step sends the rotation loss through V and up."""

import argparse
import json
from pathlib import Path
import sys

import torch

from closed_k25v2 import Lab5, RUN, SingletonUp, assign, read, write
from path_factorization import parameters_for
from value_relay import Relay

K18 = RUN.parent / "k18_training_cause_20261111_01"
sys.path.insert(0, str(RUN.parent / "k18_parameter_groups_20261111_01"))
sys.path.insert(0, str(K18))
from attribute import (OFFSETS, PROBE_INITS, ROOT, LeRobotDataset, balanced_indices, capture_rng,
                       fields, masks_for, restore_rng)
from measurement import checkpoint, load, make_bank
from rotation_bridge import compare, real_step


def random_removal(keep, seed):
    # Match the rotation mask's three deleted action channels at each action position.
    rng = torch.Generator(device="cpu").manual_seed(seed)
    columns = torch.rand(keep.shape, generator=rng).argsort(-1)[..., :3].to(keep.device)
    return torch.ones_like(keep).scatter_(-1, columns, False)


def score(values, parent_d):
    d = values[:, :10, :2].flatten(1)
    axis = parent_d[:, :10, :2].flatten(1)
    return ((d * axis).sum(1) / axis.square().sum(1).clamp_min(1e-12)).tolist()


def merged(base, donor, selected):
    return {n: donor[n] if n in selected else w for n, w in base.items()}


def path_gains(lab, family, part, full, without, bank, parent_d):
    # Current weights are the no-rotation branch. Inject outputs of the full-step
    # projection at the same current features, as in the inference mechanism tests.
    rt = lab.rt
    relay = Relay(lab, full, without) if family == "v" else SingletonUp(lab, full, without)
    contrasts = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _, _, _, obs, x, t, _, _ in bank:
            pair = relay.prefixes(obs)
            if family == "v":
                values = relay.cache(pair)
                outputs = []
                for q in "AB":
                    with relay.edit(values, q, part, reverse=True):
                        outputs.append(rt.query("F", pair[q], x, t).float())
            else:
                outputs = [w.float() for w in relay.query(pair, x, t, part, reverse=True)]
            a, b = outputs
            contrasts.append((a[..., :7] - b[..., :7]) * .5)
    return score(torch.cat(contrasts, dim=0), parent_d)


def run(step):
    config = read(K18 / "config.json")
    path = ROOT / config["data"]["train_output"]
    conversion = read(path / "r0_conversion.json")
    phase = read(K18 / "front_phases.json")
    assert phase["split_sha256"] == conversion["spec"]["split_sha256"]
    dataset = LeRobotDataset("local/cl_binding_r0_train", root=path, video_backend="pyav",
                             delta_timestamps={"action": [i / config["data"]["fps"] for i in range(50)]})
    ends = []
    for episode in conversion["episodes"]:
        ends.extend([len(ends) + episode["frames"]] * episode["frames"])
    labels = torch.tensor(phase["labels"], device="cuda")
    assert len(dataset) == len(ends) == len(labels)
    groups = {"front": list(range(len(dataset)))}
    references = {r["offset"]: r for r in map(json.loads, (K18 / f"attribution_marginal_{step:06d}.jsonl").read_text().splitlines())}
    lab = Lab5()
    try:
        policy = lab.rt.models["F"]
        named = {n: p for n, p in policy.named_parameters() if p.requires_grad}
        params = list(named.values())
        bank = [x for x in make_bank(lab, PROBE_INITS) if x[1] == "A" and x[2] == 4]
        parent_d = torch.cat([(x[-2] - x[-1]) * .5 for x in bank], dim=0)[..., :7]
        ckpt = checkpoint(lab, step)
        load(lab, ckpt)
        initial = [p.detach().clone() for p in params]
        before = fields(lab, bank)
        saved = torch.load(ckpt / "training_state.pt", map_location="cpu", weights_only=False)
        assert saved["step"] == step and saved["config"] == config
        state, original = saved["optimizer"], (saved["torch_rng"], saved["cuda_rng"])
        v_names = set(parameters_for("cross_v", named)[1])
        up_names = set(parameters_for("mlp_up", named)[1])
        with (RUN / f"training_{step:06d}.jsonl").open("x", encoding="utf-8") as out:
            for offset in OFFSETS:
                if offset == 0:
                    restore_rng(original)
                else:
                    torch.manual_seed(config["seed"] + 1000 * step + offset)
                indices = balanced_indices(groups, 6, config["seed"], step + offset)
                assert indices == references[offset]["indices"]
                batch = lab.rt.pre["F"](torch.utils.data.default_collate(dataset.__getitems__(indices)))
                rng = capture_rng()
                _, masks = masks_for(batch, indices, labels, ends, "marginal")
                keep = ~masks["channel:R"]
                variants = {"full": None, "without:channel:R": keep,
                            **{f"random_{i}": random_removal(keep, 20261120 + 1000 * step + offset + 100 * i)
                               for i in (1, 2)}}
                response, metrics, weights = {}, {}, {}
                for name, mask in variants.items():
                    tensors, diag = real_step(policy, params, initial, state, config, batch, rng, mask)
                    y = fields(lab, bank)
                    response[name] = score(y[1], parent_d)
                    metrics[name] = diag
                    if name in ("full", "without:channel:R"):
                        weights[name] = dict(zip(named, tensors, strict=True))
                        ref = references[offset]["branches"][name]
                        assert all(abs(ref[k] - v) < 1e-6 for k, v in diag.items()), (step, offset, name)
                        reproduced = compare(parent_d, before, y, 50, slice(0, 7))
                        assert max(abs(a - b) for a, b in zip(reproduced, ref["gain_delta"], strict=True)) < 2e-6
                full, without = (weights[k] for k in ("full", "without:channel:R"))
                candidates = {"only_v": merged(without, full, v_names),
                              "only_up": merged(without, full, up_names),
                              "only_v_up": merged(without, full, v_names | up_names),
                              "remove_v_up": merged(full, without, v_names | up_names)}
                for name, w in candidates.items():
                    assign(lab, "F", w)
                    response[name] = score(fields(lab, bank)[1], parent_d)
                assign(lab, "F", without)
                for family in ("v", "up"):
                    for part in ("mean", "diff"):
                        name = f"{family}_{part}"
                        response[name] = path_gains(lab, family, part, full, without, bank, parent_d)
                row = {"step": step, "offset": offset, "indices": indices, "metrics": metrics,
                       "gain_h": response, "historical_training_step_reproduced": True}
                out.write(json.dumps(row, allow_nan=False) + "\n")
                out.flush()
                print("STEP", step, offset, flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN / f"training_{step:06d}_COMPLETE.json", {"step": step, "offsets": list(OFFSETS),
                                                      "v_tensors": sorted(v_names), "up_tensors": sorted(up_names)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=(30, 100), type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.step)
