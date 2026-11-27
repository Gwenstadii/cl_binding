"""Predict and intervene on realized AdamW updates in R1 → Front."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys

import torch
from safetensors.torch import load_file

RUN = Path(__file__).resolve().parent
K20 = RUN.parent / "k20_route_margin_20261112_01"
K18 = RUN.parent / "k18_training_cause_20261111_01"
sys.path.insert(0, str(K20))
from train_route import generated
from train_weak import ORIG, ROOT, balanced_indices, open_data, read, setup, write


def dot(a, b):
    return torch.stack([(x * y).sum() for x, y in zip(a, b, strict=True)]).sum()


def response(policy, pair, offset, scale, coef):
    a, b = pair
    batch = {k: torch.cat((v, b["batch"][k])).cuda() for k, v in a["batch"].items()}
    assert torch.equal(a["noise"], b["noise"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actions = generated(policy, batch, a["noise"].cuda().expand(2, -1, -1).contiguous())
    move = (actions[..., :2] * scale + offset).clamp(-1, 1).sum(1) @ coef
    pos = move.new_tensor(a["pos"])
    centers = move.new_tensor(a["centers"])
    progress = (centers - pos).norm(dim=1) - (centers - pos - move[:, None]).norm(dim=2)
    targets = [a["target"], b["target"]]
    scores = []
    for j, target in enumerate(targets):
        other = progress[j, [i for i in range(4) if i != target]].max()
        scores.extend([progress[j, target], progress[j, target] - other])
    mu, d = move.mean(0), (move[0] - move[1]) * .5
    return torch.cat([mu, d, torch.stack(scores)])


def pairs():
    bank = torch.load(K20 / "route_bank.pt", weights_only=True, map_location="cpu")
    by = {(r["init"], r["q"]): r for r in bank}
    return {i: (by[i, "A"], by[i, "B"]) for i in sorted({r["init"] for r in bank})}


def floor_for(value, pair, device):
    floor = torch.tensor(pair[0]["floor"] + pair[1]["floor"], device=device)
    # Prevent further loss at an already infeasible query; otherwise preserve the margin floor.
    return torch.minimum(value[4:].detach(), floor)


def differentials(value, params):
    result = []
    for i, v in enumerate(value):
        grad = torch.autograd.grad(v, params, retain_graph=i + 1 < len(value), allow_unused=True)
        result.append(tuple(g.detach().float() if g is not None else torch.zeros_like(p, dtype=torch.float32)
                            for g, p in zip(grad, params, strict=True)))
    return result


def project(delta, gradients, pre, floor):
    # Minimum-change half-space projection in the span of the four score gradients.
    gram = torch.stack([torch.stack([dot(g, h) for h in gradients]) for g in gradients])
    lin = torch.stack([dot(g, delta) for g in gradients])
    dual = torch.zeros_like(lin)
    for _ in range(32):
        for i in range(len(gradients)):
            if gram[i, i] > 1e-18:
                dual[i] = (dual[i] + (floor[i] - pre[i] - lin[i] - gram[i] @ dual) / gram[i, i]).clamp_min(0)
    return [u + sum(w * g[k] for w, g in zip(dual, gradients, strict=True))
            for k, u in enumerate(delta)], dual.tolist()


@torch.no_grad()
def set_weights(params, initial, delta=None):
    for i, (p, w) in enumerate(zip(params, initial, strict=True)):
        p.copy_(w if delta is None else w.float() + delta[i])


def run(anchor, offsets, verify):
    torch.set_num_threads(2)
    seed = 20261021
    policy, pre = setup(seed)
    config = read(K20 / "config.json")
    ds = open_data(ROOT / config["data"]["train_output"])
    groups = {"front": list(range(len(ds)))}
    named = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    assert [n for n, _ in named] == read(ORIG / "trainable_parameters.json")["names"]
    params = [p for _, p in named]
    checkpoint = K18 / "train/checkpoints" / f"{anchor:06d}"
    policy.load_state_dict(load_file(str(checkpoint / "model.safetensors")), strict=True)
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    history = state["optimizer"]
    initial = [p.detach().clone() for p in params]
    historic = [json.loads(line) for line in (ORIG / "train_steps.jsonl").read_text().splitlines()]
    bank = pairs()
    ids = list(bank)
    transform = read(K20 / "action_affine_0.json")
    offset, scale = (torch.tensor(transform[x][:2], device="cuda") for x in ("offset", "scale"))
    coef = torch.tensor(read(K20 / "calibration.json")["report"]["coef"], device="cuda")
    selections = ids[:12]
    held = (ids[12], ids[14], ids[16], ids[18])
    fields = ["mu_x", "mu_y", "d_x", "d_y", "left_progress", "left_margin", "right_progress", "right_margin"]
    tag = "verified" if verify else "anchor"
    with (RUN / f"{tag}_{anchor:06d}.jsonl").open("x", encoding="utf-8") as handle:
        for index, batch_offset in enumerate(offsets):
            set_weights(params, initial)
            batch_ids = balanced_indices(groups, 6, seed, anchor + batch_offset)
            batch = pre(copy.deepcopy(torch.utils.data.default_collate(ds.__getitems__(batch_ids))))
            if batch_offset == 0:
                torch.set_rng_state(state["torch_rng"])
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            else:
                torch.manual_seed(seed + anchor * 1000 + batch_offset)
            torch_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            policy.train()
            opt = torch.optim.AdamW(params, lr=history["param_groups"][0]["lr"])
            opt.load_state_dict(copy.deepcopy(history))
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = policy(batch, reduction="none")[0].mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, policy.config.optimizer_grad_clip_norm,
                                                  error_if_nonfinite=True)
            if batch_offset == 0:
                ref = historic[anchor]
                assert ref["batch_indices"] == batch_ids and abs(float(loss.detach()) - ref["loss"]) < 1e-6
                assert abs(float(norm) - ref["gradient_norm"]) < 1e-6
            opt.step()
            proposal = [p.detach().float() - w.float() for p, w in zip(params, initial, strict=True)]
            opt.zero_grad(set_to_none=True)
            del opt
            set_weights(params, initial)
            policy.eval()
            controlled = selections[(anchor + index * 3) % len(selections)]
            # One control query and two untouched queries with J·delta predictions.
            query_ids = (controlled, held[index % 4], held[(index + 1) % 4])
            values, grads, floors = {}, {}, {}
            for i in query_ids:
                out = response(policy, bank[i], offset, scale, coef)
                values[i] = out.detach()
                floors[i] = floor_for(out, bank[i], "cuda")
                grads[i] = differentials(out, params)
            corrected, dual = project(proposal, grads[controlled][4:], values[controlled][4:], floors[controlled])
            proposal_norm = float(dot(proposal, proposal).sqrt())
            correction_norm = float(dot(corrected, corrected).sqrt())
            ratio = correction_norm / max(proposal_norm, 1e-12)
            arms = {"actual": proposal, "projected": corrected,
                    "scaled": [u * ratio for u in proposal]}
            if verify:
                for label, direction in (("original", proposal), ("corrected", corrected)):
                    for fraction in (.5, .25, .125):
                        arms[f"{label}_{fraction:g}"] = [u * fraction for u in direction]
                arms["zero"] = [torch.zeros_like(u) for u in proposal]
            updates = {}
            for arm, delta in arms.items():
                set_weights(params, initial, delta)
                realized = [p.detach().float() - w.float() for p, w in zip(params, initial, strict=True)]
                after = {}
                with torch.no_grad():
                    for i in query_ids:
                        value = response(policy, bank[i], offset, scale, coef)
                        after[i] = {"before": values[i].tolist(), "floor": floors[i].tolist(),
                                    "predicted_delta": [float(dot(g, realized)) for g in grads[i]],
                                    "actual_delta": (value - values[i]).tolist()}
                    policy.train()
                    torch.set_rng_state(torch_state)
                    torch.cuda.set_rng_state_all(cuda_state)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        new_loss = float(policy(batch, reduction="none")[0].mean())
                    policy.eval()
                updates[arm] = {"norm": float(dot(realized, realized).sqrt()),
                                "front_loss_same_batch": new_loss, "queries": after}
            row = {"anchor": anchor, "offset": batch_offset, "controlled": controlled, "held": query_ids[1:],
                   "front_ids": batch_ids, "front_before": float(loss.detach()), "dual": dual,
                   "metric_order": fields, "updates": updates}
            if verify:
                def passes(arm):
                    q = updates[arm]["queries"][controlled]
                    return all(b + d >= f - .0001 for b, d, f in zip(
                        q["before"][4:], q["actual_delta"][4:], q["floor"], strict=True))

                choices = {
                    "project_with_check": ("projected", "corrected_0.5", "corrected_0.25", "corrected_0.125", "zero"),
                    "scale_with_check": ("actual", "original_0.5", "original_0.25", "original_0.125", "zero"),
                }
                row["accepted"] = {k: next(a for a in variants if passes(a)) for k, variants in choices.items()}
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            print("K24", anchor, batch_offset, "protected", controlled, "dual", dual, flush=True)
    write(RUN / f"{tag}_{anchor:06d}_COMPLETE.json", {"anchor": anchor, "offsets": offsets})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("anchor", type=int, choices=(30, 100))
    p.add_argument("--offsets", nargs="+", type=int, default=(0, 13, 41, 79, 127, 193, 271, 389))
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    run(args.anchor, args.offsets, args.verify)
