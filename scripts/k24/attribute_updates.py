"""K24 matched AdamW branches and fixed-draft response attribution."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "k24_update_feasibility_20261117_01"))
from update_probe import K18, K20, dot, pairs, response, set_weights
from train_route import make_att_2d_masks, make_prefix, constants
from train_weak import ORIG, ROOT, balanced_indices, cross_kv, open_data, read, setup, write

SEED = 20261021
OFFSETS = (0, 17, 53, 89, 137, 211, 293, 433)
CHANNELS = (slice(0, 2), slice(2, 3), slice(3, 6), slice(6, 7))
METRICS = ("gain_h", "gain_all", "mu_x", "mu_y", "mu_z", "mu_rotation_rms",
           "mu_h_rms", "mu_z_rms", "mu_grip_rms", "off_h_rms", "off_all_rms",
           "left_progress", "left_margin", "right_progress", "right_margin")
FLOOR_TOL = .0001


def prefix(policy, pair):
    a, b = pair
    batch = {k: torch.cat((v, b["batch"][k])).cuda() for k, v in a["batch"].items()}
    images, masks = policy.prepare_images(batch)
    return make_prefix(policy, (images, masks, batch[constants.OBS_LANGUAGE_TOKENS],
                               batch[constants.OBS_LANGUAGE_ATTENTION_MASK], policy.prepare_state(batch)),
                       make_att_2d_masks)[:2]


def field(policy, pref, draft, t):
    pad, cache = pref
    return policy.model.denoise_step(pad, copy.deepcopy(cache), draft.clone(), t)


@torch.no_grad()
def reference(policy, bank):
    policy.eval()
    result = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, pair in bank.items():
            assert torch.equal(pair[0]["noise"], pair[1]["noise"])
            pref = prefix(policy, pair)
            draft = pair[0]["noise"].cuda().expand(2, -1, -1).contiguous()
            for k in range(4):
                t = draft.new_full((2,), 1. - .1 * k)
                draft = draft + (-.1) * field(policy, pref, draft, t)
            draft = draft[:1].expand(2, -1, -1).contiguous()
            t = draft.new_full((2,), .6)
            old = field(policy, pref, draft, t)[:, :10, :7].float()
            result[i] = draft, t, old.mean(0), (old[0] - old[1]) * .5
    return result


def fixed(policy, pair, ref):
    draft, t, old_mu, axis = ref
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pad, cache = prefix(policy, pair)
        # This fresh differentiable cache is consumed once by the fixed-draft query.
        v = policy.model.denoise_step(pad, cache, draft.clone(), t)[:, :10, :7].float()
    mu, d = v.mean(0), (v[0] - v[1]) * .5
    drift = mu - old_mu
    gains, off = [], []
    for dimensions in (CHANNELS[0], slice(0, 7)):
        x, y = d[:, dimensions], axis[:, dimensions]
        gain = (x * y).sum() / y.square().sum().clamp_min(1e-12)
        gains.append(gain)
        off.append((x - gain * y).square().mean().clamp_min(1e-14).sqrt())
    rms = [drift[:, c].square().mean().clamp_min(1e-14).sqrt() for c in CHANNELS]
    return torch.stack([*gains, *drift[:, :3].mean(0).unbind(), rms[2], rms[0], rms[1], rms[3], *off])


def batch_loss(policy, batch, mode, noise_rng):
    torch.set_rng_state(noise_rng[0])
    torch.cuda.set_rng_state_all(noise_rng[1])
    original = policy.model.forward
    if mode != "full":
        keep = torch.ones(7, device="cuda", dtype=torch.bool)
        if mode == "rotation":
            keep[:3] = False
            keep[6] = False
        elif mode == "without_rotation":
            keep[3:6] = False
        else:
            keep[[0, 2, 6]] = False

        def masked(*a, **kw):
            raw = original(*a, **kw)
            return torch.cat((raw[..., :7] * keep, raw[..., 7:]), -1)

        policy.model.forward = masked
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return policy(batch, reduction="none")[0].mean()
    finally:
        policy.model.forward = original


def proposed(policy, params, initial, history, grad, mode):
    set_weights(params, initial)
    opt = torch.optim.AdamW(params, lr=history["param_groups"][0]["lr"], betas=policy.config.optimizer_betas,
                            eps=policy.config.optimizer_eps, weight_decay=policy.config.optimizer_weight_decay)
    if mode != "reset":
        opt.load_state_dict(copy.deepcopy(history))
    for p, g in zip(params, grad, strict=True):
        p.grad = torch.zeros_like(g) if mode == "history_only" else g
    opt.step()
    opt.zero_grad(set_to_none=True)
    return [p.detach().float() - w.float() for p, w in zip(params, initial, strict=True)]


def select(delta, names, match, invert=False):
    return [v if match(n) != invert else torch.zeros_like(v) for n, v in zip(names, delta, strict=True)]


def norm(delta):
    return float(dot(delta, delta).sqrt())


def active_projection(delta, gradients, gaps, allowed=None):
    if allowed is not None:
        gradients = [[g if yes else torch.zeros_like(g) for g, yes in zip(row, allowed, strict=True)]
                     for row in gradients]
    gram = torch.stack([torch.stack([dot(g, h) for h in gradients]) for g in gradients])
    dual = torch.zeros(len(gaps), device="cuda")
    for _ in range(24):
        for j in range(len(dual)):
            if gram[j, j] > 1e-18:
                dual[j] = (dual[j] + (gaps[j] - gram[j] @ dual) / gram[j, j]).clamp_min(0)
    return [u + sum(w * g[k] for w, g in zip(dual, gradients, strict=True))
            for k, u in enumerate(delta)]


def run(anchor, export=False):
    torch.set_num_threads(2)
    policy, pre = setup(SEED)
    bank = pairs()
    ids = list(bank)
    refs = reference(policy, bank)
    config = read(K20 / "config.json")
    dataset = open_data(ROOT / config["data"]["train_output"])
    groups = {"front": list(range(len(dataset)))}
    checkpoint = K18 / "train/checkpoints" / f"{anchor:06d}"
    policy.load_state_dict(load_file(str(checkpoint / "model.safetensors")), strict=True)
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    names, params = zip(*[(n, p) for n, p in policy.named_parameters() if p.requires_grad], strict=True)
    assert list(names) == read(ORIG / "trainable_parameters.json")["names"]
    initial = [p.detach().clone() for p in params]
    history = state["optimizer"]
    hist = [json.loads(l) for l in (ORIG / "train_steps.jsonl").read_text().splitlines()]
    affine = read(K20 / "action_affine_0.json")
    offset, scale = [torch.tensor(affine[k][:2], device="cuda") for k in ("offset", "scale")]
    coef = torch.tensor(read(K20 / "calibration.json")["report"]["coef"], device="cuda")

    def block(n):
        return n.split(".layers.")[1].split(".")[0] if ".lm_expert.layers." in n else ""

    def mlp(n):
        return bool(block(n)) and ".mlp." in n

    def cross(n):
        return block(n) in {"1", "3", "5", "7", "9", "11", "13", "15"} and ".self_attn." in n

    def mlp47(n):
        return mlp(n) and block(n) in {"4", "5", "6", "7"}

    def both(n):
        return cross_kv(n) or mlp47(n)

    masks = {"only_cross": cross, "only_mlp": mlp, "only_kv": cross_kv,
             "without_kv": cross_kv, "without_mlp47": mlp47, "without_both": both}
    assert sum(map(cross_kv, names)) == 16 and all(any(map(m, names)) for m in masks.values())

    def host(delta):
        return [v.cpu() for v in delta]

    def measure(i):
        return torch.cat([fixed(policy, bank[i], refs[i]),
                          response(policy, bank[i], offset, scale, coef)[4:]])

    tag = "export" if export else "attribution"
    offsets = OFFSETS[:1] if export else OFFSETS
    with (HERE / f"{tag}_{anchor}.jsonl").open("x", encoding="utf-8") as handle:
        for k, skip in enumerate(offsets):
            set_weights(params, initial)
            batch_ids = balanced_indices(groups, 6, SEED, anchor + skip)
            batch = pre(copy.deepcopy(torch.utils.data.default_collate(dataset.__getitems__(batch_ids))))
            if skip == 0:
                rng = state["torch_rng"], state["cuda_rng"]
            else:
                torch.manual_seed(SEED + anchor * 1000 + skip)
                rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            policy.train()
            proposals = {}
            for mode in ("full", "rotation", "without_rotation", "drop_three_nonrotation"):
                set_weights(params, initial)
                policy.zero_grad(set_to_none=True)
                loss = batch_loss(policy, batch, mode, rng)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, policy.config.optimizer_grad_clip_norm,
                                                    error_if_nonfinite=True)
                g = [p.grad.detach().clone() for p in params]
                if mode == "full":
                    base_loss = float(loss.detach())
                    if skip == 0:
                        assert batch_ids == hist[anchor]["batch_indices"]
                        assert abs(base_loss - hist[anchor]["loss"]) < 1e-6
                        assert abs(float(gn) - hist[anchor]["gradient_norm"]) < 1e-6
                    for name in ("full", "reset", "history_only"):
                        proposals[name] = proposed(policy, params, initial, history, g, name)
                else:
                    proposals[mode] = proposed(policy, params, initial, history, g, "full")
            set_weights(params, initial)
            policy.eval()
            full = proposals["full"]
            values = {m: host(select(full, names, fn, invert=m.startswith("without"))) for m, fn in masks.items()}
            values.update({k:host(v) for k,v in proposals.items()})
            for key in ("reset", "history_only", "rotation", "without_rotation", "drop_three_nonrotation"):
                v = proposals[key]
                factor = norm(full) / max(norm(v), 1e-15)
                values[key + "_matched"] = [(x * factor).cpu() for x in v]
            del proposals, v, g
            applied = (ids[(2 * k + 1) % 12],)
            held = (ids[12 + k % 8], ids[12 + (k + 3) % 8])
            query_ids = applied + held
            with torch.no_grad():
                base = {i: measure(i) for i in query_ids}
            # Predict fixed-draft components for every tested state. Route derivatives are
            # taken only for the control query to keep the held observations out of selection.
            jac = {}
            keys = (0, 2, 3, 10, 11, 12, 13, 14)
            for i in query_ids:
                out = measure(i) if i in applied else fixed(policy, bank[i], refs[i])
                ix = keys if i in applied else keys[:4]
                jac[i] = {}
                for j, index in enumerate(ix):
                    grads = torch.autograd.grad(out[index], params, retain_graph=j + 1 < len(ix),
                                                allow_unused=True)
                    jac[i][index] = [g.detach().float() if g is not None else torch.zeros_like(p, dtype=torch.float32)
                                     for g, p in zip(grads, params, strict=True)]
            i = applied[0]
            floor = torch.minimum(base[i][-4:], base[i].new_tensor(
                bank[i][0]["floor"] + bank[i][1]["floor"]))
            vg = [jac[i][j] for j in keys[-4:]]
            gap = floor - base[i][-4:] - torch.stack([dot(g, full) for g in vg])
            corrected = active_projection(full, vg, gap)
            focused = active_projection(full, vg, gap, list(map(both, names)))
            for name, direction in (("project", corrected), ("focused", focused), ("scale", full)):
                for factor in (1., .5, .25, .125):
                    values[f"{name}_{factor:g}"] = [(factor * x).cpu() for x in direction]
            del full, corrected, focused, direction, vg
            arms = {}
            for name, delta in values.items():
                set_weights(params, initial, [x.cuda() for x in delta])
                realized = [p.detach().float() - w.float() for p, w in zip(params, initial, strict=True)]
                with torch.no_grad():
                    measurements = {str(i): {"before": base[i].tolist(),
                                             "delta": (measure(i) - base[i]).tolist(),
                                             "prediction": {METRICS[j]: float(dot(g, realized)) for j, g in jac[i].items()}}
                                    for i in query_ids}
                    policy.train()
                    front = float(batch_loss(policy, batch, "full", rng))
                    policy.eval()
                arms[name] = {"norm": norm(realized), "front_delta": front - base_loss,
                              "measurements": measurements}
            picks = {}
            for name in ("project", "focused", "scale"):
                choices = [f"{name}_{a:g}" for a in (1., .5, .25, .125)]
                picks[name] = next((key for key in choices if all(
                    b + d >= f - FLOOR_TOL for b, d, f in zip(
                        base[i][-4:].tolist(), arms[key]["measurements"][str(i)]["delta"][-4:],
                        floor.tolist(), strict=True))), None)
            if export:
                for name, key in {"anchor": None, "full": "full", **picks}.items():
                    set_weights(params, initial, None if key is None else [x.cuda() for x in values[key]])
                    save_file({n: p.detach().cpu().contiguous() for n, p in zip(names, params, strict=True)},
                              str(HERE / f"weights_{anchor}_{name}.safetensors"))
            row = {"anchor": anchor, "offset": skip, "front_ids": batch_ids, "metrics": METRICS,
                   "controlled": i, "held": held, "floor": floor.tolist(),
                   "thresholds": {str(j): torch.minimum(base[j][-4:], base[j].new_tensor(
                       bank[j][0]["floor"] + bank[j][1]["floor"])).tolist() for j in query_ids},
                   "arms": arms, "picks": picks}
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            print("K24 attribution", anchor, skip, picks, flush=True)
            del jac, values, realized, grads, out
    write(HERE / f"{tag}_{anchor}_COMPLETE.json",
          {"anchor": anchor, "offsets": offsets, "modules": {k: [n for n in names if f(n)] for k, f in masks.items()}})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("anchor", type=int, choices=(30, 100))
    p.add_argument("--export", action="store_true")
    a = p.parse_args()
    run(a.anchor, a.export)
