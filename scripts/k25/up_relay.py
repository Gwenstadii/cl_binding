"""Test local up-content corrections through all gated MLPs on paired fixed drafts."""

from contextlib import contextmanager
import copy
from statistics import median

import torch
from torch.nn import functional as F

from path_factorization import RUN, Lab5, assign, cases, checkpoint_files, configuration, fixed_bank, read, write
from value_relay import Relay, rms


def stack_prefix(pair):
    left, right = (pair[q] for q in "AB")
    cache = copy.deepcopy(left[1])
    for to, li, ri in zip(cache.layers, left[1].layers, right[1].layers, strict=True):
        to.keys = torch.cat((li.keys, ri.keys), dim=0)
        to.values = torch.cat((li.values, ri.values), dim=0)
    return torch.cat((left[0], right[0]), dim=0), cache


class UpRelay:
    def __init__(self, lab, full, anchor):
        self.policy = lab.rt.models["F"].model
        self.modules = [b.mlp for b in self.policy.vlm_with_expert.lm_expert.layers]
        self.weights = []
        for i, m in enumerate(self.modules):
            assert all(p.bias is None for p in (m.gate_proj, m.up_proj, m.down_proj))
            key = f"model.vlm_with_expert.lm_expert.layers.{i}.mlp.up_proj.weight"
            self.weights.append(tuple(w[key].to(m.up_proj.weight.device) for w in (full, anchor)))

    @contextmanager
    def edit(self, mode, reverse):
        previous = [m.forward for m in self.modules]
        if mode != "identity":
            for m, weights in zip(self.modules, self.weights, strict=True):
                other = weights[0 if reverse else 1]

                def layer(x, *, m=m, other=other):
                    new = F.linear(x, other)
                    gate = m.act_fn(m.gate_proj(x))
                    if mode == "whole":
                        return m.down_proj(gate * new)
                    current = m.up_proj(x)
                    assert current.shape[0] == 2
                    base = m.down_proj(gate * current)
                    difference = new.float() - current.float()
                    shared = difference.mean(dim=0, keepdim=True)
                    selected = shared if mode == "mean" else difference - shared
                    with torch.autocast("cuda", enabled=False):
                        correction = F.linear(gate.float() * selected, m.down_proj.weight.float())
                    return (base.float() + correction).to(base.dtype)

                m.forward = layer
        try:
            yield
        finally:
            for m, f in zip(self.modules, previous, strict=True):
                m.forward = f

    def query(self, prefix, x, t, mode, reverse):
        pad, cache = prefix
        with self.edit(mode, reverse):
            result = self.policy.denoise_step(pad, cache, x.repeat(2, 1, 1), t.repeat(2))
        assert all(layer.keys.shape[-2] == pad.shape[1] for layer in cache.layers)
        return result.float()


def measure(value, parent):
    gap = (value[0, :10, :2] - value[1, :10, :2]) * .5
    return float((gap * parent).sum() / parent.square().sum().clamp_min(1e-12))


def run():
    assert read(RUN / "selected.json")["mlp"] == "up"
    full, anchor = checkpoint_files()
    healthy = "mlp_gate_down"
    adapted = "mlp_gate_up_down"
    plan = {"base_adapted": ("identity", adapted), "base_protected": ("identity", healthy),
            "repair_whole": ("whole", adapted), "repair_mean": ("mean", adapted),
            "repair_diff": ("diff", adapted), "reverse_mean": ("mean", healthy),
            "reverse_diff": ("diff", healthy)}
    lab = Lab5()
    up = UpRelay(lab, full, anchor)
    prefix_relay = Relay(lab, full, anchor)
    results = {a: [] for a in plan}
    batch_checks = {}
    references = {}
    try:
        bank = fixed_bank(lab, cases())
        for arm, (mode, base) in plan.items():
            assign(lab, "F", configuration(base, full, anchor))
            case = None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i, route, step, obs, x, t, old_a, old_b in bank:
                    if i != case:
                        case = i
                        pair = prefix_relay.prefixes(obs)
                        p = stack_prefix(pair)
                    axis = (old_a[0, :10, :2] - old_b[0, :10, :2]) * .5
                    value = up.query(p, x, t, mode, base == healthy)
                    g = measure(value, axis)
                    sent = (i, route, step)
                    if arm == "base_protected":
                        references[sent] = value
                    if arm == "repair_whole":
                        assert torch.equal(value, references[sent]), (sent, rms(value - references[sent]))
                    if arm in ("base_protected", "base_adapted"):
                        singleton = torch.cat([lab.rt.query("F", pair[q], x, t).float() for q in "AB"], 0)
                        batch_checks.setdefault(arm, []).append({
                            "init": i, "rmse_first10": rms((value - singleton)[:, :10, :7]),
                            "h_gain_abs_difference": abs(g - measure(singleton, axis))})
                    results[arm].append({"init": i, "draft": route, "step": step, "h_gain": g})
            print("UP_RELAY", arm, flush=True)
    finally:
        lab.ctx.env.close()
    for arm, rows in batch_checks.items():
        assert median(r["h_gain_abs_difference"] for r in rows) < .005, arm
    write(RUN / "up_relay.json", {"results": results, "batch_checks": batch_checks, "whole_bitwise": True})


if __name__ == "__main__":
    torch.set_num_threads(2)
    run()
