"""Trace realized training updates to cross-attention query inputs."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from statistics import median
import sys

import torch

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parent / "k25v3_20261120_01"))
from v_attribution_v3 import (DEV, Lab5, assign, checkpoint_files, fixed_bank, handoff,
                             horizontal, make_bank, measure_episode, read, records, rms, write)
from path_factorization import CROSS, group
from window_eval import INSTR, WINDOW

GROUPS = ("action_time", "state", "self_attention", "mlp", "kv", "cross_other")
ALL = (1 << len(GROUPS)) - 1


def inventory(full, anchor):
    assert len(full) == 155 and full.keys() == anchor.keys()
    result = {key: [] for key in (*GROUPS, "action_output", "expert_norm")}
    for name in full:
        result[group(name)].append(name)
    assert all(torch.equal(full[n], anchor[n]) for n in result["expert_norm"])
    return result


def rolled(base, source, names):
    names = set(names)
    assert names <= base.keys()
    return {n: source[n] if n in names else w for n, w in base.items()}


class Queries:
    def __init__(self, lab):
        self.rt = lab.rt
        self.sites = {r: {i: self.rt.models[r].model.vlm_with_expert.lm_expert.layers[i].self_attn.q_proj
                          for i in CROSS} for r in "FR"}

    @contextmanager
    def hooks(self, role, injected=None):
        observed = {}

        def observer(layer):
            def hook(_, args):
                assert layer not in observed
                observed[layer] = args[0]
                if injected is not None:
                    value = injected[layer]
                    assert args[0].shape == value.shape and args[0].dtype == value.dtype
                    return (value,)
            return hook

        handles = [m.register_forward_pre_hook(observer(i)) for i, m in self.sites[role].items()]
        try:
            yield observed
        finally:
            for h in handles:
                h.remove()
        assert observed.keys() == self.sites[role].keys()

    def capture(self, role, prefix, x, t):
        with self.hooks(role) as h:
            value = self.rt.query(role, prefix, x, t)
        return value, h

    def replace(self, prefix, x, t, h):
        with self.hooks("F", h):
            return self.rt.query("F", prefix, x, t)

    def chunk(self, obs, q, seed, step):
        f, r = (self.rt.prefix(role, obs, INSTR[q]) for role in "FR")
        assert torch.equal(f[0], r[0]) and f[2] == r[2]

        def velocity(x, t):
            _, h = self.capture("R", r, x, t)
            return self.replace(f, x, t, h)

        return self.rt.integrate(velocity, self.rt.noise(seed, step), 10)[..., :7]


def same_role_gate(lab, sites, rec, full, anchor, groups):
    obs = handoff(lab, rec)
    start = len(rec["controls"])
    rt = lab.rt
    x, t = rt.noise(rec["seed"], start), torch.ones(1, device="cuda")
    assign(lab, "F", full)
    assign(lab, "R", rolled(full, anchor, groups["action_output"]))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        for q in "AB":
            f, r = (rt.prefix(role, obs, INSTR[q]) for role in "FR")
            expected, hf = sites.capture("F", f, x, t)
            _, hr = sites.capture("R", r, x, t)
            assert all(torch.equal(hf[i], hr[i]) for i in CROSS)
            candidate = sites.replace(f, x, t, hr)
            assert torch.equal(candidate, expected), (q, rms(candidate-expected))
    return {"init": rec["init"], "same_query_is_native_bitwise": True, "action_output_cannot_change_fixed_query": True}


def to_tensor(h):
    return torch.stack([h[i][:, :10].float() for i in CROSS])


def d_h(a, b):
    return (a[:, :10, :2].float() - b[:, :10, :2].float()) * .5


def drift(h):
    return (h[0] + h[1]) * .5, (h[0] - h[1]) * .5


def ratio(move, direction):
    return float((move * direction).sum() / direction.square().sum().clamp_min(1e-12))


class Fixed:
    def __init__(self, lab, bank, full, anchor):
        self.lab, self.bank = lab, bank
        self.rt = lab.rt
        self.sites = Queries(lab)
        assign(lab, "F", full)
        assign(lab, "R", anchor)
        self.references = []
        last = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i, route, step, obs, x, t, a0, b0 in bank:
                if i != last:
                    f, r = ({q: self.rt.prefix(role, obs, INSTR[q]) for q in "AB"} for role in "FR")
                    last = i
                native = [self.sites.capture("F", f[q], x, t) for q in "AB"]
                early = [self.sites.capture("R", r[q], x, t) for q in "AB"]
                h_f = torch.stack([to_tensor(h) for _, h in native])
                direction = torch.stack([to_tensor(h) for _, h in early]) - h_f
                axis = d_h(a0, b0)
                base_gain = horizontal(d_h(native[0][0], native[1][0]), axis)
                injected = [self.sites.replace(f[q], x, t, example[1]) for q, example in zip("AB", early, strict=True)]
                anchor_gain = horizontal(d_h(*injected), axis)
                self.references.append((f, h_f, direction, axis, base_gain, anchor_gain))

    def measure(self, candidates):
        assign(self.lab, "R", candidates)
        last = None
        rows = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for (i, route, step, obs, x, t, a0, b0), reference in zip(self.bank, self.references, strict=True):
                f, h_f, direction, axis, base_gain, anchor_gain = reference
                if i != last:
                    r = {q: self.rt.prefix("R", obs, INSTR[q]) for q in "AB"}
                    assert all(torch.equal(f[q][0], r[q][0]) for q in "AB")
                    last = i
                candidate = [self.sites.capture("R", r[q], x, t) for q in "AB"]
                move = torch.stack([to_tensor(h) for _, h in candidate]) - h_f
                injected = [self.sites.replace(f[q], x, t, example[1]) for q, example in zip("AB", candidate, strict=True)]
                qgain = horizontal(d_h(*injected), axis)
                ngain = horizontal(d_h(candidate[0][0], candidate[1][0]), axis)
                split = zip(drift(move), drift(direction), strict=True)
                layer_votes = ((move * direction).sum(dim=(0, 2, 3, 4)) /
                               direction.square().sum(dim=(0, 2, 3, 4)).clamp_min(1e-12))
                rows.append({"init": i, "route": route, "step": step,
                             "query_projection": ratio(move, direction),
                             **{key: ratio(delta, axis) for key, (delta, axis) in
                                zip(("query_mean_projection", "query_diff_projection"), split, strict=True)},
                             "by_layer": layer_votes.tolist(), "q_gain": qgain, "native_gain": ngain,
                             "q_gain_change": qgain-base_gain, "anchor_q_change": anchor_gain-base_gain,
                             "q_mu_h_to_r1": rms((injected[0][:, :10, :2].float() + injected[1][:, :10, :2].float()
                                                    - a0[:, :10, :2].float() - b0[:, :10, :2].float()) * .5)})
        return rows


def coarse(lane, lanes):
    full, anchor = checkpoint_files()
    groups = inventory(full, anchor)
    if lane == 0:
        write(RUN / "inventory.json", groups)
    lab = Lab5()
    try:
        bank = make_bank(lab, DEV)
        fixed = Fixed(lab, bank, full, anchor)
        with (RUN / f"coarse_{lane}.jsonl").open("x", encoding="utf-8") as out:
            for mask in range(lane, ALL+1, lanes):
                names = [n for j, g in enumerate(GROUPS) if mask & 1 << j for n in groups[g]]
                data = fixed.measure(rolled(full, anchor, names))
                if mask in (0, ALL):
                    for row in data:
                        expected = 0. if mask == 0 else 1.
                        assert abs(row["query_projection"] - expected) < 1e-6
                        assert abs(row["q_gain_change"] - expected*row["anchor_q_change"]) < 1e-6
                out.write(json.dumps({"mask": mask, "rows": data}, allow_nan=False) + "\n")
                out.flush()
                print("COARSE", lane, mask, median(x["q_gain_change"] for x in data), flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN / f"coarse_{lane}_COMPLETE.json", {"lane": lane, "lanes": lanes})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lane", type=int, required=True)
    p.add_argument("--lanes", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(2)
    coarse(a.lane, a.lanes)
