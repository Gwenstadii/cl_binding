"""Intervene on shared and prompt-dependent V corrections before cross-attention reads."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import torch
from torch.nn.functional import linear

from path_factorization import (CROSS, Lab5, assign, cases, checkpoint_files, configuration,
                                fixed_bank, measure_episode, read, write)
from window_eval import INSTR, WINDOW
from deep_selection import handoff

RUN = Path(__file__).resolve().parent
REPAIR = {"repair_whole": ("whole", "AB"), "repair_mean": ("mean", "AB"),
          "repair_diff": ("diff", "AB"), "repair_flip": ("flip", "B")}
REVERSE = {"reverse_mean": ("mean", "B"), "reverse_diff": ("diff", "B")}
PLAN = {**REPAIR, **REVERSE}


def rms(value):
    return float(value.float().square().mean().sqrt())


def payload_input(prefix, layer, projection="v"):
    cache = prefix[1].layers[layer]
    values = cache.values if projection == "v" else cache.keys
    return values.transpose(1, 2).reshape(values.shape[0], values.shape[2], -1)


def in_gqa(expert, x):
    batch, tokens, n_kv, dim = x.shape
    groups = expert.num_attention_heads // n_kv
    return x[:, :, :, None, :].expand(batch, tokens, n_kv, groups, dim).reshape(
        batch, tokens, n_kv * groups, dim).transpose(1, 2)


def probabilities(expert, mask, q, k, dtype):
    score = q.float().transpose(1, 2) @ in_gqa(expert, k).float().transpose(-1, -2)
    score *= q.shape[-1] ** -.5
    score = score.float()
    return torch.where(mask[:, None], score, torch.finfo(score.dtype).min).softmax(-1).to(dtype)


class Relay:
    def __init__(self, lab, full, anchor):
        self.lab = lab
        expert = lab.rt.models["F"].model.vlm_with_expert
        self.expert = expert
        self.layers = {i: expert.lm_expert.layers[i].self_attn.v_proj for i in CROSS}
        assert len(expert.lm_expert.layers) == 16 and expert.self_attn_every_n_layers == 2
        self.weights = {}
        for i, mod in self.layers.items():
            assert mod.bias is None
            key = f"model.vlm_with_expert.lm_expert.layers.{i}.self_attn.v_proj.weight"
            self.weights[i] = tuple(w[key].to(device=mod.weight.device) for w in (full, anchor))

    def prefixes(self, obs):
        rt = self.lab.rt
        pair = {q: rt.prefix("F", obs, INSTR[q]) for q in "AB"}
        assert torch.equal(pair["A"][0], pair["B"][0]) and pair["A"][2] == pair["B"][2]
        return pair

    def cache(self, pair):
        result = {}
        for layer, (late, early) in self.weights.items():
            result[layer] = {}
            for q, p in pair.items():
                raw = payload_input(p, layer).to(late.dtype)
                result[layer][q] = linear(raw, late), linear(raw, early)
        return result

    @staticmethod
    def factor(parts):
        change = {q: (v[1].float() - v[0].float()) for q, v in parts.items()}
        mean, diff = (change["A"] + change["B"]) * .5, (change["A"] - change["B"]) * .5
        return mean, diff

    @contextmanager
    def whole(self, projections, q, reverse, verify=False):
        handles = []
        visited = set()
        anchor_index = int(reverse)
        try:
            for layer, mod in self.layers.items():
                source = projections[layer][q][anchor_index]
                replacement = projections[layer][q][1 - anchor_index]

                def hook(_mod, _inputs, out, *, layer=layer, source=source, replacement=replacement):
                    if verify:
                        assert layer not in visited
                        if not torch.equal(out, source):
                            calculated = linear(_inputs[0], _mod.weight.detach())
                            raise AssertionError((layer, out.dtype, _mod.weight.dtype,
                                                  rms(out - source), rms(out - calculated),
                                                  rms(calculated - source)))
                        visited.add(layer)
                    return replacement

                handles.append(mod.register_forward_hook(hook))
            yield
            if verify:
                assert visited == set(self.layers)
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def edit(self, projections, q, mode, reverse, verify=False):
        if mode == "whole":
            with self.whole(projections, q, reverse, verify):
                yield
            return

        # Accumulate PΔV in FP32 at the weighted sum, before the output projection.
        # Rounding a shared V correction independently in two BF16 prefixes
        # otherwise introduces an unintended contrast at the V site.
        expert = self.expert
        original_cross, original_eager = expert.forward_cross_attn_layer, expert.eager_attention_forward
        active = set()
        site = None
        sign = 1 if q == "A" else -1
        scaled = {}
        for layer, values in projections.items():
            shared, differential = self.factor(values)
            choose = {"mean": shared, "diff": sign * differential, "flip": -sign * differential}
            scaled[layer] = -choose[mode] if reverse else choose[mode]

        def cross(model_layers, inputs_embeds, layer_idx, *args, **kwargs):
            nonlocal site
            site = layer_idx
            try:
                return original_cross(model_layers, inputs_embeds, layer_idx, *args, **kwargs)
            finally:
                site = None

        def eager(mask, batch, head_dim, query, key, value):
            if site not in scaled:
                return original_eager(mask, batch, head_dim, query, key, value)
            p = probabilities(expert, mask, query, key, value.dtype)
            original = (p @ in_gqa(expert, value)).transpose(1, 2).reshape(batch, query.shape[1], -1)
            if verify:
                assert site not in active and torch.equal(
                    original_eager(mask, batch, head_dim, query, key, value), original)
                active.add(site)
            with torch.autocast("cuda", enabled=False):
                adjusted = p.float() @ in_gqa(expert, scaled[site].view_as(value))
                adjusted = adjusted.transpose(1, 2).reshape_as(original)
                return (original.float() + adjusted).to(original.dtype)

        expert.forward_cross_attn_layer = cross
        expert.eager_attention_forward = eager
        try:
            yield
        finally:
            expert.forward_cross_attn_layer = original_cross
            expert.eager_attention_forward = original_eager
        if verify:
            assert active == set(self.layers)

    def chunk(self, obs, q, seed, chunk, mode, reverse, verify=False):
        rt = self.lab.rt
        pair = self.prefixes(obs)
        values = self.cache(pair)
        visited = False

        def field(x, t):
            nonlocal visited
            with self.edit(values, q, mode, reverse, verify and not visited):
                result = rt.query("F", pair[q], x, t)
            visited = True
            return result

        return rt.integrate(field, rt.noise(seed, chunk), 10)[..., :7]


def check_equivalence(lab, relay, rec, full, anchor):
    obs = handoff(lab, rec)
    seed, chunk = rec["seed"], len(rec["controls"])
    rolled = configuration("cross_v", full, anchor)
    report = {}
    # The engineering gate compares multiple weight assignments in one context.
    # Disable autocast's parameter cache so every assignment is seen immediately.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        for reverse in (False, True):
            donors = rolled if not reverse else full
            base = full if not reverse else rolled
            assign(lab, "F", donors)
            expected = {q: lab.native("F", INSTR[q], obs, seed, chunk) for q in "AB"}
            assign(lab, "F", base)
            for q in "AB":
                patched = relay.chunk(obs, q, seed, chunk, "whole", reverse, verify=True)
                assert torch.equal(expected[q], patched), (reverse, q, rms(expected[q] - patched))
            report[f"reverse_{reverse}_whole_chunk_bitwise"] = True
        assign(lab, "F", full)
        pair = relay.prefixes(obs)
        examples = relay.cache(pair)
        x = lab.rt.noise(seed, chunk)
        t = torch.ones(1, device=x.device)
        for q in "AB":
            for mode in ("mean", "diff"):
                with relay.edit(examples, q, mode, reverse=False, verify=True):
                    lab.rt.query("F", pair[q], x, t)
        ids = [lab.rt.batch("F", obs, INSTR[q])[lab.rt.constants.OBS_LANGUAGE_TOKENS] for q in "AB"]
        differing = torch.where((ids[0] != ids[1])[0])[0].tolist()
        quotient = []
        reconstruction_roundoff = []
        for parts in examples.values():
            mu, d = relay.factor(parts)
            corrections = [v[1].double() - v[0].double() for v in parts.values()]
            exact_mean, exact_diff = (corrections[0] + corrections[1]) * .5, (corrections[0] - corrections[1]) * .5
            for sign, v in zip((1, -1), parts.values(), strict=True):
                accurate = (v[0].double() + exact_mean + sign * exact_diff).to(v[1].dtype)
                assert torch.equal(accurate, v[1])
                error = float((v[0].float() + mu + sign * d - v[1].float()).abs().max())
                assert error <= max(float(v[1].abs().max()), 1.) * 1e-6
                reconstruction_roundoff.append(error)
            qa, qb = ((parts[q][0].float() + mu).to(parts[q][0].dtype).float() for q in "AB")
            discrepancy = (qa - qb) - (parts["A"][0].float() - parts["B"][0].float())
            quotient.append(rms(discrepancy) / max(rms(2 * d), 1e-12))
        report.update(init=rec["init"], token_difference_indices=differing, groups=pair["A"][2],
                      active_prefix_slots=int(pair["A"][0].sum()), value_decomposition_bitwise=True,
                      fp32_split_readout_verified=True,
                      fp32_reconstruction_max_abs_error=max(reconstruction_roundoff),
                      pre_weighted_sum_bf16_rounded_mean_leak_rms_relative_to_correction=max(quotient))
    return report


def evaluate(lane, lanes):
    assert read(RUN / "selected.json")["cross"] == "v"
    records = cases()[lane::lanes]
    full, anchor = checkpoint_files()
    lab = Lab5()
    relay = Relay(lab, full, anchor)
    native = lab.native
    try:
        verification = check_equivalence(lab, relay, records[0], full, anchor)
        with (RUN / f"relay_path_{lane}.jsonl").open("x", encoding="utf-8") as out:
            for arm, (mode, goals) in PLAN.items():
                reverse = arm in REVERSE
                assign(lab, "F", configuration("cross_v", full, anchor) if reverse else full)
                for rec in records:
                    start = len(rec["controls"])
                    for q in goals:
                        def sample(role, text, obs, seed, chunk):
                            if chunk < start + WINDOW:
                                return relay.chunk(obs, q, seed, chunk, mode, reverse)
                            return native(role, text, obs, seed, chunk)

                        lab.native = sample
                        row = {"arm": arm, **measure_episode(lab, rec, q)}
                        out.write(json.dumps(row, allow_nan=False) + "\n")
                        out.flush()
                        print("EVAL", lane, arm, rec["init"], q, row["first"], flush=True)
    finally:
        lab.native = native
        lab.ctx.env.close()
    write(RUN / f"relay_path_{lane}_COMPLETE.json", {"lane": lane, "lanes": lanes, "gate": verification})


def run_probe():
    lab = Lab5()
    full, anchor = checkpoint_files()
    relay = Relay(lab, full, anchor)
    try:
        records = cases()
        write(RUN / "relay_path_gates.json", [check_equivalence(lab, relay, r, full, anchor) for r in records[:2]])
        bank = fixed_bank(lab, records)
        modes = {"base_full": ("identity", False), "base_rolled": ("identity", True),
                 **{a: (m, a in REVERSE) for a, (m, _) in PLAN.items()}}
        for arm, (mode, reverse) in modes.items():
            assign(lab, "F", configuration("cross_v", full, anchor) if reverse else full)
            captured = []
            case = None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i, route, index, obs, x, t, old_a, old_b in bank:
                    if i != case:
                        case = i
                        pair = relay.prefixes(obs)
                        values = relay.cache(pair)
                    output = {}
                    for q in "AB":
                        if mode == "identity":
                            output[q] = lab.rt.query("F", pair[q], x, t).float()
                        else:
                            with relay.edit(values, q, mode, reverse):
                                output[q] = lab.rt.query("F", pair[q], x, t).float()
                    left, right = output["A"][:, :10], output["B"][:, :10]
                    a, b = old_a[:, :10], old_b[:, :10]
                    h, ref = ((left - right) * .5)[..., :2], ((a - b) * .5)[..., :2]
                    delta = (left + right - a - b) * .5
                    captured.append({"init": i, "route": route, "denoise_index": index,
                                     "h_gain": float((h * ref).sum() / ref.square().sum().clamp_min(1e-12)),
                                     "mu_h": rms(delta[..., :2]), "mu_z": rms(delta[..., 2:3]),
                                     "mu_r": rms(delta[..., 3:6])})
            write(RUN / f"relay_path_probe_{arm}.json", captured)
            print("PROBE", arm, flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN / "relay_path_probe_COMPLETE.json", {"arms": list(modes)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("eval", "probe"))
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "probe":
        run_probe()
    else:
        evaluate(args.lane, args.lanes)
