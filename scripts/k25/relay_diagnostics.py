"""Read matched local computations and test V × MLP training-update combinations."""

import argparse
from contextlib import contextmanager
import json

import torch
from torch.nn import functional as F

from path_factorization import (CROSS, Lab5, RUN, assign, cases, checkpoint_files, configuration,
                                fixed_bank, measure_episode, parameters_for, responses, write)
from value_relay import Relay, in_gqa, payload_input, probabilities, rms
from window_eval import INSTR

COMBOS = ("v_on_anchor", "v_mlp_on_anchor")


class ObserveAttention:
    def __init__(self, expert):
        self.expert = expert
        self.cross = expert.forward_cross_attn_layer
        self.eager = expert.eager_attention_forward
        self.site = None

    def wrapped_cross(self, model_layers, inputs_embeds, layer_idx, *args, **kwargs):
        self.site = layer_idx
        try:
            return self.cross(model_layers, inputs_embeds, layer_idx, *args, **kwargs)
        finally:
            self.site = None

    def wrapped_eager(self, mask, batch, dim, query, key, value):
        output = self.eager(mask, batch, dim, query, key, value)
        if self.site in CROSS:
            assert self.site not in self.records
            weights = probabilities(self.expert, mask, query, key, value.dtype)
            if self.verify:
                expected = (weights @ in_gqa(self.expert, value)).transpose(1, 2).reshape_as(output)
                assert torch.equal(expected, output), (self.site, rms(expected - output))
            self.records[self.site] = (mask, query, key, value, weights)
        return output

    @contextmanager
    def record(self, verify):
        self.records = {}
        self.verify = verify
        self.expert.forward_cross_attn_layer = self.wrapped_cross
        self.expert.eager_attention_forward = self.wrapped_eager
        try:
            yield self.records
        finally:
            self.expert.forward_cross_attn_layer = self.cross
            self.expert.eager_attention_forward = self.eager
        assert set(self.records) == set(CROSS)


@contextmanager
def observe_mlp(layers):
    hidden = {}

    def recorder(i):
        def hook(_module, args):
            assert i not in hidden
            hidden[i] = args[0]
        return hook

    hooks = [m.register_forward_pre_hook(recorder(i)) for i, m in enumerate(layers)]
    try:
        yield hidden
    finally:
        for hook in hooks:
            hook.remove()
    assert set(hidden) == set(range(len(layers)))


def signed_share(part, total):
    return float((part * total).sum() / total.square().sum().clamp_min(1e-12))


def trace_v(expert, pair, captured, values, layer, anchor_key):
    record = [captured[q][layer] for q in "AB"]
    probs = [r[-1][:, :, :10] for r in record]
    shifts = []
    for q, (mask, query, key, value, original) in zip("AB", record, strict=True):
        shaped = payload_input(pair[q], layer, "k").to(anchor_key.dtype)
        alternate = F.linear(shaped, anchor_key).view_as(key)
        switched = probabilities(expert, mask, query, alternate, value.dtype)
        shifts.append(float((switched[:, :, :10].float() - original[:, :, :10].float())
                            .abs().sum(-1).mean() * .5))
    mean_v, diff_v = Relay.factor(values[layer])
    with torch.autocast("cuda", enabled=False):
        amin, aplus = (probs[0].float() - probs[1].float()) * .5, (probs[0].float() + probs[1].float()) * .5
        head_shape = record[0][3].shape
        mean_v, diff_v = (in_gqa(expert, z.view(head_shape)) for z in (mean_v, diff_v))
        p_mean, p_diff = amin @ mean_v, aplus @ diff_v
        total = p_mean + p_diff
        expected = .5 * (probs[0].float() @ (mean_v + diff_v) - probs[1].float() @ (mean_v - diff_v))
        assert torch.allclose(expected, total, atol=1e-5, rtol=1e-4)
        other = {}
        for modality, (start, stop) in pair["A"][2].items():
            pm = amin[..., start:stop] @ mean_v[:, :, start:stop]
            pd = aplus[..., start:stop] @ diff_v[:, :, start:stop]
            other[modality] = {"total_signed_share": signed_share(pm + pd, total),
                               "mean_signed_share": signed_share(pm, total),
                               "diff_signed_share": signed_share(pd, total)}
    return {"local_v_correction_rms": rms(total),
            "local_mean_rms": rms(p_mean), "local_diff_rms": rms(p_diff),
            "local_mean_signed_share": signed_share(p_mean, total),
            "local_diff_signed_share": signed_share(p_diff, total),
            "k_changes_attention_total_variation": sum(shifts) / len(shifts), "spans": other}


def trace_mlp(module, left, right, anchor_up):
    g = [module.act_fn(module.gate_proj(h)).float() for h in (left, right)]
    delta_up = [F.linear(h, anchor_up).float() - module.up_proj(h).float() for h in (left, right)]
    with torch.autocast("cuda", enabled=False):
        gate_mean, gate_diff = (g[0] + g[1]) * .5, (g[0] - g[1]) * .5
        up_mean, up_diff = (delta_up[0] + delta_up[1]) * .5, (delta_up[0] - delta_up[1]) * .5
        w = module.down_proj.weight.float()
        mean = F.linear(gate_diff * up_mean, w)[:, :10]
        diff = F.linear(gate_mean * up_diff, w)[:, :10]
        total = mean + diff
        expected = F.linear((g[0] * delta_up[0] - g[1] * delta_up[1]) * .5, w)[:, :10]
        assert torch.allclose(expected, total, atol=1e-5, rtol=1e-4)
    return {"local_up_correction_rms": rms(total),
            "local_mean_rms": rms(mean), "local_diff_rms": rms(diff),
            "local_mean_signed_share": signed_share(mean, total),
            "local_diff_signed_share": signed_share(diff, total)}


def trace(family):
    lab = Lab5()
    full, anchor = checkpoint_files()
    expert = lab.rt.models["F"].model.vlm_with_expert
    try:
        bank = fixed_bank(lab, cases())
        mlps = [block.mlp for block in expert.lm_expert.layers]
        observer = ObserveAttention(expert)
        relay = Relay(lab, full, anchor)
        layers, arm = (CROSS, "full") if family == "v" else (tuple(range(16)), "mlp_gate_up_down")
        assign(lab, "F", configuration(arm, full, anchor))
        suffix = "self_attn.k_proj.weight" if family == "v" else "mlp.up_proj.weight"
        prior = {i: anchor[f"model.vlm_with_expert.lm_expert.layers.{i}.{suffix}"].cuda() for i in layers}
        case = None
        results = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index, (i, route, step, obs, x, t, _, _) in enumerate(bank):
                if i != case:
                    case = i
                    pair = relay.prefixes(obs)
                    values = relay.cache(pair) if family == "v" else None
                traces = {}
                for q in "AB":
                    context = observer.record(index == 0) if family == "v" else observe_mlp(mlps)
                    with context as traces[q]:
                        lab.rt.query("F", pair[q], x, t)
                for j in layers:
                    data = trace_v(expert, pair, traces, values, j, prior[j]) if family == "v" else trace_mlp(
                        mlps[j], traces["A"][j], traces["B"][j], prior[j])
                    results.append({"init": i, "draft": route, "step": step, "layer": j, **data})
    finally:
        lab.ctx.env.close()
    write(RUN / f"trace_{family}.json", results)
    print("TRACE", family, len(results), flush=True)


def combo(arm, full, anchor):
    names = set(parameters_for("cross_v", full)[1])
    if arm == "v_mlp_on_anchor":
        names.update(parameters_for("mlp_gate_up_down", full)[1])
    return {n: full[n] if n in names else v for n, v in anchor.items()}


def combo_probe():
    lab = Lab5()
    full, anchor = checkpoint_files()
    try:
        bank = fixed_bank(lab, cases())
        for arm in COMBOS:
            assign(lab, "F", combo(arm, full, anchor))
            write(RUN / f"probe_held_{arm}.json", responses(lab, bank, True))
            print("COMBO_PROBE", arm, flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN / "combo_probe_COMPLETE.json", {"arms": list(COMBOS)})


def combo_eval(lane, lanes):
    lab = Lab5()
    full, anchor = checkpoint_files()
    try:
        with (RUN / f"combo_{lane}.jsonl").open("x", encoding="utf-8") as out:
            for arm in COMBOS:
                assign(lab, "F", combo(arm, full, anchor))
                for rec in cases()[lane::lanes]:
                    for q in "AB":
                        row = {"arm": arm, **measure_episode(lab, rec, q)}
                        out.write(json.dumps(row, allow_nan=False) + "\n")
                        out.flush()
                        print("EVAL", lane, arm, rec["init"], q, row["first"], flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN / f"combo_{lane}_COMPLETE.json", {"lane": lane, "lanes": lanes, "arms": list(COMBOS)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("v", "mlp", "combo_probe", "combo_eval"))
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode in ("v", "mlp"):
        trace(args.mode)
    elif args.mode == "combo_probe":
        combo_probe()
    else:
        combo_eval(args.lane, args.lanes)
