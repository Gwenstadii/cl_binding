"""Causally separate cross-attention probabilities into query and key paths."""

import argparse
from contextlib import contextmanager
import json

import torch

from v_attribution_v3 import (Lab5, RUN, assign, checkpoint_files, fixed_bank, handoff, horizontal,
                             measure_episode, records, rms, state, write)
from relay_diagnostics import ObserveAttention, signed_share
from path_factorization import CROSS
from value_relay import Relay, in_gqa, probabilities
from window_eval import INSTR, WINDOW

MAIN = {
    "p_restore": ("full", "anchor", "p"),
    "p_query": ("full", "anchor", "q"),
    "p_key": ("full", "anchor", "k"),
    "pv_restore": ("full", "anchor", "pv"),
    "p_reverse": ("anchor", "full", "p"),
    "pv_reverse": ("anchor", "full", "pv"),
}
QUERY = {"p_query_hidden": ("full", "anchor", "h"), "q_weights": ("q_weights", None, "native")}


def weight_state(name, full, anchor):
    if name != "q_weights":
        return state(name, full, anchor)
    keys = {f"model.vlm_with_expert.lm_expert.layers.{i}.self_attn.q_proj.weight" for i in CROSS}
    assert keys <= full.keys()
    return {key: anchor[key] if key in keys else val for key, val in full.items()}


class ReadingBridge:
    def __init__(self, lab, full, anchor):
        self.rt = lab.rt
        self.relay = Relay(lab, full, anchor)
        self.recipient = self.rt.models["F"].model.vlm_with_expert
        self.donor = self.rt.models["R"].model.vlm_with_expert
        self.capture = ObserveAttention(self.donor)

    def prefixes(self, obs, prompts):
        pairs = {(role, q): self.rt.prefix(role, obs, INSTR[q]) for role in "FR" for q in prompts}
        for q in prompts:
            f, r = (pairs[role, q] for role in "FR")
            assert torch.equal(f[0], r[0]) and f[2] == r[2]
        return pairs

    def read_donor(self, prefix, x, t, hidden):
        inputs = {}

        def recorder(site):
            def hook(_, args):
                assert site not in inputs
                inputs[site] = args[0]
            return hook

        hooks = [self.donor.lm_expert.layers[i].self_attn.q_proj.register_forward_pre_hook(recorder(i))
                 for i in CROSS] if hidden else []
        try:
            with self.capture.record(verify=False) as captured:
                self.rt.query("R", prefix, x, t)
        finally:
            for hook in hooks:
                hook.remove()
        if hidden:
            assert set(inputs) == set(CROSS)
        return captured, inputs

    @contextmanager
    def edit(self, captured, inputs, v_values, mode, verify):
        if mode == "h":
            def exchange(site):
                def hook(_, args):
                    assert args[0].shape == inputs[site].shape
                    return (inputs[site],)
                return hook

            handles = [self.recipient.lm_expert.layers[i].self_attn.q_proj.register_forward_pre_hook(exchange(i))
                       for i in CROSS]
            try:
                yield
            finally:
                for handle in handles:
                    handle.remove()
            return

        expert = self.recipient
        original_cross, original_eager = expert.forward_cross_attn_layer, expert.eager_attention_forward
        site = None
        visited = set()

        def cross(model_layers, inputs_embeds, layer_idx, *args, **kwargs):
            nonlocal site
            site = layer_idx
            try:
                return original_cross(model_layers, inputs_embeds, layer_idx, *args, **kwargs)
            finally:
                site = None

        def eager(mask, batch, dim, q, k, v):
            if site not in captured:
                return original_eager(mask, batch, dim, q, k, v)
            donor_mask, donor_q, donor_k, _, p = captured[site]
            if verify:
                assert site not in visited and torch.equal(donor_mask, mask)
                visited.add(site)
            if mode == "q":
                return original_eager(mask, batch, dim, donor_q, k, v)
            if mode == "k":
                return original_eager(mask, batch, dim, q, donor_k, v)
            if mode == "pv":
                v = v_values[site].view_as(v)
            return (p.to(v.dtype) @ in_gqa(expert, v)).transpose(1, 2).reshape(batch, q.shape[1], -1)

        expert.forward_cross_attn_layer, expert.eager_attention_forward = cross, eager
        try:
            yield
        finally:
            expert.forward_cross_attn_layer, expert.eager_attention_forward = original_cross, original_eager
        if verify:
            assert visited == set(CROSS)

    def value_table(self, prefixes, prompts, index):
        by_layer = self.relay.cache({q: prefixes["F", q] for q in prompts})
        return {q: {i: parts[q][index] for i, parts in by_layer.items()} for q in prompts}

    def query(self, prefixes, q, x, t, mode, table=None, verify=False):
        if mode == "native":
            return self.rt.query("F", prefixes["F", q], x, t)
        captured, inputs = self.read_donor(prefixes["R", q], x, t, mode == "h")
        with self.edit(captured, inputs, table[q] if table else None, mode, verify):
            return self.rt.query("F", prefixes["F", q], x, t)

    def chunk(self, obs, q, seed, chunk, mode, reverse):
        prefixes = self.prefixes(obs, q)
        table = self.value_table(prefixes, q, 0 if reverse else 1) if mode == "pv" else None
        return self.rt.integrate(lambda x, t: self.query(prefixes, q, x, t, mode, table),
                                 self.rt.noise(seed, chunk), 10)[..., :7]


def gate(lab, bridge, rec, full):
    for role in "FR":
        assign(lab, role, full)
    obs = handoff(lab, rec)
    x, t = lab.rt.noise(rec["seed"], len(rec["controls"])), torch.ones(1, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        pairs = bridge.prefixes(obs, "AB")
        identity = bridge.value_table(pairs, "AB", 0)
        for q in "AB":
            ref = bridge.query(pairs, q, x, t, "native")
            for mode in ("p", "q", "k", "pv", "h"):
                returned = bridge.query(pairs, q, x, t, mode, identity, verify=True)
                assert torch.equal(returned, ref), (q, mode, rms(returned - ref))
    return {"init": rec["init"], "same_weights_native_bitwise_for": ["p", "q", "k", "pv", "h"]}


def run_probe(suite):
    lab = Lab5()
    full, anchor = checkpoint_files()
    bridge = ReadingBridge(lab, full, anchor)
    chosen = records()[:20] if suite == "old" else records()[20:]
    try:
        bank = fixed_bank(lab, chosen)
        verification = gate(lab, bridge, chosen[0], full)
        result = {}
        for arm, (base, donor, mode) in (MAIN | QUERY).items():
            assign(lab, "F", weight_state(base, full, anchor))
            if donor:
                assign(lab, "R", weight_state(donor, full, anchor))
            last = None
            rows = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i, route, step, obs, x, t, a0, b0 in bank:
                    if i != last:
                        prefixes = bridge.prefixes(obs, "AB")
                        table = bridge.value_table(prefixes, "AB", 0 if base == "anchor" else 1) if mode == "pv" else None
                        last = i
                    a, b = (bridge.query(prefixes, q, x, t, mode, table)[:, :10, :2].float() for q in "AB")
                    a0, b0 = a0[:, :10, :2].float(), b0[:, :10, :2].float()
                    rows.append({"init": i, "route": route, "step": step,
                                 "h_gain": horizontal((a - b) * .5, (a0 - b0) * .5),
                                 "mu_h_to_r1": rms((a + b - a0 - b0) * .5)})
            result[arm] = rows
            print("P_PROBE", suite, arm, flush=True)
        write(RUN / f"p_probe_{suite}.json", {"arms": result, "gate": verification})
    finally:
        lab.ctx.env.close()


def trace():
    lab = Lab5()
    full, anchor = checkpoint_files()
    bridge = ReadingBridge(lab, full, anchor)
    receiver_capture = ObserveAttention(bridge.recipient)
    samples = []
    try:
        bank = fixed_bank(lab, records()[20:])
        assign(lab, "F", full)
        assign(lab, "R", anchor)
        last = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i, route, step, obs, x, t, _, _ in bank:
                if i != last:
                    prefixes = bridge.prefixes(obs, "AB")
                    last = i
                traces = {}
                for q in "AB":
                    with receiver_capture.record(verify=i == 400 and route == "A" and step == 0) as fs:
                        bridge.query(prefixes, q, x, t, "native")
                    rs, _ = bridge.read_donor(prefixes["R", q], x, t, False)
                    traces[q] = (fs, rs)
                for layer in CROSS:
                    f = [traces[q][0][layer] for q in "AB"]
                    r = [traces[q][1][layer] for q in "AB"]
                    alternatives = []
                    for (mask, qf, kf, vf, pf), (other_mask, qr, kr, _, pr) in zip(f, r, strict=True):
                        assert torch.equal(mask, other_mask)
                        pq = probabilities(bridge.recipient, mask, qr, kf, vf.dtype)
                        pk = probabilities(bridge.recipient, mask, qf, kr, vf.dtype)
                        alternatives.append([p[:, :, :10].float() - pf[:, :, :10].float() for p in (pq, pk, pr)])
                    with torch.autocast("cuda", enabled=False):
                        vl, vr = (in_gqa(bridge.recipient, v[3].float()) for v in f)
                        dp, dq, dk = [], [], []
                        for a, b in zip(*alternatives, strict=True):
                            dp.append((a + b) * .5)
                            dq.append((a - b) * .5)
                            dk.append((a @ vl - b @ vr) * .5)
                        shared = dp[2] @ ((vl - vr) * .5)
                        differential = dq[2] @ ((vl + vr) * .5)
                        assert torch.allclose(shared + differential, dk[2], atol=2e-5, rtol=1e-4)
                        samples.append({"init": i, "route": route, "step": step, "layer": layer,
                                        "direct_change_d_rms": rms(dk[2]),
                                        "query_share": signed_share(dk[0], dk[2]),
                                        "key_share": signed_share(dk[1], dk[2]),
                                        "common_probability_share": signed_share(shared, dk[2]),
                                        "differential_probability_share": signed_share(differential, dk[2])})
    finally:
        lab.ctx.env.close()
    write(RUN / "p_trace.json", samples)
    print("P_TRACE", len(samples), flush=True)


def evaluate(lane, lanes, query_only):
    plan = QUERY if query_only else MAIN
    tag = "q_eval" if query_only else "p_eval"
    cohort = records()[20:] if query_only else records()
    chosen = cohort[lane::lanes]
    full, anchor = checkpoint_files()
    lab = Lab5()
    native = lab.native
    try:
        bridge = ReadingBridge(lab, full, anchor)
        checked = gate(lab, bridge, chosen[0], full)
        print("P_GATE", checked, flush=True)
        with (RUN / f"{tag}_{lane}.jsonl").open("x", encoding="utf-8") as out:
            for arm, (base, donor, mode) in plan.items():
                assign(lab, "F", weight_state(base, full, anchor))
                if donor:
                    assign(lab, "R", weight_state(donor, full, anchor))
                for rec in chosen:
                    start = len(rec["controls"])

                    def sample(role, text, obs, seed, chunk):
                        if mode != "native" and start <= chunk < start + WINDOW:
                            return bridge.chunk(obs, "B", seed, chunk, mode, base == "anchor")
                        return native(role, text, obs, seed, chunk)

                    lab.native = sample
                    row = {"arm": arm, **measure_episode(lab, rec, "B")}
                    out.write(json.dumps(row, allow_nan=False) + "\n")
                    out.flush()
                    print("P_EVAL", lane, arm, rec["init"], row["first"], flush=True)
    finally:
        lab.native = native
        lab.ctx.env.close()
    write(RUN / f"{tag}_{lane}_COMPLETE.json", {"lane": lane, "lanes": lanes, "gate": checked})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("probe", "trace", "eval", "query_eval"))
    p.add_argument("--suite", choices=("old", "fresh"), default="fresh")
    p.add_argument("--lane", type=int, default=0)
    p.add_argument("--lanes", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.mode == "probe":
        run_probe(a.suite)
    elif a.mode == "trace":
        trace()
    else:
        evaluate(a.lane, a.lanes, a.mode == "query_eval")
