"""Close the MLP up pathway and replicate both pathways on fresh grasps."""

import argparse
import copy
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

RUN = Path(__file__).resolve().parent
PATHS = RUN.parent / "k25_paths_20261119_01"
sys.path.insert(0, str(PATHS))
from path_factorization import Lab5, assign, cases, checkpoint_files, configuration, fixed_bank, measure_episode, read, write
from value_relay import Relay, check_equivalence as check_v, rms
from window_eval import INSTR, WINDOW
from deep_selection import handoff

UP_ADAPTED = "mlp_gate_up_down"
UP_PROTECTED = "mlp_gate_down"
UP_OLD = {
    "up_restore_whole": (UP_ADAPTED, "whole", "AB"),
    "up_restore_mean": (UP_ADAPTED, "mean", "AB"),
    "up_restore_diff": (UP_ADAPTED, "diff", "AB"),
    "up_reverse_whole": (UP_PROTECTED, "whole", "B"),
    "up_reverse_mean": (UP_PROTECTED, "mean", "B"),
    "up_reverse_diff": (UP_PROTECTED, "diff", "B"),
}
UP_FRESH = {
    "up_adapted": (UP_ADAPTED, "native", "AB"),
    "up_protected": (UP_PROTECTED, "native", "AB"),
    **{arm: (base, component, "AB" if arm == "up_restore_mean" else "B")
       for arm, (base, component, _) in UP_OLD.items() if arm != "up_reverse_whole"},
}
V_FRESH = {
    "v_adapted": ("full", "native", "AB"),
    "v_protected": ("cross_v", "native", "AB"),
    "v_restore_whole": ("full", "whole", "B"),
    "v_restore_mean": ("full", "mean", "AB"),
    "v_restore_diff": ("full", "diff", "B"),
    "v_reverse_mean": ("cross_v", "mean", "B"),
    "v_reverse_diff": ("cross_v", "diff", "B"),
}
PLANS = {"old_up": UP_OLD, "fresh_up": UP_FRESH, "fresh_v": V_FRESH}


class SingletonUp:
    """Coordinate two singleton forwards at each MLP, using native attention."""

    def __init__(self, lab, full, anchor):
        self.rt = lab.rt
        self.policy = self.rt.models["F"].model
        self.expert = self.policy.vlm_with_expert
        self.layers = self.expert.get_model_layers([self.expert.get_vlm_model().text_model,
                                                    self.expert.lm_expert])
        assert len(self.layers[1]) == 16 and "cross" in self.expert.attention_mode
        assert self.expert.self_attn_every_n_layers == 2 and self.policy.config.use_cache
        self.up_weights = []
        for i, block in enumerate(self.layers[1]):
            mod = block.mlp
            assert mod.up_proj.bias is None and mod.down_proj.bias is None
            key = f"model.vlm_with_expert.lm_expert.layers.{i}.mlp.up_proj.weight"
            self.up_weights.append(tuple(w[key].to(mod.up_proj.weight.device) for w in (full, anchor)))

    def prefixes(self, obs):
        pair = [self.rt.prefix("F", obs, INSTR[q]) for q in "AB"]
        assert torch.equal(pair[0][0], pair[1][0]) and pair[0][2] == pair[1][2]
        return pair

    def query(self, pair, x, t, mode, reverse):
        model, expert = self.policy, self.expert
        embeddings, suffix_pad, suffix_att = model.embed_suffix(x.clone(), t.clone())
        states = [embeddings.clone(), embeddings.clone()]
        pad = pair[0][0]
        caches = [copy.deepcopy(p[1]) for p in pair]
        prefix_mask = pad[:, None, :].expand(1, suffix_pad.shape[1], pad.shape[1])
        mask = torch.cat([prefix_mask, self.rt.make_masks(suffix_pad, suffix_att)], dim=2)
        pos = pad.sum(-1)[:, None] + suffix_pad.cumsum(dim=1) - 1
        head_dim = expert.vlm.config.text_config.head_dim
        for i, block in enumerate(self.layers[1]):
            attention = expert.forward_cross_attn_layer if i % 2 else expert.forward_attn_layer
            residuals = []
            normalized = []
            for branch, hidden in enumerate(states):
                messages, caches[branch] = attention(self.layers, [None, hidden], i, pos, mask, 1, head_dim,
                                                      use_cache=True, past_key_values=caches[branch])
                assert len(messages) == 1
                res = block.self_attn.o_proj(messages[0].to(block.self_attn.o_proj.weight.dtype))
                res += hidden
                residuals.append(res.clone())
                normalized.append(block.post_attention_layernorm(res))
            mod = block.mlp
            if mode == "native":
                outcomes = [mod(h) for h in normalized]
            else:
                gates = [mod.act_fn(mod.gate_proj(h)) for h in normalized]
                donor = self.up_weights[i][0 if reverse else 1]
                other = [F.linear(h, donor) for h in normalized]
                if mode == "whole":
                    outcomes = [mod.down_proj(g * u) for g, u in zip(gates, other, strict=True)]
                else:
                    current = [mod.up_proj(h) for h in normalized]
                    delta = [alt.float() - src.float() for alt, src in zip(other, current, strict=True)]
                    shared = (delta[0] + delta[1]) * .5
                    parts = [shared, shared] if mode == "mean" else [d - shared for d in delta]
                    outcomes = []
                    for g, u, part in zip(gates, current, parts, strict=True):
                        out = mod.down_proj(g * u)
                        with torch.autocast("cuda", enabled=False):
                            message = F.linear(g.float() * part, mod.down_proj.weight.float())
                            outcomes.append((out.float() + message).to(out.dtype))
            for out, res in zip(outcomes, residuals, strict=True):
                out += res
            states = outcomes
        for cache in caches:
            cache.crop(pad.shape[1])
            assert all(slot.keys.shape[-2] == pad.shape[1] for slot in cache.layers)
        return [model.action_out_proj(expert.lm_expert.norm(h)[:, -model.config.chunk_size:].float())
                for h in states]

    def chunk(self, obs, q, seed, chunk, mode, reverse):
        pair = self.prefixes(obs)
        q_idx = "AB".index(q)

        def velocity(x, t):
            return self.query(pair, x, t, mode, reverse)[q_idx]

        return self.rt.integrate(velocity, self.rt.noise(seed, chunk), 10)[..., :7]


def check_up(lab, up, record, full, anchor):
    obs = handoff(lab, record)
    seed, chunk = record["seed"], len(record["controls"])
    natives = {}
    for base in (UP_ADAPTED, UP_PROTECTED):
        assign(lab, "F", configuration(base, full, anchor))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            natives[base] = {q: lab.native("F", INSTR[q], obs, seed, chunk) for q in "AB"}
            for q in "AB":
                x = up.chunk(obs, q, seed, chunk, "native", False)
                assert torch.equal(natives[base][q], x), (base, q, rms(natives[base][q] - x))
    for reverse, (base, donor) in enumerate(((UP_ADAPTED, UP_PROTECTED), (UP_PROTECTED, UP_ADAPTED))):
        assign(lab, "F", configuration(base, full, anchor))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            for q in "AB":
                x = up.chunk(obs, q, seed, chunk, "whole", bool(reverse))
                assert torch.equal(natives[donor][q], x), (base, q, rms(natives[donor][q] - x))
    return {"init": record["init"], "identity_bitwise_both_bases": True, "whole_bitwise_both_directions": True}


def records(suite):
    rows = cases() if suite == "old_up" else read(RUN / "holdout.json")["selected"]
    assert len(rows) == 20
    assert all(400 <= r["init"] < 448 for r in rows) if suite != "old_up" else True
    return rows


def evaluate(suite, lane, lanes):
    plan = PLANS[suite]
    rows = records(suite)[lane::lanes]
    full, anchor = checkpoint_files()
    lab = Lab5()
    native = lab.native
    try:
        is_v = suite == "fresh_v"
        runner = Relay(lab, full, anchor) if is_v else SingletonUp(lab, full, anchor)
        gate = (check_v if is_v else check_up)(lab, runner, rows[0], full, anchor)
        print("GATE", gate, flush=True)
        with (RUN / f"{suite}_{lane}.jsonl").open("x", encoding="utf-8") as out:
            for arm, (base, mode, qs) in plan.items():
                assign(lab, "F", configuration(base, full, anchor))
                reverse = "_reverse_" in arm
                for record in rows:
                    start = len(record["controls"])
                    for q in qs:
                        def sample(role, text, obs, seed, chunk):
                            if mode != "native" and start <= chunk < start + WINDOW:
                                assert text == INSTR[q]
                                return runner.chunk(obs, q, seed, chunk, mode, reverse)
                            return native(role, text, obs, seed, chunk)

                        lab.native = sample
                        result = {"arm": arm, **measure_episode(lab, record, q)}
                        out.write(json.dumps(result, allow_nan=False) + "\n")
                        out.flush()
                        print("EVAL", suite, lane, arm, record["init"], q, result["first"], flush=True)
    finally:
        lab.native = native
        lab.ctx.env.close()
    write(RUN / f"{suite}_{lane}_COMPLETE.json", {"suite": suite, "lane": lane, "lanes": lanes, "gate": gate})


def probes(suite):
    assert suite in ("fresh_v", "fresh_up")
    lab = Lab5()
    full, anchor = checkpoint_files()
    is_v = suite == "fresh_v"
    runner = Relay(lab, full, anchor) if is_v else SingletonUp(lab, full, anchor)
    try:
        bank = fixed_bank(lab, records(suite))
        for arm, (base, mode, _) in PLANS[suite].items():
            assign(lab, "F", configuration(base, full, anchor))
            outputs = []
            last = None
            reverse = "_reverse_" in arm
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i, route, step, obs, x, t, old_a, old_b in bank:
                    if i != last:
                        last = i
                        pair = runner.prefixes(obs)
                        if is_v:
                            projections = runner.cache(pair)
                    if is_v:
                        vs = []
                        for q in "AB":
                            if mode == "native":
                                v = lab.rt.query("F", pair[q], x, t)
                            else:
                                with runner.edit(projections, q, mode, reverse):
                                    v = lab.rt.query("F", pair[q], x, t)
                            vs.append(v.float())
                    else:
                        vs = [v.float() for v in runner.query(pair, x, t, mode, reverse)]
                    a, b = (v[:, :10] for v in vs)
                    oa, ob = old_a[:, :10], old_b[:, :10]
                    d, d0 = (a - b)[..., :2] * .5, (oa - ob)[..., :2] * .5
                    mu_error = ((a + b) - (oa + ob)) * .5
                    outputs.append({"init": i, "route": route, "step": step,
                                    "h_gain": float((d * d0).sum() / d0.square().sum().clamp_min(1e-12)),
                                    "mu_h": rms(mu_error[..., :2])})
            write(RUN / f"probe_{arm}.json", outputs)
            print("PROBE", suite, arm, flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN / f"probe_{suite}_COMPLETE.json", {"arms": list(PLANS[suite])})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("eval", "probe", "gate"))
    parser.add_argument("--suite", choices=PLANS, required=True)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=4)
    a = parser.parse_args()
    torch.set_num_threads(2)
    if a.mode == "eval":
        evaluate(a.suite, a.lane, a.lanes)
    elif a.mode == "probe":
        probes(a.suite)
    else:
        lab = Lab5()
        full, anchor = checkpoint_files()
        try:
            gate = check_up(lab, SingletonUp(lab, full, anchor), records(a.suite)[0], full, anchor)
            write(RUN / f"{a.suite}_GATE.json", gate)
            print("GATE", gate, flush=True)
        finally:
            lab.ctx.env.close()
