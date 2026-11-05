"""K4: where does the instruction stop controlling the action head?

Read (attention) and write (token-level causal ablation) analysis of the SmolVLA action expert
at one early physical state, for the parent R0 and the adapted FT4 (SCENE2 book -> right).
"""
import argparse
import json
import sys
import time
from pathlib import Path

K1 = Path("/home/lzx/cl_binding/.server/k1_screen_20261101_01")
OUT = Path("/home/lzx/cl_binding/.server/k4_text_attention_20261102_01")
sys.path.insert(0, str(K1 / "code"))
sys.path.insert(0, str(OUT / "code"))

import k1_main as KM
from stage2k_entry_handoff import _snapshot, _restore_snapshot

EFF = 7
INSTR = {
    "A": "pick up the book and place it in the left compartment of the caddy",
    "B": "pick up the book and place it in the front compartment of the caddy",
    "N": "pick up the book and place it in the right compartment of the caddy",
    "S": "pick up the book and place it in the back compartment of the caddy",
    "U": "the weather is sunny today",
}
SLOT_WORDS = ("left", "front", "right", "back")


def jdump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n")


class Tap:
    def __init__(self, policy, torch):
        self.torch = torch
        self.expert = policy.model.vlm_with_expert
        self.original_cross = self.expert.forward_cross_attn_layer
        self.original_eager = self.expert.eager_attention_forward
        layers = self.expert.get_model_layers(
            [self.expert.get_vlm_model().text_model, self.expert.lm_expert])
        self.rows = []
        for index, layer in enumerate(layers[1]):
            if layer is None:
                continue
            expert_index = next(i for i, c in enumerate(self.expert.lm_expert.layers) if c is layer)
            kind = ("SA" if "cross" not in self.expert.attention_mode
                    or (self.expert.self_attn_every_n_layers > 0
                        and index % self.expert.self_attn_every_n_layers == 0) else "CA")
            self.rows.append({"vlm_layer": index, "expert_layer": expert_index, "kind": kind,
                              "module": layer})
        self.ca = [r["vlm_layer"] for r in self.rows if r["kind"] == "CA"]
        self.by_layer = {r["vlm_layer"]: r["module"] for r in self.rows}
        self.active = False
        self.record = False
        self.edit = None
        self.current = None
        self.in_cross = False
        self.group = None
        self.slot_pos = None
        self.mass = {}
        self.M = {}
        self.counts = {"cross": 0, "active_cross": 0, "record": 0}
        self.handles = [self.by_layer[i].self_attn.o_proj.register_forward_hook(self._o(i)) for i in self.ca]
        self.expert.forward_cross_attn_layer = self._cross
        self.expert.eager_attention_forward = self._eager

    def _o(self, layer):
        def hook(module, inputs, output):
            if self.active and self.record:
                self.M[layer] = output.detach().float().cpu()
            return None
        return hook

    def _cross(self, *args, **kwargs):
        layer = kwargs.get("layer_idx", args[2] if len(args) > 2 else None)
        embeds = kwargs.get("inputs_embeds", args[1] if len(args) > 1 else None)
        is_cross = isinstance(embeds, (list, tuple)) and len(embeds) == 2 and embeds[0] is None
        previous = self.current, self.in_cross
        self.current, self.in_cross = layer, is_cross
        self.counts["cross"] += int(is_cross)
        self.counts["active_cross"] += int(is_cross and self.active)
        self.counts["record"] += int(is_cross and self.active and self.record)
        try:
            if is_cross and self.active and self.edit is not None:
                layers = self.edit["layers"]
                if layers == "all" or layer in layers:
                    mask = kwargs.get("mask", args[4] if len(args) > 4 else None)
                    if mask is None:
                        raise RuntimeError("No attention mask at cross-attention layer")
                    new = mask.clone()
                    for pos in self.edit["positions"]:
                        new[..., pos] = False
                    if "mask" in kwargs:
                        kwargs = dict(kwargs)
                        kwargs["mask"] = new
                    else:
                        args = list(args)
                        args[4] = new
                        args = tuple(args)
            return self.original_cross(*args, **kwargs)
        finally:
            self.current, self.in_cross = previous

    def _eager(self, mask, batch_size, head_dim, query, key, value):
        output = self.original_eager(mask, batch_size, head_dim, query, key, value)
        if self.active and self.record and self.in_cross and self.group is not None:
            torch = self.torch
            groups = self.expert.num_attention_heads // self.expert.num_key_value_heads
            expanded = key[:, :, :, None, :].expand(
                batch_size, key.shape[1], self.expert.num_key_value_heads, groups, head_dim)
            expanded = expanded.reshape(batch_size, key.shape[1], -1, head_dim)
            scores = torch.matmul(query.float().transpose(1, 2),
                                  expanded.float().transpose(1, 2).transpose(2, 3))
            scores *= head_dim ** -0.5
            scores = scores.to(dtype=torch.float32)
            scores = torch.where(mask[:, None, :, :], scores, torch.finfo(scores.dtype).min)
            p = scores.softmax(-1)
            g = self.group
            image = p[..., g["image"][0]:g["image"][1]].sum(-1)
            text = p[..., g["text"][0]:g["text"][1]].sum(-1)
            state = p[..., g["state"][0]:g["state"][1]].sum(-1)
            slot = p[..., self.slot_pos] if self.slot_pos is not None else torch.zeros_like(image)
            self.mass[self.current] = {
                "image": float(image.mean()), "text": float(text.mean()), "state": float(state.mean()),
                "slot": float(slot.mean()),
                "slot_by_head": [round(float(x), 6) for x in slot.mean(dim=(0, 2))],
                "text_share": float(text.mean() / max(float(text.mean()) + float(image.mean()), 1e-9)),
            }
        return output

    def begin(self, edit=None, group=None, slot_pos=None):
        self.active = True
        self.edit = edit
        self.group = group
        self.slot_pos = slot_pos
        self.record = False
        self.mass = {}
        self.M = {}

    def end(self):
        self.active = False
        self.record = False
        self.edit = None
        return self.mass, self.M

    def close(self):
        self.expert.forward_cross_attn_layer = self.original_cross
        self.expert.eager_attention_forward = self.original_eager
        for h in self.handles:
            h.remove()


def token_spans(policy, text):
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
    ids = tk(text, return_tensors="pt").input_ids[0].tolist()
    pieces = [tk.decode([i]) for i in ids]
    slot = None
    for i, piece in enumerate(pieces):
        if piece.strip().lower() in SLOT_WORDS:
            slot = i
            break
    return ids, pieces, slot


def flow(rt, role, obs, text, seed, r, tap, edit=None, group=None, slot_pos=None):
    pre = rt.prefix(role, obs, text)
    pad, cache, groups = pre
    x0 = rt.noise(seed, r)
    calls = {"n": 0}
    tap.begin(edit, group, slot_pos)

    def velocity(x_t, t):
        tap.record = calls["n"] == 0
        calls["n"] += 1
        out = rt.query(role, pre, x_t, t)
        tap.record = False
        return out

    try:
        chunk = rt.integrate(velocity, x0, 10)[..., :EFF]
    finally:
        tap.end()
    return chunk, groups


def physical(rt, chunk):
    return rt.post[sorted(rt.post)[0]](chunk.clone()).float().cpu().numpy()[0]


def run_state(ctx, rt, init, roll_steps):
    seed = KM.noise_seed("scene2", init)
    obs, _ = ctx.reset(init)
    texts = {"A": INSTR["A"], "B": INSTR["A"]}
    with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
        for r in range(roll_steps // 10):
            chunk, _ = rt.sample("NAT-R0", "A", obs, texts, seed, r)
            controls = rt.physical(chunk)
            for k in range(10):
                obs, _ = ctx.step(controls[k])
                if (r * 10 + k + 1) % 10 == 0:
                    obs = ctx.fresh(ctx.env)
    arrays, meta = _snapshot(ctx.env, obs)
    return obs, arrays, meta, seed


def main(args):
    P = json.loads(Path("/home/lzx/cl_binding/.server/k3_e2_20261102_01/prereg/k3_e2_prereg.json").read_text())
    pair = {"R0": P["families"]["scene2"]["R0"], "FT4": P["families"]["scene2"]["FT4"]}
    ctx = KM.Ctx("scene2")
    rtR = KM.make_runtime(ctx, pair, ["R"])
    rtF = KM.make_runtime(ctx, pair, ["F"])
    rts = {"R0": rtR, "FT4": rtF}
    roles = {"R0": "R", "FT4": "F"}
    taps = {m: Tap(rts[m].models[roles[m]], ctx.torch) for m in rts}
    report = {"roll_steps": args.roll, "instructions": INSTR,
              "ca_layers": taps["FT4"].ca, "parts": {}}
    started = time.monotonic()

    for init in args.inits:
        obs, arrays, meta, seed = run_state(ctx, rtR, init, args.roll)
        jdump(OUT / "runs" / ("state_%d.json" % init),
              {"init": init, "signature": ctx.signature(obs), "steps": args.roll, "seed": seed})

        att, base_chunks = {}, {}
        for model in ("R0", "FT4"):
            rt, role, tap = rts[model], roles[model], taps[model]
            for key, text in INSTR.items():
                _, pieces, slot = token_spans(rt.models[role], text)
                with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                    chunk, groups = flow(rt, role, obs, text, seed, 0, tap)
                # second pass with the group and slot positions, to record attention masses
                slot_pos = None if slot is None else groups["text"][0] + slot
                with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                    chunk2, _ = flow(rt, role, obs, text, seed, 0, tap, group=groups, slot_pos=slot_pos)
                att["%s|%s" % (model, key)] = {
                    "slot_word": None if slot is None else pieces[slot], "slot_token_index": slot,
                    "text_tokens": len(pieces), "mass": tap.mass, "repeat_max_abs": float((chunk - chunk2).abs().max())}
                base_chunks["%s|%s" % (model, key)] = chunk
        report["parts"].setdefault("attention", {})[str(init)] = att

        abl = {}
        for model in ("R0", "FT4"):
            rt, role, tap = rts[model], roles[model], taps[model]
            for key in ("A", "B", "N", "S"):
                _, _, slot = token_spans(rt.models[role], INSTR[key])
                with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                    chunk, groups = flow(rt, role, obs, INSTR[key], seed, 0, tap)
                base_phys = physical(rt, chunk)
                slot_pos = groups["text"][0] + slot
                rows = {}
                edits = {
                    "slot_all": {"layers": "all", "positions": [slot_pos]},
                    "text_all": {"layers": "all", "positions": list(range(groups["text"][0], groups["text"][1]))},
                    "image_all": {"layers": "all", "positions": list(range(groups["image"][0], groups["image"][1]))},
                }
                for name, edit in edits.items():
                    with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                        c2, _ = flow(rt, role, obs, INSTR[key], seed, 0, tap, edit=edit)
                    phys = physical(rt, c2)
                    rows[name] = {"first10_max_abs": float((c2[:, :10] - chunk[:, :10]).abs().max()),
                                  "physical_first10_l2": float((((phys[:10] - base_phys[:10]) ** 2).sum()) ** 0.5)}
                for layer in tap.ca:
                    with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                        c3, _ = flow(rt, role, obs, INSTR[key], seed, 0, tap, edit={"layers": {layer}, "positions": [slot_pos]})
                    phys = physical(rt, c3)
                    rows["slot_layer_%d" % layer] = {
                        "first10_max_abs": float((c3[:, :10] - chunk[:, :10]).abs().max()),
                        "physical_first10_l2": float((((phys[:10] - base_phys[:10]) ** 2).sum()) ** 0.5)}
                abl["%s|%s" % (model, key)] = rows
        report["parts"].setdefault("ablation_open", {})[str(init)] = abl

        dist = {}
        for model in ("R0", "FT4"):
            rt, role, tap = rts[model], roles[model], taps[model]
            msgs = {}
            for key, text in INSTR.items():
                with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                    flow(rt, role, obs, text, seed, 0, tap)
                msgs[key] = {k: v.mean(dim=1)[0].double().numpy() for k, v in tap.M.items()}
            pairs = {}
            for a, b in (("A", "B"), ("A", "N"), ("N", "S"), ("A", "U")):
                per_layer = {}
                for layer in tap.ca:
                    va, vb = msgs[a][layer], msgs[b][layer]
                    denom = max(float(((va ** 2).sum()) ** 0.5), 1e-9)
                    per_layer[str(layer)] = round(float((((va - vb) ** 2).sum()) ** 0.5) / denom, 4)
                pairs["%s_vs_%s" % (a, b)] = per_layer
            dist[model] = pairs
        report["parts"].setdefault("message_distance", {})[str(init)] = dist
        report["seconds"] = time.monotonic() - started
        jdump(OUT / "runs" / ("part123_%d.json" % init), report["parts"])

        if args.closed:
            loop = {}
            conditions = [("FT4", "N", "slot_all"), ("FT4", "N", "text_all"), ("FT4", "N", None),
                          ("FT4", "U", None), ("FT4", "A", None), ("FT4", "S", None),
                          ("R0", "A", "slot_all"), ("R0", "A", None),
                          ("R0", "B", "slot_all"), ("R0", "B", None)]
            for model, key, how in conditions:
                rt, role, tap = rts[model], roles[model], taps[model]
                _, _, slot = token_spans(rt.models[role], INSTR[key])
                obs2, _ = _restore_snapshot(ctx.env, ctx.xml, arrays, meta, seed)
                with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                    _, groups = flow(rt, role, obs2, INSTR[key], seed, args.roll // 10, tap)
                if how == "slot_all":
                    edit = {"layers": "all", "positions": [groups["text"][0] + slot]}
                elif how == "text_all":
                    edit = {"layers": "all", "positions": list(range(groups["text"][0], groups["text"][1]))}
                else:
                    edit = None
                obs2, _ = _restore_snapshot(ctx.env, ctx.xml, arrays, meta, seed)
                rt.models[role].reset()
                rows = []
                tap.begin(edit)
                try:
                    with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                        for r in range(args.roll // 10, 40):
                            chunk, _ = rt.sample({"R": "NAT-R0", "F": "NAT-FT4"}[role], "A", obs2,
                                                 {"A": INSTR[key], "B": INSTR[key]}, seed, r)
                            controls = rt.physical(chunk)
                            for k in range(10):
                                step = r * 10 + k + 1
                                obs2, _ = ctx.step(controls[k])
                                if step % 10 == 0:
                                    obs2 = ctx.fresh(ctx.env)
                                rows.append(ctx.row(step, obs2, controls[k], controls[k]))
                finally:
                    tap.end()
                z0 = rows[0]["obj"][2]
                mv = ctx.geo.movement("scene2", ctx.geometry, rows, z0, ctx.plate_xy(obs2))
                rec = {"first_targets": mv["first_targets"], "first_step": mv["first_step"],
                       "success": {g: next((x["step"] for x in rows if x["pred"][g]), None)
                                   for g in ctx.geo.GOALS}}
                loop["%s|%s|%s" % (model, key, how or "none")] = rec
                print(json.dumps({"closed": "%s|%s|%s" % (model, key, how), **rec}), flush=True)
            report["parts"].setdefault("closed_loop", {})[str(init)] = loop
            jdump(OUT / "runs" / ("part1234_%d.json" % init), report["parts"])

    report["seconds"] = time.monotonic() - started
    jdump(OUT / "runs" / "report.json", report)
    print(json.dumps({"done": True, "parts": list(report["parts"]), "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inits", nargs="+", type=int, default=[4])
    parser.add_argument("--roll", type=int, default=40)
    parser.add_argument("--closed", action="store_true")
    main(parser.parse_args())