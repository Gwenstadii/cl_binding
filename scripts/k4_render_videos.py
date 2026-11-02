"""Render the K4 masking counterfactuals as videos.

Same 40-step state, same instruction (old LEFT), same noise; only the attention mask changes:
  nomask     nothing masked
  maskTEXT   every text token blocked in all 8 cross-attention layers
  maskSLOT   only the slot word ("left") blocked in all 8 cross-attention layers
400 native steps, agentview camera, 20 fps, outcome label burned in.
"""
import json
import sys
from pathlib import Path

K1 = Path("/home/lzx/cl_binding/.server/k1_screen_20261101_01")
K4 = Path("/home/lzx/cl_binding/.server/k4_text_attention_20261102_01")
OUT = Path("/home/lzx/cl_binding/video/2026-11-02_K4_文字屏蔽对照")
sys.path.insert(0, str(K1 / "code"))
sys.path.insert(0, str(K4 / "code"))

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw
import k1_main as KM
import k4_text_attention as K4M
from stage2k_entry_handoff import _restore_snapshot

CASES = [("FT4", 4), ("FT4", 6), ("R0", 4), ("R0", 6)]
VARIANTS = ["nomask", "maskTEXT", "maskSLOT"]


def label(frame, lines):
    img = Image.fromarray(frame).resize((384, 384), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 384, 20 * len(lines) + 4], fill=(0, 0, 0))
    for i, text in enumerate(lines):
        draw.text((4, 3 + 20 * i), text, fill=(255, 255, 255))
    return np.asarray(img)


def main():
    P = json.loads(Path("/home/lzx/cl_binding/.server/k3_e2_20261102_01/prereg/k3_e2_prereg.json").read_text())
    pair = {"R0": P["families"]["scene2"]["R0"], "FT4": P["families"]["scene2"]["FT4"]}
    ctx = KM.Ctx("scene2")
    rtR = KM.make_runtime(ctx, pair, ["R"])
    rtF = KM.make_runtime(ctx, pair, ["F"])
    rts = {"R0": rtR, "FT4": rtF}
    roles = {"R0": "R", "FT4": "F"}
    taps = {m: K4M.Tap(rts[m].models[roles[m]], ctx.torch) for m in rts}
    OUT.mkdir(parents=True, exist_ok=True)
    states = {}
    manifest = {"instruction": K4M.INSTR["A"], "fps": 20, "steps": 400, "state_rule":
                "R0 rolls 40 native steps from the development initial state with instruction LEFT",
                "variants": {"nomask": "no mask", "maskTEXT": "all text tokens blocked in all 8 CA layers",
                             "maskSLOT": "only the slot word 'left' blocked in all 8 CA layers"},
                "videos": []}
    for init in (4, 6):
        obs, arrays, meta, seed = K4M.run_state(ctx, rtR, init, 40)
        states[init] = (arrays, meta, seed)

    for model, init in CASES:
        arrays, meta, seed = states[init]
        rt, role, tap = rts[model], roles[model], taps[model]
        _, _, slot = K4M.token_spans(rt.models[role], K4M.INSTR["A"])
        for variant in VARIANTS:
            obs2, _ = _restore_snapshot(ctx.env, ctx.xml, arrays, meta, seed)
            with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                _, groups = K4M.flow(rt, role, obs2, K4M.INSTR["A"], seed, 4, tap)
            if variant == "maskTEXT":
                edit = {"layers": "all", "positions": list(range(groups["text"][0], groups["text"][1]))}
            elif variant == "maskSLOT":
                edit = {"layers": "all", "positions": [groups["text"][0] + slot]}
            else:
                edit = None
            obs2, _ = _restore_snapshot(ctx.env, ctx.xml, arrays, meta, seed)
            rt.models[role].reset()
            frames, rows = [], []
            tap.begin(edit)
            try:
                with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                    for r in range(4, 40):
                        chunk, _ = rt.sample({"R": "NAT-R0", "F": "NAT-FT4"}[role], "A", obs2,
                                             {"A": K4M.INSTR["A"], "B": K4M.INSTR["A"]}, seed, r)
                        controls = rt.physical(chunk)
                        for k in range(10):
                            step = r * 10 + k + 1
                            obs2, _ = ctx.step(controls[k])
                            if step % 10 == 0:
                                obs2 = ctx.fresh(ctx.env)
                            rows.append(ctx.row(step, obs2, controls[k], controls[k]))
                            frames.append(np.ascontiguousarray(obs2["agentview_image"][::-1, ::-1]))
            finally:
                tap.end()
            z0 = rows[0]["obj"][2]
            mv = ctx.geo.movement("scene2", ctx.geometry, rows, z0, ctx.plate_xy(obs2))
            first = ",".join(mv["first_targets"]) or "none"
            succ = next((x["step"] for x in rows if x["pred"]["right"]), None)
            succ_left = next((x["step"] for x in rows if x["pred"]["left"]), None)
            name = "%s_init%02d_instrLEFT_%s.mp4" % (model, init, variant)
            head = ["%s init%d  instruction: LEFT (old goal)" % (model, init),
                    "mask: %s" % ("NONE" if variant == "nomask" else ("ALL TEXT" if variant == "maskTEXT" else "WORD 'left'")),
                    "first goal reached: %s   right@%s  left@%s" % (first, succ, succ_left)]
            labs = [label(f, head) for f in frames]
            imageio.mimsave(str(OUT / name), labs, fps=20, macro_block_size=None)
            manifest["videos"].append({"file": name, "model": model, "init": init, "variant": variant,
                                       "first_targets": mv["first_targets"], "first_step": mv["first_step"],
                                       "right_success_step": succ, "left_success_step": succ_left,
                                       "seconds": 20.0})
            print(json.dumps({"file": name, "first": first, "right": succ, "left": succ_left}), flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
