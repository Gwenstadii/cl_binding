"""Render the ALL-16 whole-layer transplant for the three instructions (init 4)."""
import json
import sys
from pathlib import Path

K1 = Path("/home/lzx/cl_binding/.server/k1_screen_20261101_01")
K4 = Path("/home/lzx/cl_binding/.server/k4_text_attention_20261102_01")
OUT = Path("/home/lzx/cl_binding/video/2026-11-04_K7_整层全换_三条指令")
sys.path.insert(0, str(K1 / "code"))
sys.path.insert(0, str(K4 / "code"))

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw
import k1_main as KM
import k4_text_attention as K4M

INIT = 4
KEYS = ["A", "B", "N"]


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
    OUT.mkdir(parents=True, exist_ok=True)
    er = rtF.models["F"].model.vlm_with_expert.lm_expert
    ed = rtR.models["R"].model.vlm_with_expert.lm_expert
    saved = [(i, er.layers[i]) for i in range(16)]
    for i, layer in saved:
        er.layers[i] = ed.layers[i]
    manifest = {"init": INIT, "swap": "all 16 expert layers from R0", "model": "FT4",
                "instruction": {}, "paths": {}}
    try:
        for key in KEYS:
            text = K4M.INSTR[key]
            obs, _ = ctx.reset(INIT)
            rtF.models["F"].reset()
            frames, obj, eef, preds = [], [], [], []
            seed = KM.noise_seed("scene2", INIT)
            with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                for r in range(40):
                    chunk, _ = rtF.sample("NAT-FT4", "A", obs, {"A": text, "B": text}, seed, r)
                    controls = rtF.physical(chunk)
                    for k in range(10):
                        step = r * 10 + k + 1
                        obs, _ = ctx.step(controls[k])
                        if step % 10 == 0:
                            obs = ctx.fresh(ctx.env)
                        frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                        obj.append([float(x) for x in obs["black_book_1_pos"]])
                        eef.append([float(x) for x in obs["robot0_eef_pos"]])
                        preds.append(ctx.flags())
            z0 = obj[0][2]
            first = None
            for s, p in enumerate(preds, start=1):
                hit = ctx.geo.hits("scene2", ctx.geometry, obj[s - 1], z0, None)
                if hit:
                    # 5-step rule
                    run = 0
                    for t in range(s, min(len(preds), s + 4)):
                        if ctx.geo.hits("scene2", ctx.geometry, obj[t], z0, None):
                            run += 1
                    if run >= 4:
                        first = {"step": s, "targets": hit}
                        break
            succ = {g: next((i + 1 for i, p in enumerate(preds) if p[g]), None) for g in ctx.geo.GOALS}
            head = ["ALL-16 layer swap | init %d" % INIT,
                    "instr: %s" % {"A": "LEFT", "B": "FRONT", "N": "RIGHT"}[key],
                    "first: %s  L@%s F@%s R@%s" % (
                        (first["targets"][0] if first else "none"), succ["left"], succ["front"], succ["right"])]
            name = "ALL16_init%02d_instr%s.mp4" % (INIT, {"A": "LEFT", "B": "FRONT", "N": "RIGHT"}[key])
            imageio.mimsave(str(OUT / name), [label(f, head) for f in frames], fps=20, macro_block_size=None)
            manifest["instruction"][key] = {"text": text, "file": name, "first": first, "success": succ}
            manifest["paths"][key] = {"obj": obj, "eef": eef}
            print(json.dumps({"key": key, "first": first, "succ": {k: v for k, v in succ.items() if v}}), flush=True)
    finally:
        for i, layer in saved:
            er.layers[i] = layer
    (OUT / "paths.json").write_text(json.dumps(manifest, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()