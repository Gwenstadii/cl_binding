"""K5b control: same layer groups but swap only the attention projections (keep the receiver's
input layernorm). Tests whether the hybrid's off-distribution layernorm explains the lack of rescue."""
import json
import sys
from pathlib import Path

K1 = Path("/home/lzx/cl_binding/.server/k1_screen_20261101_01")
K4 = Path("/home/lzx/cl_binding/.server/k4_text_attention_20261102_01")
K5 = Path("/home/lzx/cl_binding/.server/k5_attention_combination_20261104_01")
sys.path.insert(0, str(K1 / "code"))
sys.path.insert(0, str(K4 / "code"))
sys.path.insert(0, str(K5 / "code"))

import k1_main as KM
import k4_text_attention as K4M
import k5_combination as K5M
from stage2k_entry_handoff import _restore_snapshot

CA = [1, 3, 5, 7, 9, 11, 13, 15]
CONDS = [("CA-ALL-proj", CA), ("SEG-ALL-proj", list(range(16)))]


def swap_proj_only(receiver, donor, layers):
    er, ed = K5M.expert(receiver), K5M.expert(donor)
    saved = []
    for i in layers:
        lr, ld = er.layers[i], ed.layers[i]
        saved.append((lr, lr.self_attn))
        lr.self_attn = ld.self_attn
    return saved


def main():
    P = json.loads(Path("/home/lzx/cl_binding/.server/k3_e2_20261102_01/prereg/k3_e2_prereg.json").read_text())
    pair = {"R0": P["families"]["scene2"]["R0"], "FT4": P["families"]["scene2"]["FT4"]}
    ctx = KM.Ctx("scene2")
    rtR = KM.make_runtime(ctx, pair, ["R"])
    rtF = KM.make_runtime(ctx, pair, ["F"])
    out = {}
    for init in (4, 6, 8):
        obs, arrays, meta, seed = K4M.run_state(ctx, rtR, init, 40)
        for name, layers in CONDS:
            for key in ("A", "B"):
                text = K4M.INSTR[key]
                obs2, _ = _restore_snapshot(ctx.env, ctx.xml, arrays, meta, seed)
                rtF.models["F"].reset()
                saved = swap_proj_only(rtF.models["F"], rtR.models["R"], layers)
                rows = []
                try:
                    with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                        for r in range(4, 40):
                            chunk, _ = rtF.sample("NAT-FT4", "A", obs2, {"A": text, "B": text}, seed, r)
                            controls = rtF.physical(chunk)
                            for k in range(10):
                                step = r * 10 + k + 1
                                obs2, _ = ctx.step(controls[k])
                                if step % 10 == 0:
                                    obs2 = ctx.fresh(ctx.env)
                                rows.append(ctx.row(step, obs2, controls[k], controls[k]))
                finally:
                    for layer, attn in saved:
                        layer.self_attn = attn
                z0 = rows[0]["obj"][2]
                mv = ctx.geo.movement("scene2", ctx.geometry, rows, z0, ctx.plate_xy(obs2))
                rec = {"first_targets": mv["first_targets"], "first_step": mv["first_step"],
                       "success": {g: next((x["step"] for x in rows if x["pred"][g]), None) for g in ctx.geo.GOALS}}
                out["%d|%s|%s" % (init, name, key)] = rec
                print(json.dumps({"init": init, "cond": name, "instr": key, "first": rec["first_targets"]}), flush=True)
    (K5 / "runs" / "variant_proj_only.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()