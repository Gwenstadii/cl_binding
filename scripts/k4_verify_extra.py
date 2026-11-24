"""Two extra closed-loop controls: text+image blocked; unrelated text with text attention blocked."""
import json
import sys
from pathlib import Path

K1 = Path("/home/lzx/cl_binding/.server/k1_screen_20261101_01")
OUT = Path("/home/lzx/cl_binding/.server/k4_text_attention_20261102_01")
sys.path.insert(0, str(K1 / "code"))
sys.path.insert(0, str(OUT / "code"))

import k1_main as KM
import k4_text_attention as K4
from stage2k_entry_handoff import _restore_snapshot


def main():
    P = json.loads(Path("/home/lzx/cl_binding/.server/k3_e2_20261102_01/prereg/k3_e2_prereg.json").read_text())
    pair = {"R0": P["families"]["scene2"]["R0"], "FT4": P["families"]["scene2"]["FT4"]}
    ctx = KM.Ctx("scene2")
    rtR = KM.make_runtime(ctx, pair, ["R"])
    rtF = KM.make_runtime(ctx, pair, ["F"])
    rt = rtF
    role = "F"
    tap = K4.Tap(rt.models[role], ctx.torch)
    obs, arrays, meta, seed = K4.run_state(ctx, rtR, 4, 40)
    loop = {}
    for key, how in (("N", "text_image"), ("U", "text_only")):
        obs2, _ = _restore_snapshot(ctx.env, ctx.xml, arrays, meta, seed)
        with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
            _, groups = K4.flow(rt, role, obs2, K4.INSTR[key], seed, 4, tap)
        if how == "text_image":
            pos = list(range(groups["image"][0], groups["image"][1])) + list(range(groups["text"][0], groups["text"][1]))
        else:
            pos = list(range(groups["text"][0], groups["text"][1]))
        edit = {"layers": "all", "positions": pos}
        obs2, _ = _restore_snapshot(ctx.env, ctx.xml, arrays, meta, seed)
        rt.models[role].reset()
        rows = []
        tap.begin(edit)
        try:
            with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                for r in range(4, 40):
                    chunk, _ = rt.sample("NAT-FT4", "A", obs2, {"A": K4.INSTR[key], "B": K4.INSTR[key]}, seed, r)
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
        loop["%s|%s" % (key, how)] = {"first_targets": mv["first_targets"], "first_step": mv["first_step"],
                                      "success": {g: next((x["step"] for x in rows if x["pred"][g]), None)
                                                  for g in ctx.geo.GOALS}}
        print(json.dumps({"closed": "%s|%s" % (key, how), **loop["%s|%s" % (key, how)]}), flush=True)
    (OUT / "runs" / "verify2.json").write_text(json.dumps(loop, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
