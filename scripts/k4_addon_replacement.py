"""K4 add-on: the replacement cell -- old LEFT instruction, language masked.

Same state, same instruction A (old left), same noise; only the mask changes.
FT4 with A is the replacement condition (it goes to the new right goal), so this asks whether
that replacement still happens when the action head cannot read the text tokens / the slot word.
R0 with A is the parent control.
"""
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

CONDITIONS = [("FT4", "A", "text_all"), ("FT4", "A", "slot_all"), ("R0", "A", "text_all")]


def main():
    P = json.loads(Path("/home/lzx/cl_binding/.server/k3_e2_20261102_01/prereg/k3_e2_prereg.json").read_text())
    pair = {"R0": P["families"]["scene2"]["R0"], "FT4": P["families"]["scene2"]["FT4"]}
    ctx = KM.Ctx("scene2")
    rtR = KM.make_runtime(ctx, pair, ["R"])
    rtF = KM.make_runtime(ctx, pair, ["F"])
    rts = {"R0": rtR, "FT4": rtF}
    roles = {"R0": "R", "FT4": "F"}
    taps = {m: K4.Tap(rts[m].models[roles[m]], ctx.torch) for m in rts}
    out = {}
    for init in (4, 6):
        obs, arrays, meta, seed = K4.run_state(ctx, rtR, init, 40)
        for model, key, how in CONDITIONS:
            rt, role, tap = rts[model], roles[model], taps[model]
            _, _, slot = K4.token_spans(rt.models[role], K4.INSTR[key])
            obs2, _ = _restore_snapshot(ctx.env, ctx.xml, arrays, meta, seed)
            with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                _, groups = K4.flow(rt, role, obs2, K4.INSTR[key], seed, 4, tap)
            if how == "text_all":
                edit = {"layers": "all", "positions": list(range(groups["text"][0], groups["text"][1]))}
            else:
                edit = {"layers": "all", "positions": [groups["text"][0] + slot]}
            obs2, _ = _restore_snapshot(ctx.env, ctx.xml, arrays, meta, seed)
            rt.models[role].reset()
            rows = []
            tap.begin(edit)
            try:
                with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                    for r in range(4, 40):
                        chunk, _ = rt.sample({"R": "NAT-R0", "F": "NAT-FT4"}[role], "A", obs2,
                                             {"A": K4.INSTR[key], "B": K4.INSTR[key]}, seed, r)
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
                   "success": {g: next((x["step"] for x in rows if x["pred"][g]), None) for g in ctx.geo.GOALS}}
            out["i%d|%s|%s|%s" % (init, model, key, how)] = rec
            print(json.dumps({"case": "i%d|%s|%s|%s" % (init, model, key, how), **rec}), flush=True)
    (OUT / "runs" / "addon_replacement.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
