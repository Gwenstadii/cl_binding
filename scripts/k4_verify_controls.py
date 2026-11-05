"""K4 verification: does the text-token mask really remove text attention, and what do
token-level controls / image blocking do in closed loop?"""
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
    rts = {"R0": rtR, "FT4": rtF}
    roles = {"R0": "R", "FT4": "F"}
    taps = {m: K4.Tap(rts[m].models[roles[m]], ctx.torch) for m in rts}
    report = {}

    obs, arrays, meta, seed = K4.run_state(ctx, rtR, 4, 40)

    # 1) mask really removes text attention
    rt, role, tap = rts["FT4"], roles["FT4"], taps["FT4"]
    _, _, slot = K4.token_spans(rt.models[role], K4.INSTR["N"])
    with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
        chunk, groups = K4.flow(rt, role, obs, K4.INSTR["N"], seed, 0, tap)
    text_edit = {"layers": "all", "positions": list(range(groups["text"][0], groups["text"][1]))}
    with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
        c2, _ = K4.flow(rt, role, obs, K4.INSTR["N"], seed, 0, tap, edit=text_edit,
                        group=groups, slot_pos=groups["text"][0] + slot)
    report["text_mask_check"] = {
        "text_mass_after_block": {str(k): v["text"] for k, v in tap.mass.items()},
        "image_mass_after_block": {str(k): v["image"] for k, v in tap.mass.items()},
        "chunk_change": float((c2[:, :10] - chunk[:, :10]).abs().max()),
    }

    # 2) control: block one non-slot word ("the", token index 2) at all layers, open loop
    controls = {}
    for model, key in (("FT4", "N"), ("R0", "A")):
        rt, role, tap = rts[model], roles[model], taps[model]
        ids, pieces, slot = K4.token_spans(rt.models[role], K4.INSTR[key])
        with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
            base, groups = K4.flow(rt, role, obs, K4.INSTR[key], seed, 0, tap)
        base_phys = K4.physical(rt, base)
        for name, edit in (("the", {"layers": "all", "positions": [groups["text"][0] + 2]}),
                           ("slot", {"layers": "all", "positions": [groups["text"][0] + slot]})):
            with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                c, _ = K4.flow(rt, role, obs, K4.INSTR[key], seed, 0, tap, edit=edit)
            phys = K4.physical(rt, c)
            controls["%s|%s|%s" % (model, key, name)] = float((((phys[:10] - base_phys[:10]) ** 2).sum()) ** 0.5)
    report["token_controls_open"] = controls

    # 3) closed loop controls on the same state
    loop = {}
    conditions = [("FT4", "N", "image_all"), ("FT4", "N", "the_all"), ("R0", "A", "the_all"),
                  ("FT4", "N", "slot_layer_13"), ("FT4", "N", "slot_layer_7")]
    for model, key, how in conditions:
        rt, role, tap = rts[model], roles[model], taps[model]
        _, _, slot = K4.token_spans(rt.models[role], K4.INSTR[key])
        obs2, _ = _restore_snapshot(ctx.env, ctx.xml, arrays, meta, seed)
        with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
            _, groups = K4.flow(rt, role, obs2, K4.INSTR[key], seed, 4, tap)
        if how == "image_all":
            edit = {"layers": "all", "positions": list(range(groups["image"][0], groups["image"][1]))}
        elif how == "the_all":
            edit = {"layers": "all", "positions": [groups["text"][0] + 2]}
        else:
            edit = {"layers": {int(how.split("_")[-1])}, "positions": [groups["text"][0] + slot]}
        obs2, _ = _restore_snapshot(ctx.env, ctx.xml, arrays, meta, seed)
        rt.models[role].reset()
        rows = []
        tap.begin(edit)
        try:
            with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                for r in range(4, 40):
                    chunk, _ = rt.sample({"R": "NAT-R0", "F": "NAT-FT4"}[role], "A", obs2,
                                         {"A": K4.INSTR[key], "B": K4.INSTR[key]}, seed, r)
                    controls_c = rt.physical(chunk)
                    for k in range(10):
                        step = r * 10 + k + 1
                        obs2, _ = ctx.step(controls_c[k])
                        if step % 10 == 0:
                            obs2 = ctx.fresh(ctx.env)
                        rows.append(ctx.row(step, obs2, controls_c[k], controls_c[k]))
        finally:
            tap.end()
        z0 = rows[0]["obj"][2]
        mv = ctx.geo.movement("scene2", ctx.geometry, rows, z0, ctx.plate_xy(obs2))
        loop["%s|%s|%s" % (model, key, how)] = {
            "first_targets": mv["first_targets"], "first_step": mv["first_step"],
            "success": {g: next((x["step"] for x in rows if x["pred"][g]), None) for g in ctx.geo.GOALS}}
        print(json.dumps({"closed": "%s|%s|%s" % (model, key, how), **loop["%s|%s|%s" % (model, key, how)]}), flush=True)
    report["closed_loop_controls"] = loop
    (OUT / "runs" / "verify.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(report, ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()
