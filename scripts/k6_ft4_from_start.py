"""K6 (exploratory): same interventions but with FT4 controlling from step 0 -- no R0 prefix.

NAT-FT4 / NAT-R0 baselines, CA text mask, slot-word mask, and attention-block swaps, all from
the reset state to step 400, both old instructions.
"""
import json
import sys
import time
from pathlib import Path

K1 = Path("/home/lzx/cl_binding/.server/k1_screen_20261101_01")
K4 = Path("/home/lzx/cl_binding/.server/k4_text_attention_20261102_01")
K5 = Path("/home/lzx/cl_binding/.server/k5_attention_combination_20261104_01")
OUT = Path("/home/lzx/cl_binding/.server/k6_ft4_from_start_20261104_01")
sys.path.insert(0, str(K1 / "code"))
sys.path.insert(0, str(K4 / "code"))
sys.path.insert(0, str(K5 / "code"))

import k1_main as KM
import k4_text_attention as K4M
import k5_combination as K5M

CA = [1, 3, 5, 7, 9, 11, 13, 15]
CONDS = ["NAT-FT4", "NAT-R0", "maskTEXT", "maskSLOT", "swapCA", "swapSEG"]


def run_episode(ctx, rt, role, init, text, tap=None, mask=None, swap=None):
    obs, sha0 = ctx.reset(init)
    rt.models[role].reset()
    with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
        _, groups = K4M.flow(rt, role, obs, text, 0, 0, tap) if tap is not None else (None, None)
    rows = []
    saved = []
    if swap is not None:
        saved = K5M.apply_swap(rt.models[role], swap[0], swap[1])
    if tap is not None:
        tap.begin(mask)
    try:
        with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
            for r in range(40):
                chunk, _ = rt.sample("NAT-R0" if role == "R" else "NAT-FT4", "A", obs,
                                     {"A": text, "B": text}, 0, r)
                controls = rt.physical(chunk)
                for k in range(10):
                    step = r * 10 + k + 1
                    obs, _ = ctx.step(controls[k])
                    if step % 10 == 0:
                        obs = ctx.fresh(ctx.env)
                    rows.append(ctx.row(step, obs, controls[k], controls[k]))
    finally:
        if tap is not None:
            tap.end()
        if saved:
            K5M.undo_swap(saved)
    z0 = rows[0]["obj"][2]
    mv = ctx.geo.movement("scene2", ctx.geometry, rows, z0, ctx.plate_xy(obs))
    return {"first_targets": mv["first_targets"], "first_step": mv["first_step"],
            "success": {g: next((x["step"] for x in rows if x["pred"][g]), None) for g in ctx.geo.GOALS},
            "first_close": next((x["step"] for x in rows if x["a"][6] > 0), None)}


def main():
    P = json.loads(Path("/home/lzx/cl_binding/.server/k3_e2_20261102_01/prereg/k3_e2_prereg.json").read_text())
    pair = {"R0": P["families"]["scene2"]["R0"], "FT4": P["families"]["scene2"]["FT4"]}
    ctx = KM.Ctx("scene2")
    rtR = KM.make_runtime(ctx, pair, ["R"])
    rtF = KM.make_runtime(ctx, pair, ["F"])
    tap = K4M.Tap(rtF.models["F"], ctx.torch)
    report = {"note": "exploratory follow-up; FT4 controls from step 0, no R0 prefix", "states": {}}
    started = time.monotonic()
    for init in (4, 6, 8):
        state = {"runs": {}}
        for text_key in ("A", "B"):
            text = K4M.INSTR[text_key]
            probe, _ = ctx.reset(init)
            with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                _, groups = K4M.flow(rtF, "F", probe, text, 0, 0, tap)
            _, _, slot = K4M.token_spans(rtF.models["F"], text)
            slot_pos = groups["text"][0] + slot
            runs = {
                "NAT-FT4": (rtF, "F", None, None),
                "NAT-R0": (rtR, "R", None, None),
                "maskTEXT": (rtF, "F", tap, {"layers": "all", "positions": list(range(groups["text"][0], groups["text"][1]))}),
                "maskSLOT": (rtF, "F", tap, {"layers": "all", "positions": [slot_pos]}),
                "swapCA": (rtF, "F", None, None),
                "swapSEG": (rtF, "F", None, None),
            }
            for name, (rt, role, tp, mask) in runs.items():
                swap = None
                if name == "swapCA":
                    swap = (rtR.models["R"], CA)
                elif name == "swapSEG":
                    swap = (rtR.models["R"], list(range(16)))
                res = run_episode(ctx, rt, role, init, text, tap=tp, mask=mask, swap=swap)
                state["runs"]["%s|%s" % (name, text_key)] = res
                print(json.dumps({"init": init, "cond": name, "instr": text_key, "first": res["first_targets"],
                                  "succ": {g: v for g, v in res["success"].items() if v}}), flush=True)
        state["D"] = {name: (state["runs"][name + "|A"]["first_targets"] == ["left"]
                             and state["runs"][name + "|B"]["first_targets"] == ["front"]) for name in CONDS}
        report["states"][str(init)] = state
        print(json.dumps({"init": init, "D": state["D"]}, ensure_ascii=False), flush=True)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "runs" / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print("K6_COMPLETE", round(time.monotonic() - started, 1), flush=True)


if __name__ == "__main__":
    main()
