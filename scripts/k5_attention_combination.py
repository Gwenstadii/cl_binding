"""K5: attention-layer combination swap -- does restoring donor attention restore old-goal control?

FT4 is the receiver. Selected expert attention blocks (input_layernorm + self_attn) are replaced by
the donor's modules, so the donor attention is computed on the receiver's own hidden states.
Closed loop from a 40-step state, both old instructions, first-target and D.
"""
import json
import sys
import time
from pathlib import Path

K1 = Path("/home/lzx/cl_binding/.server/k1_screen_20261101_01")
K4 = Path("/home/lzx/cl_binding/.server/k4_text_attention_20261102_01")
OUT = Path("/home/lzx/cl_binding/.server/k5_attention_combination_20261104_01")
sys.path.insert(0, str(K1 / "code"))
sys.path.insert(0, str(K4 / "code"))

import k1_main as KM
import k4_text_attention as K4M
from stage2k_entry_handoff import _restore_snapshot

CA = [1, 3, 5, 7, 9, 11, 13, 15]
CONDITIONS = [
    ("NAT-FT4", None, "FT4"),
    ("NAT-R0", None, "R0"),
    ("SELF-ALL", ("FT4", list(range(16))), "FT4"),
    ("CA-shallow", ("R0", [1, 3, 5]), "FT4"),
    ("CA-mid", ("R0", [7, 9, 11]), "FT4"),
    ("CA-deep", ("R0", [13, 15]), "FT4"),
    ("CA-ALL", ("R0", CA), "FT4"),
    ("CA-mismatch", ("R0", list(zip(CA, list(reversed(CA))))), "FT4"),
    ("SEG-0-5", ("R0", list(range(0, 6))), "FT4"),
    ("SEG-6-10", ("R0", list(range(6, 11))), "FT4"),
    ("SEG-11-15", ("R0", list(range(11, 16))), "FT4"),
    ("SEG-ALL", ("R0", list(range(16))), "FT4"),
]


def expert(policy):
    return policy.model.vlm_with_expert.lm_expert


def apply_swap(receiver, donor, spec):
    """spec: list of layer indices, or list of (receiver_layer, donor_layer) pairs."""
    er, ed = expert(receiver), expert(donor)
    pairs = [(i, i) for i in spec] if all(isinstance(i, int) for i in spec) else list(spec)
    saved = []
    for ri, di in pairs:
        lr, ld = er.layers[ri], ed.layers[di]
        saved.append((lr, lr.self_attn, lr.input_layernorm))
        lr.self_attn = ld.self_attn
        lr.input_layernorm = ld.input_layernorm
    return saved


def undo_swap(saved):
    for layer, attn, norm in saved:
        layer.self_attn = attn
        layer.input_layernorm = norm


def run_continue(ctx, rt, role, arrays, meta, seed, text, swap=None):
    obs, _ = _restore_snapshot(ctx.env, ctx.xml, arrays, meta, seed)
    rt.models[role].reset()
    saved = []
    rows = []
    if swap is not None:
        donor, spec = swap
        receiver = rt.models[role]
        saved = apply_swap(receiver, donor, spec)
    try:
        with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
            for r in range(4, 40):
                chunk, _ = rt.sample("NAT-R0" if role == "R" else "NAT-FT4", "A", obs,
                                     {"A": text, "B": text}, seed, r)
                controls = rt.physical(chunk)
                for k in range(10):
                    step = r * 10 + k + 1
                    obs, _ = ctx.step(controls[k])
                    if step % 10 == 0:
                        obs = ctx.fresh(ctx.env)
                    rows.append(ctx.row(step, obs, controls[k], controls[k]))
    finally:
        if saved:
            undo_swap(saved)
    z0 = rows[0]["obj"][2]
    mv = ctx.geo.movement("scene2", ctx.geometry, rows, z0, ctx.plate_xy(obs))
    return {"first_targets": mv["first_targets"], "first_step": mv["first_step"],
            "success": {g: next((x["step"] for x in rows if x["pred"][g]), None) for g in ctx.geo.GOALS},
            "actions": [[round(float(v), 6) for v in row["a"]] for row in rows]}


def main():
    P = json.loads(Path("/home/lzx/cl_binding/.server/k3_e2_20261102_01/prereg/k3_e2_prereg.json").read_text())
    pair = {"R0": P["families"]["scene2"]["R0"], "FT4": P["families"]["scene2"]["FT4"]}
    ctx = KM.Ctx("scene2")
    rtR = KM.make_runtime(ctx, pair, ["R"])
    rtF = KM.make_runtime(ctx, pair, ["F"])
    rts = {"R0": rtR, "FT4": rtF}
    roles = {"R0": "R", "FT4": "F"}
    report = {"conditions": [c[0] for c in CONDITIONS], "states": {}}
    started = time.monotonic()

    for init in (4, 6, 8):
        obs, arrays, meta, seed = K4M.run_state(ctx, rtR, init, 40)
        state = {"init": init, "seed": seed, "signature": ctx.signature(obs), "runs": {}}
        for name, swap_spec, model in CONDITIONS:
            for key, text in (("A", K4M.INSTR["A"]), ("B", K4M.INSTR["B"])):
                if swap_spec is None:
                    res = run_continue(ctx, rts[model], roles[model], arrays, meta, seed, text)
                else:
                    donor_name, spec = swap_spec
                    donor = rts[donor_name].models["R"] if donor_name == "R0" else rts["FT4"].models["F"]
                    res = run_continue(ctx, rts["FT4"], "F", arrays, meta, seed, text, swap=(donor, spec))
                state["runs"]["%s|%s" % (name, key)] = res
                print(json.dumps({"init": init, "cond": name, "instr": key,
                                  "first": res["first_targets"],
                                  "succ": {g: v for g, v in res["success"].items() if v}}), flush=True)
        state["D"] = {name: (state["runs"][name + "|A"]["first_targets"] == ["left"]
                             and state["runs"][name + "|B"]["first_targets"] == ["front"])
                      for name, _, _ in CONDITIONS}
        state["seconds"] = time.monotonic() - started
        report["states"][str(init)] = state
        (OUT / "runs").mkdir(parents=True, exist_ok=True)
        (OUT / "runs" / ("state_%d.json" % init)).write_text(json.dumps(state, ensure_ascii=False) + "\n")
        print(json.dumps({"init": init, "D": state["D"]}, ensure_ascii=False), flush=True)

    (OUT / "runs" / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print("K5_COMPLETE", json.dumps({"seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
