"""K7: whole-layer transplant. Replace entire expert decoder layers (attention + MLP + norms)
from R0 into FT4, FT4 controls the whole episode from step 0.

Conditions: all 16 layers, four layer quarters, two cumulative prefixes, plus native baselines.
10 development initial states x 2 old instructions; competence check with the new instruction
on NAT-FT4 and ALL-16.
"""
import argparse
import json
import sys
import time
from pathlib import Path

K1 = Path("/home/lzx/cl_binding/.server/k1_screen_20261101_01")
K4 = Path("/home/lzx/cl_binding/.server/k4_text_attention_20261102_01")
OUT = Path("/home/lzx/cl_binding/.server/k7_whole_layer_20261104_01")
sys.path.insert(0, str(K1 / "code"))
sys.path.insert(0, str(K4 / "code"))

import k1_main as KM
import k4_text_attention as K4M

CONDS = {
    "NAT-FT4": None,
    "NAT-R0": None,
    "ALL-16": list(range(16)),
    "Q1-0-3": [0, 1, 2, 3],
    "Q2-4-7": [4, 5, 6, 7],
    "Q3-8-11": [8, 9, 10, 11],
    "Q4-12-15": [12, 13, 14, 15],
    "CUM-0-7": list(range(8)),
    "CUM-0-11": list(range(12)),
}
INITS = list(range(10))
NEW = "N"
NEW_CONDS = ("NAT-FT4", "ALL-16")


def swap_layers(receiver, donor, layers):
    er, ed = receiver.model.vlm_with_expert.lm_expert, donor.model.vlm_with_expert.lm_expert
    saved = []
    for i in layers:
        saved.append((i, er.layers[i]))
        er.layers[i] = ed.layers[i]
    return saved


def undo(saved):
    er = None
    for i, layer in saved:
        layer_owner = layer
        # restored by the caller through the list captured at swap time
        pass


def run_episode(ctx, rt, role, init, text, layers, donor_policy):
    obs, _ = ctx.reset(init)
    rt.models[role].reset()
    saved = []
    er = rt.models[role].model.vlm_with_expert.lm_expert
    if layers:
        ed = donor_policy.model.vlm_with_expert.lm_expert
        for i in layers:
            saved.append((i, er.layers[i]))
            er.layers[i] = ed.layers[i]
    rows = []
    seed = KM.noise_seed("scene2", init)
    try:
        with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
            for r in range(40):
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
        for i, layer in saved:
            er.layers[i] = layer
    z0 = rows[0]["obj"][2]
    mv = ctx.geo.movement("scene2", ctx.geometry, rows, z0, ctx.plate_xy(obs))
    return {"first_targets": mv["first_targets"], "first_step": mv["first_step"],
            "success": {g: next((x["step"] for x in rows if x["pred"][g]), None) for g in ctx.geo.GOALS},
            "first_close": next((x["step"] for x in rows if x["a"][6] > 0), None)}


def main(args):
    P = json.loads(Path("/home/lzx/cl_binding/.server/k3_e2_20261102_01/prereg/k3_e2_prereg.json").read_text())
    pair = {"R0": P["families"]["scene2"]["R0"], "FT4": P["families"]["scene2"]["FT4"]}
    ctx = KM.Ctx("scene2")
    rtR = KM.make_runtime(ctx, pair, ["R"])
    rtF = KM.make_runtime(ctx, pair, ["F"])
    jobs = []
    for init in INITS:
        for cond in CONDS:
            for key in ("A", "B"):
                jobs.append((init, cond, key))
        for cond in NEW_CONDS:
            jobs.append((init, cond, NEW))
    mine = jobs[args.worker::args.workers]
    runs = OUT / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    jsonl = runs / ("w%d.jsonl" % args.worker)
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["init"], row["cond"], row["instr"]))
    status = runs / ("w%d_status.json" % args.worker)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    for init, cond, key in mine:
        if (init, cond, key) in done:
            continue
        text = K4M.INSTR[key]
        if cond == "NAT-R0":
            res = run_episode(ctx, rtR, "R", init, text, None, None)
        else:
            layers = CONDS[cond]
            res = run_episode(ctx, rtF, "F", init, text, layers, rtR.models["R"])
        rec = {"init": init, "cond": cond, "instr": key, **res}
        with jsonl.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        done.add((init, cond, key))
        status.write_text(json.dumps({"worker": args.worker, "done": len(done), "total": len(mine),
                                      "started": started, "last": [init, cond, key],
                                      "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False))
        print(json.dumps({"init": init, "cond": cond, "instr": key, "first": res["first_targets"],
                          "succ": {g: v for g, v in res["success"].items() if v}}), flush=True)
    print("WORKER_COMPLETE", args.worker, len(done), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    main(parser.parse_args())
