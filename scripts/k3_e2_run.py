"""K3-E2: native rollouts with substituted test instructions (no training, no hooks).

Reads prereg/k3_e2_prereg.json. Every episode is one model, one instruction, one fixed
initial state, 400 native control steps, the same replanning-noise stream as K1.

Modes
  gate   reproduce an archived K1 prefix rollout bit for bit, and check determinism
  pilot  one timed episode
  run    shard the full job list across workers
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

K1 = Path("/home/lzx/cl_binding/.server/k1_screen_20261101_01")
OUT = Path("/home/lzx/cl_binding/.server/k3_e2_20261102_01")
sys.path.insert(0, str(K1 / "code"))
sys.path.insert(0, str(OUT / "code"))

import k1_main as KM


def jdump(path, value):
    KM.jdump(path, value)


def prereg(path=None):
    return json.loads(Path(path or (OUT / "prereg/k3_e2_prereg.json")).read_text())


def jobs(P):
    out = []
    for family in P["design"]["families"]:
        fam = family.split(" ")[0]
        for model in P["design"]["models"]:
            model = model.split(" ")[0]
            for key in P["design"]["instruction_sets"][fam]:
                for init in range(10):
                    out.append((fam, model, key, init))
    return out


def episode(ctx, rt, family, model, key, text, init, horizon=400):
    seed = KM.noise_seed(family, init)
    obs, sha0 = ctx.reset(init)
    plate = ctx.plate_xy(obs)
    rows = []
    identity = []
    t0 = time.monotonic()
    texts = {"A": text, "B": text}
    cond = "NAT-" + model
    with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
        for r in range(horizon // 10):
            chunk, _ = rt.sample(cond, "A", obs, texts, seed, r)
            controls = rt.physical(chunk)
            for k in range(10):
                step = r * 10 + k + 1
                obs, executed = ctx.step(controls[k])
                if step % 10 == 0:
                    obs = ctx.fresh(ctx.env)
                rows.append(ctx.row(step, obs, controls[k], executed))
    z0 = rows[0]["obj"][2]
    mv = ctx.geo.movement(family, ctx.geometry, rows, z0, plate)
    success = {g: next((x["step"] for x in rows if x["pred"][g]), None) for g in ctx.geo.GOALS}
    closes = [x["step"] for x in rows if x["a"][6] > 0]
    lifted = [x["step"] for x in rows if x["obj"][2] - z0 > ctx.geo.LIFT]
    first = mv["first_targets"]
    summary = {
        "family": family, "model": model, "instruction_key": key, "instruction": text, "init": init,
        "noise_seed": seed, "initial_sha": sha0, "plate_xy": plate,
        "first_targets": first, "first_target_step": mv["first_step"],
        "target_times": mv["times"], "native_success_step": success,
        "first_close": closes[0] if closes else None, "attempted": bool(closes),
        "lifted_steps": len(lifted), "lifted": bool(lifted),
        "terminal_pred": rows[-1]["pred"],
        "path": {str(s): {"obj": rows[s - 1]["obj"], "eef": rows[s - 1]["eef"],
                          "pred": rows[s - 1]["pred"]} for s in (40, 80, 120, 200, 300, 400)},
        "seconds": time.monotonic() - t0, "identity": identity,
    }
    return summary, rows


def open_logs(worker, tag=""):
    runs = OUT / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    done = set()
    jsonl = runs / f"{tag}w{worker:02d}.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["family"], row["model"], row["instruction_key"], row["init"]))
    events = runs / f"{tag}w{worker:02d}_events.jsonl.gz"
    return done, jsonl, events


def status(worker, workers, done, started, last, error=None, tag=""):
    path = OUT / "runs" / f"{tag}w{worker:02d}_status.json"
    jdump(path, {"worker": worker, "workers": workers, "episodes_done": len(done),
                 "started": started, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "last": last, "error": error})


def mode_gate(args):
    P = prereg()
    family, model, key, init = "scene2", "FT4", "A", 4
    text = P["design"]["instruction_sets"][family][key]
    ctx = KM.Ctx(family)
    pair = {"R0": P["families"][family]["R0"], "FT4": P["families"][family]["FT4"]}
    rt = KM.make_runtime(ctx, pair, ["F"])
    sum1, rows1 = episode(ctx, rt, family, model, key, text, init)
    sum2, rows2 = episode(ctx, rt, family, model, key, text, init)
    det = max((max(abs(x - y) for x, y in zip(a["a"], b["a"])) for a, b in zip(rows1, rows2)), default=0.0)
    with gzip.open(K1 / "prefix" / family / f"FT4_scene2_s1_i{init}.json.gz", "rt") as f:
        ref = json.load(f)
    act = max((max(abs(x - y) for x, y in zip(a["a"], b["a"])) for a, b in zip(rows1, ref["rows"])), default=0.0)
    pos = max((max(max(abs(x - y) for x, y in zip(a["obj"], b["obj"])),
                   max(abs(x - y) for x, y in zip(a["eef"], b["eef"]))) for a, b in zip(rows1, ref["rows"])),
              default=0.0)
    pred = sum(1 for a, b in zip(rows1, ref["rows"]) if a["pred"] != b["pred"])
    out = {"family": family, "model": model, "instruction_key": key, "init": init,
           "reference": str(K1 / "prefix" / family / f"FT4_scene2_s1_i{init}.json.gz"),
           "reference_instruction": ref["instruction"], "determinism_max_abs": det,
           "action_max_abs_vs_prefix": act, "object_eef_max_abs_vs_prefix": pos,
           "predicate_rows_differing": pred, "seconds": sum1["seconds"],
           "first_targets": sum1["first_targets"], "reference_first": ref["native_success_step"],
           "passed": det == 0.0 and act == 0.0 and pos <= 1e-9 and pred == 0}
    jdump(OUT / "engineering" / "gate_prefix_reproduction.json", out)
    print(json.dumps(out, ensure_ascii=False), flush=True)
    ctx.env.close()


def mode_pilot(args):
    P = prereg()
    ctx = KM.Ctx(args.family)
    pair = {"R0": P["families"][args.family]["R0"], "FT4": P["families"][args.family]["FT4"]}
    rt = KM.make_runtime(ctx, pair, ["F", "R"])
    text = P["design"]["instruction_sets"][args.family][args.instruction]
    for model in ("R0", "FT4"):
        s, _ = episode(ctx, rt, args.family, model, args.instruction, text, args.init)
        print(json.dumps({k: s[k] for k in ("model", "instruction_key", "first_targets", "first_close",
                                           "native_success_step", "seconds")}), flush=True)
    ctx.env.close()


def mode_run(args):
    P = prereg(args.prereg)
    all_jobs = jobs(P)
    mine = all_jobs[args.worker::args.workers]
    done, jsonl, events = open_logs(args.worker, args.tag)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    pending = [j for j in mine if j not in done]
    print(json.dumps({"worker": args.worker, "jobs": len(mine), "pending": len(pending)}), flush=True)
    by_family = {}
    for fam, model, key, init in pending:
        by_family.setdefault(fam, []).append((model, key, init))
    for family, items in by_family.items():
        ctx = KM.Ctx(family)
        pair = {"R0": P["families"][family]["R0"], "FT4": P["families"][family]["FT4"]}
        rts = {m: KM.make_runtime(ctx, pair, ["F" if m == "FT4" else "R"]) for m in ("R0", "FT4")}
        with gzip.open(events, "at") as ev:
            for model, key, init in items:
                text = P["design"]["instruction_sets"][family][key]
                s, rows = episode(ctx, rts[model], family, model, key, text, init)
                ev.write(json.dumps({"family": family, "model": model, "instruction_key": key,
                                     "init": init, "rows": rows}) + "\n")
                ev.flush()
                with jsonl.open("a") as f:
                    f.write(json.dumps(s) + "\n")
                done.add((family, model, key, init))
                status(args.worker, args.workers, done, started,
                       f"{family}|{model}|{key}|i{init}", tag=args.tag)
                print(json.dumps({"family": family, "model": model, "key": key, "init": init,
                                  "first": s["first_targets"], "close": s["first_close"],
                                  "sec": round(s["seconds"], 1)}), flush=True)
        for rt in rts.values():
            del rt
        import gc
        gc.collect()
        ctx.torch.cuda.empty_cache()
        ctx.env.close()
    jdump(OUT / "runs" / f"{args.tag}w{args.worker:02d}_done.json",
          {"worker": args.worker, "episodes": len(done), "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    print("WORKER_COMPLETE", args.worker, len(done), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("gate")
    p = sub.add_parser("pilot")
    p.add_argument("--family", default="scene2")
    p.add_argument("--instruction", default="N")
    p.add_argument("--init", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--worker", type=int, required=True)
    r.add_argument("--workers", type=int, required=True)
    r.add_argument("--prereg", default=None)
    r.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.mode == "gate":
        mode_gate(args)
    elif args.mode == "pilot":
        mode_pilot(args)
    else:
        mode_run(args)