"""Supplement: full R1 response in the same short window after adapted prefixes."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import time

from k11_reference import SERVER, SITES, WINDOW, km, modules, velocity_pair

INSTR = {g: f"pick up the book and place it in the {name} compartment of the caddy"
         for g, name in (("A", "left"), ("B", "right"), ("N", "front"))}
LABEL = {"A": "left", "B": "right", "N": "front"}
PAIR = {"R0": str(SERVER / "r1_left_right_20261104_01/train_r1/checkpoints/024000"),
        "FT4": str(SERVER / "r1_then_front_v1/train/checkpoints/012000")}
EXPECTED = {"R": "225322ba28d0ad152fcced19950c43e3e40a435f1835cf64a664226da7830003",
            "F": "8ff1bf8f372d728a3218101a3cc93fb6fb9d484e8fefb39c48d81a0c8ab2c515"}


BOUNDARIES = dict.fromkeys(range(20, 40), 60)
BOUNDARIES[22] = 50  # Preserve the K12 timing for every overlapping initial state.
CONDITIONS = ("oracle",)


def append(handle, record):
    handle.write(json.dumps(record, allow_nan=False) + "\n")
    handle.flush()


def run(output, init):
    boundary = BOUNDARIES[init]
    ctx = km.Ctx("scene2")
    torch = ctx.torch
    torch.set_num_threads(2)
    rt = km.make_runtime(ctx, PAIR, ["R", "F"])
    sites = {role: modules(model) for role, model in rt.models.items()}
    assert rt.sha == EXPECTED
    seed = km.noise_seed("scene2", init)
    result = output / f"i{init}.jsonl"
    trace_path = output / f"i{init}_traces.jsonl"

    def native(goal, obs, r, role="F"):
        return rt.sample("NAT-R0" if role == "R" else "NAT-FT4", "A", obs,
                         {"A": INSTR[goal], "B": INSTR[goal]}, seed, r)[0]

    def execute(obs, controls, start):
        events = []
        for k, action in enumerate(controls[:10]):
            obs, executed = ctx.step(action)
            if k == 9:
                obs = ctx.fresh(ctx.env)
            events.append(ctx.row(start + k + 1, obs, action, executed))
        return obs, events

    def fixed_anchor(obs, goal):
        """Read the exact first denoising input of the intervention window."""
        r = boundary // 10
        x = rt.noise(seed, r)
        t = torch.ones(1, dtype=torch.float32, device=x.device)
        prefixes = {(role, g): rt.prefix(role, obs, INSTR[g]) for role in "RF" for g in "AB"}
        raw = {key: rt.query(key[0], pre, x, t)[0, :10, :3].double().flatten() for key, pre in prefixes.items()}
        new_goal = rt.query("F", rt.prefix("F", obs, INSTR["N"]), x, t)[0, :10, :3].double().flatten()
        mu = {role: (raw[role, "A"] + raw[role, "B"]) / 2 for role in "RF"}
        d = {role: (raw[role, "A"] - raw[role, "B"]) / 2 for role in "RF"}
        ref_distance = (mu["R"] - new_goal).norm()
        ref_diff = d["R"].norm()
        assert min(float(ref_distance), float(ref_diff)) > 1e-9
        edits = {}
        for condition in CONDITIONS[1:]:
            diagnostics = defaultdict(list)
            velocity_pair(rt, sites, prefixes, x, t, condition, seed * 10000 + r * 100 + 2, diagnostics)
            edits[condition] = {key: vals[0] for key, vals in diagnostics.items()}
        return {
            "prefix_goal": goal,
            "adapted_mean_distance_to_front_over_r1": float((mu["F"] - new_goal).norm() / ref_distance),
            "adapted_diff_norm_over_r1": float(d["F"].norm() / ref_diff),
            "adapted_diff_cosine_to_r1": float((d["F"] @ d["R"]) / (d["F"].norm() * ref_diff).clamp_min(1e-9)),
            "adapted_diff_projection_over_r1": float((d["F"] @ d["R"]) / ref_diff.square()),
            "edits": edits,
        }

    def prepare(goal, role="F"):
        obs, _ = ctx.reset(init)
        start = ctx.signature(obs)
        z0 = float(obs[ctx.obj][2])
        controls, rows = [], []
        for r in range(boundary // 10):
            controls.append(rt.physical(native(goal, obs, r, role)))
            obs, events = execute(obs, controls[-1], r * 10)
            rows.extend(events)
        held = all(row["obj"][2] - z0 > .03 and row["a"][6] > 0
                   and math.dist(row["obj"], row["eef"]) < .15 for row in rows[-5:])
        pretarget = not any(ctx.geo.hits("scene2", ctx.geometry, row["obj"], z0, None)
                            or any(row["pred"].values()) for row in rows)
        return {"controls": controls, "rows": rows, "reset_signature": start, "boundary_signature": ctx.signature(obs),
                "z0": z0, "held_at_boundary": held, "pretarget_at_boundary": pretarget,
                "fixed_anchor": None}

    def episode(condition, goal, prep):
        started = time.monotonic()
        role = "R" if condition == "r1" else "F"
        obs, _ = ctx.reset(init)
        assert ctx.signature(obs) == prep["reset_signature"]
        rows = []
        for r, controls in enumerate(prep["controls"]):
            obs, events = execute(obs, controls, r * 10)
            rows.extend(events)
        assert rows == prep["rows"] and ctx.signature(obs) == prep["boundary_signature"]
        diagnostics = defaultdict(list)
        for r in range(boundary // 10, 40):
            if condition not in ("ft4", "r1") and r * 10 < boundary + WINDOW:
                prefixes = {(role, g): rt.prefix(role, obs, INSTR[g]) for role in "RF" for g in "AB"}
                iteration = 0

                def velocity(x, t):
                    nonlocal iteration
                    iteration += 1
                    return velocity_pair(rt, sites, prefixes, x, t, condition,
                                         seed * 10000 + r * 100 + iteration * 2, diagnostics)[goal]

                chunk = rt.integrate(velocity, rt.noise(seed, r), 10)[..., :7]
                assert iteration == 10
            else:
                chunk = native(goal, obs, r, role)
            obs, events = execute(obs, rt.physical(chunk), r * 10)
            rows.extend(events)
        assert len(rows) == 400
        assert all(len(values) == WINDOW for values in diagnostics.values())
        movement = ctx.geo.movement("scene2", ctx.geometry, rows, prep["z0"], None)
        info = {key: prep[key] for key in ("z0", "reset_signature", "boundary_signature", "held_at_boundary", "pretarget_at_boundary")}
        return {"init": init, "condition": condition, "goal": goal, "requested": LABEL[goal], "seed": seed,
                "boundary": boundary, "window": WINDOW,
                "prefix_model": "R1" if role == "R" else "Front-12000", "prefix_goal": goal,
                **info, "fixed_anchor": prep["fixed_anchor"] if condition == "ft4" else None,
                "first": movement["first_targets"], "first_step": movement["first_step"],
                "first_times": movement["times"], "terminal": rows[-1]["pred"],
                "ever": {g: any(row["pred"][g] for row in rows) for g in ctx.geo.GOALS},
                "diagnostics": {k: mean(values) for k, values in diagnostics.items()},
                "seconds": time.monotonic() - started}, rows

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), \
            result.open("x", encoding="utf-8") as results, trace_path.open("x", encoding="utf-8") as traces:
        prefixes = {goal: prepare(goal) for goal in "AB"}
        assert all(p["reset_signature"] == prefixes["A"]["reset_signature"] for p in prefixes.values())
        for condition in CONDITIONS:
            for goal in ("ABN" if condition == "ft4" else "AB"):
                row, events = episode(condition, goal, prefixes[goal])
                append(traces, {"condition": condition, "goal": goal, "rows": events})
                append(results, row)
                print(f"{init} {condition} {goal} {row['first']} {row['seconds']:.1f}s", flush=True)
    ctx.env.close()
    (output / f"i{init}_COMPLETE.json").write_text(json.dumps({
        "init": init, "boundary": boundary, "window": WINDOW, "conditions": CONDITIONS, "sites": SITES,
        "model_sha256": rt.sha, "model_paths": rt.paths, "instructions": INSTR,
        "processor_check": rt.processor_check, "episode_count": 2,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "intervention_sha256": hashlib.sha256((Path(__file__).parent / "k11_reference.py").read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init", type=int, choices=BOUNDARIES, required=True)
    args = parser.parse_args()
    run(args.output, args.init)
