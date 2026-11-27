"""Matched closed-loop test of shared velocity drift and two candidate differential sites."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys
import time


SERVER = Path("/home/lzx/cl_binding/.server")
sys.path[:0] = [str(SERVER / name / "code") for name in (
    "k1_screen_20261101_01", "k4_text_attention_20261102_01")]
import k1_main as km
from k4_text_attention import INSTR


INITS = (*range(8), *range(20, 28))
SITES = ("CA-7", "MLP-9")
LABEL = {"A": "left", "B": "front", "N": "right"}
WINDOW = 60
CONDITIONS = ("r0", "ft4", "mean", "sites", "joint", "anti", "random", "oracle")
REVERSE = ("reverse_mean", "reverse_sites", "reverse_joint", "reverse_oracle")
TARGET_MEAN = {"mean", "joint", "anti", "random", "oracle", "reverse_mean", "reverse_joint", "reverse_oracle"}


def modules(model):
    layers = model.model.vlm_with_expert.lm_expert.layers
    assert len(layers) == 16
    return dict(zip(SITES, (layers[7].self_attn.o_proj, layers[9].mlp), strict=True))


def readout(rt, role, prefix, x, t, sites, patch=None):
    captured = {}

    def tap(name):
        def hook(_module, _inputs, out):
            assert name not in captured
            captured[name] = out.detach().clone()
            if patch is not None:
                return (out.float() + patch[name]).to(out.dtype)
        return hook

    handles = [module.register_forward_hook(tap(name)) for name, module in sites.items()]
    try:
        velocity = rt.query(role, prefix, x, t)
    finally:
        for handle in handles:
            handle.remove()
    assert tuple(sorted(captured)) == tuple(sorted(sites))
    return velocity, captured


def velocity_pair(rt, sites, prefixes, x, t, condition, seed, diagnostics):
    torch = rt.torch
    reverse = condition.startswith("reverse_")
    receiver, donor = ("R", "F") if reverse else ("F", "R")
    needs_sites = condition in ("sites", "joint", "anti", "random", "reverse_sites", "reverse_joint")
    states = {(role, goal): readout(rt, role, prefixes[role, goal], x, t, sites[role] if needs_sites else {})
              for role in (receiver, donor) for goal in "AB"}
    native = {role: [states[role, goal][0].float() for goal in "AB"] for role in (receiver, donor)}
    mu = {role: (a + b) / 2 for role, (a, b) in native.items()}
    d = {role: (a - b) / 2 for role, (a, b) in native.items()}
    target_mu = mu[donor] if condition in TARGET_MEAN else mu[receiver]
    if needs_sites:
        patches = {}
        for i, name in enumerate(SITES):
            contrast = {role: (states[role, "A"][1][name].float() - states[role, "B"][1][name].float()) / 2
                        for role in (receiver, donor)}
            delta = contrast[donor] - contrast[receiver]
            if condition == "anti":
                delta = -delta
            elif condition == "random":
                rng = torch.Generator(device=x.device).manual_seed(seed + i)
                control = torch.randn(delta.shape, generator=rng, device=x.device)
                delta = control * (delta.norm() / control.norm().clamp_min(1e-9))
            patches[name] = delta
        edited = [readout(rt, receiver, prefixes[receiver, g], x, t, sites[receiver],
                          {name: sign * delta for name, delta in patches.items()})[0].float()
                  for g, sign in (("A", 1), ("B", -1))]
        difference = (edited[0] - edited[1]) / 2
    else:
        difference = d[donor if condition.endswith("oracle") else receiver]
    # Recenter both outputs to the same prescribed mean, including after nonlinear propagation.
    out = {g: (target_mu + sign * difference).to(rt.vdtype) for g, sign in (("A", 1), ("B", -1))}
    ref_role = "R"
    reference = d[ref_role][0, :10, :3].flatten()
    effective = ((out["A"].float() - out["B"].float()) / 2)[0, :10, :3].flatten()
    target = target_mu[0, :10, :3]
    actual_mean = ((out["A"].float() + out["B"].float()) / 2)[0, :10, :3]
    diagnostics["old_direction_projection"].append(float((effective @ reference) / (reference @ reference).clamp_min(1e-9)))
    diagnostics["output_mean_roundoff_rms"].append(float((actual_mean - target).square().mean().sqrt()))
    if condition.endswith("oracle"):
        for g in "AB":
            assert torch.equal(out[g], states[donor, g][0]), "Oracle should reproduce native donor field exactly"
    return out


def run(output, worker, workers, reverse):
    pair = json.loads((SERVER / "k3_e2_20261102_01/prereg/k3_e2_prereg.json").read_text())["families"]["scene2"]
    ctx = km.Ctx("scene2")
    torch, np = ctx.torch, ctx.np
    torch.set_num_threads(2)
    rt = km.make_runtime(ctx, pair, ["R", "F"])
    sites = {role: modules(model) for role, model in rt.models.items()}
    expected = json.loads((SERVER / "k10_fixed_input_projection_20261105_01/fields.json").read_text())["model_sha256"]
    assert rt.sha == expected
    names = REVERSE if reverse else CONDITIONS
    suffix = "reverse" if reverse else "forward"
    result = output / f"{suffix}_{worker}.jsonl"
    traces = output / f"{suffix}_{worker}_traces.jsonl"
    if result.exists() or traces.exists():
        raise FileExistsError(result)

    def native(role, goal, obs, seed, r):
        return rt.sample("NAT-R0" if role == "R" else "NAT-FT4", "A", obs,
                         {"A": INSTR[goal], "B": INSTR[goal]}, seed, r)[0]

    def execute(obs, controls, start):
        events = []
        for k in range(10):
            obs, executed = ctx.step(controls[k])
            if k == 9:
                obs = ctx.fresh(ctx.env)
            events.append(ctx.row(start + k + 1, obs, controls[k], executed))
        return obs, events

    def prepare(init, seed):
        obs, _ = ctx.reset(init)
        start = np.asarray(obs[ctx.obj]).copy()
        rows, controls = [], []
        for r in range(6):
            chunk = native("R", "A", obs, seed, r)
            controls.append(rt.physical(chunk))
            obs, events = execute(obs, controls[-1], r * 10)
            rows.extend(events)
            if r < 3:
                continue
            # The end-effector coordinate is about 10–14 cm from the centre of a held book.
            lifted = all(row["obj"][2] - start[2] > .03 and row["a"][6] > 0
                         and math.dist(row["obj"], row["eef"]) < .15 for row in rows[-5:])
            pretarget = not any(ctx.geo.hits("scene2", ctx.geometry, row["obj"], start[2], None)
                                or any(row["pred"].values()) for row in rows)
            travel = float(np.linalg.norm(np.asarray(obs[ctx.obj])[:2] - start[:2]))
            if lifted and pretarget and travel <= .10:
                return controls, ctx.signature(obs), float(start[2]), travel
        return None

    def episode(init, seed, prepared, condition, goal):
        started = time.monotonic()
        controls, signature, z0, travel = prepared
        obs, _ = ctx.reset(init)
        for r, control in enumerate(controls):
            obs, _ = execute(obs, control, r * 10)
        assert ctx.signature(obs) == signature, "Snapshot reconstruction differed from the recorded prefix"
        boundary = len(controls) * 10
        base = "R" if condition == "r0" or condition.startswith("reverse_") else "F"
        rows = []
        diagnostics = defaultdict(list)
        for r in range(len(controls), 40):
            edit = condition not in ("r0", "ft4") and r * 10 < boundary + WINDOW
            if edit:
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
                chunk = native(base, goal, obs, seed, r)
            obs, events = execute(obs, rt.physical(chunk), r * 10)
            rows.extend(events)
        movement = ctx.geo.movement("scene2", ctx.geometry, rows, z0, None, start=boundary + 1)
        assert len(rows) == 400 - boundary
        assert all(len(values) == (0 if condition in ("r0", "ft4") else WINDOW) for values in diagnostics.values())
        with traces.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"init": init, "condition": condition, "goal": goal, "rows": rows}, allow_nan=False) + "\n")
        return {"init": init, "condition": condition, "goal": goal, "requested": LABEL[goal], "seed": seed,
                "boundary": boundary, "boundary_xy_m": travel, "signature": signature,
                "first": movement["first_targets"], "first_step": movement["first_step"],
                "ever": {g: any(r["pred"][g] for r in rows) for g in ctx.geo.GOALS},
                "terminal": rows[-1]["pred"], "seconds": time.monotonic() - started,
                "diagnostics": {k: mean(v) for k, v in diagnostics.items()}}

    manifest = {"worker": worker, "workers": workers, "model_sha256": rt.sha, "conditions": names,
                "inits": list(INITS[worker::workers]), "window": WINDOW, "sites": SITES,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "skipped_inits": []}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), result.open("x", encoding="utf-8") as handle:
        for init in manifest["inits"]:
            seed = km.noise_seed("scene2", init)
            prepared = prepare(init, seed)
            if prepared is None:
                manifest["skipped_inits"].append(init)
                print(f"{init} no eligible pretarget grasp", flush=True)
                continue
            for condition in names:
                for goal in ("ABN" if condition == "ft4" else "AB"):
                    row = episode(init, seed, prepared, condition, goal)
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    handle.flush()
                    print(f"{init} {condition} {goal} {row['first']} {row['seconds']:.1f}s", flush=True)
    ctx.env.close()
    (output / f"{suffix}_{worker}_COMPLETE.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    run(args.output, args.worker, args.workers, args.reverse)
