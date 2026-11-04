"""K7 follow-up: paired external-component and reverse layer transplants."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time


SERVER = Path("/home/lzx/cl_binding/.server")
K7 = SERVER / "k7_whole_layer_20261104_01"
COMPONENTS = {
    "output": ("action_out_proj",),
    "input": ("state_proj", "action_in_proj", "action_time_mlp_in", "action_time_mlp_out"),
    "norm": ("vlm_with_expert.lm_expert.norm",),
}
QUARTERS = {f"reverse-q{q + 1}": tuple(range(q * 4, q * 4 + 4)) for q in range(4)}
ALL_LAYERS = tuple(range(16))
GOALS = {"A": "left", "B": "front", "N": "right"}


def jobs():
    return [(i, condition, instruction)
            for i in range(10)
            for condition, instructions in (
                ("r0-native", "ABN"),
                ("all16-output", "ABN"), ("all16-input", "ABN"), ("all16-norm", "ABN"),
                ("all16-external", "ABN"),
                *( (name, "AB") for name in QUARTERS ),
                ("reverse-all16", "AB"),
            )
            for instruction in instructions]


def resolve(model, path):
    parts = path.split(".")
    for part in parts[:-1]:
        model = getattr(model, part)
    return model, parts[-1]


@contextmanager
def swap(receiver, donor, layers, components):
    rec = receiver.model
    src = donor.model
    er = rec.vlm_with_expert.lm_expert.layers
    ed = src.vlm_with_expert.lm_expert.layers
    saved_layers, saved_modules = [], []
    try:
        for i in layers:
            saved_layers.append((i, er[i]))
            er[i] = ed[i]
        for path in components:
            owner, attr = resolve(rec, path)
            donor_owner, donor_attr = resolve(src, path)
            saved_modules.append((owner, attr, getattr(owner, attr)))
            setattr(owner, attr, getattr(donor_owner, donor_attr))
        yield
    finally:
        for owner, attr, module in reversed(saved_modules):
            setattr(owner, attr, module)
        for i, layer in reversed(saved_layers):
            er[i] = layer


def append(path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")


def run(out, worker, workers, gate_only):
    sys.path[:0] = [str(SERVER / folder / "code") for folder in (
        "k1_screen_20261101_01", "k4_text_attention_20261102_01")]
    import k1_main as km
    from k4_text_attention import INSTR

    pair = json.loads((SERVER / "k3_e2_20261102_01/prereg/k3_e2_prereg.json").read_bytes())["families"]["scene2"]
    ctx = km.Ctx("scene2")
    runtimes = {role: km.make_runtime(ctx, pair, [role]) for role in ("R", "F")}
    policies = {role: rt.models[role] for role, rt in runtimes.items()}
    all_external = tuple(path for paths in COMPONENTS.values() for path in paths)

    def episode(init, condition, instruction):
        start = time.monotonic()
        role = "R" if condition == "r0-native" or condition.startswith("reverse-") else "F"
        donor = "F" if role == "R" else "R"
        layers = (ALL_LAYERS if condition.startswith("all16") or condition == "reverse-all16"
                  else QUARTERS.get(condition, ()))
        component = condition.removeprefix("all16-") if condition.startswith("all16-") else ""
        outside = all_external if component == "external" else COMPONENTS.get(component, ())
        rt = runtimes[role]
        obs, _ = ctx.reset(init)
        rt.models[role].reset()
        seed = km.noise_seed("scene2", init)
        rows = []
        with swap(policies[role], policies[donor], layers, outside):
            with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                for r in range(40):
                    text = INSTR[instruction]
                    chunk, _ = rt.sample("NAT-R0" if role == "R" else "NAT-FT4", "A", obs,
                                         {"A": text, "B": text}, seed, r)
                    controls = rt.physical(chunk)
                    for k in range(10):
                        step = r * 10 + k + 1
                        obs, _ = ctx.step(controls[k])
                        if k == 9:
                            obs = ctx.fresh(ctx.env)
                        rows.append(ctx.row(step, obs, controls[k], controls[k]))
        mv = ctx.geo.movement("scene2", ctx.geometry, rows, rows[0]["obj"][2], ctx.plate_xy(obs))
        trace = json.dumps(rows, sort_keys=True, allow_nan=False).encode()
        return {
            "init": init, "condition": condition, "instruction": instruction, "seed": seed,
            "first_targets": mv["first_targets"], "first_step": mv["first_step"],
            "success": {g: next((x["step"] for x in rows if x["pred"][g]), None) for g in ctx.geo.GOALS},
            "terminal_pred": rows[-1]["pred"],
            "first_close": next((x["step"] for x in rows if x["a"][6] > 0), None),
            "trace_sha256": hashlib.sha256(trace).hexdigest(),
            "seconds": time.monotonic() - start,
        }

    reference = {}
    for path in sorted((K7 / "runs").glob("w[01].jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            reference[row["init"], row["cond"], row["instr"]] = row
    compare_keys = ("first_targets", "first_step", "success", "first_close")
    if gate_only:
        if len(reference) != 200:
            raise ValueError("K7 reference matrix is incomplete")
        gate_rows = []
        for init, instruction in ((4, "A"), (6, "B"), (8, "N")):
            results = {cond: episode(init, cond, instruction)
                       for cond in ("r0-native", "ft4-native", "all16", "all16-external")}
            for new, old in (("r0-native", "NAT-R0"), ("ft4-native", "NAT-FT4"), ("all16", "ALL-16")):
                if old == "NAT-R0" and instruction == "N":
                    continue
                if any(results[new][k] != reference[init, old, instruction][k] for k in compare_keys):
                    raise ValueError(f"K7 baseline mismatch at {init}, {instruction}, {new}")
            if results["r0-native"]["trace_sha256"] != results["all16-external"]["trace_sha256"]:
                raise ValueError(f"Full R0 reconstruction differs at {init}, {instruction}")
            gate_rows.append(results)
            print(f"Gate passed: {init} {instruction}", flush=True)
        (out / "gate.json").write_text(json.dumps({
            "status": "passed", "cases": gate_rows,
            "models": {r: {"path": str(rt.paths[r]), "sha256": rt.sha[r]} for r, rt in runtimes.items()},
            "reference_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((K7 / "runs").glob("w[01].jsonl"))},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }, indent=2) + "\n", encoding="utf-8")
        return

    if json.loads((out / "gate.json").read_bytes())["status"] != "passed":
        raise ValueError("Engineering gate required")
    result = out / f"worker{worker}.jsonl"
    planned = jobs()[worker::workers]
    for init, condition, instruction in planned:
        row = episode(init, condition, instruction)
        append(result, row)
        print(f"{init} {condition} {instruction} {row['first_targets']}", flush=True)
    (out / f"worker{worker}.done").write_text(f"{len(planned)}\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    run(args.output, args.worker, args.workers, args.gate_only)
