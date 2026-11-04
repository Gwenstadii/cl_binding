"""K7/K8 missing factorial cell: native FT4 with the R0 action output projection."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys

from k8_transplant_attribution import GOALS, K7, SERVER, append, swap


CONDITIONS = ("ft4-native", "ft4-r0-output")
JOBS = [(init, key) for init in range(10) for key in GOALS]


def run(output, worker):
    sys.path[:0] = [str(SERVER / folder / "code") for folder in (
        "k1_screen_20261101_01", "k4_text_attention_20261102_01")]
    import k1_main as km
    from k4_text_attention import INSTR

    pair = json.loads((SERVER / "k3_e2_20261102_01/prereg/k3_e2_prereg.json").read_bytes())["families"]["scene2"]
    ctx = km.Ctx("scene2")
    rt = km.make_runtime(ctx, pair, ["F", "R"])
    ft4, r0 = rt.models["F"], rt.models["R"]
    proj = ft4.model.action_out_proj
    assert proj.weight.shape == r0.model.action_out_proj.weight.shape
    assert not ctx.torch.equal(proj.weight, r0.model.action_out_proj.weight)
    reference = {}
    for path in sorted((K7 / "runs").glob("w[01].jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["cond"] == "NAT-FT4":
                reference[row["init"], row["instr"]] = row
    assert set(reference) == set(JOBS)
    (output / f"worker{worker}_meta.json").write_text(json.dumps({
        "model_sha256": rt.sha,
        "projection_shape": [proj.in_features, proj.out_features],
        "chunk_size": ft4.config.chunk_size,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")

    for init, key in JOBS[worker::2]:
        seed = km.noise_seed("scene2", init)
        text = INSTR[key]
        for condition in CONDITIONS:
            obs, _ = ctx.reset(init)
            ft4.reset()
            intervention = (swap(ft4, r0, (), ("action_out_proj",))
                            if condition == "ft4-r0-output" else nullcontext())
            rows = []
            with intervention, ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                expected = r0.model.action_out_proj if condition == "ft4-r0-output" else proj
                assert ft4.model.action_out_proj is expected
                for r in range(40):
                    chunk, _ = rt.sample("NAT-FT4", "A", obs, {"A": text, "B": text}, seed, r)
                    controls = rt.physical(chunk)
                    for k in range(10):
                        step = r * 10 + k + 1
                        obs, _ = ctx.step(controls[k])
                        if k == 9:
                            obs = ctx.fresh(ctx.env)
                        rows.append(ctx.row(step, obs, controls[k], controls[k]))
            assert ft4.model.action_out_proj is proj
            movement = ctx.geo.movement("scene2", ctx.geometry, rows, rows[0]["obj"][2], ctx.plate_xy(obs))
            row = {
                "init": init, "instruction": key, "condition": condition, "seed": seed,
                "first_targets": movement["first_targets"], "first_step": movement["first_step"],
                "success": {g: next((r["step"] for r in rows if r["pred"][g]), None) for g in ctx.geo.GOALS},
                "terminal_pred": rows[-1]["pred"],
                "first_close": next((r["step"] for r in rows if r["a"][6] > 0), None),
                "trace_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            }
            if condition == "ft4-native":
                assert all(row[k] == reference[init, key][k] for k in (
                    "first_targets", "first_step", "success", "first_close")), (init, key, row)
            append(output / f"worker{worker}.jsonl", row)
            print(init, key, condition, row["first_targets"], flush=True)
    (output / f"worker{worker}.done").write_text("30\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=int, choices=(0, 1))
    args = parser.parse_args()
    run(args.output, args.worker)
