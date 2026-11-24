"""Fixed-input responses and native evaluation for K18."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

RUN = Path(__file__).resolve().parent
ORIGINAL = Path("/home/lzx/cl_binding/.server/r1_then_front_v1/train")
K15 = ORIGINAL.parents[1] / "k15_background_gain_20261109_01"
assert hashlib.sha256((K15 / "k15.py").read_bytes()).hexdigest() == "8e928d79232cf9346be64965fa77dfcf4c32b2906b3715e13bd278cf700c77a6"
sys.path.insert(0, str(K15))
from k15 import Lab5, SOURCE, INSTR

BLOCKS = {"H": slice(0, 2), "Z": slice(2, 3), "R": slice(3, 6), "G": slice(6, 7), "P": slice(7, 32)}
STEPS = (0, 10, 30, 100, 300, 1000, 3000, 6000, 12000)


def checkpoint(lab, step):
    if step == 0:
        return Path(lab.rt.paths["R"])
    if step == 12000:
        return ORIGINAL / "checkpoints/012000"
    path = RUN / "train/checkpoints" / f"{step:06d}"
    while not (path / "CHECKPOINT_COMPLETE.json").exists():
        time.sleep(5)
    return path


def load(lab, path):
    from safetensors.torch import load_file
    model = lab.rt.models["F"]
    model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
    model.eval()


def snapshots():
    return {r["init"]: r for r in json.loads((SOURCE / "snapshots/snapshots.json").read_text())["records"]}


def rebuild(lab, snap):
    obs, rows = lab.rebuild(snap["init"], [lab.np.asarray(x, dtype=lab.np.float32) for x in snap["controls"]], snap["z0"])
    assert lab.ctx.signature(obs) == snap["signature"]
    return obs, rows


def make_bank(lab, inits):
    torch = lab.torch
    bank = []
    saved = snapshots()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for init in inits:
            snap = saved[init]
            obs, _ = rebuild(lab, snap)
            obs = copy.deepcopy(obs)
            prefix = {g: lab.rt.prefix("R", obs, INSTR[g]) for g in "AB"}
            for route in "AB":
                drafts = []
                def field(x, t):
                    drafts.append((x.clone(), t.clone()))
                    return lab.rt.query("R", prefix[route], x, t)
                lab.rt.integrate(field, lab.rt.noise(snap["seed"], snap["boundary"] // 10), 10)
                for index in (0, 4, 8):
                    x, t = drafts[index]
                    values = [lab.rt.query("R", prefix[g], x, t).float() for g in "AB"]
                    bank.append((init, route, index, obs, x, t, *values))
    return bank


def responses(lab, bank, by_channel=False):
    torch = lab.torch
    result = []
    current = None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for init, route, index, obs, x, t, old_a, old_b in bank:
            if current != init:
                prefix = {g: lab.rt.prefix("F", obs, INSTR[g]) for g in "AB"}
                current = init
            a, b = [lab.rt.query("F", prefix[g], x, t).float() for g in "AB"]
            old_d, d = (old_a - old_b) / 2, (a - b) / 2
            mu = (a + b) / 2
            drift = mu - (old_a + old_b) / 2
            row = {"init": init, "route": route, "denoise_index": index}
            for n in (10, 50):
                ref = old_d[:, :n, :7].flatten()
                cur = d[:, :n, :7].flatten()
                norm = float(ref.norm())
                inner = float((ref * cur).sum())
                gain = inner / max(float(ref.square().sum()), 1e-12)
                row[str(n)] = {
                    "gain": gain,
                    "norm_ratio": float(cur.norm()) / max(norm, 1e-6),
                    "cosine": inner / max(norm * float(cur.norm()), 1e-12),
                    "noncollinear_ratio": float((cur - gain * ref).norm()) / max(norm, 1e-6),
                    "parent_contrast_rms": float(ref.square().mean().sqrt()),
                    "mu_drift_rms": {k: float(drift[:, :n, v].square().mean().sqrt()) for k, v in BLOCKS.items()},
                }
                if by_channel:
                    row[str(n)]["contrast_by_channel"] = {}
                    for k in "HZRG":
                        block = BLOCKS[k]
                        block_ref, block_cur = old_d[:, :n, block], d[:, :n, block]
                        energy = float(block_ref.square().sum())
                        dot = float((block_ref * block_cur).sum())
                        row[str(n)]["contrast_by_channel"][k] = {
                            "gain": dot / max(energy, 1e-12),
                            "energy_fraction": energy / max(float(ref.square().sum()), 1e-12),
                            "parent_contrast_rms": float(block_ref.square().mean().sqrt()),
                        }
            result.append(row)
    return result


def probe(lab, steps, by_channel=False):
    bank = make_bank(lab, range(20, 40))
    for step in steps:
        path = checkpoint(lab, step)
        load(lab, path)
        rows = responses(lab, bank, by_channel)
        assert len(rows) == 120
        if step == 0:
            error = max(abs(r["50"]["gain"] - 1) for r in rows)
            drift = max(v for r in rows for v in r["50"]["mu_drift_rms"].values())
            assert error < 1e-6 and drift == 0, (error, drift)
        prefix = "probe_channels" if by_channel else "probe"
        (RUN / f"{prefix}_{step:06d}.json").write_text(json.dumps(rows, allow_nan=False))
        print("PROBE", step, len(rows), flush=True)


def evaluate(lab, steps, lane, lanes):
    torch = lab.torch
    saved = snapshots()
    for step in steps:
        load(lab, checkpoint(lab, step))
        with (RUN / f"native_{step:06d}_w{lane}.jsonl").open("x") as handle:
            for init in list(range(20, 40))[lane::lanes]:
                snap = saved[init]
                for goal in "ABN":
                    if goal == "N":
                        obs, _ = lab.ctx.reset(init)
                        z0 = float(obs[lab.ctx.obj][2])
                        rows = []
                    else:
                        obs, rows = rebuild(lab, snap)
                        z0 = snap["z0"]
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        for r in range(len(rows) // 10, 40):
                            chunk = lab.native("F", INSTR[goal], obs, snap["seed"], r)
                            obs, part = lab.execute(obs, lab.rt.physical(chunk), r * 10, z0)
                            rows.extend(part)
                    assert len(rows) == 400
                    move = lab.ctx.geo.movement("scene2", lab.ctx.geometry, rows, z0, None)
                    row = {"init": init, "step": step, "goal": goal, "first": move["first_targets"], "first_step": move["first_step"],
                           "ever": {g: any(x["pred"][g] for x in rows) for g in lab.ctx.geo.GOALS}, "terminal": rows[-1]["pred"]}
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    handle.flush()
                    print("NATIVE", step, init, goal, row["first"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("probe", "native"))
    p.add_argument("--steps", nargs="+", type=int)
    p.add_argument("--lane", type=int, default=0)
    p.add_argument("--lanes", type=int, default=1)
    p.add_argument("--by-channel", action="store_true")
    args = p.parse_args()
    lab = Lab5()
    try:
        if args.mode == "probe":
            probe(lab, args.steps or STEPS, args.by_channel)
        else:
            evaluate(lab, args.steps or (0, 1000, 3000, 6000, 12000), args.lane, args.lanes)
    finally:
        lab.ctx.env.close()
