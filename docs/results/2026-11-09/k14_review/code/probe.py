"""Engineering probe on dev inits 0-9 (Front-12000 executing Front; no intervention outcomes).
Logs grasp signals per step and d padding / temporal spectrum at boundaries."""
import json, math, sys
from pathlib import Path
SERVER = Path("/home/lzx/cl_binding/.server")
sys.path[:0] = [str(SERVER / "k13_r1_front_20_20261107_01")]
from k11_reference import km
from r1_front_start import INSTR, PAIR, EXPECTED
OUT = Path(__file__).resolve().parent.parent / "probe"
OUT.mkdir(exist_ok=True)
ctx = km.Ctx("scene2"); torch, np = ctx.torch, ctx.np
torch.set_num_threads(2)
rt = km.make_runtime(ctx, PAIR, ["R", "F"]); assert rt.sha == EXPECTED
def contacts():
    sim = ctx.env.env.sim; m = sim.model
    gid = {m.geom_id2name(i): i for i in range(m.ngeom)}
    F1 = {gid["gripper0_finger1_collision"], gid["gripper0_finger1_pad_collision"]}
    F2 = {gid["gripper0_finger2_collision"], gid["gripper0_finger2_pad_collision"]}
    BOOK = {gid[n] for n in ("black_book_1", "black_book_1_g0", "black_book_1_g1") if n in gid}
    c1 = c2 = False
    for c in sim.data.contact[:sim.data.ncon]:
        pair = {c.geom1, c.geom2}
        if pair & BOOK:
            c1 |= bool(pair & F1); c2 |= bool(pair & F2)
    return c1, c2
D = np.array([[math.cos(math.pi * (n + .5) * k / 50) for n in range(50)] for k in range(50)])
D /= np.linalg.norm(D, axis=1, keepdims=True)
def dstats(obs, seed, r):
    x = rt.noise(seed, r); t = torch.ones(1, dtype=torch.float32, device=x.device)
    v = {(role, g): rt.query(role, rt.prefix(role, obs, INSTR[g]), x, t)[0].double().cpu().numpy() for role in "RF" for g in "AB"}
    out = {}
    for role in "RF":
        d = (v[role, "A"] - v[role, "B"]) / 2
        real, pad = d[:, :7], d[:, 7:]
        spec = (D @ real) ** 2
        cum = np.cumsum(spec.sum(1)) / spec.sum()
        out[role] = {"a": float(np.linalg.norm(real)), "pad_over_real": float(np.linalg.norm(pad) / np.linalg.norm(real)),
                     "chan_rms": [float(x) for x in np.sqrt((real ** 2).mean(0))],
                     "dct_cum_k": [float(cum[k]) for k in (1, 2, 3, 4, 5, 7, 10, 15)]}
    return out
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for init in range(10):
        seed = km.noise_seed("scene2", init)
        obs, _ = ctx.reset(init)
        z0 = float(obs[ctx.obj][2]); rows = []; bstats = {}
        for r in range(20):
            if 3 <= r <= 15:
                bstats[r * 10] = dstats(obs, seed, r)
            chunk = rt.sample("NAT-FT4", "A", obs, {"A": INSTR["N"], "B": INSTR["N"]}, seed, r)[0]
            ctl = rt.physical(chunk)
            for k in range(10):
                obs, ex = ctx.step(ctl[k])
                if k == 9: obs = ctx.fresh(ctx.env)
                row = ctx.row(r * 10 + k + 1, obs, ctl[k], ex)
                c1, c2 = contacts()
                row.update(c1=c1, c2=c2, relpos=[float(x) for x in obs["black_book_1_to_robot0_eef_pos"]],
                           relquat=[float(x) for x in obs["black_book_1_to_robot0_eef_quat"]],
                           gq=[float(x) for x in obs["robot0_gripper_qpos"]], hits=ctx.geo.hits("scene2", ctx.geometry, row["obj"], z0, None))
                rows.append(row)
        (OUT / f"i{init}.json").write_text(json.dumps({"init": init, "z0": z0, "rows": rows, "bstats": bstats}))
        print(init, "done", flush=True)
