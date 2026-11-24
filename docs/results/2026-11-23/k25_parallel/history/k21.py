"""K21 life-or-death transfer test of 'old common response + fixed gain on the adapted model's own instruction response'.
Lines = (family, parent R, adapted F, old goals A/B, new goal N). Rules: ../prereg/k21_prereg.json.
Phases: calib (alpha on dev inits, no behaviour used) | gen (fresh physical resets per family) | screen | lock | worker."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

SERVER = Path("/home/lzx/cl_binding/.server")
OUT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(SERVER / "k1_screen_20261101_01/code")]
import k1_main as km

RT = "/home/lzx/cl_binding_stage2a/"
OLD = ("left", "front", "right")
LINES = {
    "r0_s2": dict(family="scene2", R=RT + "outputs/r0/dual_left_front_20261023_01/checkpoints/012000",
                  F=RT + "outputs/stage2a/stage2a_ft4_er_20261024_01_ft4/checkpoints/012000", goals=OLD, alpha=None),
    "c_s2a": dict(family="scene2", R=RT + "outputs/k1confirm/r0_scene2_seed20261102/checkpoints/012000",
                  F=RT + "outputs/k1confirm/scene2_seed20261102_ft4/checkpoints/012000", goals=OLD, alpha=None),
    "c_s2b": dict(family="scene2", R=RT + "outputs/k1confirm/r0_scene2_seed20261103/checkpoints/012000",
                  F=RT + "outputs/k1confirm/scene2_seed20261103_ft4/checkpoints/012000", goals=OLD, alpha=None),
    "scene1b": dict(family="scene1b", R=RT + "outputs/stage3/scene1b_dual_left_front_cosine_20261025_01/checkpoints/012000",
                    F=RT + "outputs/stage2a/stage3b_scene1_ft4_er_20261025_01_ft4/checkpoints/012000", goals=OLD, alpha=None),
    "goal": dict(family="goal", R=RT + "outputs/stage3/goal_dual_constant_20261025_01/checkpoints/012000",
                 F=RT + "outputs/stage2a/stage3g_goal_ft4_er_20261025_01_ft4/checkpoints/012000", goals=OLD, alpha=None),
    "c_goal": dict(family="goal", R=RT + "outputs/k1confirm/r0_goal_seed20261102/checkpoints/012000",
                   F=RT + "outputs/k1confirm/goal_seed20261102_ft4/checkpoints/012000", goals=OLD, alpha=None),
    "r1_front": dict(family="scene2", R=str(SERVER / "r1_left_right_20261104_01/train_r1/checkpoints/024000"),
                     F=str(SERVER / "r1_then_front_v1/train/checkpoints/012000"), goals=("left", "right", "front"), alpha=2.959),
}
RESET_BASE = {"scene2": 31110000, "scene1b": 31120000, "goal": 31130000}
ID_BASE = {"scene2": 1000, "scene1b": 2000, "goal": 3000}
NOISE_BASE = 77310001
N_CAND, NEED, MIN_SNAP = 40, 20, 12
BMIN, BMAX, RETAIN, WINDOW, HORIZON, CAP = 30, 150, 30, 60, 400, 10.0
WINDOW_GROUPS = ("B1", "NG", "S", "F_old")
JOBS = [(g, k) for g in ("A_nat", "B1", "NG", "S", "F_old", "R_full") for k in "AB"] + [("S", "N")]


def jdump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False) + "\n")
    tmp.replace(path)


class Lab:
    def __init__(self, line=None, family=None, models=True):
        cfg = LINES[line] if line else {"family": family, "goals": OLD}
        self.line, self.cfg, self.family = line, cfg, cfg["family"]
        self.ctx = km.Ctx(self.family)
        self.torch, self.np = self.ctx.torch, self.ctx.np
        self.torch.set_num_threads(2)
        self.goal = dict(zip("ABN", cfg["goals"]))
        self.text = {k: self.ctx.all_texts[g] for k, g in self.goal.items()}
        self.base = self.ctx.obj[:-4]
        if models:
            self.rt = km.make_runtime(self.ctx, {"R0": cfg["R"], "FT4": cfg["F"]}, ["R", "F"])
            p = OUT / "lines" / line / "alpha.json"
            self.alpha = cfg["alpha"] if cfg["alpha"] else (json.loads(p.read_text())["alpha"] if p.exists() else None)

    # ---------------------------------------------------------------- physics
    def reset_rec(self, rec):
        ctx, np = self.ctx, self.np
        ctx.evaluation.reset_common(ctx.env, np.asarray(rec["state"], dtype=np.float64), rec["reset_seed"],
                                    ctx.root["evaluation"]["settle_steps"], ctx.xml)
        return ctx.fresh(ctx.env)

    def contacts(self):
        sim = self.ctx.env.env.sim
        m = sim.model
        f1 = f2 = obj = None
        if not hasattr(self, "_gid") or self._gid[0] != id(m):
            names = {m.geom_id2name(i): i for i in range(m.ngeom)}
            f1 = {names["gripper0_finger1_collision"], names["gripper0_finger1_pad_collision"]}
            f2 = {names["gripper0_finger2_collision"], names["gripper0_finger2_pad_collision"]}
            obj = {i for n, i in names.items() if n and (n == self.base or n.startswith(self.base + "_"))}
            assert obj, "no object geoms"
            self._gid = (id(m), f1, f2, obj)
        _, f1, f2, obj = self._gid
        c1 = c2 = False
        for c in sim.data.contact[:sim.data.ncon]:
            pair = {c.geom1, c.geom2}
            if pair & obj:
                c1 |= bool(pair & f1)
                c2 |= bool(pair & f2)
        return c1, c2

    def execute(self, obs, controls, start, z0, plate):
        ctx, events = self.ctx, []
        for k in range(10):
            obs, executed = ctx.step(controls[k])
            if k == 9:
                obs = ctx.fresh(ctx.env)
            row = ctx.row(start + k + 1, obs, controls[k], executed)
            c1, c2 = self.contacts()
            row.update(c1=c1, c2=c2, relpos=[float(x) for x in obs[self.base + "_to_robot0_eef_pos"]],
                       relquat=[float(x) for x in obs[self.base + "_to_robot0_eef_quat"]],
                       hits=ctx.geo.hits(self.family, ctx.geometry, row["obj"], z0, plate))
            events.append(row)
        return obs, events

    def native(self, role, text, obs, seed, r):
        return self.rt.sample("NAT-R0" if role == "R" else "NAT-FT4", "A", obs, {"A": text, "B": text}, seed, r)[0]

    @staticmethod
    def criterion(rows, z0, b):
        last5, last10 = rows[b - 5:b], rows[b - 10:b]
        out = {"lifted": all(r["obj"][2] - z0 > .03 for r in last5), "closed_cmd": all(r["a"][6] > 0 for r in last5),
               "near": all(math.dist(r["obj"], r["eef"]) < .15 for r in last5),
               "both_contacts": bool(rows[b - 1]["c1"] and rows[b - 1]["c2"]),
               "pretarget": not any(r["hits"] or any(r["pred"].values()) for r in rows[:b])}
        q0, q1 = last10[0]["relquat"], last10[-1]["relquat"]
        out["relpos_drift_m"] = math.dist(last10[0]["relpos"], last10[-1]["relpos"])
        out["relrot_drift_deg"] = 2 * math.degrees(math.acos(min(1.0, abs(sum(a * c for a, c in zip(q0, q1))))))
        out["stable_relpose"] = out["relpos_drift_m"] < .005 and out["relrot_drift_deg"] < 3.0
        out["ok"] = all(out[k] for k in ("lifted", "closed_cmd", "near", "both_contacts", "pretarget", "stable_relpose"))
        return out

    def rebuild(self, snap):
        obs = self.reset_rec(snap)
        rows = []
        for r, c in enumerate(snap["controls"]):
            obs, ev = self.execute(obs, self.np.asarray(c, dtype=self.np.float32), r * 10, snap["z0"], snap["plate"])
            rows.extend(ev)
        return obs, rows

    def a(self, z):
        return z[..., :7].float().flatten().norm()

    # ---------------------------------------------------------------- phases
    def gen(self):
        ctx, np = self.ctx, self.np
        refs = np.stack([np.asarray(x, dtype=np.float64)[1:] for group in ctx.states.values() for x in group])
        rows = []
        for i in range(N_CAND):
            seed = RESET_BASE[self.family] + i
            np.random.seed(seed)
            ctx.env.seed(seed)
            ctx.env.reset()
            state = np.asarray(ctx.env.get_sim_state(), dtype=np.float64).copy()
            rec = {"init": ID_BASE[self.family] + i, "state": state.tolist(), "reset_seed": seed, "seed": NOISE_BASE + 107 * i,
                   "min_dist_known": float(np.linalg.norm(refs - state[1:], axis=1).min()) if state[1:].shape == refs[0].shape else None}
            obs = self.reset_rec(rec)
            try:
                built = ctx.geo.build(self.family, ctx.env, np)
                for g in ctx.geometry:  # amendment 1: goal family has no static 'right' site (plate is dynamic)
                    x, y = built[g], ctx.geometry[g]
                    err = max(max(abs(u - v) for u, v in zip(x["centre"], y["centre"])), max(abs(u - v) for u, v in zip(x["half_size"], y["half_size"])),
                              max(abs(u - v) for ra, rb in zip(x["rotation"], y["rotation"]) for u, v in zip(ra, rb)))
                    assert err <= 1e-5, f"fixture {g} moved by {err}"
                rec["geometry_ok"] = True
            except Exception as e:  # fixture moved: candidate unusable with the fixed region geometry
                rec["geometry_ok"] = False
                rec["geometry_error"] = str(e)[:200]
            rec.update(signature=ctx.signature(obs), z0=float(obs[ctx.obj][2]), plate=ctx.plate_xy(obs),
                       obj_start=[float(x) for x in obs[ctx.obj]], pred_at_reset=ctx.flags())
            rows.append(rec)
            print("CAND", rec["init"], rec["geometry_ok"], rec["min_dist_known"], rec["obj_start"], flush=True)
        assert len({r["signature"]["sim"] for r in rows}) == len(rows)
        jdump(OUT / "states" / f"{self.family}.json", rows)

    def fields(self, obs, seed, r):
        torch = self.torch
        x = self.rt.noise(seed, r)
        t = torch.ones(1, dtype=torch.float32, device=x.device)
        v = {(role, g): self.rt.query(role, self.rt.prefix(role, obs, self.text[g]), x, t).float() for role in "RF" for g in "AB"}
        d = {role: (v[role, "A"] - v[role, "B"]) / 2 for role in "RF"}
        mu = {role: (v[role, "A"] + v[role, "B"]) / 2 for role in "RF"}
        vN = self.rt.query("F", self.rt.prefix("F", obs, self.text["N"]), x, t).float()
        u, w = d["R"][..., :7].flatten(), d["F"][..., :7].flatten()
        return {"ratio": float(u.norm() / w.norm().clamp_min(1e-30)), "cos": float((u * w).sum() / (u.norm() * w.norm()).clamp_min(1e-30)),
                "mu_shift_over_dold": float(self.a(mu["F"] - mu["R"]) / u.norm().clamp_min(1e-30)),
                "muF_to_N_over_muR_to_N": float(self.a(mu["F"] - vN) / self.a(mu["R"] - vN).clamp_min(1e-30))}

    def calib(self):
        stats = []
        with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            for init in range(10):
                seed = km.noise_seed(self.family, init)
                obs, _ = self.ctx.reset(init)
                z0, plate = float(obs[self.ctx.obj][2]), self.ctx.plate_xy(obs)
                for r in range(16):
                    if r >= 3:
                        stats.append({"init": init, "step": r * 10, **self.fields(obs, seed, r)})
                    obs, _ = self.execute(obs, self.rt.physical(self.native("F", self.text["N"], obs, seed, r)), r * 10, z0, plate)
                print("CALIB", self.line, init, flush=True)
        med = lambda k: sorted(s[k] for s in stats)[len(stats) // 2]
        alpha = min(CAP, med("ratio"))
        jdump(OUT / "lines" / self.line / "alpha.json", {"alpha": alpha, "n": len(stats), "median": {k: med(k) for k in ("ratio", "cos", "mu_shift_over_dold", "muF_to_N_over_muR_to_N")},
                                                         "rule": "median a_old/a_new, dev inits 0-9, adapted N-prefix, boundaries 30-150, first Euler input; capped 10", "stats": stats})
        print("ALPHA", self.line, alpha, flush=True)

    def screen(self, lane, lanes):
        cands = [r for r in json.loads((OUT / "states" / f"{self.family}.json").read_text()) if r["geometry_ok"] and not any(r["pred_at_reset"].values())]
        out = []
        with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            for rec in cands[lane::lanes]:
                obs = self.reset_rec(rec)
                assert self.ctx.signature(obs) == rec["signature"], "fresh reset not reproducible"
                controls, rows, sigs, chosen = [], [], {}, None
                for r in range((BMAX + RETAIN) // 10):
                    controls.append(self.rt.physical(self.native("F", self.text["N"], obs, rec["seed"], r)))
                    obs, ev = self.execute(obs, controls[-1], r * 10, rec["z0"], rec["plate"])
                    rows.extend(ev)
                    b = len(rows)
                    sigs[b] = self.ctx.signature(obs)
                    if chosen is None and BMIN <= b <= BMAX and self.criterion(rows, rec["z0"], b)["ok"]:
                        chosen = b
                    if chosen is not None and b >= chosen + RETAIN:
                        break
                res = {**rec, "boundary": chosen}
                if chosen is None:
                    lifted = any(x["obj"][2] - rec["z0"] > .03 for x in rows)
                    res["reason"] = "never_lifted" if not lifted else "region_before_stable" if any(x["hits"] or any(x["pred"].values()) for x in rows) else "unstable"
                else:
                    post = rows[chosen:chosen + RETAIN]
                    res.update(controls=[c.tolist() for c in controls[:chosen // 10]], handoff_signature=sigs[chosen],
                               retained_30=all(x["obj"][2] - rec["z0"] > .03 and x["c1"] and x["c2"] for x in post))
                    ok = []
                    for _ in range(2):
                        o, rr = self.rebuild(res)
                        ok.append(self.ctx.signature(o) == sigs[chosen] and rr == rows[:chosen])
                    res["rebuild_x2"] = ok
                    res["fields_at_handoff"] = self.fields(o, rec["seed"], chosen // 10)
                out.append(res)
                print("SCREEN", self.line, rec["init"], chosen, res.get("reason"), res.get("retained_30"), res.get("rebuild_x2"), flush=True)
        jdump(OUT / "lines" / self.line / f"screen_{lane}.json", out)

    # ---------------------------------------------------------------- recombination
    def combine(self, group, pre, key, x, t, diag):
        torch, rt = self.torch, self.rt
        if group == "F_old":
            return rt.query("R", pre["R", key], x, t)
        need_r = group != "NG"
        v = {k: rt.query(k[0], p, x, t).float() for k, p in pre.items() if k != "q" and (need_r or k[0] == "F")}
        vq = v["F", key] if key in "AB" else rt.query("F", pre["q"], x, t).float()
        muF, dF = (v["F", "A"] + v["F", "B"]) / 2, (v["F", "A"] - v["F", "B"]) / 2
        gain = 1.0 if group == "B1" else self.alpha
        if need_r:
            muR, dR = (v["R", "A"] + v["R", "B"]) / 2, (v["R", "A"] - v["R", "B"]) / 2
            out = muR + gain * (vq - muF)
            e, ref, w = (out - muR)[0, :, :7].flatten(), dR[0, :, :7].flatten(), dF[0, :, :7].flatten()
            diag["proj_on_dold"].append(float((e * ref).sum() / (ref * ref).sum().clamp_min(1e-30)))
            diag["new_over_old_norm"].append(float(w.norm() / ref.norm().clamp_min(1e-30)))
            diag["cos_new_old"].append(float((w * ref).sum() / (w.norm() * ref.norm()).clamp_min(1e-30)))
            diag["mu_shift_over_dold"].append(float(self.a(muF - muR) / ref.norm().clamp_min(1e-30)))
        else:
            out = muF + gain * (vq - muF)
        return out.to(rt.vdtype)

    def episode(self, snap, group, key):
        started = time.monotonic()
        b, z0, seed, plate = snap["boundary"], snap["z0"], snap["seed"], snap["plate"]
        obs, rows = self.rebuild(snap)
        assert self.ctx.signature(obs) == snap["handoff_signature"], "snapshot rebuild differs"
        diag, q = defaultdict(list), self.text[key]
        for r in range(b // 10, HORIZON // 10):
            if group in WINDOW_GROUPS and r * 10 < b + WINDOW:
                roles = ("F",) if group == "NG" else ("R", "F")
                pre = {(role, g): self.rt.prefix(role, obs, self.text[g]) for role in roles for g in "AB"}
                if key not in "AB":
                    pre["q"] = self.rt.prefix("F", obs, q)
                chunk = self.rt.integrate(lambda x, t: self.combine(group, pre, key, x, t, diag), self.rt.noise(seed, r), 10)[..., :7]
            else:
                chunk = self.native("R" if group == "R_full" else "F", q, obs, seed, r)
            obs, ev = self.execute(obs, self.rt.physical(chunk), r * 10, z0, plate)
            rows.extend(ev)
        assert len(rows) == HORIZON
        mv = self.ctx.geo.movement(self.family, self.ctx.geometry, rows, z0, plate)
        req = self.goal[key]
        win = rows[b:b + WINDOW]
        rec = {"line": self.line, "init": snap["init"], "group": group, "key": key, "requested": req, "boundary": b,
               "first": mv["first_targets"], "first_step": mv["first_step"],
               "ever": {g: any(x["pred"][g] for x in rows) for g in self.ctx.geo.GOALS}, "terminal": rows[-1]["pred"],
               "clip_window": sum(x["clip"] for x in win) / len(win), "contacts_window_end": [win[-1]["c1"], win[-1]["c2"]],
               "diag": {k: sum(v) / len(v) for k, v in diag.items()}, "seconds": time.monotonic() - started}
        return rec, [{k: x[k] for k in ("step", "obj", "eef", "pred", "c1", "c2")} for x in rows[b:]]

    def worker(self, lane, lanes):
        snaps = json.loads((OUT / "lines" / self.line / "snapshots.json").read_text())["selected"]
        d = OUT / "lines" / self.line / "episodes"
        d.mkdir(exist_ok=True)
        res, tr = d / f"w{lane}.jsonl", d / f"w{lane}_traces.jsonl"
        done = {(j["init"], j["group"], j["key"]) for j in map(json.loads, res.read_text().splitlines())} if res.exists() else set()
        assert self.alpha, "alpha missing"
        with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16), res.open("a") as fr, tr.open("a") as ft:
            for snap in snaps[lane::lanes]:
                for group, key in JOBS:
                    if (snap["init"], group, key) in done:
                        continue
                    rec, rows = self.episode(snap, group, key)
                    rec["alpha"] = self.alpha
                    ft.write(json.dumps({"init": snap["init"], "group": group, "key": key, "rows": rows}) + "\n"); ft.flush()
                    fr.write(json.dumps(rec, allow_nan=False) + "\n"); fr.flush()
                    print(self.line, snap["init"], group, key, rec["first"], rec["ever"][rec["requested"]], f"{rec['seconds']:.0f}s", flush=True)
        (d / f"w{lane}_COMPLETE.json").write_text(json.dumps({"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))


def lock(line, lanes):
    d = OUT / "lines" / line
    rows = sorted([r for i in range(lanes) for r in json.loads((d / f"screen_{i}.json").read_text())], key=lambda r: r["init"])
    ok = [r for r in rows if r["boundary"] and all(r["rebuild_x2"])]
    reasons = defaultdict(int)
    for r in rows:
        reasons[r.get("reason") or ("ok" if r in ok else "rebuild_failed")] += 1
    sel = ok[:NEED]
    jdump(d / "snapshots.json", {"line": line, "candidates": len(rows), "qualified": len(ok), "reasons": dict(reasons),
                                 "retained_30": sum(bool(r["retained_30"]) for r in sel), "selected": sel,
                                 "rule": "first 20 in predetermined seed order qualifying on the adapted model's N-prefix handoff"})
    print("LOCK", line, len(rows), len(ok), dict(reasons), [r["init"] for r in sel], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("calib", "gen", "screen", "lock", "worker"))
    p.add_argument("--line")
    p.add_argument("--family")
    p.add_argument("--lane", type=int, default=0)
    p.add_argument("--lanes", type=int, default=10)
    a = p.parse_args()
    if a.phase == "lock":
        lock(a.line, a.lanes)
    else:
        lab = Lab(a.line, a.family, models=a.phase != "gen")
        {"calib": lab.calib, "gen": lab.gen, "screen": lambda: lab.screen(a.lane, a.lanes), "worker": lambda: lab.worker(a.lane, a.lanes)}[a.phase]()
        lab.ctx.env.close()
