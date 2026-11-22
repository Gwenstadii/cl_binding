"""K14 batch 1: common Front-prefix snapshots; amplitude vs content of the conditional difference (R1 -> Front-12000).

Phases: screen (snapshots, no interventions) -> gates (engineering identities) -> worker (episodes) ; analysis in k14_analyze.py.
Design and locked rules: ../prereg/k14_prereg.json.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

SERVER = Path("/home/lzx/cl_binding/.server")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent
sys.path[:0] = [str(SERVER / "k13_r1_front_20_20261107_01"), str(SERVER / "mu_d_deep_audit_20261107_01")]
import k11_reference as hist_ref                 # historical aliased capture (K13)
import k11_reference_capture_fixed as pure_ref   # copy capture (11-07 audit)
from k11_reference import km, modules, SITES
from r1_front_start import INSTR, PAIR, EXPECTED

POOL, SUPPLEMENT, NEED = list(range(20, 40)), list(range(0, 20)), 20
BMIN, BMAX, RETAIN, WINDOW, HORIZON = 30, 150, 30, 60, 400
CAP, DEGEN, K_DCT = 10.0, 1e-4, 5
WINDOW_GROUPS = ("B_mean", "C_amp", "D_hist", "D_pure", "E_hist", "E_pure", "F_old", "G_rand")
GROUPS = ("A_nat",) + WINDOW_GROUPS + ("R_full",)
LABEL = {"A": "left", "B": "right", "N": "front"}


def jdump(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


class Lab:
    def __init__(self):
        self.ctx = km.Ctx("scene2")
        self.torch, self.np = self.ctx.torch, self.ctx.np
        self.torch.set_num_threads(2)
        self.rt = km.make_runtime(self.ctx, PAIR, ["R", "F"])
        assert self.rt.sha == EXPECTED
        self.sites = {role: modules(m) for role, m in self.rt.models.items()}
        np = self.np
        basis = np.array([[math.cos(math.pi * (n + .5) * k / 50) for n in range(50)] for k in range(K_DCT)])
        self.dct = self.torch.tensor(basis / np.linalg.norm(basis, axis=1, keepdims=True), dtype=self.torch.float32).cuda()

    # ------------------------------------------------------------ physics helpers
    def contacts(self):
        sim = self.ctx.env.env.sim
        m = sim.model
        gid = {m.geom_id2name(i): i for i in range(m.ngeom)}
        f1 = {gid["gripper0_finger1_collision"], gid["gripper0_finger1_pad_collision"]}
        f2 = {gid["gripper0_finger2_collision"], gid["gripper0_finger2_pad_collision"]}
        book = {gid[n] for n in ("black_book_1", "black_book_1_g0", "black_book_1_g1") if n in gid}
        c1 = c2 = False
        for c in sim.data.contact[:sim.data.ncon]:
            pair = {c.geom1, c.geom2}
            if pair & book:
                c1 |= bool(pair & f1)
                c2 |= bool(pair & f2)
        return c1, c2

    def row(self, step, obs, action, executed, z0):
        ctx = self.ctx
        row = ctx.row(step, obs, action, executed)
        c1, c2 = self.contacts()
        row.update(c1=c1, c2=c2, relpos=[float(x) for x in obs["black_book_1_to_robot0_eef_pos"]],
                   relquat=[float(x) for x in obs["black_book_1_to_robot0_eef_quat"]],
                   hits=ctx.geo.hits("scene2", ctx.geometry, row["obj"], z0, None))
        return row

    def execute(self, obs, controls, start, z0):
        events = []
        for k in range(10):
            obs, executed = self.ctx.step(controls[k])
            if k == 9:
                obs = self.ctx.fresh(self.ctx.env)
            events.append(self.row(start + k + 1, obs, controls[k], executed, z0))
        return obs, events

    def native(self, role, text, obs, seed, r):
        return self.rt.sample("NAT-R0" if role == "R" else "NAT-FT4", "A", obs, {"A": text, "B": text}, seed, r)[0]

    # ------------------------------------------------------------ snapshot criterion (locked)
    @staticmethod
    def criterion(rows, z0, b):
        last5, last10 = rows[b - 5:b], rows[b - 10:b]
        out = {
            "lifted": all(r["obj"][2] - z0 > .03 for r in last5),
            "closed_cmd": all(r["a"][6] > 0 for r in last5),
            "near": all(math.dist(r["obj"], r["eef"]) < .15 for r in last5),
            "both_contacts": bool(rows[b - 1]["c1"] and rows[b - 1]["c2"]),
            "pretarget": not any(r["hits"] or any(r["pred"].values()) for r in rows[:b]),
        }
        p0, p1 = last10[0]["relpos"], last10[-1]["relpos"]
        q0, q1 = last10[0]["relquat"], last10[-1]["relquat"]
        out["relpos_drift_m"] = math.dist(p0, p1)
        out["relrot_drift_deg"] = 2 * math.degrees(math.acos(min(1.0, abs(sum(a * c for a, c in zip(q0, q1))))))
        out["stable_relpose"] = out["relpos_drift_m"] < .005 and out["relrot_drift_deg"] < 3.0
        out["ok"] = all(out[k] for k in ("lifted", "closed_cmd", "near", "both_contacts", "pretarget", "stable_relpose"))
        return out

    def rebuild(self, init, controls, z0):
        obs, _ = self.ctx.reset(init)
        rows = []
        for r, c in enumerate(controls):
            obs, ev = self.execute(obs, c, r * 10, z0)
            rows.extend(ev)
        return obs, rows

    def screen(self):
        records, qualified = [], 0
        order = POOL + SUPPLEMENT
        for idx, init in enumerate(order):
            if idx >= len(POOL) and qualified >= NEED:
                break
            seed = km.noise_seed("scene2", init)
            obs, _ = self.ctx.reset(init)
            z0 = float(obs[self.ctx.obj][2])
            controls, rows, sigs = [], [], {}
            with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                for r in range((BMAX + RETAIN) // 10):
                    controls.append(self.rt.physical(self.native("F", INSTR["N"], obs, seed, r)))
                    obs, ev = self.execute(obs, controls[-1], r * 10, z0)
                    rows.extend(ev)
                    sigs[(r + 1) * 10] = self.ctx.signature(obs)
            checks = {b: self.criterion(rows, z0, b) for b in range(BMIN, BMAX + 1, 10)}
            chosen = next((b for b, c in checks.items() if c["ok"]), None)
            rec = {"init": init, "seed": seed, "pool": "main" if init in POOL else "supplement", "z0": z0,
                   "boundary": chosen, "checks": {str(b): c for b, c in checks.items()}}
            if chosen is None:
                first_hit = next((r["step"] for r in rows if r["hits"] or any(r["pred"].values())), None)
                ever_ok = {k: any(c[k] for c in checks.values()) for k in ("lifted", "both_contacts", "stable_relpose")}
                rec["reason"] = ("never_lifted" if not ever_ok["lifted"] else "never_both_contacts" if not ever_ok["both_contacts"]
                                 else "reached_region_before_stable" if first_hit is not None and first_hit <= BMAX else "unstable")
            else:
                post = rows[chosen:chosen + RETAIN]
                rec["retained_30"] = all(r["obj"][2] - z0 > .03 and r["c1"] and r["c2"] for r in post)
                rec["signature"] = sigs[chosen]
                rec["controls"] = [c.tolist() for c in controls[:chosen // 10]]
                rec["boundary_state"] = {k: rows[chosen - 1][k] for k in ("obj", "eef", "relpos", "relquat", "c1", "c2", "a")}
                rebuilt = []
                for _ in range(3):
                    o, rr = self.rebuild(init, controls[:chosen // 10], z0)
                    rebuilt.append(self.ctx.signature(o) == sigs[chosen] and rr == rows[:chosen])
                rec["rebuild_x3"] = rebuilt
                np = self.np
                sim_state = np.asarray(self.ctx.env.get_sim_state(), dtype=np.float64)
                np.savez_compressed(OUT / "snapshots" / f"i{init}.npz", sim_state=sim_state,
                                    agentview=o["agentview_image"], wrist=o["robot0_eye_in_hand_image"],
                                    eef_pos=o["robot0_eef_pos"], eef_quat=o["robot0_eef_quat"], gripper=o["robot0_gripper_qpos"])
                qualified += 1
            records.append(rec)
            print(init, chosen, rec.get("reason"), rec.get("retained_30"), rec.get("rebuild_x3"), flush=True)
        jdump(OUT / "snapshots" / "snapshots.json", {"records": records, "qualified": qualified,
                                                     "selected": [r["init"] for r in records if r["boundary"] is not None][:NEED]})

    # ------------------------------------------------------------ recombination at one (x, t)
    def readout(self, role, prefix, x, t, patch=None, hooked=True):
        alias, pure = {}, {}

        def tap(name):
            def hook(_m, _i, out):
                alias[name] = out.detach()          # historical semantics: aliases the later in-place residual add
                pure[name] = out.detach().clone()   # module output at hook time
                if patch is not None:
                    return (out.float() + patch[name]).to(out.dtype)
            return hook
        handles = [m.register_forward_hook(tap(n)) for n, m in self.sites[role].items()] if hooked else []
        try:
            v = self.rt.query(role, prefix, x, t)
        finally:
            for h in handles:
                h.remove()
        return v, alias, pure

    def a(self, z):
        return z[..., :7].float().flatten().norm()

    def random_diff(self, dR, seed):
        torch = self.torch
        g = torch.Generator(device="cpu").manual_seed(seed)
        coef = torch.randn((K_DCT, 7), generator=g, dtype=torch.float32).cuda()
        real = dR[0, :, :7].float()
        rms = real.square().mean(0).sqrt()
        r = (self.dct.T.unsqueeze(-1) * coef.unsqueeze(0)).sum(1)  # elementwise: immune to bf16 autocast
        r = r / r.square().mean(0).sqrt().clamp_min(1e-12) * rms
        u, f = real.flatten(), r.flatten()
        f = f - (f * u).sum() / (u * u).sum().clamp_min(1e-30) * u
        f = f * (u.norm() / f.norm().clamp_min(1e-30))
        out = dR.clone().float()
        out[0, :, :7] = f.view(50, 7)
        return out

    def combine(self, group, prefixes, x, t, seed, diag):
        torch = self.torch
        need_sites = group in ("D_hist", "D_pure", "E_hist", "E_pure")
        st = {(role, g): self.readout(role, prefixes[role, g], x, t, hooked=need_sites) for role in "RF" for g in "AB"}
        v = {k: s[0].float() for k, s in st.items()}
        mu = {role: (v[role, "A"] + v[role, "B"]) / 2 for role in "RF"}
        d = {role: (v[role, "A"] - v[role, "B"]) / 2 for role in "RF"}
        aR, aF = self.a(d["R"]), self.a(d["F"])
        cap = degen = 0
        if group == "B_mean":
            diff = d["F"]
        elif group == "F_old":
            diff = d["R"]
        elif group == "C_amp":
            if aF < DEGEN * self.a(mu["R"]):
                diff, degen = torch.zeros_like(d["F"]), 1
            else:
                s = aR / aF
                cap = int(s > CAP)
                diff = d["F"] * s.clamp(max=CAP)
        elif group == "G_rand":
            diff = self.random_diff(d["R"], seed)
        else:
            idx = 1 if group.endswith("hist") else 2
            patches = {}
            for name in SITES:
                c = {role: (st[role, "A"][idx][name].float() - st[role, "B"][idx][name].float()) / 2 for role in "RF"}
                patches[name] = c["R"] - c["F"]
            w = [self.readout("F", prefixes["F", g], x, t, {n: sign * p for n, p in patches.items()})[0].float()
                 for g, sign in (("A", 1), ("B", -1))]
            diff = (w[0] - w[1]) / 2
            diag["patch_over_old_norm"].append(float(self.a(diff) / aR))
            if group.startswith("E_"):
                ap = self.a(diff)
                if ap < DEGEN * self.a(mu["R"]):
                    diff, degen = torch.zeros_like(diff), 1
                else:
                    s = aR / ap
                    cap = int(s > CAP)
                    diff = diff * s.clamp(max=CAP)
        out = {g: (mu["R"] + sign * diff).to(self.rt.vdtype) for g, sign in (("A", 1), ("B", -1))}
        eff = ((out["A"].float() - out["B"].float()) / 2)[0, :, :7].flatten()
        ref = d["R"][0, :, :7].flatten()
        diag["old_proj_full"].append(float((eff * ref).sum() / (ref * ref).sum().clamp_min(1e-30)))
        diag["old_cos_full"].append(float((eff * ref).sum() / (eff.norm() * ref.norm()).clamp_min(1e-30)))
        diag["norm_over_old"].append(float(eff.norm() / ref.norm().clamp_min(1e-30)))
        e10, r10 = ((out["A"].float() - out["B"].float()) / 2)[0, :10, :3].flatten(), d["R"][0, :10, :3].flatten()
        diag["old_proj_10x3"].append(float((e10 * r10).sum() / (r10 * r10).sum().clamp_min(1e-30)))
        diag["new_over_old_norm"].append(float(aF / aR))
        diag["pad_over_real_diff"].append(float(diff[..., 7:].float().norm() / self.a(diff).clamp_min(1e-30)))
        diag["cap_hit"].append(cap)
        diag["degenerate"].append(degen)
        if group == "F_old":
            for g in "AB":
                assert torch.equal(out[g], st["R", g][0]), "F_old must equal native R1 exactly"
        return out

    # ------------------------------------------------------------ episodes
    def episode(self, snap, group, goal):
        started = time.monotonic()
        init, b, z0, seed = snap["init"], snap["boundary"], snap["z0"], snap["seed"]
        controls = [self.np.asarray(c, dtype=self.np.float32) for c in snap["controls"]]
        obs, rows = self.rebuild(init, controls, z0)
        assert self.ctx.signature(obs) == snap["signature"], "snapshot rebuild differs"
        diag = defaultdict(list)
        text = INSTR["N"] if group == "N_nat" else INSTR[goal]
        for r in range(b // 10, HORIZON // 10):
            if group in WINDOW_GROUPS and r * 10 < b + WINDOW:
                prefixes = {(role, g): self.rt.prefix(role, obs, INSTR[g]) for role in "RF" for g in "AB"}
                it = 0

                def velocity(x, t):
                    nonlocal it
                    it += 1
                    return self.combine(group, prefixes, x, t, seed * 10000 + r * 100 + it * 2, diag)[goal]
                chunk = self.rt.integrate(velocity, self.rt.noise(seed, r), 10)[..., :7]
                assert it == 10
            else:
                chunk = self.native("R" if group == "R_full" else "F", text, obs, seed, r)
            obs, ev = self.execute(obs, self.rt.physical(chunk), r * 10, z0)
            rows.extend(ev)
        assert len(rows) == HORIZON
        assert not any(r["hits"] or any(r["pred"].values()) for r in rows[:b])
        mv = self.ctx.geo.movement("scene2", self.ctx.geometry, rows, z0, None)
        req = "front" if group == "N_nat" else LABEL[goal]
        win = rows[b:b + WINDOW]
        rec = {"init": init, "group": group, "goal": goal, "requested": req, "seed": seed, "boundary": b,
               "first": mv["first_targets"], "first_step": mv["first_step"], "first_times": mv["times"],
               "ever": {g: any(r["pred"][g] for r in rows) for g in self.ctx.geo.GOALS},
               "ever_step": next((r["step"] for r in rows if r["pred"][req]), None),
               "terminal": rows[-1]["pred"],
               "clip_rate_window": sum(r["clip"] for r in win) / len(win), "clip_rate_all": sum(r["clip"] for r in rows[b:]) / (HORIZON - b),
               "contacts_at_window_end": [rows[b + WINDOW - 1]["c1"], rows[b + WINDOW - 1]["c2"]],
               "diagnostics": {k: [sum(v) / len(v), min(v), max(v)] for k, v in diag.items()},
               "n_window_queries": len(diag.get("cap_hit", [])), "seconds": time.monotonic() - started}
        return rec, rows

    def worker(self, inits, tag):
        snaps = {r["init"]: r for r in json.loads((OUT / "snapshots/snapshots.json").read_text())["records"]}
        res, tr = OUT / "episodes" / f"w{tag}.jsonl", OUT / "episodes" / f"w{tag}_traces.jsonl"
        done = set()
        if res.exists():
            done = {(j["init"], j["group"], j["goal"]) for j in map(json.loads, res.read_text().splitlines())}
        with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16), \
                res.open("a") as fr, tr.open("a") as ft:
            for init in inits:
                snap = snaps[init]
                jobs = [(g, q) for g in GROUPS for q in "AB"] + [("N_nat", "N")]
                for group, goal in jobs:
                    if (init, group, goal) in done:
                        continue
                    rec, rows = self.episode(snap, group, goal)
                    ft.write(json.dumps({"init": init, "group": group, "goal": goal, "rows": rows}) + "\n")
                    ft.flush()
                    fr.write(json.dumps(rec, allow_nan=False) + "\n")
                    fr.flush()
                    print(init, group, goal, rec["first"], rec["ever"][rec["requested"]], rec["terminal"][rec["requested"]],
                          f"{rec['seconds']:.0f}s", flush=True)
        (OUT / "episodes" / f"w{tag}_COMPLETE.json").write_text(json.dumps({"inits": inits, "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()}))

    # ------------------------------------------------------------ engineering gates
    def gates(self, inits):
        torch = self.torch
        snaps = {r["init"]: r for r in json.loads((OUT / "snapshots/snapshots.json").read_text())["records"]}
        rep = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for init in inits:
                s = snaps[init]
                controls = [self.np.asarray(c, dtype=self.np.float32) for c in s["controls"]]
                obs, _ = self.rebuild(init, controls, s["z0"])
                assert self.ctx.signature(obs) == s["signature"]
                r = s["boundary"] // 10
                x = self.rt.noise(s["seed"], r)
                t = torch.ones(1, dtype=torch.float32, device=x.device)
                pre = {(role, g): self.rt.prefix(role, obs, INSTR[g]) for role in "RF" for g in "AB"}
                row = {"init": init}
                for mine, ref_mod, ref_cond in (("D_hist", hist_ref, "joint"), ("D_pure", pure_ref, "joint"), ("B_mean", hist_ref, "mean")):
                    ours = self.combine(mine, pre, x, t, 0, defaultdict(list))
                    theirs = ref_mod.velocity_pair(self.rt, self.sites, pre, x, t, ref_cond, 0, defaultdict(list))
                    row[mine] = {g: {"bitwise": bool(torch.equal(ours[g], theirs[g])),
                                     "max_abs": float((ours[g].float() - theirs[g].float()).abs().max())} for g in "AB"}
                for g in ("A", "B"):
                    nat = self.rt.query("R", pre["R", g], x, t)
                    row.setdefault("F_old", {})[g] = bool(torch.equal(self.combine("F_old", pre, x, t, 0, defaultdict(list))[g], nat))
                dg = defaultdict(list)
                self.combine("G_rand", pre, x, t, 12345, dg)
                row["G_rand"] = {"cos": dg["old_cos_full"][0], "norm_ratio": dg["norm_over_old"][0]}
                dc = defaultdict(list)
                self.combine("C_amp", pre, x, t, 0, dc)
                row["C_amp"] = {"norm_ratio": dc["norm_over_old"][0], "cos": dc["old_cos_full"][0], "cap": dc["cap_hit"][0]}
                de = defaultdict(list)
                self.combine("E_pure", pre, x, t, 0, de)
                row["E_pure"] = {"norm_ratio": de["norm_over_old"][0], "cos": de["old_cos_full"][0]}
                hp = [defaultdict(list), defaultdict(list)]
                self.combine("D_hist", pre, x, t, 0, hp[0]); self.combine("D_pure", pre, x, t, 0, hp[1])
                row["patch_norm_over_old"] = {"hist": hp[0]["patch_over_old_norm"][0], "pure": hp[1]["patch_over_old_norm"][0]}
                rep.append(row)
                print(json.dumps(row), flush=True)
        jdump(OUT / "gates.json", rep)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("screen", "gates", "worker"))
    p.add_argument("--inits", type=int, nargs="*", default=[])
    p.add_argument("--tag", default="0")
    a = p.parse_args()
    (OUT / "snapshots").mkdir(exist_ok=True)
    (OUT / "episodes").mkdir(exist_ok=True)
    lab = Lab()
    if a.phase == "screen":
        lab.screen()
    elif a.phase == "gates":
        lab.gates(a.inits)
    else:
        lab.worker(a.inits, a.tag)
    lab.ctx.env.close()
