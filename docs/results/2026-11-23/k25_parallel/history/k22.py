"""K22 extensions after K21 (rules: ../prereg/k22_prereg.json).
p2: on K21 common snapshots, dissect the amplified residual: gain sweep, parallel / orthogonal split w.r.t. the old difference, norm matching, sign-only.
p1: from-reset protocol (no snapshot selection; window = first 150 steps) on all 7 lines incl. the bowl family."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

K21 = Path("/home/lzx/cl_binding/.server/k21_transfer_20261113_01")
OUT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(K21 / "code"))
import k21
from k21 import Lab, HORIZON, WINDOW, CAP

P2 = {"a1.5": ("full", 1.5), "a2": ("full", 2.0), "a4.5": ("full", 4.5), "par": ("par", None), "perp": ("perp", None), "norm": ("norm", None), "sign": ("sign", None)}
P1_GROUPS, P1_WINDOW, P1_N = ("A_nat", "B1", "NG", "S", "F_old", "R_full"), 150, 20


class Lab2(Lab):
    def combine2(self, mode, alpha, pre, key, x, t, diag):
        torch, rt = self.torch, self.rt
        v = {k: rt.query(k[0], p, x, t).float() for k, p in pre.items()}
        muF, muR = (v["F", "A"] + v["F", "B"]) / 2, (v["R", "A"] + v["R", "B"]) / 2
        dR = (v["R", "A"] - v["R", "B"]) / 2
        resid = v["F", key] - muF
        e = (alpha or self.alpha) * resid
        ref, real = dR[0, :, :7], resid[0, :, :7]
        c1 = (real * ref).sum() / (ref * ref).sum().clamp_min(1e-30)   # coefficient of the un-amplified residual on d_old
        c = (alpha or self.alpha) * c1
        diag["c1"].append(float(c1))
        diag["cos"].append(float((real * ref).sum() / (real.norm() * ref.norm()).clamp_min(1e-30)))
        diag["perp_over_old"].append(float((self.alpha * (real - c1 * ref)).norm() / ref.norm().clamp_min(1e-30)))
        if mode == "full":
            out = muR + e
        elif mode == "par":
            out = muR + c * dR
        elif mode == "perp":
            out = muR + e - c * dR
        elif mode == "norm":
            out = muR + resid * (ref.norm() / real.norm().clamp_min(1e-30)).clamp(max=CAP)
        elif mode == "sign":
            out = muR + torch.sign(c1) * dR
        return out.to(rt.vdtype)

    def finish(self, rows, rec, key, z0, plate, b, diag, started, extra):
        mv = self.ctx.geo.movement(self.family, self.ctx.geometry, rows, z0, plate)
        req = self.goal[key]
        return {"line": self.line, "init": rec["init"], "key": key, "requested": req, "boundary": b, "first": mv["first_targets"], "first_step": mv["first_step"],
                "ever": {g: any(x["pred"][g] for x in rows) for g in self.ctx.geo.GOALS}, "terminal": rows[-1]["pred"],
                "diag": {k: [sum(v) / len(v), sum(x > 0 for x in v) / len(v)] for k, v in diag.items()}, "seconds": time.monotonic() - started, **extra}

    def ep2(self, snap, name, key):
        started, (mode, alpha) = time.monotonic(), P2[name]
        b, z0, seed, plate = snap["boundary"], snap["z0"], snap["seed"], snap["plate"]
        obs, rows = self.rebuild(snap)
        assert self.ctx.signature(obs) == snap["handoff_signature"]
        diag = defaultdict(list)
        for r in range(b // 10, HORIZON // 10):
            if r * 10 < b + WINDOW:
                pre = {(role, g): self.rt.prefix(role, obs, self.text[g]) for role in "RF" for g in "AB"}
                chunk = self.rt.integrate(lambda x, t: self.combine2(mode, alpha, pre, key, x, t, diag), self.rt.noise(seed, r), 10)[..., :7]
            else:
                chunk = self.native("F", self.text[key], obs, seed, r)
            obs, ev = self.execute(obs, self.rt.physical(chunk), r * 10, z0, plate)
            rows.extend(ev)
        return self.finish(rows, snap, key, z0, plate, b, diag, started, {"group": name})

    def ep1(self, rec, group, key):
        started = time.monotonic()
        z0, seed, plate = rec["z0"], rec["seed"], rec["plate"]
        obs = self.reset_rec(rec)
        assert self.ctx.signature(obs) == rec["signature"]
        rows, diag = [], defaultdict(list)
        for r in range(HORIZON // 10):
            inwin = r * 10 < P1_WINDOW
            if inwin and group in ("B1", "NG", "S"):
                roles = ("F",) if group == "NG" else ("R", "F")
                pre = {(role, g): self.rt.prefix(role, obs, self.text[g]) for role in roles for g in "AB"}
                chunk = self.rt.integrate(lambda x, t: self.combine(group, pre, key, x, t, diag), self.rt.noise(seed, r), 10)[..., :7]
            else:
                role = "R" if group == "R_full" or (group == "F_old" and inwin) else "F"
                chunk = self.native(role, self.text[key], obs, seed, r)
            obs, ev = self.execute(obs, self.rt.physical(chunk), r * 10, z0, plate)
            rows.extend(ev)
        lifted = any(x["obj"][2] - z0 > .03 for x in rows)
        return self.finish(rows, rec, key, z0, plate, 0, diag, started, {"group": group, "lifted": lifted})

    def run(self, part, lane, lanes):
        if part == "p2":
            items = json.loads((K21 / "lines" / self.line / "snapshots.json").read_text())["selected"]
            jobs, fn = [(g, k) for g in P2 for k in "AB"], self.ep2
        else:
            cands = json.loads((K21 / "states" / f"{self.family}.json").read_text())
            items = [r for r in cands if r["geometry_ok"] and not any(r["pred_at_reset"].values())][:P1_N]
            jobs, fn = [(g, k) for g in P1_GROUPS for k in "AB"], self.ep1
        d = OUT / part / self.line
        d.mkdir(parents=True, exist_ok=True)
        res = d / f"w{lane}.jsonl"
        done = {(j["init"], j["group"], j["key"]) for j in map(json.loads, res.read_text().splitlines())} if res.exists() else set()
        with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16), res.open("a") as fr:
            for it in items[lane::lanes]:
                for g, k in jobs:
                    if (it["init"], g, k) in done:
                        continue
                    rec = fn(it, g, k)
                    rec["alpha"] = self.alpha
                    fr.write(json.dumps(rec, allow_nan=False) + "\n"); fr.flush()
                    print(part, self.line, it["init"], g, k, rec["first"], f"{rec['seconds']:.0f}s", flush=True)
        (d / f"w{lane}_COMPLETE.json").write_text(json.dumps({"sha": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("part", choices=("p1", "p2"))
    p.add_argument("--line", required=True)
    p.add_argument("--lane", type=int, default=0)
    p.add_argument("--lanes", type=int, default=10)
    a = p.parse_args()
    lab = Lab2(a.line)
    lab.run(a.part, a.lane, a.lanes)
    lab.ctx.env.close()
