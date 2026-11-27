"""K14 §5: language routing without target-sign selection.
v(q) = mu_old + ALPHA * (v_new(q) - mu_new); mu_old, mu_new from the canonical Left/Right pair only; ALPHA fixed
(dev probe inits 0-9, median a_old/a_new = 2.959). q = the full text actually given; no code reads q's target to choose a sign.
Phases: verify (R1 from reset, inits 0-9, canonical + paraphrase templates) -> main (20 common snapshots)."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from k14 import Lab, OUT, INSTR, HORIZON, WINDOW, km

ALPHA = 2.959
S5 = OUT / "s5"
TEMPLATES = {
    "P1": "put the book into the {s} compartment of the caddy",
    "P2": "place the book in the caddy's {s} compartment",
    "P3": "move the book to the {s} section of the caddy",
    "P4": "pick up the book and put it in the {s} slot of the caddy",
    # amendment 1 (after verification round 1, before any main episode): one element changed at a time
    "Q1": "pick up the book and place it in the {s} slot of the caddy",
    "Q2": "pick up the book and place it into the {s} compartment of the caddy",
    "Q3": "grab the book and place it in the {s} compartment of the caddy",
    "Q4": "take the book and put it in the {s} compartment of the caddy",
}
ROUND = {"canon": 1, "P1": 1, "P2": 1, "P3": 1, "P4": 1, "Q1": 2, "Q2": 2, "Q3": 2, "Q4": 2}
NEUTRAL = "pick up the book and place it in the caddy"
SIDE = {"L": "left", "R": "right"}
KEEP_MAX = 3


def text(tpl, side):
    return INSTR["A" if side == "L" else "B"] if tpl == "canon" else TEMPLATES[tpl].format(s=SIDE[side])


class Lab5(Lab):
    def corrected(self, prefixes, qpre, x, t, diag):
        v = {k: self.rt.query(k[0], p, x, t).float() for k, p in prefixes.items()}
        vq = self.rt.query("F", qpre, x, t).float()
        muR, muF = (v["R", "A"] + v["R", "B"]) / 2, (v["F", "A"] + v["F", "B"]) / 2
        dR = (v["R", "A"] - v["R", "B"]) / 2
        out = muR + ALPHA * (vq - muF)
        e, r = (out - muR)[0, :, :7].flatten(), dR[0, :, :7].flatten()
        diag["proj_on_dold"].append(float((e * r).sum() / (r * r).sum().clamp_min(1e-30)))
        diag["norm_over_old"].append(float(e.norm() / r.norm().clamp_min(1e-30)))
        return out.to(self.rt.vdtype)

    def from_reset(self, init, q):
        seed = km.noise_seed("scene2", init)
        obs, _ = self.ctx.reset(init)
        z0 = float(obs[self.ctx.obj][2])
        rows = []
        for r in range(HORIZON // 10):
            chunk = self.native("R", q, obs, seed, r)
            obs, ev = self.execute(obs, self.rt.physical(chunk), r * 10, z0)
            rows.extend(ev)
        mv = self.ctx.geo.movement("scene2", self.ctx.geometry, rows, z0, None)
        return {"first": mv["first_targets"], "ever": {g: any(x["pred"][g] for x in rows) for g in self.ctx.geo.GOALS},
                "terminal": rows[-1]["pred"]}

    def verify(self, inits, tag):
        S5.mkdir(exist_ok=True)
        path = S5 / f"verify_r{self.round}_w{tag}.jsonl"
        with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16), path.open("a") as f:
            for init in inits:
                for tpl in [t for t in ("canon", *TEMPLATES) if ROUND[t] == self.round]:
                    for side in "LR":
                        q = text(tpl, side)
                        rec = {"init": init, "tpl": tpl, "side": side, "text": q, **self.from_reset(init, q)}
                        f.write(json.dumps(rec) + "\n"); f.flush()
                        print(init, tpl, side, rec["first"], rec["ever"][SIDE[side]], flush=True)
        (S5 / f"verify_r{self.round}_w{tag}_COMPLETE.json").write_text("{}")

    def episode5(self, snap, mode, q):
        started = time.monotonic()
        init, b, z0, seed = snap["init"], snap["boundary"], snap["z0"], snap["seed"]
        controls = [self.np.asarray(c, dtype=self.np.float32) for c in snap["controls"]]
        obs, rows = self.rebuild(init, controls, z0)
        assert self.ctx.signature(obs) == snap["signature"]
        diag = defaultdict(list)
        for r in range(b // 10, HORIZON // 10):
            if mode == "S" and r * 10 < b + WINDOW:
                prefixes = {(role, g): self.rt.prefix(role, obs, INSTR[g]) for role in "RF" for g in "AB"}
                qpre = self.rt.prefix("F", obs, q)
                chunk = self.rt.integrate(lambda x, t: self.corrected(prefixes, qpre, x, t, diag), self.rt.noise(seed, r), 10)[..., :7]
            else:
                chunk = self.native("F", q, obs, seed, r)
            obs, ev = self.execute(obs, self.rt.physical(chunk), r * 10, z0)
            rows.extend(ev)
        assert len(rows) == HORIZON
        mv = self.ctx.geo.movement("scene2", self.ctx.geometry, rows, z0, None)
        return {"first": mv["first_targets"], "first_step": mv["first_step"],
                "ever": {g: any(x["pred"][g] for x in rows) for g in self.ctx.geo.GOALS}, "terminal": rows[-1]["pred"],
                "diagnostics": {k: [sum(v) / len(v), min(v), max(v)] for k, v in diag.items()},
                "clip_rate_all": sum(x["clip"] for x in rows[b:]) / (HORIZON - b), "seconds": time.monotonic() - started}, rows

    def main5(self, inits, tag):
        kept = json.loads((S5 / "verified_templates.json").read_text())["kept"]
        snaps = {r["init"]: r for r in json.loads((OUT / "snapshots/snapshots.json").read_text())["records"]}
        jobs = [("S", "canon", s) for s in "LR"]
        for tpl in kept:
            jobs += [("S", tpl, s) for s in "LR"] + [("N", tpl, s) for s in "LR"]
        jobs += [("S", "neutral", "-"), ("S", "front", "-")]
        res, tr = S5 / f"main_w{tag}.jsonl", S5 / f"main_w{tag}_traces.jsonl"
        done = set()
        if res.exists():
            done = {(j["init"], j["mode"], j["tpl"], j["side"]) for j in map(json.loads, res.read_text().splitlines())}
        with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16), res.open("a") as fr, tr.open("a") as ft:
            for init in inits:
                for mode, tpl, side in jobs:
                    if (init, mode, tpl, side) in done:
                        continue
                    q = NEUTRAL if tpl == "neutral" else INSTR["N"] if tpl == "front" else text(tpl, side)
                    rec, rows = self.episode5(snaps[init], mode, q)
                    rec = {"init": init, "mode": mode, "tpl": tpl, "side": side, "text": q, **rec}
                    ft.write(json.dumps({"init": init, "mode": mode, "tpl": tpl, "side": side, "rows": rows}) + "\n"); ft.flush()
                    fr.write(json.dumps(rec, allow_nan=False) + "\n"); fr.flush()
                    print(init, mode, tpl, side, rec["first"], f"{rec['seconds']:.0f}s", flush=True)
        (S5 / f"main_w{tag}_COMPLETE.json").write_text(json.dumps({"inits": inits, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))


def select_templates():
    """Locked: template kept iff R1 from reset on dev inits 0-9 has first==[side] >= 8/10 AND ever[side] >= 7/10 on BOTH sides; first KEEP_MAX in P1..P4 order."""
    recs = [json.loads(l) for f in sorted(S5.glob("verify_r*_w*.jsonl")) for l in f.read_text().splitlines()]
    assert len(recs) == 10 * 9 * 2, len(recs)
    stat = {}
    for tpl in ("canon", *TEMPLATES):
        stat[tpl] = {s: {"first": sum(r["first"] == [SIDE[s]] for r in recs if r["tpl"] == tpl and r["side"] == s),
                         "ever": sum(r["ever"][SIDE[s]] for r in recs if r["tpl"] == tpl and r["side"] == s)} for s in "LR"}
    ok = [t for t in TEMPLATES if all(stat[t][s]["first"] >= 8 and stat[t][s]["ever"] >= 7 for s in "LR")]
    out = {"stats": stat, "passed": ok, "kept": ok[:KEEP_MAX]}
    (S5 / "verified_templates.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("verify", "select", "main"))
    p.add_argument("--inits", type=int, nargs="*", default=[])
    p.add_argument("--tag", default="0")
    p.add_argument("--round", type=int, default=1)
    a = p.parse_args()
    if a.phase == "select":
        select_templates()
    else:
        lab = Lab5()
        lab.round = a.round
        (lab.verify if a.phase == "verify" else lab.main5)(a.inits, a.tag)
        lab.ctx.env.close()
