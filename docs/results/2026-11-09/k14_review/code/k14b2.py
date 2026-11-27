"""K14 batch 2 (plan §6): old mean held to step 400; only the source and duration of the conditional difference vary.
H: d_new to 400 | I: candidate 60 steps then d_new | J: candidate to 400 | K: matched random (G rule) to 400 | L = batch-1 R_full (identical computation).
Candidate fixed by batch-1 locked rule (judgement/k14_judgement.json verdicts.batch2_pick)."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import k14
from k14 import Lab, OUT, INSTR, LABEL, HORIZON, WINDOW

B2 = OUT / "batch2"


def schedule(group, cand, r, b):
    if group == "H_new400":
        return "B_mean"
    if group == "I_cand60":
        return cand if r * 10 < b + WINDOW else "B_mean"
    if group == "J_cand400":
        return cand
    if group == "K_rand400":
        return "G_rand"
    raise ValueError(group)


class Lab2(Lab):
    def episode2(self, snap, group, goal, cand):
        started = time.monotonic()
        init, b, z0, seed = snap["init"], snap["boundary"], snap["z0"], snap["seed"]
        controls = [self.np.asarray(c, dtype=self.np.float32) for c in snap["controls"]]
        obs, rows = self.rebuild(init, controls, z0)
        assert self.ctx.signature(obs) == snap["signature"]
        diag = defaultdict(list)
        used = []
        for r in range(b // 10, HORIZON // 10):
            spec = schedule(group, cand, r, b)
            used.append(spec)
            prefixes = {(role, g): self.rt.prefix(role, obs, INSTR[g]) for role in "RF" for g in "AB"}
            it = 0

            def velocity(x, t):
                nonlocal it
                it += 1
                return self.combine(spec, prefixes, x, t, seed * 10000 + r * 100 + it * 2, diag)[goal]
            chunk = self.rt.integrate(velocity, self.rt.noise(seed, r), 10)[..., :7]
            assert it == 10
            obs, ev = self.execute(obs, self.rt.physical(chunk), r * 10, z0)
            rows.extend(ev)
        assert len(rows) == HORIZON
        mv = self.ctx.geo.movement("scene2", self.ctx.geometry, rows, z0, None)
        req = LABEL[goal]
        return {"init": init, "group": group, "goal": goal, "requested": req, "candidate": cand, "boundary": b, "seed": seed,
                "schedule": used, "first": mv["first_targets"], "first_step": mv["first_step"], "first_times": mv["times"],
                "ever": {g: any(x["pred"][g] for x in rows) for g in self.ctx.geo.GOALS},
                "ever_step": next((x["step"] for x in rows if x["pred"][req]), None), "terminal": rows[-1]["pred"],
                "clip_rate_all": sum(x["clip"] for x in rows[b:]) / (HORIZON - b),
                "diagnostics": {k: [sum(v) / len(v), min(v), max(v)] for k, v in diag.items()},
                "n_queries": len(diag["cap_hit"]), "seconds": time.monotonic() - started}, rows

    def worker2(self, inits, tag, cand):
        snaps = {r["init"]: r for r in json.loads((OUT / "snapshots/snapshots.json").read_text())["records"]}
        B2.mkdir(exist_ok=True)
        res, tr = B2 / f"w{tag}.jsonl", B2 / f"w{tag}_traces.jsonl"
        done = set()
        if res.exists():
            done = {(j["init"], j["group"], j["goal"]) for j in map(json.loads, res.read_text().splitlines())}
        with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16), res.open("a") as fr, tr.open("a") as ft:
            for init in inits:
                for group in ("H_new400", "I_cand60", "J_cand400", "K_rand400"):
                    for goal in "AB":
                        if (init, group, goal) in done:
                            continue
                        rec, rows = self.episode2(snaps[init], group, goal, cand)
                        ft.write(json.dumps({"init": init, "group": group, "goal": goal, "rows": rows}) + "\n"); ft.flush()
                        fr.write(json.dumps(rec, allow_nan=False) + "\n"); fr.flush()
                        print(init, group, goal, rec["first"], rec["ever"][rec["requested"]], rec["terminal"][rec["requested"]], f"{rec['seconds']:.0f}s", flush=True)
        (B2 / f"w{tag}_COMPLETE.json").write_text(json.dumps({"inits": inits, "candidate": cand,
                                                              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inits", type=int, nargs="+", required=True)
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    v = json.loads((OUT / "judgement/k14_judgement.json").read_text())["verdicts"]
    pre = json.loads((OUT / "prereg/k14_prereg.json").read_text())
    assert v["batch2_run"] or any(m.get("decision", "").startswith("user approved opening batch 2") for m in pre["amendments"]), \
        "batch 2 neither opened by the locked rule nor by a disclosed amendment"
    lab = Lab2()
    lab.worker2(a.inits, a.tag, v["batch2_pick"])
    lab.ctx.env.close()
