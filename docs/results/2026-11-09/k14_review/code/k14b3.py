"""K14 batch 3 (plan §7): from states where the fixed candidate policy (batch-2 J: mu_old +/- d_new*a_old/a_new) has carried the
book over the requested compartment but not yet placed it, branch and restore old-difference content channel by channel.
Phases: select (from J traces, no new rollouts) -> worker. Rules: ../prereg/k14_b3_prereg.json."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from k14 import Lab, OUT, INSTR, LABEL, HORIZON, CAP, DEGEN, K_DCT

B3 = OUT / "batch3"
SETS = {"z": [2], "rotgrip": [3, 4, 5, 6], "nonhoriz": [2, 3, 4, 5, 6], "xy": [0, 1]}
GROUPS = ("T0_cand", "T1_z", "T2_rotgrip", "T3_nonhoriz", "T4_xy", "T5_full", "T1r_z", "T2r_rotgrip", "T3r_nonhoriz")
CHAN = {"T1_z": "z", "T2_rotgrip": "rotgrip", "T3_nonhoriz": "nonhoriz", "T4_xy": "xy",
        "T1r_z": "z", "T2r_rotgrip": "rotgrip", "T3r_nonhoriz": "nonhoriz"}


def select():
    """Locked: J episode of side s; t_a = first step of the first 5-step lifted run inside the requested region;
    b3 = first planning boundary >= t_a + 4 (event confirmed without lookahead); eligible iff at step b3 the book is lifted
    inside the requested region with both finger contacts, and the requested native predicate was never true at steps <= b3."""
    import k1_geometry as geo
    snaps = {r["init"]: r for r in json.loads((OUT / "snapshots/snapshots.json").read_text())["records"]}
    geometry = json.loads((OUT / "batch3_geometry.json").read_text())
    out, report = [], []
    for f in sorted((OUT / "batch2").glob("w*_traces.jsonl")):
        for line in f.read_text().splitlines():
            t = json.loads(line)
            if t["group"] != "J_cand400":
                continue
            init, goal = t["init"], t["goal"]
            req, rows, z0 = LABEL[goal], t["rows"], snaps[init]["z0"]
            inside = [req in geo.hits("scene2", geometry, r["obj"], z0, None) for r in rows]
            ta = next((rows[i]["step"] for i in range(len(rows) - 4) if all(inside[i:i + 5])), None)
            rec = {"init": init, "goal": goal, "requested": req, "t_arrive": ta}
            if ta is None:
                rec["reason"] = "never_arrived"
            else:
                b3 = -(-(ta + 4) // 10) * 10
                row = rows[b3 - 1]
                rec["b3"] = b3
                if b3 >= HORIZON:
                    rec["reason"] = "too_late"
                elif any(r["pred"][req] for r in rows[:b3]):
                    rec["reason"] = "already_placed"
                elif not (inside[b3 - 1] and row["c1"] and row["c2"]):
                    rec["reason"] = "not_held_over_target"
                else:
                    rec["reason"] = None
                    rec["actions"] = [r["a"] for r in rows[snaps[init]["boundary"]:b3]]
                    rec["rows_to_b3"] = rows[:b3]
                    rec["j_rows_after"] = [{k: r[k] for k in ("step", "obj", "pred")} for r in rows[b3:]]
            (out if rec.get("reason") is None else report).append(rec)
    out.sort(key=lambda r: (r["goal"], r["init"]))
    B3.mkdir(exist_ok=True)
    (B3 / "b3_snapshots.json").write_text(json.dumps({"eligible": out, "excluded": report}))
    print("eligible", [(r["init"], r["goal"], r["b3"]) for r in out])
    print("excluded", [(r["init"], r["goal"], r["reason"]) for r in report])


class Lab3(Lab):
    def combine3(self, group, prefixes, x, t, seed, goal, diag):
        torch = self.torch
        st = {(role, g): self.rt.query(role, prefixes[role, g], x, t).float() for role in "RF" for g in "AB"}
        muR = (st["R", "A"] + st["R", "B"]) / 2
        dR = (st["R", "A"] - st["R", "B"]) / 2
        dF = (st["F", "A"] - st["F", "B"]) / 2
        aR, aF = self.a(dR), self.a(dF)
        if aF < DEGEN * self.a(muR):
            cand = torch.zeros_like(dF); diag["degenerate"].append(1)
        else:
            s = aR / aF
            diag["cap_hit"].append(int(s > CAP))
            cand = dF * s.clamp(max=CAP)
        if group == "T0_cand":
            diff = cand
        elif group == "T5_full":
            diff = dR
        else:
            S = SETS[CHAN[group]]
            diff = cand.clone()
            delta = dR[0, :, S] - cand[0, :, S]
            if group[2] == "r":
                g = torch.Generator(device="cpu").manual_seed(seed)
                coef = torch.randn((K_DCT, len(S)), generator=g, dtype=torch.float32).cuda()
                r = (self.dct.T.unsqueeze(-1) * coef.unsqueeze(0)).sum(1)
                r = r / r.square().mean(0).sqrt().clamp_min(1e-12) * delta.square().mean(0).sqrt()
                diag["rand_cos_to_delta"].append(float((r * delta).sum() / (r.norm() * delta.norm()).clamp_min(1e-30)))
                diag["rand_norm_over_delta"].append(float(r.norm() / delta.norm().clamp_min(1e-30)))
                diff[0, :, S] = cand[0, :, S] + r
            else:
                diff[0, :, S] = dR[0, :, S]
            diag["delta_norm_over_old"].append(float(delta.norm() / aR.clamp_min(1e-30)))
        sign = 1 if goal == "A" else -1
        out = (muR + sign * diff).to(self.rt.vdtype)
        e, ref = diff[0, :, :7].flatten(), dR[0, :, :7].flatten()
        diag["old_cos_full"].append(float((e * ref).sum() / (e.norm() * ref.norm()).clamp_min(1e-30)))
        diag["norm_over_old"].append(float(e.norm() / ref.norm().clamp_min(1e-30)))
        if group == "T5_full":
            assert torch.equal(out, self.rt.query("R", prefixes["R", goal], x, t)), "T5 must equal native R1"
        return out

    def episode3(self, snap, s3, group):
        started = time.monotonic()
        init, b, z0, seed = snap["init"], snap["boundary"], snap["z0"], snap["seed"]
        goal, b3 = s3["goal"], s3["b3"]
        controls = [self.np.asarray(c, dtype=self.np.float32) for c in snap["controls"]]
        obs, rows = self.rebuild(init, controls, z0)
        assert self.ctx.signature(obs) == snap["signature"]
        acts = self.np.asarray(s3["actions"], dtype=self.np.float32)
        for k, r in enumerate(range(b // 10, b3 // 10)):
            obs, ev = self.execute(obs, acts[k * 10:(k + 1) * 10], r * 10, z0)
            rows.extend(ev)
        assert rows == s3["rows_to_b3"], "replay of the candidate trajectory to b3 differs"
        diag = defaultdict(list)
        for r in range(b3 // 10, HORIZON // 10):
            prefixes = {(role, g): self.rt.prefix(role, obs, INSTR[g]) for role in "RF" for g in "AB"}
            it = 0

            def velocity(x, t):
                nonlocal it
                it += 1
                return self.combine3(group, prefixes, x, t, seed * 10000 + r * 100 + it * 2, goal, diag)
            chunk = self.rt.integrate(velocity, self.rt.noise(seed, r), 10)[..., :7]
            obs, ev = self.execute(obs, self.rt.physical(chunk), r * 10, z0)
            rows.extend(ev)
        req = LABEL[goal]
        after = rows[b3:]
        same_as_j = [{k: x[k] for k in ("step", "obj", "pred")} for x in after] == s3["j_rows_after"] if group == "T0_cand" else None
        return {"init": init, "goal": goal, "requested": req, "group": group, "b3": b3,
                "ever_after": any(x["pred"][req] for x in after), "ever_step": next((x["step"] for x in after if x["pred"][req]), None),
                "terminal": rows[-1]["pred"], "term_req": rows[-1]["pred"][req],
                "min_z_rel_after": min(x["obj"][2] - z0 for x in after),
                "contacts_end": [rows[-1]["c1"], rows[-1]["c2"]], "t0_reproduces_J": same_as_j,
                "clip_rate": sum(x["clip"] for x in after) / len(after),
                "diagnostics": {k: [sum(v) / len(v), min(v), max(v)] for k, v in diag.items()},
                "seconds": time.monotonic() - started}, rows

    def worker3(self, keys, tag):
        snaps = {r["init"]: r for r in json.loads((OUT / "snapshots/snapshots.json").read_text())["records"]}
        el = {(r["init"], r["goal"]): r for r in json.loads((B3 / "b3_snapshots.json").read_text())["eligible"]}
        res, tr = B3 / f"w{tag}.jsonl", B3 / f"w{tag}_traces.jsonl"
        done = set()
        if res.exists():
            done = {(j["init"], j["goal"], j["group"]) for j in map(json.loads, res.read_text().splitlines())}
        with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16), res.open("a") as fr, tr.open("a") as ft:
            for init, goal in keys:
                for group in GROUPS:
                    if (init, goal, group) in done:
                        continue
                    rec, rows = self.episode3(snaps[init], el[init, goal], group)
                    ft.write(json.dumps({"init": init, "goal": goal, "group": group, "rows": rows[rec["b3"]:]}) + "\n"); ft.flush()
                    fr.write(json.dumps(rec, allow_nan=False) + "\n"); fr.flush()
                    print(init, goal, group, rec["ever_after"], rec["term_req"], rec["t0_reproduces_J"], f"{rec['seconds']:.0f}s", flush=True)
        (B3 / f"w{tag}_COMPLETE.json").write_text(json.dumps({"keys": keys, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("geometry", "select", "worker"))
    p.add_argument("--keys", nargs="*", default=[], help="init:goal")
    p.add_argument("--tag", default="0")
    a = p.parse_args()
    if a.phase == "select":
        select()
    else:
        lab = Lab3()
        if a.phase == "geometry":
            (OUT / "batch3_geometry.json").write_text(json.dumps(lab.ctx.geometry))
        else:
            lab.worker3([(int(k.split(":")[0]), k.split(":")[1]) for k in a.keys], a.tag)
        lab.ctx.env.close()
