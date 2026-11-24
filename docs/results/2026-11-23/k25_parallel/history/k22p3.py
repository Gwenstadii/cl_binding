"""K22 p3 (amendment 1): from reset, first 150 steps, old common response + (par | sign) of the adapted residual. All 7 lines."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from k22 import Lab2, K21, OUT, HORIZON, P1_WINDOW, P1_N


class Lab3(Lab2):
    def ep3(self, rec, group, key):
        started = time.monotonic()
        z0, seed, plate = rec["z0"], rec["seed"], rec["plate"]
        obs = self.reset_rec(rec)
        assert self.ctx.signature(obs) == rec["signature"]
        rows, diag = [], defaultdict(list)
        for r in range(HORIZON // 10):
            if r * 10 < P1_WINDOW:
                pre = {(role, g): self.rt.prefix(role, obs, self.text[g]) for role in "RF" for g in "AB"}
                chunk = self.rt.integrate(lambda x, t: self.combine2(group, None, pre, key, x, t, diag), self.rt.noise(seed, r), 10)[..., :7]
            else:
                chunk = self.native("F", self.text[key], obs, seed, r)
            obs, ev = self.execute(obs, self.rt.physical(chunk), r * 10, z0, plate)
            rows.extend(ev)
        return self.finish(rows, rec, key, z0, plate, 0, diag, started, {"group": group})

    def run3(self, lane, lanes):
        cands = json.loads((K21 / "states" / f"{self.family}.json").read_text())
        items = [r for r in cands if r["geometry_ok"] and not any(r["pred_at_reset"].values())][:P1_N]
        d = OUT / "p3" / self.line
        d.mkdir(parents=True, exist_ok=True)
        res = d / f"w{lane}.jsonl"
        done = {(j["init"], j["group"], j["key"]) for j in map(json.loads, res.read_text().splitlines())} if res.exists() else set()
        with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16), res.open("a") as fr:
            for it in items[lane::lanes]:
                for g in ("par", "sign"):
                    for k in "AB":
                        if (it["init"], g, k) in done:
                            continue
                        rec = self.ep3(it, g, k)
                        rec["alpha"] = self.alpha
                        fr.write(json.dumps(rec, allow_nan=False) + "\n"); fr.flush()
                        print("p3", self.line, it["init"], g, k, rec["first"], f"{rec['seconds']:.0f}s", flush=True)
        (d / f"w{lane}_COMPLETE.json").write_text("{}")


def summary():
    L = ["## p3: from reset, old mu + par / sign (dir /n; A/B first-correct; place,term /2n; sign-correct query rate)", "| line | n | par | sign | sign-correct A/B | median c1 A/B |", "|---|---|---|---|---|---|"]
    for line in ("r0_s2", "c_s2a", "c_s2b", "scene1b", "goal", "c_goal", "r1_front"):
        recs = [json.loads(l) for f in sorted((OUT / "p3" / line).glob("w*.jsonl")) for l in f.read_text().splitlines()] if (OUT / "p3" / line).exists() else []
        if not recs:
            continue
        E = {(r["init"], r["group"], r["key"]): r for r in recs}
        inits = sorted({r["init"] for r in recs if all((r["init"], g, k) in E for g in ("par", "sign") for k in "AB")})
        n, cells = len(inits), []
        for g in ("par", "sign"):
            a = [E[i, g, "A"] for i in inits]; b = [E[i, g, "B"] for i in inits]
            gA, gB = a[0]["requested"], b[0]["requested"]
            cells.append(f"{sum(x['first'] == [gA] and y['first'] == [gB] for x, y in zip(a, b))} ({sum(x['first'] == [gA] for x in a)}/{sum(y['first'] == [gB] for y in b)}; "
                         f"{sum(x['ever'][gA] for x in a) + sum(y['ever'][gB] for y in b)},{sum(x['terminal'][gA] for x in a) + sum(y['terminal'][gB] for y in b)})")
        sa = sum(E[i, "par", "A"]["diag"]["c1"][1] for i in inits) / n; sb = sum(1 - E[i, "par", "B"]["diag"]["c1"][1] for i in inits) / n
        ca = sorted(E[i, "par", "A"]["diag"]["c1"][0] for i in inits)[n // 2]; cb = sorted(E[i, "par", "B"]["diag"]["c1"][0] for i in inits)[n // 2]
        L.append(f"| {line} | {n} | {cells[0]} | {cells[1]} | {sa:.3f}/{sb:.3f} | {ca:.2f}/{cb:.2f} |")
    (OUT / "judgement/k22_p3.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--line")
    p.add_argument("--lane", type=int, default=0)
    p.add_argument("--lanes", type=int, default=10)
    p.add_argument("--summary", action="store_true")
    a = p.parse_args()
    if a.summary:
        summary()
    else:
        lab = Lab3(a.line)
        lab.run3(a.lane, a.lanes)
        lab.ctx.env.close()
