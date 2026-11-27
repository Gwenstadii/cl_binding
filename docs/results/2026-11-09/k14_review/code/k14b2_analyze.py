"""K14 batch-2 judgement (rules: prereg/k14_prereg.json amendment 1)."""
import json
from pathlib import Path
import random

OUT = Path(__file__).resolve().parent.parent
DELTA, CLOSE, NB, SEED = 0.15, 0.10, 10000, 20261108


def metrics(a, b):
    return {"dir": [int(x["first"] == ["left"] and y["first"] == ["right"]) for x, y in zip(a, b)],
            "place": [int(x["ever"]["left"]) + int(y["ever"]["right"]) for x, y in zip(a, b)],
            "term": [int(x["terminal"]["left"]) + int(y["terminal"]["right"]) for x, y in zip(a, b)],
            "L_ever": [int(x["ever"]["left"]) for x in a], "R_ever": [int(y["ever"]["right"]) for y in b],
            "L_term": [int(x["terminal"]["left"]) for x in a], "R_term": [int(y["terminal"]["right"]) for y in b],
            "front_term": [int(x["terminal"]["front"]) + int(y["terminal"]["front"]) for x, y in zip(a, b)]}


def main():
    recs = [json.loads(l) for f in sorted((OUT / "batch2").glob("w*.jsonl")) if "traces" not in f.name for l in f.read_text().splitlines()]
    b1 = json.loads((OUT / "judgement/k14_judgement.json").read_text())
    inits = b1["inits"]
    E = {(r["init"], r["group"], r["goal"]): r for r in recs}
    assert len(E) == len(recs) == 160, len(recs)
    per = {g: metrics([E[i, g, "A"] for i in inits], [E[i, g, "B"] for i in inits]) for g in ("H_new400", "I_cand60", "J_cand400", "K_rand400")}
    per["L_Rfull"] = b1["per_snapshot"]["R_full"]
    per["b1_C_amp"] = b1["per_snapshot"]["C_amp"]
    per["b1_F_old"] = b1["per_snapshot"]["F_old"]
    den = {"dir": 20, "place": 40, "term": 40}
    rng = random.Random(SEED)
    boots = [[rng.randrange(len(inits)) for _ in inits] for _ in range(NB)]

    def cmp(x, y, m):
        diff = [p - q for p, q in zip(per[x][m], per[y][m])]
        pt = sum(diff) / den[m]
        dist = sorted(sum(diff[k] for k in idx) / den[m] for idx in boots)
        lo, hi = dist[int(.025 * NB)], dist[int(.975 * NB) - 1]
        return {"x": x, "y": y, "metric": m, "point": pt, "ci": [lo, hi], "clearly_better": pt >= DELTA - 1e-12 and lo > 0, "close": abs(pt) <= CLOSE + 1e-12}
    C = {}
    for x, y in [("J_cand400", "I_cand60"), ("J_cand400", "H_new400"), ("J_cand400", "K_rand400"), ("L_Rfull", "J_cand400"),
                 ("I_cand60", "b1_C_amp"), ("J_cand400", "b1_C_amp"), ("H_new400", "K_rand400"), ("L_Rfull", "I_cand60")]:
        for m in ("dir", "place", "term"):
            C[f"{x}>{y}:{m}"] = cmp(x, y, m)
    c = lambda x, y: C[f"{x}>{y}:term"]
    V = {"B2_duration": c("J_cand400", "I_cand60")["clearly_better"] and c("J_cand400", "H_new400")["clearly_better"] and c("J_cand400", "K_rand400")["clearly_better"],
         "B2_content_incomplete": c("L_Rfull", "J_cand400")["clearly_better"],
         "B2_nonspecific": c("J_cand400", "H_new400")["close"] or c("J_cand400", "K_rand400")["close"],
         "B2_both_fail": sum(per["J_cand400"]["term"]) < 20 and sum(per["L_Rfull"]["term"]) < 20}
    diag = {}
    for g in ("H_new400", "I_cand60", "J_cand400", "K_rand400"):
        rows = [E[i, g, q] for i in inits for q in "AB"]
        diag[g] = {"cap_hits": sum(round(r["diagnostics"]["cap_hit"][0] * r["n_queries"]) for r in rows),
                   "clip_all_mean": sum(r["clip_rate_all"] for r in rows) / len(rows),
                   "old_cos_full_median": sorted(r["diagnostics"]["old_cos_full"][0] for r in rows)[20]}
    out = {"summary": {g: {k: sum(v) for k, v in per[g].items()} for g in per}, "per_snapshot": per, "comparisons": C, "verdicts": V, "diagnostics": diag}
    (OUT / "judgement/k14_b2_judgement.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    L = ["| 组 | 成对分流 /20 | 左曾成功 | 右曾成功 | 曾成功 /40 | 左终态 | 右终态 | 终态 /40 | 前格终态 |", "|---|" + "---:|" * 8]
    for g, s in out["summary"].items():
        L.append(f"| {g} | {s['dir']} | {s['L_ever']} | {s['R_ever']} | {s['place']} | {s['L_term']} | {s['R_term']} | {s['term']} | {s['front_term']} |")
    L += ["", "| 比较 | 指标 | 差值 | 95% CI | 明确更好 | 接近 |", "|---|---|---:|---|---|---|"]
    for v in C.values():
        L.append(f"| {v['x']} − {v['y']} | {v['metric']} | {v['point']:+.3f} | [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | {'是' if v['clearly_better'] else ''} | {'是' if v['close'] else ''} |")
    L += ["", "```json", json.dumps({"verdicts": V, "diagnostics": diag}, ensure_ascii=False, indent=1), "```"]
    (OUT / "judgement/k14_b2_judgement.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(json.dumps(V))


if __name__ == "__main__":
    main()
