"""K14 §5 judgement (rules: ../prereg/k14_s5_prereg.json)."""
import json
from pathlib import Path
import random

OUT = Path(__file__).resolve().parent.parent
S5 = OUT / "s5"
DELTA, CLOSE, NB, SEED = 0.15, 0.10, 10000, 20261108


def main():
    recs = [json.loads(l) for f in sorted(S5.glob("main_w*.jsonl")) if "traces" not in f.name for l in f.read_text().splitlines()]
    kept = json.loads((S5 / "verified_templates.json").read_text())["kept"]
    b1 = json.loads((OUT / "judgement/k14_judgement.json").read_text())
    inits = b1["inits"]
    E = {(r["init"], r["mode"], r["tpl"], r["side"]): r for r in recs}
    assert len(E) == len(recs) == len(inits) * (2 + 4 * len(kept) + 2), len(recs)
    groups = {"S_canon": ("S", "canon")} | {f"S_{k}": ("S", k) for k in kept} | {f"N_{k}": ("N", k) for k in kept}
    per = {}
    for name, (mode, tpl) in groups.items():
        a = [E[i, mode, tpl, "L"] for i in inits]; b = [E[i, mode, tpl, "R"] for i in inits]
        per[name] = {"dir": [int(x["first"] == ["left"] and y["first"] == ["right"]) for x, y in zip(a, b)],
                     "place": [int(x["ever"]["left"]) + int(y["ever"]["right"]) for x, y in zip(a, b)],
                     "term": [int(x["terminal"]["left"]) + int(y["terminal"]["right"]) for x, y in zip(a, b)],
                     "L_first": [int(x["first"] == ["left"]) for x in a], "R_first": [int(y["first"] == ["right"]) for y in b],
                     "front_first": [int("front" in x["first"]) + int("front" in y["first"]) for x, y in zip(a, b)],
                     "L_ever": [int(x["ever"]["left"]) for x in a], "R_ever": [int(y["ever"]["right"]) for y in b],
                     "L_term": [int(x["terminal"]["left"]) for x in a], "R_term": [int(y["terminal"]["right"]) for y in b]}
    for g in ("A_nat", "C_amp", "F_old"):
        per["b1_" + g] = b1["per_snapshot"][g]
    den = {"dir": 20, "place": 40, "term": 40}
    rng = random.Random(SEED)
    boots = [[rng.randrange(len(inits)) for _ in inits] for _ in range(NB)]

    def cmp(x, y, m):
        diff = [p - q for p, q in zip(per[x][m], per[y][m])]
        pt = sum(diff) / den[m]
        dist = sorted(sum(diff[k] for k in idx) / den[m] for idx in boots)
        lo, hi = dist[int(.025 * NB)], dist[int(.975 * NB) - 1]
        return {"x": x, "y": y, "metric": m, "point": pt, "ci": [lo, hi], "clearly_better": pt >= DELTA - 1e-12 and lo > 0,
                "close": abs(pt) <= CLOSE + 1e-12}
    C = {}
    pairs = [("S_canon", "b1_A_nat"), ("S_canon", "b1_C_amp"), ("b1_F_old", "S_canon")]
    for k in kept:
        pairs += [(f"S_{k}", f"N_{k}"), ("S_canon", f"S_{k}"), (f"S_{k}", "b1_A_nat")]
    for x, y in pairs:
        for m in ("dir", "place", "term"):
            C[f"{x}>{y}:{m}"] = cmp(x, y, m)
    V = {"L1_canonical_fixed_gain": C["S_canon>b1_A_nat:dir"]["clearly_better"]}
    l2 = {k: C[f"S_{k}>N_{k}:dir"]["clearly_better"] and not C[f"S_canon>S_{k}:dir"]["clearly_better"] for k in kept}
    V["L2_per_template"] = l2
    V["L2_paraphrase_routing"] = "supported" if len(kept) >= 2 and all(l2.values()) else "partial" if any(l2.values()) else "not supported"
    V["L3_fixed_sentence_only"] = V["L1_canonical_fixed_gain"] and not any(l2.values())
    side = {}
    for tpl in ("neutral", "front"):
        rs = [E[i, "S", tpl, "-"] for i in inits]
        side[tpl] = {"first": {g: sum(r["first"] == [g] for r in rs) for g in ("left", "right", "front", "back")} | {"none": sum(not r["first"] for r in rs)},
                     "terminal": {g: sum(r["terminal"][g] for r in rs) for g in ("left", "right", "front", "back")},
                     "proj_on_dold_median": sorted(r["diagnostics"]["proj_on_dold"][0] for r in rs)[10]}
    V["neutral_side_bias"] = max(v for k, v in side["neutral"]["first"].items() if k != "none") >= 15
    proj = {}
    for name, (mode, tpl) in groups.items():
        if mode == "S":
            for s in "LR":
                vals = sorted(E[i, mode, tpl, s]["diagnostics"]["proj_on_dold"][0] for i in inits)
                proj[f"{name}_{s}"] = vals[10]
    out = {"kept": kept, "summary": {g: {k: sum(v) for k, v in per[g].items()} for g in per}, "per_snapshot": per,
           "comparisons": C, "verdicts": V, "neutral_front": side, "proj_on_dold_median": proj}
    (OUT / "judgement/k14_s5_judgement.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    L = ["| 组 | 左先到左 | 右先到右 | 成对分流 /20 | 先到前 /40 | 左曾成功 | 右曾成功 | 曾成功 /40 | 终态 /40 |", "|---|" + "---:|" * 8]
    for g, s in out["summary"].items():
        L.append(f"| {g} | {s['L_first']} | {s['R_first']} | {s['dir']} | {s['front_first']} | {s['L_ever']} | {s['R_ever']} | {s['place']} | {s['term']} |")
    L += ["", "| 比较 | 指标 | 差值 | 95% CI | 明确更好 | 接近 |", "|---|---|---:|---|---|---|"]
    for v in C.values():
        L.append(f"| {v['x']} − {v['y']} | {v['metric']} | {v['point']:+.3f} | [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | {'是' if v['clearly_better'] else ''} | {'是' if v['close'] else ''} |")
    L += ["", "```json", json.dumps({"verdicts": V, "neutral_front": side, "proj": proj}, ensure_ascii=False, indent=1), "```"]
    (OUT / "judgement/k14_s5_judgement.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:len(per) + 2]))
    print(json.dumps({"verdicts": V, "neutral_front": side, "proj": proj}, ensure_ascii=False))


if __name__ == "__main__":
    main()
