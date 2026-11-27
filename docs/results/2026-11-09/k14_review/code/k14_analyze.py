"""K14 batch-1 judgement, rules locked in ../prereg/k14_prereg.json (delta 0.15, close 0.10, cluster bootstrap)."""
import json
from pathlib import Path
import random

OUT = Path(__file__).resolve().parent.parent
GROUPS = ("A_nat", "B_mean", "C_amp", "D_hist", "D_pure", "E_hist", "E_pure", "F_old", "G_rand", "R_full")
SHORT = {g: g.split("_")[0] for g in GROUPS}
DELTA, CLOSE, NB, SEED = 0.15, 0.10, 10000, 20261108


def load():
    recs = []
    for f in sorted((OUT / "episodes").glob("w*.jsonl")):
        if f.name.endswith("_traces.jsonl"):
            continue
        recs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    keys = [(r["init"], r["group"], r["goal"]) for r in recs]
    assert len(keys) == len(set(keys)), "duplicate episodes"
    return recs


def main():
    recs = load()
    snaps = json.loads((OUT / "snapshots/snapshots.json").read_text())
    inits = snaps["selected"]
    assert len(recs) == len(inits) * (len(GROUPS) * 2 + 1), f"incomplete: {len(recs)}"
    E = {(r["init"], r["group"], r["goal"]): r for r in recs}
    per = {}  # group -> metric -> list over inits
    for g in GROUPS:
        a = [E[i, g, "A"] for i in inits]; b = [E[i, g, "B"] for i in inits]
        per[g] = {
            "dir": [int(x["first"] == ["left"] and y["first"] == ["right"]) for x, y in zip(a, b)],
            "place": [int(x["ever"]["left"]) + int(y["ever"]["right"]) for x, y in zip(a, b)],
            "term": [int(x["terminal"]["left"]) + int(y["terminal"]["right"]) for x, y in zip(a, b)],
            "L_first": [int(x["first"] == ["left"]) for x in a], "R_first": [int(y["first"] == ["right"]) for y in b],
            "front_first": [int("front" in x["first"]) + int("front" in y["first"]) for x, y in zip(a, b)],
            "none_first": [int(not x["first"]) + int(not y["first"]) for x, y in zip(a, b)],
            "L_ever": [int(x["ever"]["left"]) for x in a], "R_ever": [int(y["ever"]["right"]) for y in b],
            "L_term": [int(x["terminal"]["left"]) for x in a], "R_term": [int(y["terminal"]["right"]) for y in b],
            "front_term": [int(x["terminal"]["front"]) + int(y["terminal"]["front"]) for x, y in zip(a, b)],
            "both_term": [int(x["terminal"]["left"] and y["terminal"]["right"]) for x, y in zip(a, b)],
        }
    den = {"dir": len(inits), "place": 2 * len(inits), "term": 2 * len(inits)}
    rng = random.Random(SEED)
    boots = [[rng.randrange(len(inits)) for _ in inits] for _ in range(NB)]

    def cmp(x, y, m):
        dx, dy = per[x][m], per[y][m]
        diff = [p - q for p, q in zip(dx, dy)]
        point = sum(diff) / den[m]
        dist = sorted(sum(diff[k] for k in idx) / den[m] for idx in boots)
        lo, hi = dist[int(.025 * NB)], dist[int(.975 * NB) - 1]
        return {"x": x, "y": y, "metric": m, "point": point, "ci": [lo, hi],
                "clearly_better": point >= DELTA - 1e-12 and lo > 0, "close": abs(point) <= CLOSE + 1e-12,
                "gained": sum(1 for p, q in zip(dx, dy) if p > q), "lost": sum(1 for p, q in zip(dx, dy) if p < q)}
    C = {}
    for x, y in [("F_old", "C_amp"), ("E_hist", "C_amp"), ("E_pure", "C_amp"), ("F_old", "E_hist"), ("F_old", "E_pure"),
                 ("E_hist", "B_mean"), ("E_pure", "B_mean"), ("E_hist", "G_rand"), ("E_pure", "G_rand"),
                 ("D_hist", "D_pure"), ("E_hist", "E_pure"), ("C_amp", "B_mean"), ("C_amp", "G_rand"),
                 ("D_hist", "B_mean"), ("D_pure", "B_mean"), ("D_hist", "G_rand"), ("D_pure", "G_rand"),
                 ("F_old", "D_hist"), ("F_old", "D_pure"), ("F_old", "B_mean"), ("F_old", "G_rand"), ("R_full", "F_old"),
                 ("B_mean", "A_nat"), ("G_rand", "B_mean"), ("E_hist", "D_hist"), ("E_pure", "D_pure")]:
        for m in ("dir", "place", "term"):
            C[f"{x}>{y}:{m}"] = cmp(x, y, m)
    c = lambda x, y, m: C[f"{x}>{y}:{m}"]
    V = {}
    V["V1_amplitude_sufficient"] = c("F_old", "C_amp", "place")["close"] and not any(c(e, "C_amp", "place")["clearly_better"] for e in ("E_hist", "E_pure"))
    V["V2_content_beyond_amplitude"] = {e: [m for m in ("dir", "place") if c(e, "C_amp", m)["clearly_better"]] for e in ("E_hist", "E_pure")}
    V["V3_partial_content"] = {e: c("F_old", e, "dir")["close"] and c("F_old", e, "place")["clearly_better"] for e in ("E_hist", "E_pure")}
    V["V4_local_specificity_weak"] = {e: not any(c(e, "B_mean", m)["clearly_better"] and c(e, "G_rand", m)["clearly_better"] for m in ("dir", "place")) for e in ("E_hist", "E_pure")}
    fplace = sum(per["F_old"]["place"])
    V["V5_snapshot_cannot_carry"] = fplace < 20
    V["H_capture"] = {k: {m: C[f"{k}:{m}"] for m in ("dir", "place", "term")} for k in ("D_hist>D_pure", "E_hist>E_pure")}
    cand = [g for g in ("C_amp", "E_pure", "E_hist", "D_pure", "D_hist")
            if any(c(g, "B_mean", m)["clearly_better"] and c(g, "G_rand", m)["clearly_better"] for m in ("dir", "place"))]
    order = {g: k for k, g in enumerate(("C_amp", "E_pure", "E_hist", "D_pure", "D_hist"))}
    cand.sort(key=lambda g: (-sum(per[g]["place"]), -sum(per[g]["dir"]), order[g]))
    pick = cand[0] if cand else None
    V["batch2_candidates"] = cand
    V["batch2_pick"] = pick
    V["batch2_run"] = bool(pick) and c("F_old", pick, "place")["clearly_better"]
    V["s5_language_routing_run"] = any(c("C_amp", "B_mean", m)["clearly_better"] for m in ("dir", "place"))
    # diagnostics
    diag = {}
    for g in GROUPS:
        rows = [E[i, g, q] for i in inits for q in "AB"]
        d = {}
        for k in ("old_proj_full", "old_cos_full", "norm_over_old", "old_proj_10x3", "new_over_old_norm", "pad_over_real_diff", "patch_over_old_norm"):
            vals = [r["diagnostics"][k][0] for r in rows if k in r["diagnostics"]]
            if vals:
                d[k] = sorted(vals)[len(vals) // 2]
        d["cap_hits"] = sum(round(r["diagnostics"]["cap_hit"][0] * r["n_window_queries"]) for r in rows if "cap_hit" in r["diagnostics"])
        d["degenerate"] = sum(round(r["diagnostics"]["degenerate"][0] * r["n_window_queries"]) for r in rows if "degenerate" in r["diagnostics"])
        d["queries"] = sum(r["n_window_queries"] for r in rows)
        d["clip_window_mean"] = sum(r["clip_rate_window"] for r in rows) / len(rows)
        d["clip_all_mean"] = sum(r["clip_rate_all"] for r in rows) / len(rows)
        d["both_contacts_window_end"] = sum(all(r["contacts_at_window_end"]) for r in rows)
        diag[g] = d
    nat = [E[i, "N_nat", "N"] for i in inits]
    summary = {g: {k: sum(v) for k, v in per[g].items()} for g in GROUPS}
    summary["N_nat"] = {"front_first": sum(r["first"] == ["front"] for r in nat), "front_ever": sum(r["ever"]["front"] for r in nat),
                        "front_term": sum(r["terminal"]["front"] for r in nat)}
    out = {"inits": inits, "boundaries": {r["init"]: r["boundary"] for r in snaps["records"]}, "summary": summary,
           "per_snapshot": per, "comparisons": C, "verdicts": V, "diagnostics": diag,
           "snapshot_screen": {"candidates": len(snaps["records"]), "qualified": snaps["qualified"],
                               "retained_30": sum(bool(r.get("retained_30")) for r in snaps["records"]),
                               "rebuild_ok": sum(all(r.get("rebuild_x3", [])) for r in snaps["records"])}}
    (OUT / "judgement").mkdir(exist_ok=True)
    (OUT / "judgement/k14_judgement.json").write_text(json.dumps(out, indent=1))
    L = ["| 组 | 左先到左 | 右先到右 | 成对分流 /20 | 先到前 /40 | 左曾成功 | 右曾成功 | 曾成功 /40 | 左终态 | 右终态 | 终态 /40 | 前格终态 /40 |", "|---|" + "---:|" * 11]
    for g in GROUPS:
        s = summary[g]
        L.append(f"| {g} | {s['L_first']} | {s['R_first']} | {s['dir']} | {s['front_first']} | {s['L_ever']} | {s['R_ever']} | {s['place']} | {s['L_term']} | {s['R_term']} | {s['term']} | {s['front_term']} |")
    L.append(""); L.append("| 比较 | 指标 | 差值 | 95% CI | 明确更好 | 接近 | 增/减 |"); L.append("|---|---|---:|---|---|---|---|")
    for k, v in C.items():
        L.append(f"| {v['x']} − {v['y']} | {v['metric']} | {v['point']:+.3f} | [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | {'是' if v['clearly_better'] else ''} | {'是' if v['close'] else ''} | {v['gained']}/{v['lost']} |")
    L.append(""); L.append("```json"); L.append(json.dumps(V | {"H_capture": "see table"}, ensure_ascii=False, indent=1)); L.append("```")
    L.append(""); L.append("```json"); L.append(json.dumps(diag, indent=1)); L.append("```")
    (OUT / "judgement/k14_judgement.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:len(GROUPS) + 2]))
    print(json.dumps(V | {"H_capture": None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
