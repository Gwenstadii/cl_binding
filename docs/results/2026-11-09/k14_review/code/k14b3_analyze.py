"""K14 batch-3 judgement (rules: ../prereg/k14_b3_prereg.json)."""
import json
from pathlib import Path
import random

OUT = Path(__file__).resolve().parent.parent
B3 = OUT / "batch3"
GROUPS = ("T0_cand", "T1_z", "T2_rotgrip", "T3_nonhoriz", "T4_xy", "T5_full", "T1r_z", "T2r_rotgrip", "T3r_nonhoriz")
DELTA, CLOSE, NB, SEED = 0.15, 0.10, 10000, 20261108


def main():
    recs = [json.loads(l) for f in sorted(B3.glob("w*.jsonl")) if "traces" not in f.name for l in f.read_text().splitlines()]
    el = json.loads((B3 / "b3_snapshots.json").read_text())["eligible"]
    states = [(r["init"], r["goal"]) for r in el]
    E = {(r["init"], r["goal"], r["group"]): r for r in recs}
    assert len(E) == len(recs) == len(states) * len(GROUPS), len(recs)
    inits = sorted({i for i, _ in states})
    per = {g: {m: {s: int(E[s[0], s[1], g][k]) for s in states} for m, k in (("term", "term_req"), ("ever", "ever_after"))} for g in GROUPS}
    n = len(states)
    rng = random.Random(SEED)
    boots = [[rng.choice(inits) for _ in inits] for _ in range(NB)]
    by_init = {i: [s for s in states if s[0] == i] for i in inits}

    def cmp(x, y, m, side=None):
        ss = [s for s in states if side is None or s[1] == side]
        d = {s: per[x][m][s] - per[y][m][s] for s in ss}
        pt = sum(d.values()) / len(ss)
        out = {"x": x, "y": y, "metric": m, "side": side or "pooled", "n": len(ss), "point": pt,
               "gained": sum(v > 0 for v in d.values()), "lost": sum(v < 0 for v in d.values())}
        if side is None:
            dist = []
            for idx in boots:
                picked = [s for i in idx for s in by_init[i]]
                dist.append(sum(d[s] for s in picked) / len(picked))
            dist.sort()
            out["ci"] = [dist[int(.025 * NB)], dist[int(.975 * NB) - 1]]
            out["clearly_better"] = pt >= DELTA - 1e-12 and out["ci"][0] > 0
            out["close"] = abs(pt) <= CLOSE + 1e-12
        return out
    pairs = [("T1_z", "T0_cand"), ("T2_rotgrip", "T0_cand"), ("T3_nonhoriz", "T0_cand"), ("T4_xy", "T0_cand"), ("T5_full", "T0_cand"),
             ("T1_z", "T1r_z"), ("T2_rotgrip", "T2r_rotgrip"), ("T3_nonhoriz", "T3r_nonhoriz"),
             ("T5_full", "T3_nonhoriz"), ("T5_full", "T4_xy"), ("T5_full", "T1_z"), ("T5_full", "T2_rotgrip")]
    C = {}
    for x, y in pairs:
        for m in ("term", "ever"):
            C[f"{x}>{y}:{m}"] = cmp(x, y, m)
            for side in "AB":
                C[f"{x}>{y}:{m}:{side}"] = cmp(x, y, m, side)
    c = lambda x, y, m: C[f"{x}>{y}:{m}"]
    V = {}
    for S, T, Tr in (("z", "T1_z", "T1r_z"), ("rotgrip", "T2_rotgrip", "T2r_rotgrip"), ("nonhoriz", "T3_nonhoriz", "T3r_nonhoriz")):
        V[f"channel_{S}_carries_content"] = [m for m in ("term", "ever") if c(T, "T0_cand", m)["clearly_better"] and c(T, Tr, m)["clearly_better"]]
    V["xy_carries_content"] = [m for m in ("term", "ever") if c("T4_xy", "T0_cand", m)["clearly_better"]]
    V["not_channel_separable"] = c("T5_full", "T3_nonhoriz", "term")["clearly_better"] and c("T5_full", "T4_xy", "term")["clearly_better"]
    summ = {g: {"term": sum(per[g]["term"].values()), "ever": sum(per[g]["ever"].values()),
                "L_term": sum(v for s, v in per[g]["term"].items() if s[1] == "A"), "R_term": sum(v for s, v in per[g]["term"].items() if s[1] == "B"),
                "L_ever": sum(v for s, v in per[g]["ever"].items() if s[1] == "A"), "R_ever": sum(v for s, v in per[g]["ever"].items() if s[1] == "B")} for g in GROUPS}
    diag = {}
    for g in GROUPS:
        rows = [E[s[0], s[1], g] for s in states]
        diag[g] = {k: sorted(r["diagnostics"][k][0] for r in rows)[n // 2] for k in rows[0]["diagnostics"] if k not in ("cap_hit", "degenerate")}
        diag[g]["cap_hits"] = sum(r["diagnostics"].get("cap_hit", [0])[2] for r in rows)
        diag[g]["clip_mean"] = sum(r["clip_rate"] for r in rows) / n
        diag[g]["min_z_rel_median"] = sorted(r["min_z_rel_after"] for r in rows)[n // 2]
    t0_repro = sum(bool(E[s[0], s[1], "T0_cand"]["t0_reproduces_J"]) for s in states)
    out = {"n_states": n, "n_left": sum(s[1] == "A" for s in states), "n_right": sum(s[1] == "B" for s in states),
           "t0_reproduces_J": t0_repro, "summary": summ, "per_state": {g: {f"{s[0]}{s[1]}": per[g]["term"][s] for s in states} for g in GROUPS},
           "comparisons": C, "verdicts": V, "diagnostics": diag}
    (OUT / "judgement/k14_b3_judgement.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    L = [f"状态数 {n}（左 {out['n_left']}，右 {out['n_right']}）；T0 逐步复现 J：{t0_repro}/{n}", "",
         "| 组 | 终态 | 左终态 | 右终态 | 分叉后曾成功 | 左 | 右 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for g in GROUPS:
        s = summ[g]
        L.append(f"| {g} | {s['term']} | {s['L_term']} | {s['R_term']} | {s['ever']} | {s['L_ever']} | {s['R_ever']} |")
    L += ["", "| 比较 | 指标 | 差值 | 95% CI | 明确更好 | 接近 | 增/减 |", "|---|---|---:|---|---|---|---|"]
    for v in C.values():
        if v["side"] == "pooled":
            L.append(f"| {v['x']} − {v['y']} | {v['metric']} | {v['point']:+.3f} | [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | {'是' if v['clearly_better'] else ''} | {'是' if v['close'] else ''} | {v['gained']}/{v['lost']} |")
    L += ["", "```json", json.dumps({"verdicts": V, "diagnostics": diag}, ensure_ascii=False, indent=1), "```"]
    (OUT / "judgement/k14_b3_judgement.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:len(GROUPS) + 4]))
    print(json.dumps(V))


if __name__ == "__main__":
    main()
