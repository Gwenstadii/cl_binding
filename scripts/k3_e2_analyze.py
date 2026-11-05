"""K3-E2 analysis: capture of the new goal as a function of the test instruction.

Reads prereg/k3_e2_prereg.json and runs/w*.jsonl (E2 only; the exploratory b_ files are excluded).
Writes judgement/e2_summary.json and judgement/e2_report.md.
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("/home/lzx/cl_binding/.server/k3_e2_20261102_01")
KEYS = ["A", "B", "N", "P", "S", "U", "Z"]
FENCE = chr(96) * 3


def load(prefix="w"):
    rows = []
    for path in sorted((OUT / "runs").glob(prefix + "*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rate(n, d):
    return None if d == 0 else round(n / d, 3)


def sign_test(pairs):
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p": None}
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"b": b, "c": c, "p": round(min(1.0, p), 4)}


def summarize(rows, P, keys, new_goal):
    by = defaultdict(list)
    for r in rows:
        by[(r["family"], r["model"], r["instruction_key"])].append(r)
    cells = {}
    for family in P["design"]["families"]:
        for model in P["design"]["models"]:
            for key in keys:
                rs = by[(family, model, key)]
                if not rs:
                    continue
                n = len(rs)
                dist = Counter(t for r in rs for t in (r["first_targets"] or ["none"]))
                steps = sorted(r["first_target_step"] for r in rs if r["first_target_step"])
                cells[family + "|" + model + "|" + key] = {
                    "family": family, "model": model, "key": key,
                    "instruction": rs[0]["instruction"], "n": n,
                    "capture": rate(sum(1 for r in rs if r["first_targets"] == [new_goal[family]]), n),
                    "any_target": rate(sum(1 for r in rs if r["first_targets"]), n),
                    "lifted": rate(sum(1 for r in rs if r["lifted"]), n),
                    "first_target_distribution": dict(dist),
                    "success_new_goal": rate(sum(1 for r in rs
                                                 if r["native_success_step"][new_goal[family]] is not None), n),
                    "median_first_target_step": steps[len(steps) // 2] if steps else None,
                }
    return cells, by


def paired(by, family, model, a, b, goal):
    ra = {r["init"]: r for r in by[(family, model, a)]}
    rb = {r["init"]: r for r in by[(family, model, b)]}
    common = sorted(set(ra) & set(rb))
    pairs = [(ra[i]["first_targets"] == [goal], rb[i]["first_targets"] == [goal]) for i in common]
    return {"n": len(pairs), **sign_test(pairs)}


def main():
    P = json.loads((OUT / "prereg/k3_e2_prereg.json").read_text())
    rows = load()
    fams = P["design"]["families"]
    goal = {f: P["families"][f]["new_goal"] for f in fams}
    cells, by = summarize(rows, P, KEYS, goal)

    contrasts = {}
    for family in fams:
        for model in ("R0", "FT4"):
            for a, b in (("N", "U"), ("N", "Z"), ("N", "P"), ("N", "S"), ("A", "B")):
                contrasts[family + "|" + model + "|" + a + "_vs_" + b] = paired(by, family, model, a, b, goal[family])

    ft4 = {f: {k: cells.get(f + "|FT4|" + k, {}).get("capture") for k in KEYS} for f in fams}
    r0 = {f: {k: cells.get(f + "|R0|" + k, {}).get("capture") for k in KEYS} for f in fams}

    def all_at(table, keys, op):
        vals = [(f, k, table[f][k]) for f in fams for k in keys]
        ok = [(f, k, v) for f, k, v in vals if v is not None and op(v)]
        return {"values": {f + "|" + k: v for f, k, v in vals},
                "passing": len(ok), "total": len(vals), "all": len(ok) == len(vals)}

    verdict = {
        "H_vis_every_instruction_captures": all_at(ft4, KEYS, lambda v: v >= 0.8),
        "H_sim_form_like_captures_unrelated_does_not": {
            "similar_ge_0.8": all_at(ft4, ["N", "P", "S"], lambda v: v >= 0.8),
            "unrelated_le_0.3": all_at(ft4, ["U", "Z"], lambda v: v <= 0.3)},
        "P3_R0_capture_le_0.2": all_at(r0, KEYS, lambda v: v <= 0.2),
        "P4_replication": {
            "R0_first_under_A": {f: cells.get(f + "|R0|A", {}).get("first_target_distribution") for f in fams},
            "R0_first_under_B": {f: cells.get(f + "|R0|B", {}).get("first_target_distribution") for f in fams},
            "FT4_capture_A_B": {f: (ft4[f]["A"], ft4[f]["B"]) for f in fams}},
        "P5_lift_under_UZ": {f: {k: cells.get(f + "|FT4|" + k, {}).get("lifted") for k in ("U", "Z")} for f in fams},
        "family_pattern": {f: [ft4[f][k] for k in KEYS] for f in fams},
    }
    out = {"episodes": len(rows), "expected": P["design"]["episodes"],
           "complete": len(rows) == P["design"]["episodes"],
           "cells": cells, "contrasts": contrasts, "verdict": verdict}
    (OUT / "judgement").mkdir(exist_ok=True)
    (OUT / "judgement/e2_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")

    lines = ["# K3-E2 结果：换测试指令下的新目标捕获", "",
             "回合 " + str(len(rows)) + "/" + str(P["design"]["episodes"]) + "，每格 10 个开发初态（0-9），"
             "固定噪声流。捕获 = 首次进入的目标恰为新目标；任何目标 = 至少进入一个目标格；抬起 = 物体中心抬高超过 3cm。", "",
             "| 家族 | 模型 | 指令 | 指令文本 | 捕获 | 任何目标 | 抬起 | 首次目标分布 | 新目标终态成功 |",
             "|---|---|---|---|---:|---:|---:|---|---:|"]
    for f in fams:
        for m in ("R0", "FT4"):
            for k in KEYS:
                c = cells.get(f + "|" + m + "|" + k)
                if not c:
                    continue
                lines.append("| " + " | ".join([f, m, k, c["instruction"], str(c["capture"]), str(c["any_target"]),
                                                 str(c["lifted"]), str(c["first_target_distribution"]),
                                                 str(c["success_new_goal"])]) + " |")
    lines += ["", "## 配对对比（同家族、同模型、同初态，捕获的符号检验）", ""]
    for name, c in contrasts.items():
        lines.append("- " + name + ": n=" + str(c["n"]) + " b=" + str(c["b"]) + " c=" + str(c["c"]) + " p=" + str(c["p"]))
    lines += ["", "## 预登记判定", "", FENCE + "json",
              json.dumps(verdict, ensure_ascii=False, indent=1), FENCE, ""]
    (OUT / "judgement/e2_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"episodes": len(rows), "complete": len(rows) == P["design"]["episodes"],
                      "family_pattern_A_B_N_P_S_U_Z": verdict["family_pattern"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
