"""Exploratory E2b analysis: text controls that keep the object/scene nouns.

Post-hoc block; never pooled with the pre-registered E2 cells.
"""
import json
from collections import Counter
from pathlib import Path

import k3_e2_analyze as A

OUT = Path("/home/lzx/cl_binding/.server/k3_e2_20261102_01")
KEYS = ["W1", "W2", "W3"]


def main():
    P = json.loads((OUT / "prereg/k3_e2b_prereg.json").read_text())
    rows = A.load("b_")
    goal = {f: P["families"][f]["new_goal"] for f in P["design"]["families"]}
    cells, _ = A.summarize(rows, P, KEYS, goal)
    out = {"episodes": len(rows), "expected": P["design"]["episodes"], "cells": cells}
    (OUT / "judgement").mkdir(exist_ok=True)
    (OUT / "judgement/e2b_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    lines = ["# K3-E2b（事后补充）：保留物体名词的文本控制", "",
             "回合 " + str(len(rows)) + "/" + str(P["design"]["episodes"]) + "。事后补充块，不与 E2 合并判定。", "",
             "| 家族 | 模型 | 指令 | 文本 | 捕获 | 任何目标 | 抬起 | 首次目标分布 |",
             "|---|---|---|---|---:|---:|---:|---|"]
    for f in P["design"]["families"]:
        for m in ("R0", "FT4"):
            for k in KEYS:
                c = cells.get(f + "|" + m + "|" + k)
                if not c:
                    continue
                lines.append("| " + " | ".join([f, m, k, c["instruction"], str(c["capture"]), str(c["any_target"]),
                                                 str(c["lifted"]), str(c["first_target_distribution"])]) + " |")
    (OUT / "judgement/e2b_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: (v["capture"], v["any_target"], v["lifted"]) for k, v in cells.items()},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
