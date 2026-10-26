"""Cost curves of ordinary replay: goal substitution vs. execution success as the per-goal memory shrinks.

Numbers are counts over the 40 old-instruction episodes (left + front, 20 each, paired states 20-39) taken from the
committed comparison/failure-phase files; budget 0 is the matched FT4 branch.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BUDGETS = [0, 1, 4, 16, 64, 256]


def main(args):
    data = json.loads(args.data.read_text())
    if args.font:
        font_manager.fontManager.addfont(str(args.font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(args.font)).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=200)
    colors = ["#3b915d", "#4f6d9a", "#c0504d", "#8a6db1", "#d08a2e"]
    xs = range(len(BUDGETS))
    for (setting, rows), color in zip(data["settings"].items(), colors):
        for ax, key in zip(axes, ("substituted", "success")):
            points = [(i, rows[str(b)][key] / 40) for i, b in enumerate(BUDGETS) if str(b) in rows]
            ax.plot([p[0] for p in points], [p[1] for p in points], "-o", color=color, lw=1.8, ms=4, label=setting)
    for ax, key in zip(axes, ("substituted", "success")):
        for label, rows in data.get("extra_points", {}).items():
            for b, v in rows.items():
                if key in v:
                    ax.scatter(BUDGETS.index(int(b)), v[key] / 40, marker="*", s=110, color="#1f2328", zorder=5,
                               label=label if b == next(iter(rows)) else None)
    titles = ("旧指令被送去新目标的比例（选择失败）", "旧指令真正完成的比例（执行）")
    for ax, title in zip(axes, titles):
        ax.set_xticks(list(xs), ["0\n(FT4)", "1", "4", "16", "64", "256"])
        ax.set_xlabel("每个旧目标的回放记忆帧数")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", color="#e3e5e8", lw=0.8)
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--font", type=Path, help="CJK font file (e.g. a Noto CJK .ttc)")
    main(p.parse_args())
