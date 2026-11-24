"""Plot physical-state-level first arrival and terminal placement separately."""

import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"native": "#777777", "forward": "#2468a0", "reverse": "#be633b"}


def plot(summary, output):
    report = json.loads(summary.read_text(encoding="utf-8"))
    lines = list(report["lines"].items())
    if "reserve_outcomes" in report:
        lines.append(("lr_front_1000", dict(goals=("left", "right", "front"), gain=1.,
                                          outcomes={"main": report["reserve_outcomes"]}, reserve=True)))
    fig, axes = plt.subplots(len(lines), 2, figsize=(11.5, 2.6*len(lines)+.8), sharey=True,
                             squeeze=False, layout="constrained")
    for i, (line, item) in enumerate(lines):
        a, b, new = item["goals"]
        step = line.rsplit("_", 1)[-1]
        title = f"{a.title()} + {b.title()} -> {new.title()}, step {step}, gain {item['gain']:.4g}"
        title += " [confirmation set 900]" if item.get("reserve") else " [set 800]"
        for j, field in enumerate(("first_correct", "terminal_correct")):
            ax = axes[i, j]
            joint = "joint_first" if j == 0 else "joint_terminal"
            for k, arm in enumerate(COLORS):
                result = item["outcomes"]["main"][arm]
                vals = [result["per_goal"][g][field] for g in "AB"]+[result[joint]]
                positions = np.arange(3)+(k-1)*.25
                ax.bar(positions, vals, width=.23, color=COLORS[arm], label=arm)
                for x, y in zip(positions, vals):
                    ax.text(x, y+.35, str(y), ha="center", va="bottom", fontsize=9)
            ax.set_xticks(range(3), (a.title(), b.title(), "Both old goals"))
            ax.set_ylim(0, 22)
            ax.set_yticks(range(0, 21, 5))
            ax.set_title(("First correct arrival" if j == 0 else "Terminal correct placement"), loc="left", fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel("Physical starts / 20")
                ax.text(0, 1.18, title, transform=ax.transAxes, fontsize=11, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Single-policy V contrast repair on fresh paired starts\n"
                 "20 paired starts per set; steps 81-180; gains selected on 8 development starts",
                 fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.summary, args.output)
