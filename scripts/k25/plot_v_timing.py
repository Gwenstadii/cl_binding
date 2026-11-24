"""Make the complete timing grid visible without selecting a winning window."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

import v_timing_protocol as spec

LABELS = ["Native", "1–100", "41–140", "81–180", "161–260", "1–400", "81–400", "161–400"]
LINE_LABELS = {"lr_front_1000": "Left + Right → Front", "lf_right_1000": "Left + Front → Right"}


def plot(folder, output):
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharey=True)
    x = np.arange(8)
    for row, line in enumerate(spec.LINES):
        item = summary["lines"][line]
        for col, goal in enumerate("AB"):
            ax = axes[row, col]
            ax.axvspan(.5, 4.5, color="#edf4fa", zorder=-5)
            ax.axvspan(4.5, 7.5, color="#f7f1e8", zorder=-5)
            for arm, shift, color in (("forward", -.10, "#176da0"), ("reverse", .10, "#b14b37")):
                groups = [item["native"], *[item["windows"][w][arm] for w in spec.WINDOWS]]
                for metric, marker, style in (("first_correct", "o", "-"), ("terminal_correct", "s", "--")):
                    values = [s["per_goal"][goal][metric] for s in groups]
                    ax.plot(x+shift, values, marker=marker, ms=6, color=color, ls=style, lw=1,
                            markerfacecolor=color if metric == "first_correct" else "white")
            # Native is shared, not a polarity experiment.
            for metric, marker in (("first_correct", "o"), ("terminal_correct", "s")):
                ax.scatter([0], [item["native"]["per_goal"][goal][metric]], marker=marker,
                           s=45, c="#333333", zorder=5)
            ax.set_title(f"{LINE_LABELS[line]}  |  old {spec.GOALS[line][goal].title()}", loc="left", pad=12)
            ax.set_xticks(x, LABELS, rotation=35, ha="right")
            ax.set_yticks(range(0, 21, 5))
            ax.set_ylim(-.8, 21)
            ax.set_xlim(-.4, 7.4)
            ax.grid(axis="y", alpha=.2)
            if col == 0:
                ax.set_ylabel("Correct episodes / 20 starts")
    handles = [Line2D([], [], color=c, marker=m, ls=ls, markerfacecolor=c if m == "o" else "white", label=label)
               for c, m, ls, label in (("#176da0", "o", "-", "Forward: first arrival"),
                   ("#176da0", "s", "--", "Forward: final placement"),
                   ("#b14b37", "o", "-", "Reverse: first arrival"),
                   ("#b14b37", "s", "--", "Reverse: final placement"))]
    fig.legend(handles=handles, loc="lower center", ncols=4, frameon=False, bbox_to_anchor=(.5, .052))
    fig.suptitle("K25-V: fixed V repair, varied intervention timing", x=.06, ha="left", fontsize=17)
    fig.text(.06, .022, "Both checkpoints: step 1000; eight V differences: 30→1000; α=5.294. Same 20 starts and canonical prompts.\n"
             "Blue background: equal 100-step windows. Beige: sustained repair (duration differs). Lines guide the eye; no fitted trend.", fontsize=9)
    fig.tight_layout(rect=(.02, .11, 1, .95), h_pad=2.0)
    for extension in ("png", "pdf"):
        path = output / ("k25_v_timing."+extension)
        fig.savefig(path, dpi=200, metadata={"Title": "K25-V intervention timing"})
        print(path)
    plt.close(fig)


def trajectories(folder, output):
    """The lowest numbered initial state is an illustration, not a selected win."""
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    cases = {r["init"]: r for r in json.loads((folder / "cases.json").read_text(encoding="utf-8"))}
    initial = min(spec.IDS)
    selected = {}
    names = ("native", "early_100", "delayed_100", "full_400")
    for path in folder.glob("*.jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for text in stream:
                row = json.loads(text)
                if row["init"] == initial and row["goal"] == "B" and row["window"] in names and row["arm"] in ("native", "forward"):
                    selected[row["line"], row["window"]] = row
    assert len(selected) == 8
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = {"native": "#444444", "early_100": "#c17c27", "delayed_100": "#176da0", "full_400": "#7c4b9d"}
    labels = {"native": "Native", "early_100": "Forward 1–100", "delayed_100": "Forward 81–180", "full_400": "Forward 1–400"}
    for col, line in enumerate(spec.LINES):
        xy, height = axes[0, col], axes[1, col]
        for g, geometry in protocol["geometry"].items():
            c = geometry["centre"]
            ext = [sum(abs(geometry["rotation"][i][k])*geometry["half_size"][k] for k in range(3)) for i in (0, 1)]
            xy.add_patch(Rectangle((c[0]-ext[0], c[1]-ext[1]), 2*ext[0], 2*ext[1],
                facecolor="#e9eef1", edgecolor="#a0adb5", lw=.8))
            xy.text(c[0], c[1], g.title(), ha="center", va="center", fontsize=9, color="#4f5c65")
        for w in names:
            row = selected[line, w]
            points = np.asarray([row["trace"]["start"]["object"], *row["trace"]["object"]])
            color = colors[w]
            xy.plot(points[:, 0], points[:, 1], color=color, lw=1.4, alpha=.9)
            xy.scatter(points[80, 0], points[80, 1], marker="x", c=color, s=45, zorder=5)
            xy.scatter(points[-1, 0], points[-1, 1], marker="s", facecolors="none", edgecolors=color, s=40, zorder=5)
            height.plot(range(401), 1000*(points[:, 2]-cases[initial]["z0"]), color=color, lw=1.3)
            if row["grasp_confirmed_step"] is not None:
                step = row["grasp_confirmed_step"]
                height.scatter([step], [1000*(points[step, 2]-cases[initial]["z0"])], color=color, marker="o", s=18)
        height.axhline(30, ls=":", color="#999999", lw=.8)
        height.axvline(80, ls=":", color="#555555", lw=.8)
        height.axvline(180, ls=":", color="#555555", lw=.8)
        xy.set_aspect("equal", adjustable="datalim")
        xy.set_xlabel("Book x (m)")
        xy.set_ylabel("Book y (m)")
        xy.set_title(f"{LINE_LABELS[line]}\nold {spec.GOALS[line]['B'].title()}, initial state {initial}", loc="left", pad=10)
        height.set_xlabel("Executed control step")
        height.set_ylabel("Book height above reset (mm)")
        height.grid(alpha=.15)
    handles = [Line2D([], [], color=colors[w], lw=2, label=labels[w]) for w in names]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .06), ncols=4, frameon=False)
    fig.text(.06, .02, "Illustration: lowest numbered state, old goal B, same checkpoints and noise. ×: step 80; □: step 400; small dot: confirmed grasp.\n"
             "Horizontal regions are the scoring bounds. Trajectories do not replace the aggregate results over all 20 states.", fontsize=9)
    fig.suptitle("K25-V: book trajectories and grasp timing", x=.06, ha="left", fontsize=16)
    fig.tight_layout(rect=(0, .13, 1, .95), h_pad=2)
    for extension in ("png", "pdf"):
        path = output / ("k25_v_trajectories."+extension)
        fig.savefig(path, dpi=200, metadata={"Title": "K25-V paired trajectories, state 800"})
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.folder, args.output)
    trajectories(args.folder, args.output)
