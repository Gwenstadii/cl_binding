"""Generate a standalone K18 figure from the verified summary."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import font_manager
import numpy as np


def main(summary_path, output_path):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    navy, green, orange, gray = "#214f86", "#12786e", "#b74c0d", "#858da4"
    fig, (probe, native, adam) = plt.subplots(3, 1, figsize=(9.5, 11), layout="constrained")

    def tune(ax, title, ylabel):
        ax.set_title(title, loc="left", fontsize=15, weight="bold", pad=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e5e8ed", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    steps = list(summary["probe"])
    x = np.log10(1 + np.array(steps, dtype=int))
    base = [summary["probe"][s]["10"] for s in steps]
    horizontal = [summary["probe_channels"][s]["10"]["contrast_by_channel"]["H"]["gain"]["median"]
                  for s in steps]
    probe.fill_between(x, [d["gain"]["q1"] for d in base], [d["gain"]["q3"] for d in base],
                       color=navy, alpha=0.14, label="跨场景中间 50%")
    probe.plot(x, [d["gain"]["median"] for d in base], "o-", color=navy, lw=2.2, label="有效 7 维旧方向增益")
    probe.plot(x, horizontal, "s--", color=green, lw=1.7, label="水平动作旧方向增益")
    probe.plot(x, [d["cosine"]["median"] for d in base], "^:", color=gray, lw=1.7, label="方向余弦")
    tune(probe, "固定输入下的旧指令区别", "归一化读数")
    probe.set_xticks(x, steps)
    probe.set_xlabel("Front 续训步数（横轴为对数间距）")
    probe.legend(loc="lower left", fontsize=9, ncols=2, frameon=False)

    steps = list(summary["native"])
    positions = np.arange(len(steps))
    for name, label, color, marker in (("left", "左", navy, "o"), ("right", "右", orange, "s"),
                                        ("front", "前", green, "^")):
        rows = [summary["native"][s][name] for s in steps]
        native.plot(positions, [r["first"].get(name, 0) for r in rows], marker=marker, color=color,
                    lw=2, label=label)
        native.plot(positions, [r["terminal"][name] for r in rows], ":", marker=marker, color=color,
                    lw=1.8, alpha=0.7)
    tune(native, "闭环表现　实线为首次到对目标，点线为终态成功", "次数 / 20")
    native.set_xticks(positions, steps)
    native.set_yticks(range(0, 21, 5))
    native.set_ylim(-1, 21)
    native.set_xlabel("Front 续训步数")
    native.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), frameon=False, ncols=3)

    by_step = [(str(int(k.split("_")[-1])), v["branches"]) for k, v in summary["attribution"].items()
               if k.startswith("marginal_")]
    positions = np.arange(len(by_step))
    for offset, key, label, color in ((-0.18, "full", "完整 Front 梯度", navy),
                                      (0.18, "zero_gradient", "当前梯度置零", orange)):
        values = [b[key]["gain_delta"]["median"] * 1000 for _, b in by_step]
        adam.bar(positions + offset, values, width=0.34, label=label, color=color)
    adam.axhline(0, color="#4e5664", linewidth=0.8)
    tune(adam, "同一个锚点实际执行一步 AdamW", "旧方向增益变化 × 1000")
    adam.set_xticks(positions, [s for s, _ in by_step])
    adam.set_xlabel("锚点步数")
    adam.legend(loc="best", frameon=False)
    fig.savefig(output_path, dpi=165, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.summary, args.output)
