"""Plot the primary K18-C comparisons from the audit summary."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np


def main(source, output):
    summary = json.loads(source.read_text(encoding="utf-8"))
    result = summary["000030_000100"]
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    navy, orange, teal = "#244b83", "#b55b1d", "#138478"
    fig, (a,b,c) = plt.subplots(3,1,figsize=(10,10.6),layout="constrained")

    def axes(ax,title,ylabel):
        ax.set_title(title,loc="left",fontsize=14,weight="bold",pad=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y",color="#e5e8ef")
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)

    groups = result["probe"]["by_group"]
    x = np.arange(len(groups))
    for dx,metric,label,color in ((-.18,"add_loss","加入后衰减",navy),(.18,"undo_rescue","撤回后恢复",orange)):
        scores = [g[metric]["gain_H"]["value"] for g in groups.values()]
        ends = [g[metric]["gain_H"]["bootstrap_95"] for g in groups.values()]
        errors = [[max(0,s-a) for s,(a,_) in zip(scores,ends)],
                  [max(0,b-s) for s,(_,b) in zip(scores,ends)]]
        a.bar(x+dx,scores,yerr=errors,width=.34,color=color,capsize=3,label=label)
    a.axhline(0,color="#5a6677",lw=.8)
    a.set_xticks(x,["状态投影","动作／时间输入","动作专家注意力","MLP／归一化","动作输出投影"])
    a.legend(loc="upper left",frameon=False)
    axes(a,"第 30–100 步真实参数改动对旧水平区别的影响","旧方向增益变化")

    native = result["native"]["conditions"]
    conditions = list(native)
    x = np.arange(len(conditions))
    for dx,goal,label,color in ((-.24,"left","左句先到左",navy),(0,"right","右句先到右",orange),
                                (.24,"front","前句终态放对",teal)):
        scores = [v[goal]["first"].get(goal,0) if goal != "front" else v[goal]["terminal"][goal]
                  for v in native.values()]
        bars = b.bar(x+dx,scores,width=.22,color=color,label=label)
        b.bar_label(bars,padding=2,fontsize=8)
    assert len(conditions) == 6
    b.set_xticks(x,["第30步","第100步","30＋真实\n注意力","100−真实\n注意力","30＋随机\n注意力","100用随机\n方向替代"])
    b.set_ylim(0,24)
    b.set_yticks(range(0,21,5))
    b.legend(loc="upper right",ncols=3,frameon=False)
    axes(b,"第 30–100 步参数改动后的原生闭环","次数 / 20")

    later = summary["000030_001000"]["native"]["conditions"]
    x = np.arange(len(later))
    bottom = np.zeros(len(later),dtype=int)
    for key,label,color in (("right","先到右",navy),("front","先到前",orange),("none","未到区","#8c98aa"),
                            ("other","其他目标","#714f80")):
        if key == "other":
            values = [20-n for n in bottom]
        else:
            values = [v["right"]["first"].get(key,0) for v in later.values()]
        if not any(values):
            continue
        bars = c.bar(x,values,bottom=bottom,width=.54,color=color,label=label)
        c.bar_label(bars,labels=[str(v) if v else "" for v in values],label_type="center",color="white")
        bottom += values
    assert list(bottom) == [20]*4
    c.set_xticks(x,["第30步","第1000步","1000−真实注意力","1000用随机方向替代"])
    c.set_ylim(0,24)
    c.set_yticks(range(0,21,5))
    c.legend(loc="upper right",ncols=4,frameon=False)
    axes(c,"第 1000 步右句最先到哪里","次数 / 20")
    fig.savefig(output,dpi=165,facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    main(a.source,a.output)
