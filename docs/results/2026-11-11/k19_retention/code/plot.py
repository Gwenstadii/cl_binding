"""Plot development and independent confirmation endpoints for K19."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager,pyplot as plt
import numpy as np

LABELS={"parent":"R1","full":"完整续训","freeze_kv":"冻结 K/V","mu":"约束 μ","d":"约束 d",
        "both":"约束 μ＋d","replay":"稀疏回放","both_replay":"约束＋稀疏回放",
        "replay_full":"动态回放 1.0","replay_soft":"动态回放 0.35"}
SCREEN=("parent","full","freeze_kv","mu","d","both","replay","both_replay","replay_full","replay_soft")


def main(summary,output):
    s=json.loads(summary.read_text(encoding="utf-8"))
    font=Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"]=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"]=False
    fig,(a,b)=plt.subplots(2,1,figsize=(10.7,8.5),layout="constrained")

    def draw(ax,rows,labels,title):
        x=np.arange(len(rows))
        for delta,color,label,key in ((-.23,"#244b83","左句先到左",("left","first","left")),
                                      (0,"#b1581b","右句先到右",("right","first","right")),
                                      (.23,"#14776b","前句终态放对",("front","terminal","front"))):
            g,m,t=key
            bars=ax.bar(x+delta,[r[g][m].get(t,0) for r in rows],width=.22,label=label,color=color)
            ax.bar_label(bars,padding=1,fontsize=8)
        ax.set_title(title,loc="left",fontweight="bold",fontsize=14)
        ax.set_xticks(x,labels,rotation=15 if len(rows)>6 else 0,ha="right" if len(rows)>6 else "center")
        ax.set_ylabel("次数 / 20")
        ax.set_ylim(0,24.7)
        ax.set_yticks(range(0,21,5))
        ax.grid(axis="y",color="#e5e8ef")
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(loc="upper center",ncols=3,frameon=False)

    draw(a,[s["dev"][f"{0 if x=='parent' else 20261021}_{x}"] for x in SCREEN],
         [LABELS[x] for x in SCREEN],"K19 开发筛选")
    arm=s["screen"]["candidate"]
    if s["confirm"]:
        keys=["0_parent",*(f"{seed}_{name}" for seed in (20261121,20261122) for name in ("full",arm))]
        draw(b,[s["confirm"][k] for k in keys],
             ["R1","流 1 完整续训",f"流 1 {LABELS[arm]}","流 2 完整续训",f"流 2 {LABELS[arm]}"],
             "新物理初态上的两条随机流确认")
    else:
        b.set_visible(False)
    fig.savefig(output,dpi=165,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("summary",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    main(a.summary,a.output)
