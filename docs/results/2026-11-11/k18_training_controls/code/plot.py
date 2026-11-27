"""Plot K18-D outcomes by training seed and checkpoints."""
import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np

SEEDS=(20261021,20261121,20261122)
ARMS=("full","no_rot","random","norm")
STEPS=(30,100,300,1000)
LABELS=("完整监督","去旋转","同量随机遮蔽","匹配实际步长")
COLORS=("#244b83","#a94b18","#138478","#70518f")


def main(source,output):
    result=json.loads(source.read_text(encoding="utf-8"))
    font=Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"]=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"]=False
    fig,(a,b,c)=plt.subplots(3,1,figsize=(9.5,10.5),layout="constrained")

    def tune(ax,title,label):
        ax.set_title(title,loc="left",fontweight="bold",fontsize=14)
        ax.set_ylabel(label)
        ax.grid(axis="y",color="#e5e8ef")
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)

    x=np.arange(3)
    outcomes=result["native"]["outcomes"]
    for i,(arm,label,color) in enumerate(zip(ARMS,LABELS,COLORS)):
        series=[]
        for seed in SEEDS:
            row=outcomes[f"{seed}_{arm}"]
            series.append([row["left"]["first"].get("left",0),row["right"]["first"].get("right",0),
                           row["front"]["terminal"]["front"]])
        positions=x+(i-1.5)*.2
        a.bar(positions,np.mean(series,axis=0),width=.18,label=label,color=color,alpha=.8)
        for delta,values in zip((-.043,0,.043),series):
            a.scatter(positions+delta,values,s=16,color="#263242",zorder=3)
    a.set_xticks(x,("左句先到左","右句先到右","前句终态放对"))
    a.set_ylim(-.6,25)
    a.set_yticks(range(0,21,5))
    a.legend(loc="upper center",ncols=4,frameon=False)
    tune(a,"1000 步闭环　柱为三随机流均值，点为每个随机流","次数 / 20")

    for ax,key,title,label in (
        (b,lambda p:p["fixed"]["10"]["gain_H"],"同一固定输入下旧水平区别的变化","相对 R1 的水平增益"),
        (c,lambda p:p["front_demo"]["all"]["all"]["median"],"独立 Front 示范上的预测误差","归一化去噪 MSE"),
    ):
        for arm,arm_label,color in zip(ARMS,LABELS,COLORS):
            curves=[[key(result["probes"][f"{s}_{arm}_{step:06d}"]) for step in STEPS] for s in SEEDS]
            x=np.arange(len(STEPS))
            ax.plot(x,np.mean(curves,axis=0),"o-",color=color,label=arm_label)
            ax.fill_between(x,np.min(curves,axis=0),np.max(curves,axis=0),color=color,alpha=.10)
        tune(ax,title,label)
        ax.set_xticks(x,STEPS)
        ax.set_xlabel("续训步数")
        ax.legend(loc="best",ncols=2,frameon=False)
    fig.savefig(output,dpi=165,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("summary",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    main(a.summary,a.output)
