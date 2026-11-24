"""Show the matched mu-source comparison and every predeclared control."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(folder,output):
    s=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    def get(family,background="late",window="native",arm="forward"):
        result=[r for r in s["groups"] if r["job"]["family"] == family and r["job"]["background"] == background
                and r["job"]["window"] == window and (family == "native" or r["job"]["arm"] == arm)]
        assert len(result) == 1
        return result[0]
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    windows=(("delayed_100","Controls 81–180"),("full_400","Controls 1–400"))
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,(w,title) in zip(axes,windows,strict=True):
        groups=[get("native"),get("mu_only","parent",w),get("v","late",w),get("v","parent",w)]
        x=np.arange(4)
        for shift,metric,color,label in ((-.18,"joint_first","#1b7099","Both old goals: first arrival"),
                                          (.18,"joint_terminal","#b66d37","Both old goals: final placement")):
            values=[r[metric] for r in groups]
            ax.bar(x+shift,values,width=.36,color=color,label=label)
            for i,v in enumerate(values):
                ax.text(i+shift,v+.25,str(v),ha="center",fontsize=10)
        ax.set_xticks(x,["Late μ\nno V","Parent μ\nno V","Late μ\n+ V contrast","Parent μ\n+ V contrast"])
        ax.set_ylim(0,22)
        ax.set_yticks(range(0,21,5))
        ax.grid(axis="y",alpha=.18)
        ax.set_axisbelow(True)
        ax.set_title(title,loc="left")
        ax.set_ylabel("Paired initial states /20")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncols=2,frameon=False,bbox_to_anchor=(.5,.08))
    fig.suptitle("LF→Right: common-response source × fixed K25 V correction",x=.05,ha="left",fontsize=15)
    fig.text(.05,.018,"Executor F1000; donor P0; V anchor30, α=5.294; residual coefficient1. Same 20 reused resets.\n"
             "Native baseline is shared across panels. Parent μ is a two-policy diagnostic.",fontsize=9)
    fig.tight_layout(rect=(0,.19,1,.93))
    for ext in ("png","pdf"):
        path=output/("mu_source_factorial."+ext)
        fig.savefig(path,dpi=200,metadata={"Title":"LF common response by fixed V correction"})
        print(path)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,7.5))
    for ax,(w,title) in zip(axes,windows,strict=True):
        groups=[s["parent_policy"],get("native"),get("mu_only","parent",w),get("v","late",w),get("v","parent",w),
                get("v","late",w,"reverse"),get("v","parent",w,"reverse"),get("own","late",w),get("own","parent",w)]
        labels=["Full parent policy*","Native late F1000","Parent μ + rF","Late μ + rF + V","Parent μ + rF + V",
                "Late μ + rF − V","Parent μ + rF − V","Late μ + 2.959 rF","Parent μ + 2.959 rF"]
        values=np.asarray([[r["per_goal"][g][k] for g in "AB" for k in ("first_correct","terminal_correct")]
                           +[r["joint_first"],r["joint_terminal"]] for r in groups])
        ax.imshow(values,vmin=0,vmax=20,cmap="Blues",aspect="auto")
        for i in range(values.shape[0]):
            for j in range(6):
                v=values[i,j]
                ax.text(j,i,str(v),ha="center",va="center",color="white" if v >= 12 else "#18303f",fontsize=10)
        ax.set_xticks(range(6),["Left first","Left final","Front first","Front final","Joint first","Joint final"],rotation=25,ha="right")
        ax.set_yticks(range(9),labels,fontsize=10)
        ax.set_title(title,loc="left",fontsize=12)
        ax.tick_params(length=0)
        for y in (.5,2.5,6.5):
            ax.axhline(y,color="#546a78",lw=1.2)
    fig.suptitle("LF→Right: all mean, V-polarity and own-residual controls",x=.03,ha="left",fontsize=15)
    fig.text(.03,.025,"Every cell is a count /20. V: original eight-V correction, α=5.294; rF: actual-query late residual.\n"
             "600 full rollouts include 200 exact reference replays; *40 prior full-parent qualifications are contextual. No new-task preservation evaluation.",fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.93),w_pad=2)
    for ext in ("png","pdf"):
        path=output/("mu_swap_controls."+ext)
        fig.savefig(path,dpi=200,metadata={"Title":"LF common response complete control grid"})
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    plot(args.folder,args.output)
