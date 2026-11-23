"""Plot donor competence separately from the complete restricted-repair grid."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import anchor_protocol as p

LABELS = {"lr_front_1000":"Left + Right → Front", "lf_right_1000":"Left + Front → Right"}


def plot(folder,output):
    data = json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes = plt.subplots(2,3,figsize=(13,7),sharey=True)
    for row,line in enumerate(p.LINES):
        item = data["lines"][line]
        for col,g in enumerate("ABN"):
            ax = axes[row,col]
            groups = [item["policies"]["0"],item["policies"]["30"],item["late_native"]]
            first = [s["per_goal"][g]["first_correct"] for s in groups]
            final = [s["per_goal"][g]["terminal_correct"] for s in groups]
            x = np.arange(3)
            ax.bar(x-.18,first,width=.36,color="#1b7099",label="First arrival")
            ax.bar(x+.18,final,width=.36,color="#b66d37",label="Final placement")
            for shift,values in ((-.18,first),(.18,final)):
                for i,v in enumerate(values):
                    ax.text(i+shift,v+.3,str(v),ha="center",fontsize=9)
            ax.set_xticks(x,["Parent 0","Step 30","Late 1000"])
            ax.set_ylim(0,22)
            ax.set_yticks(range(0,21,5))
            ax.grid(axis="y",alpha=.18)
            ax.set_axisbelow(True)
            kind = "new" if g == "N" else "old"
            ax.set_title(f"{LABELS[line]}\n{kind} {p.GOALS[line][g].title()}",loc="left",fontsize=11)
            if col == 0:
                ax.set_ylabel("Correct episodes / 20 starts")
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncols=2,frameon=False,bbox_to_anchor=(.5,.04))
    fig.suptitle("K25 anchor audit: complete policy competence",x=.06,ha="left",fontsize=16)
    fig.text(.06,.012,"Parent and step 30: newly executed. Late 1000: frozen previous references. Same 20 starts; full 400-step rollouts.",fontsize=9)
    fig.tight_layout(rect=(.01,.105,1,.94),h_pad=2)
    for ext in ("png","pdf"):
        path = output/("anchor_policy_competence."+ext)
        fig.savefig(path,dpi=200,metadata={"Title":"K25 anchor full-policy qualification"})
        print(path)
    plt.close(fig)

    fig,axes = plt.subplots(1,2,figsize=(15,12))
    images = []
    for ax,line in zip(axes,p.LINES,strict=True):
        item = data["lines"][line]
        groups = [item["late_native"],item["policies"]["0"],item["policies"]["30"],item["v_weights"]["0"],item["v_weights"]["30"]]
        rows = ["Native late 1000","Complete parent 0","Complete step 30","Eight V ← parent 0","Eight V ← step 30"]
        for a in p.ANCHORS:
            for gain in p.GAINS:
                for window in p.WINDOWS:
                    for arm in p.ARMS:
                        low,high = p.temporal.WINDOWS[window]
                        imported = " *" if p.is_reused(a,gain) else ""
                        rows.append(f"S{a} α{gain:.3g} {low+1}–{high} {'+' if arm == 'forward' else '−'}{imported}")
                        groups.append(item["delta"][str(a)][str(gain)][window][arm])
        values = np.asarray([[s["per_goal"][g][metric] for g in "AB" for metric in ("first_correct","terminal_correct")]+[s["joint_first"],s["joint_terminal"]] for s in groups])
        im = ax.imshow(values,vmin=0,vmax=20,cmap="Blues",aspect="auto")
        images.append(im)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i,j]
                ax.text(j,i,str(v),ha="center",va="center",fontsize=9,color="white" if v >= 12 else "#18303f")
        a,b = (p.GOALS[line][g].title() for g in "AB")
        ax.set_xticks(range(6),[a+" first",a+" final",b+" first",b+" final","Joint first","Joint final"],rotation=30,ha="right")
        ax.set_yticks(range(len(rows)),rows,fontsize=9)
        ax.set_title(LABELS[line],loc="left",fontsize=13,pad=12)
        for boundary in (2.5,4.5,12.5):
            ax.axhline(boundary,color="#546a78",lw=1.4)
        ax.tick_params(length=0)
    fig.suptitle("K25 anchor audit: full donors, V-only weights, and μ-preserving delta repair",x=.04,ha="left",fontsize=15)
    fig.text(.04,.025,"All cells are counts /20. S0/S30: source of eight historical V tensors. +/−: injection polarity. *: 320 imported K25-V controls.\n"
             "Full-donor competence and eight-V compatibility are separate tests. Two fixed gains; no winning dose/window selected on this set.",fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.95),w_pad=3)
    for ext in ("png","pdf"):
        path = output/("anchor_repair_grid."+ext)
        fig.savefig(path,dpi=200,metadata={"Title":"K25 anchor factorial repair audit"})
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    plot(args.folder,args.output)
