"""Present audited closed-loop outcomes; no simulator or model queries."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(audit_path):
    data=json.loads(audit_path.read_text(encoding="utf-8"))
    if data["status"] != "passed":
        raise ValueError("Passed independent behavioral audit required")
    return {"status":"audited_summary", "audit_sha256":hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            **{key:value for key,value in data.items() if key not in
               ("status","audit_source_sha256","source_files","episodes")}}


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig,axes=plt.subplots(1,2,figsize=(13,6.8))
    colors={"correct":"#33846d","new_right":"#c75c37","other_wrong":"#c7a35f",
            "unfinished":"#b9c0c6","ambiguous":"#555d65"}
    labels={"correct":"Requested target at end","new_right":"Wrong new-right target at end",
            "other_wrong":"Other wrong target at end","unfinished":"No target at end",
            "ambiguous":"Multiple targets at end"}
    for ax,role in zip(axes,("FT4","R0")):
        rows=[r for r in summary["per_condition"] if r["model"]==role]
        keys=["correct","new_right","other_wrong","unfinished","ambiguous"]
        values={key:[] for key in keys}
        for row in rows:
            counts=row["terminal_counts"]
            values["correct"].append(counts["correct"])
            values["new_right"].append(row["new_right_terminal"])
            values["other_wrong"].append(counts["wrong_goal"]-row["new_right_terminal"])
            values["unfinished"].append(counts["unfinished"])
            values["ambiguous"].append(counts["ambiguous"])
        left=[0]*len(rows)
        for key in keys:
            ax.barh(range(len(rows)),values[key],left=left,color=colors[key],height=.68,label=labels[key])
            left=[a+b for a,b in zip(left,values[key])]
        ax.scatter([r["correct_ever"] for r in rows],range(len(rows)),marker="D",s=25,
                   facecolors="none",edgecolors="#151515",label="Requested target ever reached",zorder=4)
        ax.set_yticks(range(len(rows)),[r["condition"]+" / "+r["requested_goal"] for r in rows])
        ax.invert_yaxis();ax.set_xticks(range(5));ax.set_xlim(-.1,4.1)
        ax.set(title=role,xlabel="Episodes (4 per request and condition)")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc="lower center",ncol=3,frameon=False,fontsize=8.5)
    fig.suptitle("Closed-loop test: terminal outcomes and ever success remain separate",fontsize=13)
    fig.text(.5,.105,"2 reused physical starts x 2 noise seeds. j9 whole M, first 120 / 400 steps. Development evidence only.",
             ha="center",fontsize=9,color="#444444")
    fig.tight_layout(rect=(0,.13,1,.94))
    fig.savefig(output,dpi=180);plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    target=args.output_dir/"summary.json"
    if target.exists():raise FileExistsError("Existing summary must be preserved")
    value=summarize(args.audit)
    target.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    plot(value,args.output_dir/"closedloop_outcomes.png")
    print(json.dumps({"status":value["status"],"counts":value["counts"]}))
