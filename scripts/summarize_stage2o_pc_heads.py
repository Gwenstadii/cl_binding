"""Summarize an independently audited run, without further model queries."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def med(values):
    return statistics.median(values)


def summarize(audit_path):
    data = json.loads(audit_path.read_text(encoding="utf-8"))
    if data["status"] != "passed":
        raise ValueError("Passed independent audit required")
    matched = []
    for row in data["per_condition"]:
        if row["head"] is not None:
            continue
        records = [r for r in data["paired_responses"]
                   if r["noise_seed"] == 2026102801
                   and (r["layer"], r["mode"], r["head"]) == (row["layer"], row["mode"], None)]
        matched.append({"layer":row["layer"],"mode":row["mode"],"paired_inputs":len(records),
                        **{field+"_reduction_median":med(r["relative_primary"][field] for r in records)
                           for field in ("common_error_rms","contrast_error_rms")}})
    return {"status":"audited_summary", "audit_sha256":hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "counts":data["counts"],
            "factor_conditions":[r for r in data["per_condition"] if r["head"] is None],
            "factor_on_head_subset":matched,
            "head_conditions":[r for r in data["per_condition"] if r["head"] is not None],
            "limits":data["limits"]}


def plot(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2,3,figsize=(14,7))
    colors=("#286091","#c75c37")
    for column, layer in enumerate((1,3,9)):
        for row, field in enumerate(("common_error_rms_reduction_median","contrast_error_rms_reduction_median")):
            ax=axes[row,column]
            for offset, suffix, color, label in zip((-.18,.18),("same","wrong"),colors,("Correct source","Wrong source")):
                values=[next(r[field] for r in summary["factor_conditions"]
                             if r["layer"]==layer and r["mode"]==f"old_{part}_{suffix}")*100
                        for part in ("p","c","pc","m")]
                ax.bar([i+offset for i in range(4)],values,width=.32,color=color,label=label)
            random=next(r[field] for r in summary["factor_conditions"] if r["layer"]==layer and r["mode"]=="random_pc")*100
            ax.bar([4],[random],width=.5,color="#87949d",label="Random U delta")
            ax.axhline(0,color="#333333",lw=.8)
            ax.set_xticks(range(5),["P","C","PC","M","Random"])
            ax.set_title(f"j{layer}: {'shared' if row==0 else 'Left-Front contrast'} error")
            ax.set_ylabel("Paired error reduction (%)")
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y",alpha=.15); ax.set_axisbelow(True)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle("Layer-internal factorization: positive means closer to R0",fontsize=14)
    fig.text(.5,.01,"36 fixed inputs / 4 reused starts. Open-loop velocity evidence; no physical success claim.",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(output_dir/"pc_factorization.png",dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,field,title in zip(axes,("common_error_rms_reduction_median","contrast_error_rms_reduction_median"),
                             ("Shared response error","Left-Front contrast error")):
        matrix=[];labels=[]
        for layer in (1,3,9):
            for mode in ("head_same","head_wrong","head_random"):
                matrix.append([next(r[field] for r in summary["head_conditions"]
                                    if (r["layer"],r["mode"],r["head"])==(layer,mode,h))*100 for h in range(15)])
                labels.append(f"j{layer}: {mode.removeprefix('head_')}")
        scale=max(abs(v) for row in matrix for v in row) or 1
        im=ax.imshow(matrix,cmap="RdBu",vmin=-scale,vmax=scale,aspect="auto")
        ax.set_xticks(range(15),range(15));ax.set_yticks(range(9),labels)
        ax.set(title=title,xlabel="Actual pre-W_O head index")
        fig.colorbar(im,ax=ax,label="Median paired error reduction (%)",fraction=.05,pad=.03)
    fig.suptitle("All registered single-head results; no head selected for follow-up",fontsize=13)
    fig.text(.5,.01,"12 fixed inputs / 4 reused starts, one noise seed. Compare layers to the matched 12-input factor table.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(output_dir/"true_head_screen.png",dpi=180);plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output=args.output_dir/"summary.json"
    if output.exists():raise FileExistsError("Existing summary must be retained")
    summary=summarize(args.audit)
    output.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    plot(summary,args.output_dir)
    print(json.dumps({"status":summary["status"],"counts":summary["counts"]}))
