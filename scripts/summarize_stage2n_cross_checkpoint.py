"""Describe audited step-3 readout effects; no further model queries."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from audit_stage2n_cross_checkpoint import read_rows, reference_metrics


def med(values):
    selected = [value for value in values if value is not None]
    return statistics.median(selected) if selected else None


def summarize(run, audit_path):
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "passed":
        raise ValueError("Passed independent audit required")
    baseline, patches = [read_rows(run/name) for name in ("baseline.jsonl", "patches.jsonl")]
    lookup = {(r["state_index"], r["noise_seed"], r["flow_index"], r["model"], r["goal"]):
              r["velocity"] for r in baseline}
    rows = []
    for row in patches:
        key = row["state_index"], row["noise_seed"], row["flow_index"]
        role, goal = row["model"], row["recipient"]
        native = lookup[*key, role, goal]
        healthy = lookup[*key, "R0", goal]
        other = lookup[*key, "R0", "front" if goal == "left" else "left"]
        measures = reference_metrics(native, row["velocity"], healthy, other)["primary"]
        rows.append({
            **{name: row[name] for name in ("state_index", "model", "recipient", "layer", "mode")},
            **measures, "cross_transfer": row["metrics"]["cross_same"]["primary"]["T"],
            "absolute_change": row["metrics"]["cross_same"]["primary"]["rms_delta"],
        })
    conditions, states = [], []
    for role in ("R0", "FT4"):
        for layer in (1, 3, 9):
            for mode in ("self", "cross_same", "cross_wrong", "random_delta"):
                records = [r for r in rows if (r["model"], r["layer"], r["mode"]) == (role, layer, mode)]
                numeric = list(records[0].keys())[5:]
                conditions.append({
                    "model": role, "layer": layer, "mode": mode, "queries": len(records),
                    **{field+"_median": med(r[field] for r in records) for field in numeric},
                })
                for state in range(4):
                    subset = [r for r in records if r["state_index"] == state]
                    states.append({
                        "state_index": state, "model": role, "layer": layer, "mode": mode,
                        "queries": len(subset),
                        "healthy_distance_gain_median": med(r["healthy_distance_gain"] for r in subset),
                        "cross_transfer_median": med(r["cross_transfer"] for r in subset),
                    })
    paired = []
    anchor_fields = ("state_index", "noise_seed", "flow_index", "model", "layer")
    paired_self = {
        tuple(r[name] for name in anchor_fields): r["metrics"]["primary"]
        for r in audit["paired_responses"] if r["mode"] == "self"
    }
    for role in ("R0", "FT4"):
        for layer in (1, 3, 9):
            for mode in ("self", "cross_same", "cross_wrong", "random_delta"):
                selected = [r for r in audit["paired_responses"]
                            if (r["model"], r["layer"], r["mode"]) == (role, layer, mode)]
                records = [r["metrics"]["primary"] for r in selected]
                reductions = {}
                for field in ("common_error_rms", "contrast_error_rms"):
                    values = []
                    for r in selected:
                        native_error = paired_self[tuple(r[name] for name in anchor_fields)][field]
                        if native_error > 0:
                            values.append(1-r["metrics"]["primary"][field]/native_error)
                    name = field.removesuffix("_rms")
                    reductions.update({name+"_reduction_median": med(values),
                                       name+"_improved_inputs": sum(v > 0 for v in values),
                                       name+"_relative_valid_inputs": len(values)})
                paired.append({
                    "model": role, "layer": layer, "mode": mode, "paired_inputs": len(records),
                    **{field+"_median": med(r[field] for r in records) for field in records[0]},
                    **reductions,
                })
    return {
        "status": "audited_summary", "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "counts": audit["counts"], "per_condition": conditions, "per_state": states,
        "paired_responses": paired, "random_norm_evidence": audit["random_norm_evidence"],
        "limits": [
            "Core step3 only: no conditional extra arms, P/C, closed loop or training.",
            "Distances and projections are in generated velocity space, not target success.",
            "Common and contrast are algebraic summaries of existing paired outputs, not new interventions.",
            "Relative reductions are paired per input against native self, then median; not ratios of medians.",
            "Correct/wrong natural donor replacements need not have identical perturbation amplitude.",
            "Four reused development states and one adaptation process; no population significance claim.",
        ],
    }


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    layers = (1, 3, 9)
    colors = {"cross_same": "#286091", "cross_wrong": "#c75c37", "random_delta": "#87949d"}
    labels = {"cross_same": "Same-condition old message",
              "cross_wrong": "Wrong-condition old message", "random_delta": "Matched random delta"}
    for ax, field, title in zip(
            axes, ("common_error_reduction_median", "contrast_error_reduction_median"),
            ("Reduction of shared response error", "Reduction of Left-Front contrast error")):
        for index, mode in enumerate(colors):
            points = [next(r[field] for r in summary["paired_responses"]
                           if (r["model"], r["layer"], r["mode"]) == ("FT4", layer, mode))
                      for layer in layers]
            ax.bar([x+(index-1)*0.24 for x in range(3)], [100*p for p in points], width=0.22,
                   color=colors[mode], label=labels[mode])
        ax.axhline(0, color="#333333", linestyle="--", linewidth=1, label="No change vs native FT4")
        ax.set_xticks(range(3), ["j1: control", "j3: candidate", "j9: candidate"])
        ax.set(title=title, ylabel="Median paired error reduction (%)", xlabel="Zero-based expert layer")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Old messages compensate shared output shift; conditional recovery remains limited", fontsize=12)
    fig.text(0.5, 0.015, "Positive = closer to R0. 36 paired inputs / 4 starts. Open-loop; no target-success claim.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.plot and args.plot.exists()):
        raise FileExistsError("Summary output already exists")
    result = summarize(args.run, args.audit)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if args.plot:
        plot(result, args.plot)
    print(json.dumps({"status": result["status"], "counts": result["counts"]}))
