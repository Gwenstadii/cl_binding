"""Summarize audited open-loop measurements; no model or simulator imports."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def mean(values):
    return statistics.mean(values)


def contrast(a, b, start=0, end=7, horizon=10):
    values = [(a[i][j] - b[i][j]) ** 2 for i in range(horizon) for j in range(start, end)]
    return math.sqrt(math.fsum(values) / len(values))


def summarize(run, audit):
    verified = json.loads(audit.read_text(encoding="utf-8"))
    if verified["status"] != "passed":
        raise ValueError("An independently passed audit is required")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    launcher_path = run.parent / "launcher.json"
    launcher = json.loads(launcher_path.read_text(encoding="utf-8")) if launcher_path.is_file() else {}
    engineer_seconds = next((r["seconds"] for r in launcher.get("phases", [])
                             if r["name"] == "engineering"), None)
    baseline, paired, patches = [read_rows(run / name)
                                 for name in ("baseline.jsonl", "paired.jsonl", "patches.jsonl")]
    lookup = {(r["state_index"], r["noise_seed"], r["flow_index"], r["model"], r["goal"]): r
              for r in baseline}
    roles, flows = manifest["roles"], manifest["flow_indices"]
    anchors = list(itertools.product(manifest["indices"], manifest["seeds"], flows))
    contrasts = []
    for role in roles:
        for ga, gb in (("left", "front"), ("left", "right"), ("front", "right")):
            for i, seed, flow in anchors:
                a, b = [lookup[i, seed, flow, role, goal]["velocity"] for goal in (ga, gb)]
                contrasts.append({
                    "model": role, "pair": [ga, gb], "state_index": i,
                    "noise_seed": seed, "flow_index": flow,
                    "primary": contrast(a, b), "motion": contrast(a, b, end=6),
                    "gripper": contrast(a, b, start=6), "full50": contrast(a, b, horizon=50),
                })
    native = []
    for role in roles:
        for flow in [None, *flows]:
            rows = [r for r in contrasts if r["model"] == role and r["pair"] == ["left", "front"]
                    and (flow is None or r["flow_index"] == flow)]
            native.append({"model": role, "flow_index": flow, "queries": len(rows),
                           **{key + "_mean": mean(r[key] for r in rows)
                              for key in ("primary", "motion", "gripper", "full50")}})
    layers = sorted(manifest["ca_layers"]["R0"])
    messages, swaps = [], []
    for layer in layers:
        for role in roles:
            rows = [r for r in paired if r["model"] == role and r["layer"] == layer
                    and r["goal_pair"] == ["left", "front"]]
            messages.append({
                "model": role, "layer": layer, "queries": len(rows),
                **{name + "_difference_rms_mean": mean(r["features_primary"][name]["rms_diff"]
                                                      for r in rows)
                   for name in ("P", "PC", "M", "attention_residual", "post_mlp")},
            })
            for direction in (None, "left", "front"):
                records = [r for r in patches if r["model"] == role and r["layer"] == layer
                           and (direction is None or r["recipient"] == direction)]
                for group in ("primary", "motion", "gripper", "full50"):
                    vals = [r["metrics"][group] for r in records]
                    valid = [r["T"] for r in vals if r["valid_T"]]
                    swaps.append({
                        "model": role, "layer": layer, "recipient": direction,
                        "group": group, "queries": len(records), "valid_T": len(valid),
                        "median_T": statistics.median(valid) if valid else None,
                        "mean_T": mean(valid) if valid else None,
                        "min_T": min(valid) if valid else None,
                        "max_T": max(valid) if valid else None,
                        "median_absolute_delta_rms": statistics.median(r["rms_delta"] for r in vals),
                        "mean_absolute_delta_rms": mean(r["rms_delta"] for r in vals),
                    })
    prefix = [r for rows in manifest["prefix_audit"] for r in rows]
    return {
        "status": "audited_summary", "audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
        "counts": manifest["counts"], "physical_states": len(manifest["indices"]),
        "engineer_seconds": engineer_seconds, "analysis_seconds": manifest["elapsed_seconds"],
        "models": manifest["model_sha256"], "ca_layers": layers,
        "identity_checks": json.loads((run / "engineering.json").read_text())["checks"],
        "prefix": {
            "valid_image_text_max_abs": max(r["active"]["max_abs"] for r in prefix
                                             if r["group"] in ("image", "text")),
            "masked_text_max_abs": max(r["masked"]["max_abs"] for r in prefix if r["group"] == "text"),
            "state_max_abs": max(r["active"]["max_abs"] for r in prefix if r["group"] == "state"),
        },
        "native_left_front": native, "per_layer_messages": messages,
        "per_layer_swaps": swaps,
        "candidates": json.loads((run / "candidates.json").read_text()),
        "limits": [
            "Open-loop inference on four development starts; no behavioral target execution.",
            "T is normalized to the same model's other native response, not healthy target direction.",
            "Message statistics are recorded summaries, not independently recomputed hidden tensors.",
            "Raw C includes masked prefix positions and is intentionally excluded from content-retention claims.",
            "Natural message changes and downstream response changes are not yet isolated.",
        ],
    }


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"R0": "#286091", "FT4": "#c75c37"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    layers = summary["ca_layers"]
    for role in ("R0", "FT4"):
        native = [r for r in summary["native_left_front"] if r["model"] == role
                  and r["flow_index"] is not None]
        axes[0, 0].plot([r["flow_index"] for r in native],
                        [r["primary_mean"] for r in native],
                        marker="o", color=colors[role], label=role)
        msgs = [r for r in summary["per_layer_messages"] if r["model"] == role]
        axes[0, 1].plot(layers, [r["M_difference_rms_mean"] for r in msgs],
                        marker="o", color=colors[role], label=role)
        swp = [r for r in summary["per_layer_swaps"] if r["model"] == role
               and r["recipient"] is None and r["group"] == "primary"]
        axes[1, 0].plot(layers, [r["median_absolute_delta_rms"] for r in swp],
                        marker="o", color=colors[role], label=role)
        axes[1, 1].plot(layers, [r["median_T"] for r in swp],
                        marker="o", color=colors[role], label=role)
    specifications = [
        ("Native Left-Front velocity contrast", "Flow integration step", "Mean RMS difference"),
        ("Cross-attention message contrast", "Expert layer (zero-based)", "Mean RMS difference"),
        ("Effect of swapping Left/Front messages", "Expert layer (zero-based)", "Median absolute velocity change (RMS)"),
        ("Transfer toward the other native response", "Expert layer (zero-based)", "Median T (not success rate)"),
    ]
    for ax, (title, xlabel, ylabel) in zip(axes.flat, specifications):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    axes[0, 0].set_xticks([0, 4, 8], ["0 (t=1.0)", "4 (t=0.6)", "8 (t=0.2)"])
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set_xticks(layers)
    axes[1, 1].axhline(0, color="#888888", linewidth=0.7)
    figure.suptitle("Continual adaptation: condition messages remain influential, but their effects change",
                    fontsize=13)
    figure.text(0.5, 0.015,
                "4 development starts; first 10 actions x 7 active dimensions. Open-loop; no target-success claim.",
                ha="center", fontsize=9, color="#555555")
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.plot and args.plot.exists()):
        raise FileExistsError("Summary outputs already exist")
    result = summarize(args.run, args.audit)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                           encoding="utf-8")
    if args.plot:
        plot(result, args.plot)
    print(json.dumps({"status": result["status"], "counts": result["counts"]}))
