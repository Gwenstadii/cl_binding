"""Summarize K24 fork measurements without training dependencies."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median

TOLERANCE = .0001


def prediction(q, key):
    return any(b + d < f - TOLERANCE for b, d, f in zip(
        q["before"][4:], q[key][4:], q["floor"], strict=True))


def summarize(rows, select):
    control, held, losses, fractions, norms, ms = [], [], [], [], [], []
    for r in rows:
        arm = r.get("accepted", {}).get(select, select)
        update = r["updates"][arm]
        fractions.append(arm)
        control.append(update["queries"][str(r["controlled"])])
        held.extend(update["queries"][str(i)] for i in r["held"])
        losses.append(update["front_loss_same_batch"] - r["front_before"])
        norms.append(update["norm"])
        ms.append(control[-1]["actual_delta"][-1])
    error = [mean(abs(q["predicted_delta"][i] - q["actual_delta"][i]) for q in control + held) * 1000
             for i in range(8)]
    return {"accepted_arms": dict(Counter(fractions)), "nonzero_steps": sum(x > 0 for x in norms),
            "controlled_breaches": {"measured": sum(prediction(q, "actual_delta") for q in control),
                                    "predicted": sum(prediction(q, "predicted_delta") for q in control)},
            "held_breaches": {"measured": sum(prediction(q, "actual_delta") for q in held),
                              "predicted": sum(prediction(q, "predicted_delta") for q in held)},
            "mae_mm": dict(zip(rows[0]["metric_order"], error, strict=True)),
            "median_measured_right_margin_delta_mm": median(ms) * 1000,
            "median_front_loss_change": median(losses)}


def run(directory):
    outputs = {}
    for path in sorted(directory.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        marker = json.loads(path.with_name(path.stem + "_COMPLETE.json").read_text(encoding="utf-8"))
        assert [r["offset"] for r in rows] == marker["offsets"] and len(rows) == 8
        assert all(r["anchor"] == marker["anchor"] and len(r["held"]) == 2 and r["controlled"] not in r["held"] for r in rows)
        arms = ("actual", "projected", "scaled", *rows[0].get("accepted", {}))
        outputs[path.stem] = {"batches": len(rows), "active_projection": sum(any(r["dual"]) for r in rows),
                              "arms": {a: summarize(rows, a) for a in arms}}
    summary = json.dumps(outputs, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (directory / "summary.json").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("evidence", type=Path)
    run(p.parse_args().evidence)
