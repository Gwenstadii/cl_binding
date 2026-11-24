"""Summarize common and instruction-specific Flow Matching updates."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np

from k10_fixed_input_projection import FLOW_INDICES, ROLES


def norm(x):
    return float(np.linalg.norm(x))


def cosine(x, y):
    return float(x @ y / (norm(x) * norm(y)))


def summarize(fields, k9):
    data = json.loads(fields.read_bytes())
    ref = json.loads(k9.read_bytes())
    assert data["model_sha256"] == ref["metadata"]["model_sha256"]
    source = Path(__file__).with_name("k10_fixed_input_projection.py")
    assert data["code_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    anchors = data["anchors"]
    assert len(anchors) == 60 and data["native_readout_identity_checks"] == 360
    assert {(r["init"], r["anchor_role"], r["flow"]) for r in anchors} == {
        (i, role, f) for i in range(10) for role in ROLES for f in FLOW_INDICES
    }
    metrics = []
    for row in anchors:
        for axes, dimensions in (("translation", 3), ("motion", 6), ("effective", 7)):
            values = {}
            for stack in ROLES:
                for goal in ("A", "B", "N"):
                    for output in ROLES:
                        a = np.asarray(row["values"][stack + goal]["velocity"][output], dtype=float)
                        assert a.shape == (10, 7) and np.isfinite(a).all()
                        # Euler integration proceeds toward smaller t.
                        values[stack, output, goal] = -a[:, :dimensions].ravel()
            def mean_and_difference(stack, output):
                a, b = (values[stack, output, g] for g in ("A", "B"))
                return (a + b) / 2, (a - b) / 2

            mu0, d0 = mean_and_difference("R", "R")
            right = values["F", "F", "N"] - mu0
            assert min(norm(d0), norm(right)) > 1e-9
            for stack in ROLES:
                for output in ROLES:
                    mu, d = mean_and_difference(stack, output)
                    drift = mu - mu0
                    metrics.append({
                        **{k: row[k] for k in ("init", "anchor_role", "flow")},
                        "axes": axes, "stack": stack, "output": output,
                        "d_norm_ratio": norm(d) / norm(d0),
                        "d_cosine": cosine(d, d0) if norm(d) else None,
                        "d_signed_projection": float(d @ d0 / (d0 @ d0)),
                        "mean_drift_over_r0_d": norm(drift) / norm(d0),
                        "mean_drift_to_ft4_right": float(drift @ right / (right @ right)),
                        "mean_distance_to_ft4_right_over_r0": norm(drift - right) / norm(right),
                    })
    keys = ("d_norm_ratio", "d_cosine", "d_signed_projection", "mean_drift_over_r0_d",
            "mean_drift_to_ft4_right", "mean_distance_to_ft4_right_over_r0")
    table = {}
    for axes in ("translation", "motion", "effective"):
        table[axes] = {}
        for stack in ROLES:
            for output in ROLES:
                rows = [r for r in metrics if (r["axes"], r["stack"], r["output"]) == (axes, stack, output)]
                table[axes][stack + output] = {
                    "median": {k: median(r[k] for r in rows if r[k] is not None) for k in keys},
                    "anchors_positive_d_projection": sum(r["d_signed_projection"] > 0 for r in rows),
                    "by_draft_source": {
                        m: {k: median(r[k] for r in rows if r["anchor_role"] == m and r[k] is not None)
                            for k in keys} for m in ROLES
                    },
                }
    return {"snapshots": 10, "anchors": len(anchors), "table": table, "per_anchor": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--k9", type=Path, required=True)
    args = parser.parse_args()
    out = summarize(args.fields, args.k9)
    args.fields.with_name("summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({axis: {cell: v["median"] for cell, v in results.items()}
                      for axis, results in out["table"].items()}))
