"""Standard-library audit of fixed P/C and true-head computational experiments."""

import argparse
import itertools
import json
from pathlib import Path
import statistics

from audit_stage2n_cross_checkpoint import (
    audit_random_norms, indexed, number, paired_response_metrics, prefix_gate,
    read_json, read_rows, reference_metrics, require, sha256, vector,
)

GOALS = ("left", "front")
FACTOR = ("self_pc", "old_p_same", "old_c_same", "old_pc_same", "old_p_wrong",
          "old_c_wrong", "old_pc_wrong", "old_m_same", "old_m_wrong", "random_pc")
HEAD = ("head_same", "head_wrong", "head_random")
ANCHORS = tuple(itertools.product(range(4), (2026102801, 2026102802, 2026102803), (0, 4, 8)))


def expected_cells():
    keys = {(i, s, f, g, layer, mode, None)
            for i, s, f in ANCHORS for g in GOALS for layer in (1, 3, 9) for mode in FACTOR}
    keys.update((i, s, f, g, layer, mode, h) for i, s, f in ANCHORS if s == 2026102801
                for g in GOALS for layer in (1, 3, 9) for h in range(15) for mode in HEAD)
    return keys


def validate_head_norm_support(record):
    """A single-head random control must never change other head coordinates."""
    if record["head"] is None:
        return
    head = record["head"]
    require(type(head) is int and 0 <= head < 15, "Invalid head identity")
    fields = ["effective_block_l2"]
    if record["mode"] == "head_random":
        fields += ["target_block_l2", "desired_block_l2"]
    for field in fields:
        require(len(record[field]) == 50 and all(len(r) == 15 for r in record[field]),
                "Expected actual 15-head norm coordinates")
        for row in record[field]:
            require(all(number(x) == 0 for h, x in enumerate(row) if h != head),
                    "Single-head norm leaked to untouched head")


def relative_reduction(after, before):
    return 1-after/before if before > 1e-6 else None


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def audit(run, prior_baseline, prior_patches):
    manifest = read_json(run/"manifest.json")
    require(manifest == read_json(run/"COMPLETE.json") and manifest["status"] == "complete", "Incomplete run")
    require(manifest["script_sha256"] == sha256(Path(__file__).with_name("stage2o_pc_heads.py")),
            "Worker source differs from audited version")
    require(manifest["prior_baseline_sha256"] == sha256(prior_baseline), "Prior baseline hash changed")
    require(manifest["configuration"]["prior_patch_sha256"] == sha256(prior_patches), "Prior patch hash changed")
    files = {}
    for name, expected in manifest["files"].items():
        actual = {"bytes": (run/name).stat().st_size, "sha256": sha256(run/name)}
        require(actual == expected, f"Evidence hash mismatch: {name}")
        files[name] = actual
    require(manifest["actual_heads"] == 15 and manifest["actual_head_dim"] > 0,
            "Actual head architecture changed")
    require(manifest["no_followup_launched"], "Unexpected followup")
    require(max(manifest["attention_readout_reconstruction_max_abs"].values()) <= 1e-5,
            "Native P/C reconstruction failed")
    prefix = prefix_gate(manifest)
    gate = read_json(run/"engineering.json")
    require(gate["passed"], "Engineering gate failed")
    require(gate["script_sha256"] == manifest["script_sha256"]
            and gate["config_sha256"] == manifest["config_sha256"]
            and gate["sources"] == manifest["sources"], "Engineering provenance changed")
    indexed([r for r in gate["checks"] if r["kind"] == "record"], ("model", "goal"),
            {(role, goal) for role in ("R0", "FT4") for goal in GOALS})
    indexed([r for r in gate["checks"] if r["kind"] == "self_pc"], ("goal", "layer"),
            {(goal, layer) for goal in GOALS for layer in (1, 3, 9)})
    indexed([r for r in gate["checks"] if r["kind"] == "head_self"], ("goal", "layer", "head"),
            {(goal, layer, h) for goal in GOALS for layer in (1, 3, 9) for h in range(15)})
    require(len(gate["checks"]) == 100, "Engineering checks missing or unexpected")
    for r in gate["checks"]:
        require(number(r["max_abs"]) <= 1e-5, "Recording/self identity failed")
    baseline = indexed(read_rows(run/"baseline.jsonl"),
                       ("state_index", "noise_seed", "flow_index", "model", "goal"),
                       {(*a, role, goal) for a in ANCHORS for role in ("R0", "FT4") for goal in GOALS})
    patches = indexed(read_rows(run/"patches.jsonl"),
                      ("state_index", "noise_seed", "flow_index", "recipient", "layer", "mode", "head"),
                      expected_cells())
    norms = read_rows(run/"norms.jsonl.gz")
    norm_expected = {k for k in expected_cells() if k[5] in ("old_pc_same", "random_pc", "head_same", "head_random")}
    norm_map = indexed(norms, ("state_index", "noise_seed", "flow_index", "recipient", "layer", "mode", "head"), norm_expected)
    require(len(norm_map) == 2592, "Norm evidence count differs")
    old_baseline = indexed([r for r in read_rows(prior_baseline) if r["goal"] in GOALS],
                           ("state_index", "noise_seed", "flow_index", "model", "goal"), set(baseline))
    old_patch = {(r["anchor_id"], r["recipient"], r["layer"], r["mode"]):r
                 for r in read_rows(prior_patches) if r["model"] == "FT4"}
    baseline_error, self_error, m_error, witness_error = 0., 0., 0., 0.
    for key, row in baseline.items():
        prior = old_baseline[key]
        for field in ("anchor_id", "x_sha256", "state_sha256"):
            require(row[field] == prior[field], "Fixed baseline input identity changed")
        baseline_error = max(baseline_error, max(abs(a-b) for a, b in zip(
            vector(row["velocity"], 50), vector(prior["velocity"], 50))))
    for key, row in patches.items():
        i, seed, flow, goal, layer, mode, head = key
        base = baseline[i, seed, flow, "FT4", goal]
        require(row["model"] == "FT4", "Unexpected recipient model")
        for field in ("anchor_id", "x_sha256", "state_sha256"):
            require(row[field] == base[field], "Patch changed input identity")
        if mode == "self_pc":
            self_error = max(self_error, max(abs(a-b) for a, b in zip(
                vector(row["velocity"], 50), vector(base["velocity"], 50))))
        if mode.startswith("old_m"):
            previous = old_patch[row["anchor_id"], goal, layer, "cross_wrong" if mode.endswith("wrong") else "cross_same"]
            m_error = max(m_error, max(abs(a-b) for a, b in zip(
                vector(row["velocity"], 50), vector(previous["velocity"], 50))))
        else:
            witness_error = max(witness_error, number(row["actual_input_witness_max_abs"]))
        if head is not None:
            require(number(row["untouched_heads_max_abs"]) == 0, "Untouched actual heads changed")
    require(max(baseline_error, self_error, m_error, witness_error) <= 1e-5,
            "Baseline, self, post-M reproduction or witness identity failed")
    norm_stats = []
    for key, row in norm_map.items():
        validate_head_norm_support(row)
        if row["mode"] in ("random_pc", "head_random"):
            correct_mode = "old_pc_same" if row["head"] is None else "head_same"
            correct_key = (*key[:5], correct_mode, key[-1])
            norm_stats.append(audit_random_norms(row, norm_map[correct_key]))
    head_rows = indexed(read_rows(run/"heads.jsonl"),
                        ("state_index", "noise_seed", "flow_index", "model", "layer", "head"),
                        {(*a, role, layer, head) for a in ANCHORS for role in ("R0", "FT4")
                         for layer in (1, 3, 9) for head in range(15)})
    for row in head_rows.values():
        require(row["head_dim"] == manifest["actual_head_dim"], "Head dimension changed")
        require(number(row["P_LF_rms"]) >= 0 and number(row["U_LF_rms"]) >= 0, "Invalid descriptive head RMS")
        for masses in row["source_mass"].values():
            values = [number(v) for v in masses.values()]
            require(min(values) >= 0 and abs(sum(values)-1) <= 2**-8+1e-6, "Source mass exceeds BF16 rounding bound")
    paired = []
    for i, seed, flow in ANCHORS:
        healthy = [baseline[i, seed, flow, "R0", g]["velocity"] for g in GOALS]
        native = [baseline[i, seed, flow, "FT4", g]["velocity"] for g in GOALS]
        original = paired_response_metrics(*native, *healthy)
        for layer in (1, 3, 9):
            jobs = [(m, None) for m in FACTOR]
            if seed == 2026102801:
                jobs += [(m, h) for h in range(15) for m in HEAD]
            for mode, head in jobs:
                output = [patches[i, seed, flow, g, layer, mode, head]["velocity"] for g in GOALS]
                measures = paired_response_metrics(*output, *healthy)
                paired.append({"state_index": i, "noise_seed": seed, "flow_index": flow,
                               "layer": layer, "mode": mode, "head": head, "metrics": measures,
                               "relative_primary": {
                                   field:relative_reduction(measures["primary"][field], original["primary"][field])
                                   for field in ("common_error_rms", "contrast_error_rms")}})
    per_condition = []
    for layer in (1, 3, 9):
        for mode, head in [(m, None) for m in FACTOR]+[(m, h) for h in range(15) for m in HEAD]:
            group = [r for r in paired if (r["layer"], r["mode"], r["head"]) == (layer, mode, head)]
            gains = {field:[r["relative_primary"][field] for r in group]
                     for field in ("common_error_rms", "contrast_error_rms")}
            per_condition.append({"layer": layer, "mode": mode, "head": head, "paired_inputs": len(group),
                                  **{field+"_reduction_median":median(values) for field, values in gains.items()},
                                  **{field+"_improved_inputs":sum(v is not None and v > 0 for v in values)
                                     for field, values in gains.items()},
                                  "contrast_projection_median":median(r["metrics"]["primary"]["contrast_projection"] for r in group)})
    return {"status":"passed", "audit_source_sha256":sha256(__file__), "source_files":files,
            "counts":{"physical_states":4, "anchors":36, "baseline":len(baseline), "patches":len(patches),
                      "norm_records":len(norms), "head_observations":len(head_rows)},
            "prefix_validation":prefix, "baseline_reproduction_max_abs":baseline_error,
            "full_message_reproduction_max_abs":m_error, "self_pc_max_abs":self_error,
            "actual_input_witness_max_abs":witness_error,
            "random_norm_max_relative_actual_difference":max(r["max_relative_actual_norm_difference"] for r in norm_stats),
            "per_condition":per_condition, "paired_responses":paired,
            "limits":["Open-loop velocity evidence only; no physical target success.",
                      "Four reused physical starts; head screen uses one noise seed, twelve inputs.",
                      "P/C natural donor combinations are not all amplitude-matched.",
                      "True indexed heads share checkpoint lineage, not a proof of semantic alignment.",
                      "Single-head effects need not add up to full-layer effects.",
                      "Recorded hidden-tensor checks cannot be independently rerun from velocity files."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prior-baseline", type=Path, required=True)
    parser.add_argument("--prior-patches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Never overwrite existing audit")
    result = audit(args.run, args.prior_baseline, args.prior_patches)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"counts":result["counts"]}))
