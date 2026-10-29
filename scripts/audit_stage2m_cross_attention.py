"""Independent CPU audit: query matrices and transfer from effective velocities.

No worker/model/NumPy imports. This checks evidence, not target correctness.
"""

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ROLES = ("R0", "FT4")
GOALS = ("left", "front", "right")
GROUPS = {"primary": (10, 0, 7), "motion": (10, 0, 6),
          "gripper": (10, 6, 7), "full50": (50, 0, 7)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_nonfinite(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_nonfinite)


def read_jsonl(path):
    return [json.loads(line, parse_constant=reject_nonfinite)
            for line in Path(path).read_text(encoding="utf-8").splitlines()]


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "Nonfinite/invalid number")
    return float(value)


def finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)
    elif type(value) in (int, float):
        finite(value)


def active_vector(matrix, horizon=10, start=0, end=7):
    require(isinstance(matrix, list) and len(matrix) == 50, "Velocity must have 50 rows")
    require(0 < horizon <= 50 and 0 <= start < end <= 7, "Invalid action slice")
    require(all(isinstance(row, list) and len(row) >= 7 for row in matrix), "Missing action dimensions")
    return [finite(matrix[i][j]) for i in range(horizon) for j in range(start, end)]


def rms(values):
    return math.sqrt(math.fsum(x * x for x in values) / len(values))


def transfer_metrics(base, donor, patched, repeat_rms=0.0):
    repeat_rms = finite(repeat_rms)
    require(repeat_rms >= 0, "Negative repeat RMS")
    epsilon = max(10 * repeat_rms, 1e-6)
    result = {}
    for name, selection in GROUPS.items():
        before, target, after = [active_vector(v, *selection) for v in (base, donor, patched)]
        direction = [b - a for a, b in zip(before, target)]
        change = [b - a for a, b in zip(before, after)]
        native_rms, delta_rms = rms(direction), rms(change)
        valid = native_rms > epsilon
        coefficient = (math.fsum(a * b for a, b in zip(change, direction))
                       / math.fsum(x * x for x in direction)) if valid else None
        result[name] = {"rms_delta": delta_rms, "rms_native_contrast": native_rms,
            "T": coefficient, "epsilon": epsilon, "valid_T": valid,
            "aligned_rms": abs(coefficient) * native_rms if valid else None,
            "perp_rms": rms([a - coefficient * b for a, b in zip(change, direction)]) if valid else None}
    return result


def indexed(rows, names, expected):
    result = {}
    for row in rows:
        key = tuple(row[name] for name in names)
        require(key not in result, f"Duplicate cell: {key}")
        result[key] = row
    expected = set(expected)
    require(set(result) == expected, f"Matrix mismatch: missing={expected-set(result)}, extra={set(result)-expected}")
    return result


def close(value, reference, label):
    if reference is None:
        require(value is None, f"{label}: undefined value must be null")
    elif type(reference) is bool:
        require(type(value) is bool and value == reference, f"{label}: boolean mismatch")
    else:
        require(math.isclose(finite(value), reference, rel_tol=2e-5, abs_tol=2e-6),
                f"{label}: {value} != independently computed {reference}")


def anchor(row):
    return row["state_index"], row["noise_seed"], row["flow_index"]


def pair_metrics(a, b, selection=(50, 0, 7)):
    a, b = active_vector(a, *selection), active_vector(b, *selection)
    norm_product = math.sqrt(math.fsum(x*x for x in a) * math.fsum(x*x for x in b))
    return {"comparable": True, "rms_diff": rms([x-y for x, y in zip(a, b)]),
            "rms_common": rms([(x+y)/2 for x, y in zip(a, b)]),
            "rms_left": rms(a), "rms_front": rms(b),
            "cosine": math.fsum(x*y for x, y in zip(a, b))/norm_product
                      if norm_product > 1e-20 else None}


def audit_prefix_inputs(manifest):
    """Audit valid conditioning positions; keep masked-query differences visible."""
    input_tables, prefix_batches = [manifest[name] for name in ("prefix_input_hashes", "prefix_audit")]
    require(len(input_tables) == len(prefix_batches) == len(manifest["indices"]),
            "Missing physical-state prefix/input audit")
    prefix_layers = max(row["vlm_layer"] for row in manifest["layer_map"]["R0"]) + 1
    expected = itertools.product(GOALS, range(prefix_layers), ("keys", "values"), ("image", "text", "state"))
    expected = set(expected)
    maxima = {kind: {group: 0.0 for group in ("image", "text", "state")}
              for kind in ("all", "active", "masked")}
    total_rows = 0
    for inputs, rows in zip(input_tables, prefix_batches):
        finite_tree(inputs)
        require(set(inputs) == set(ROLES), "Missing preprocessed input model")
        for role in ROLES:
            require(set(inputs[role]) == set(GOALS), "Missing preprocessed input instruction")
            token_hashes = set()
            for goal in GOALS:
                record = inputs[role][goal]
                for name in ("images", "image_masks"):
                    require(bool(record[name]), f"Empty {name}")
                    for digest in record[name]:
                        require(isinstance(digest, str) and len(digest) == 64
                                and all(c in "0123456789abcdef" for c in digest), f"Invalid {name} hash")
                for name in ("language_tokens", "language_mask", "state_before_projection"):
                    digest = record[name]
                    require(isinstance(digest, str) and len(digest) == 64
                            and all(c in "0123456789abcdef" for c in digest), f"Invalid {name} hash")
                token_hashes.add(record["language_tokens"])
                for name in ("images", "image_masks", "state_before_projection", "image_shapes", "state_shape"):
                    require(record[name] == inputs[role]["left"][name],
                            f"Non-language input changed with instruction: {role}/{goal}/{name}")
            require(len(token_hashes) == 3, f"Instructions collapsed in preprocessing: {role}")
        for goal in GOALS:
            require(inputs["R0"][goal] == inputs["FT4"][goal],
                    f"Preprocessed input mismatch across models: {goal}")
        indexed(rows, ("goal", "layer", "tensor", "group"), expected)
        for row in rows:
            finite_tree(row)
            counts = []
            squared = []
            for label in ("active", "masked"):
                subset = row[label]
                count, elements = subset["count"], subset["element_count"]
                require(type(count) is int and type(elements) is int and count >= 0 and elements >= 0,
                        "Invalid prefix position/element count")
                require((count == 0) == (elements == 0), "Prefix position/element emptiness differs")
                require(elements == 0 or elements % count == 0, "Invalid prefix elements per position")
                maximum, error = finite(subset["max_abs"]), finite(subset["rms"])
                require(0 <= error <= maximum + 1e-7, "Invalid prefix error moments")
                if elements == 0:
                    require(maximum == error == 0, "Empty prefix subset has nonzero error")
                counts.append(elements)
                squared.append(elements * error * error)
                maxima[label][row["group"]] = max(maxima[label][row["group"]], maximum)
            require(sum(counts) > 0, "Empty prefix group")
            close(row["max_abs"], max(row["active"]["max_abs"], row["masked"]["max_abs"]),
                  "Prefix all-position maximum")
            close(row["rms"], math.sqrt(math.fsum(squared) / sum(counts)), "Prefix partition RMS")
            maxima["all"][row["group"]] = max(maxima["all"][row["group"]], row["max_abs"])
            if row["group"] in ("image", "text"):
                require(row["active"]["count"] > 0, "No valid image/text conditioning positions")
                require(row["active"]["max_abs"] <= 1e-5, "Frozen valid image/text prefix differs")
            total_rows += 1
    return {"physical_input_tables": len(input_tables), "prefix_difference_records": total_rows,
            "valid_image_text_threshold": 1e-5, "max_abs_by_position_subset": maxima,
            "preprocessed_inputs_identical_across_models": True,
            "nonlanguage_inputs_identical_across_instructions": True,
            "instruction_token_hashes_distinct": True,
            "masked_query_differences_are_not_valid_conditioning_differences": True}


def audit(run):
    run = Path(run).resolve()
    manifest, complete, engineering = [read_json(run / f"{name}.json")
                                     for name in ("manifest", "COMPLETE", "engineering")]
    require(manifest["status"] == complete["status"] == "complete", "Incomplete run")
    require(engineering["passed"] is True, "Engineering did not pass")
    indices, seeds, flows = [manifest[key] for key in ("indices", "seeds", "flow_indices")]
    require(indices == [0, 1, 2, 3], "Unexpected development states")
    require(seeds == [2026102801, 2026102802, 2026102803], "Unexpected inference seeds")
    require(flows == [0, 4, 8], "Unexpected flow grid")
    require(tuple(manifest["roles"]) == ROLES, "Unexpected checkpoint roles")
    prefix_audit = audit_prefix_inputs(manifest)
    layers = manifest["ca_layers"]
    for role in ROLES:
        require(bool(layers[role]) and len(layers[role]) == len(set(layers[role])), "Invalid CA layers")
        mapped = [r["vlm_layer"] for r in engineering["layer_map"][role]
                  if r["kind"] in ("CA", "cross_attn", "cross_attention")]
        require(mapped == layers[role], f"Runtime CA map mismatch: {role}")
    checks = engineering["checks"]
    require({r["model"] for r in checks} == set(ROLES), "Engineering missing model")
    indexed(checks, ("model", "goal"), itertools.product(ROLES, GOALS))
    for row in checks:
        for field in ("record_v_max_abs", "self_v_max_abs", "record_action_max_abs",
                      "self_action_max_abs", "attention_readout_reconstruction_max_abs"):
            require(0 <= finite(row[field]) <= 1e-5, f"Identity gate failed: {field}")
    for role in ROLES:
        require(0 <= finite(manifest["attention_readout_reconstruction_max_abs"][role]) <= 1e-5,
                f"Attention P/C readout reconstruction failed: {role}")
    repeat = finite(manifest["identity_epsilon_v"])
    require(repeat >= max(finite(row["repeat_v_rms"]) for row in checks) >= 0, "Invalid error floor")
    isolation = engineering["cache_isolation"]
    require(isolation["object_unique"] is True and isolation["tensor_storage_unique"] is True,
            "Cache isolation not established")
    baseline_rows = read_jsonl(run / "baseline.jsonl")
    baseline = indexed(baseline_rows, ("state_index", "noise_seed", "flow_index", "model", "goal"),
                       itertools.product(indices, seeds, flows, ROLES, GOALS))
    anchors, state_hashes = {}, {}
    for row in baseline_rows:
        finite_tree(row)
        key = anchor(row)
        require(row["anchor_id"] == f"i{key[0]:02d}_s{key[1]}_f{key[2]:02d}", "Anchor ID mismatch")
        require(math.isclose(row["time"], 1-key[2]/10, abs_tol=1e-6), "Flow time mismatch")
        for name in ("x_sha256", "state_sha256"):
            value = row[name]
            require(len(value) == 64 and all(c in "0123456789abcdef" for c in value), f"Invalid {name}")
        identity = (row["anchor_id"], row["x_sha256"], row["state_sha256"], row["time"])
        require(anchors.setdefault(key, identity) == identity, "Input mismatch inside anchor")
        require(state_hashes.setdefault(key[0], row["state_sha256"]) == row["state_sha256"], "State drift")
        active_vector(row["velocity"], 50)
        require(set(map(int, row["layers"])) == set(layers[row["model"]]), "Missing CA measurements")
        for measures in row["layers"].values():
            require(set(("P", "C", "PC", "M", "attention_residual", "post_mlp")) <= set(measures),
                    "Missing recorded computation stage")
            for name, stats in measures.items():
                count = math.prod(stats["shape"])
                require(count > 0 and min(stats["rms"], stats["norm"], stats["max_abs"]) >= 0,
                        f"Invalid {name} feature moments")
                require(math.isclose(stats["norm"], stats["rms"]*math.sqrt(count),
                                     rel_tol=1e-4, abs_tol=1e-5), f"Inconsistent {name} norm/RMS")
            masses = measures["P"]["mass_by_source"]
            require(set(masses) in ({"image", "text", "state"}, {"image", "language", "state"}),
                    "Incomplete prefix modality groups")
            lengths = {len(values) for values in masses.values()}
            require(len(lengths) == 1, "Different head counts in modality mass")
            for mass in zip(*masses.values()):
                require(all(-1e-6 <= value <= 1+1e-6 for value in mass), "Invalid attention mass")
                require(math.isclose(math.fsum(mass), 1, rel_tol=0, abs_tol=0.01),
                        "Prefix attention modality mass does not sum to one")
    swap_name = "swap.jsonl" if (run / "swap.jsonl").is_file() else "patches.jsonl"
    swap_rows = read_jsonl(run / swap_name)
    expected = ((i, seeds[0], f, role, goal, "front" if goal == "left" else "left", layer)
                for i, f, role, goal in itertools.product(indices, flows, ROLES, ("left", "front"))
                for layer in layers[role])
    indexed(swap_rows, ("state_index", "noise_seed", "flow_index", "model", "recipient", "donor", "layer"), expected)
    summaries = {}
    for row in swap_rows:
        finite_tree(row)
        key, role = anchor(row), row["model"]
        before, donor = [baseline[(*key, role, row[field])] for field in ("recipient", "donor")]
        for field in ("anchor_id", "x_sha256", "state_sha256", "time"):
            require(row[field] == before[field] == donor[field], f"Patch changed {field}")
        metrics = transfer_metrics(before["velocity"], donor["velocity"], row["velocity"], repeat)
        for group, expected in metrics.items():
            for field, value in expected.items():
                close(row["metrics"][group][field], value, f"{key}/{role}/{row['layer']}/{group}/{field}")
        summaries.setdefault((role, row["layer"], row["recipient"], row["donor"]), []).append(metrics["primary"])
    files = {}
    for name in ("manifest.json", "COMPLETE.json", "engineering.json", "baseline.jsonl",
                 swap_name, "paired.jsonl", "candidates.json"):
        path = run / name
        require(path.is_file(), f"Missing evidence: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if name not in ("manifest.json", "COMPLETE.json"):
            require(name in manifest["files"], f"Missing sealed file digest: {name}")
            require(files[name] == manifest["files"][name], f"File digest mismatch: {name}")
    paired = read_jsonl(run / "paired.jsonl")
    pair_keys = set()
    for row in paired:
        finite_tree(row)
        key, role, goals = anchor(row), row["model"], tuple(row["goal_pair"])
        cell = (*key, role, goals, row["layer"])
        require(cell not in pair_keys, "Duplicate paired measurement")
        pair_keys.add(cell)
        a, b = [baseline[(*key, role, goal)] for goal in goals]
        for field in ("anchor_id", "x_sha256", "state_sha256", "time"):
            require(row[field] == a[field] == b[field], f"Paired input mismatch: {field}")
        for field, value in pair_metrics(a["velocity"], b["velocity"]).items():
            close(row["velocity"][field], value, f"Paired velocity {field}")
        for name, selection in (("velocity_primary", (10, 0, 7)),
                                ("velocity_motion_primary", (10, 0, 6)),
                                ("velocity_gripper_primary", (10, 6, 7))):
            for field, value in pair_metrics(a["velocity"], b["velocity"], selection).items():
                close(row[name][field], value, f"Paired {name}/{field}")
        for name, stats in row["features"].items():
            left, front = [base["layers"][str(row["layer"])][name] for base in (a, b)]
            require(stats["comparable"] == (left["shape"] == front["shape"]), "Feature shape comparison mismatch")
            if stats["comparable"]:
                close(stats["rms_left"], left["rms"], f"Paired {name} left RMS")
                close(stats["rms_front"], front["rms"], f"Paired {name} front RMS")
    expected_pair = {
        (i, seed, f, role, goals, layer)
        for i, seed, f, role, goals in itertools.product(
            indices, seeds, flows, ROLES,
            (("left", "front"), ("left", "right"), ("front", "right")))
        for layer in layers[role]}
    require(pair_keys == expected_pair, "Incomplete paired measurement matrix")
    candidate_file = read_json(run / "candidates.json")
    expected_ranking = []
    epsilon = max(10*repeat, 1e-6)
    for layer in layers["R0"]:
        directions = {}
        for recipient, donor in (("left", "front"), ("front", "left")):
            values = summaries[("R0", layer, recipient, donor)]
            valid = [item["T"] for item in values if item["valid_T"]]
            directions[recipient] = {"valid": len(valid), "total": len(values),
                "median_T": statistics.median(valid) if valid else None,
                "median_rms_delta": statistics.median(item["rms_delta"] for item in values)}
        eligible = all(row["valid"] > 0 and row["median_T"] > 0
                       and row["median_rms_delta"] > epsilon for row in directions.values())
        score = min(row["median_T"] for row in directions.values()) if eligible else None
        expected_ranking.append({"layer": layer, "directions": directions,
                                 "eligible": eligible, "score": score})
    expected_ranking.sort(key=lambda row: row["score"] if row["score"] is not None else -math.inf,
                          reverse=True)
    require([row["layer"] for row in candidate_file["ranking"]] ==
            [row["layer"] for row in expected_ranking], "Candidate ranking differs")
    for recorded, expected in zip(candidate_file["ranking"], expected_ranking):
        for name in ("eligible", "score"):
            close(recorded[name], expected[name], f"Candidate {name}")
        for goal, fields in expected["directions"].items():
            for name, value in fields.items():
                close(recorded["directions"][goal][name], value, f"Candidate {goal}/{name}")
    selected = [row["layer"] for row in expected_ranking if row["eligible"]][:2]
    require(candidate_file["selected"] == selected and len(selected) <= 2, "Candidate cap/selection mismatch")
    for container in (manifest, complete):
        require(container["counts"]["baseline"] == len(baseline_rows), "Baseline total mismatch")
        require(container["counts"]["patches"] == len(swap_rows), "Patch total mismatch")
        require(container["counts"]["paired"] == len(paired), "Paired total mismatch")
        require(container["counts"]["reference_denoise"] == len(indices)*len(seeds)*10,
                "Hidden/missing reference generation cost")
    report = {"status": "passed",
        "scope": "Independent audit of saved effective velocities; not behavioral target control",
        "counts": {"physical_states": len(indices), "anchors": len(anchors),
                   "baseline_queries": len(baseline_rows), "patch_queries": len(swap_rows),
                   "paired_records": len(paired)},
        "registered_inputs": {"indices": indices, "seeds": seeds, "flow_indices": flows, "ca_layers": layers},
        "identity": {"threshold": 1e-5, "checks": len(checks), "repeat_v_rms": repeat, "cache_isolation": isolation},
        "prefix_input_validation": prefix_audit,
        "source_files": files, "per_layer_direction": [],
        "selected_development_candidates": selected,
        "limits": ["Four development physical states; noise repeats are not independent training seeds.",
                   "T measures transfer to the same model's other response, not correct target execution.",
                   "No model/GPU rerun; raw P/C or hidden tensors are not independently recomputed."]}
    for (role, layer, recipient, donor), items in sorted(summaries.items()):
        valid = [item["T"] for item in items if item["valid_T"]]
        report["per_layer_direction"].append({"model": role, "layer": layer,
            "recipient": recipient, "donor": donor, "queries": len(items), "valid_T": len(valid),
            "mean_T": math.fsum(valid)/len(valid) if valid else None,
            "mean_absolute_change_rms": math.fsum(item["rms_delta"] for item in items)/len(items)})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Audit output exists")
    result = audit(args.run)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, ensure_ascii=False))
