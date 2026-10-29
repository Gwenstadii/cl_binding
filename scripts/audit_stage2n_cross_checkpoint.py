"""Independent CPU audit of step-3 cross-checkpoint message interventions.

Only standard-library evidence reading and velocity arithmetic. No worker,
model, torch or simulator imports. Computational reference recovery is not
physical goal success; hidden-message norm evidence remains recorded evidence.
"""

import argparse
from collections import defaultdict
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


ROLES = ("R0", "FT4")
GOALS = ("left", "front")
MODES = ("cross_same", "cross_wrong", "random_delta", "self")
GROUPS = {"primary": (10, 0, 7), "motion": (10, 0, 6),
          "gripper": (10, 6, 7), "full50": (50, 0, 7)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def reject_nonfinite(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_nonfinite)


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line, parse_constant=reject_nonfinite) for line in stream]


def number(value):
    require(type(value) in (float, int) and math.isfinite(value), "Invalid finite number")
    return float(value)


def vector(matrix, horizon=10, start=0, end=7):
    require(isinstance(matrix, list) and len(matrix) == 50, "Expected 50 velocity positions")
    require(all(isinstance(row, list) and len(row) >= 7 for row in matrix),
            "Missing effective action dimensions")
    require(0 < horizon <= 50 and 0 <= start < end <= 7, "Invalid effective action slice")
    return [number(matrix[i][j]) for i in range(horizon) for j in range(start, end)]


def rms(values):
    return math.sqrt(math.fsum(x*x for x in values)/len(values))


def subtract(a, b):
    return [x-y for x, y in zip(a, b)]


def transfer_metrics(base, reference, patched, epsilon=1e-6):
    epsilon = number(epsilon)
    require(epsilon >= 1e-6, "Invalid final velocity error floor")
    result = {}
    for name, selection in GROUPS.items():
        before, target, after = [vector(v, *selection) for v in (base, reference, patched)]
        direction, change = subtract(target, before), subtract(after, before)
        size, delta = rms(direction), rms(change)
        valid = size > epsilon
        coefficient = (math.fsum(a*b for a, b in zip(change, direction))
                       / math.fsum(a*a for a in direction)) if valid else None
        result[name] = {
            "rms_delta": delta, "rms_native_contrast": size, "T": coefficient,
            "aligned_rms": abs(coefficient)*size if valid else None,
            "perp_rms": rms([x-coefficient*y for x, y in zip(change, direction)])
                        if valid else None,
            "epsilon": epsilon, "valid_T": valid,
        }
    return result


def reference_metrics(base, patched, healthy_same, healthy_other, epsilon=1e-6):
    """Readout distances and goal-conditioned margin, not goal success."""
    result = {}
    for name, selection in GROUPS.items():
        before, after, same, other = [vector(v, *selection) for v in
                                      (base, patched, healthy_same, healthy_other)]
        pre, post = rms(subtract(before, same)), rms(subtract(after, same))
        alternative = rms(subtract(after, other))
        healthy_contrast = rms(subtract(same, other))
        result[name] = {
            "healthy_same_distance_before": pre,
            "healthy_same_distance_after": post,
            "healthy_other_distance_after": alternative,
            "healthy_distance_gain": 1-post/pre if pre > epsilon else None,
            "healthy_preference_margin": (alternative*alternative-post*post)
                                         / (healthy_contrast*healthy_contrast)
                                         if healthy_contrast > epsilon else None,
        }
    return result


def paired_response_metrics(left, front, healthy_left, healthy_front, epsilon=1e-6):
    """Separate common shifts from instruction-contrast errors algebraically."""
    result = {}
    for name, selection in GROUPS.items():
        a, b, old_a, old_b = [vector(v, *selection) for v in
                              (left, front, healthy_left, healthy_front)]
        contrast, healthy = subtract(a, b), subtract(old_a, old_b)
        centre = [(x+y)/2 for x, y in zip(a, b)]
        old_centre = [(x+y)/2 for x, y in zip(old_a, old_b)]
        size, reference_size = rms(contrast), rms(healthy)
        dot = math.fsum(x*y for x, y in zip(contrast, healthy))
        common_error = rms(subtract(centre, old_centre))
        contrast_error = rms(subtract(contrast, healthy))
        response_error = math.sqrt((rms(subtract(a, old_a))**2
                                    + rms(subtract(b, old_b))**2)/2)
        require(math.isclose(response_error**2, common_error**2+contrast_error**2/4,
                             rel_tol=1e-10, abs_tol=1e-12),
                "Paired response decomposition identity failed")
        result[name] = {
            "contrast_rms": size, "healthy_contrast_rms": reference_size,
            "contrast_projection": dot/math.fsum(x*x for x in healthy)
                                   if reference_size > epsilon else None,
            "contrast_cosine": dot/(len(contrast)*size*reference_size)
                               if min(size, reference_size) > epsilon else None,
            "contrast_error_rms": contrast_error, "common_error_rms": common_error,
            "response_error_rms": response_error,
        }
    return result


def indexed(rows, names, expected):
    result = {}
    for row in rows:
        key = tuple(row[name] for name in names)
        require(key not in result, f"Duplicate evidence cell: {key}")
        result[key] = row
    expected = set(expected)
    require(set(result) == expected,
            f"Matrix mismatch: expected {len(expected)}, actual {len(result)}; "
            f"missing={list(expected-set(result))[:4]}, extra={list(set(result)-expected)[:4]}")
    return result


def close(value, reference, label):
    if reference is None:
        require(value is None, f"{label}: undefined value must be null")
    elif type(reference) is bool:
        require(type(value) is bool and value == reference, f"{label}: boolean mismatch")
    else:
        require(math.isclose(number(value), reference, rel_tol=2e-5, abs_tol=2e-6),
                f"{label}: recorded value differs from independent arithmetic")


def block_array(value, count=15):
    require(isinstance(value, list) and len(value) == 50, "Expected 50 message norm positions")
    require(all(isinstance(row, list) and len(row) == count for row in value),
            "Message norm block shape mismatch")
    flat = [number(x) for row in value for x in row]
    require(min(flat) >= 0, "Negative block norm")
    return flat


def audit_random_norms(record, correct_record, count=15):
    target, desired, effective, cast_error, cast_input = [
        block_array(record[name], count) for name in
        ("target_block_l2", "desired_block_l2", "effective_block_l2",
         "cast_error_block_l2", "cast_input_block_l2")]
    correct = block_array(correct_record["effective_block_l2"], count)
    addition = block_array(record.get("addition_error_block_l2",
                                      [[0.0]*count for _ in range(50)]), count)
    maximum_relative, maximum_absolute, maximum_cast = 0.0, 0.0, 0.0
    for t, d, e, error, source_size, matched, add_error in zip(
            target, desired, effective, cast_error, cast_input, correct, addition):
        close(t, matched, "Random target versus correct-source applied difference")
        require(math.isclose(d, t, rel_tol=1e-5, abs_tol=1e-6),
                "Planned random difference is not matched to correct-source difference")
        if t == 0:
            require(d <= 1e-9, "Zero target requires zero intended random difference")
        tolerance = max(2e-6, 1e-5*max(t, d, e, source_size))
        require(abs(e-t) <= error+abs(d-t)+add_error+tolerance,
                "Actual random norm mismatch exceeds the cast triangle bound")
        require(error <= (2**-8+1e-5)*source_size+1e-6,
                "Recorded BF16 cast error exceeds round-to-nearest bound")
        maximum_relative = max(maximum_relative, abs(e-t)/t if t > 1e-9 else 0)
        maximum_absolute = max(maximum_absolute, abs(e-t))
        maximum_cast = max(maximum_cast, error)
    return {"max_relative_actual_norm_difference": maximum_relative,
            "max_absolute_actual_norm_difference": maximum_absolute,
            "max_recorded_cast_error_norm": maximum_cast,
            "global_effective_to_target_norm_ratio":
                math.sqrt(math.fsum(x*x for x in effective)/math.fsum(x*x for x in target))
                if math.fsum(x*x for x in target) > 1e-18 else None,
            "norm_blocks": len(target)}


def engineering_gate(engineering, layers):
    require(engineering["passed"] is True, "Engineering gate not passed")
    rows = engineering["checks"]
    checks = indexed(rows, ("model", "goal"), itertools.product(ROLES, GOALS))
    for row in rows:
        for field in ("record_v_max_abs", "prior_velocity_max_abs", "self_max_abs",
                      "readout_reconstruction_max_abs"):
            require(0 <= number(row[field]) <= 1e-5, f"Engineering failed: {field}")
        require(set(map(int, row["site_self_max_abs"])) == set(layers),
                "Engineering missing candidate/control self replacement")
        for error in row["site_self_max_abs"].values():
            require(0 <= number(error) <= 1e-5, "Engineering site self-replacement failed")
        require(number(row["repeat_v_rms"]) >= 0, "Negative repeated-query error")
    norm_checks = engineering["norm_engineering_checks"]
    indexed(norm_checks, ("model", "recipient", "layer"),
            itertools.product(ROLES, GOALS, layers))
    for row in norm_checks:
        require(0 <= number(row["desired_norm_max_abs_error"]) <= 1e-5,
                "Engineering intended random norm failed")
        require(number(row["triangle_excess_max"]) <= 1e-6,
                "Engineering random arithmetic triangle failed")
    isolation = engineering["cache_isolation"]
    require(isolation["object_unique"] is True and isolation["tensor_storage_unique"] is True
            and isolation["deep_copy_per_query"] is True, "Cache isolation not verified")
    return max(number(row["repeat_v_rms"]) for row in checks.values())


def prefix_gate(manifest):
    tables, batches = manifest["prefix_input_hashes"], manifest["prefix_audit"]
    require(len(tables) == len(batches) == len(manifest["indices"]), "Missing state input/prefix evidence")
    layer_count = max(r["vlm_layer"] for r in manifest["layer_map"]["R0"])+1
    expected = set(itertools.product(GOALS, range(layer_count), ("keys", "values"),
                                     ("image", "text", "state")))
    total, maximum = 0, 0.0
    for inputs, rows in zip(tables, batches):
        require(set(inputs) == set(ROLES), "Missing preprocessed model inputs")
        for goal in GOALS:
            require(inputs["R0"][goal] == inputs["FT4"][goal], "Cross-model preprocessed inputs differ")
        for role in ROLES:
            require(set(inputs[role]) == set(GOALS), "Missing old instruction input")
            left, front = [inputs[role][goal] for goal in GOALS]
            for field in ("images", "image_masks", "state_before_projection", "image_shapes", "state_shape"):
                require(left[field] == front[field], "Non-language input differs across old instructions")
            require(left["language_tokens"] != front["language_tokens"], "Old instructions collapsed in tokenizer")
        indexed(rows, ("goal", "layer", "tensor", "group"), expected)
        for row in rows:
            active, masked = row["active"], row["masked"]
            n, m = active["element_count"], masked["element_count"]
            require(type(n) is int and type(m) is int and n >= 0 and m >= 0 and n+m > 0,
                    "Invalid prefix element counts")
            for subset in (active, masked):
                require(type(subset["count"]) is int and subset["count"] >= 0, "Invalid prefix position count")
                require((subset["count"] == 0) == (subset["element_count"] == 0), "Prefix empty subset mismatch")
                require(0 <= number(subset["rms"]) <= number(subset["max_abs"])+1e-7,
                        "Prefix error moments inconsistent")
            close(row["max_abs"], max(active["max_abs"], masked["max_abs"]), "Prefix all-position maximum")
            close(row["rms"], math.sqrt((n*active["rms"]**2+m*masked["rms"]**2)/(n+m)),
                  "Prefix partition RMS")
            if row["group"] in ("image", "text"):
                require(active["count"] > 0 and active["max_abs"] <= 1e-5, "Frozen valid prefix changed")
                maximum = max(maximum, active["max_abs"])
        total += len(rows)
    return {"rows": total, "valid_image_text_max_abs": maximum}


def anchor(row):
    return row["state_index"], row["noise_seed"], row["flow_index"]


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def audit(run, prior_baseline):
    run, prior_baseline = Path(run).resolve(), Path(prior_baseline).resolve()
    manifest, complete, engineering = [read_json(run/f"{name}.json")
                                     for name in ("manifest", "COMPLETE", "engineering")]
    require(manifest["status"] == complete["status"] == "complete", "Run incomplete")
    require(engineering["passed"] is True, "Engineering gate not passed")
    indices, seeds, flows = [manifest[name] for name in ("indices", "seeds", "flow_indices")]
    require(sha256(prior_baseline) == manifest["prior_baseline_sha256"],
            "Sealed prior baseline digest mismatch")
    require(indices == [0, 1, 2, 3], "Unexpected physical development states")
    require(seeds == [2026102801, 2026102802, 2026102803], "Unexpected noise seeds")
    require(flows == [0, 4, 8], "Unexpected flow steps")
    require(tuple(manifest["roles"]) == ROLES and tuple(manifest["goals"]) == GOALS,
            "Unexpected model/instruction set")
    require(set(manifest["layers"]) == {1, 3, 9}, "Candidate/control layers changed")
    require(set(manifest["modes"]) == set(MODES), "Intervention modes changed")
    require(manifest["block_count"] == 15 and manifest["block_width"] == 48,
            "Output-coordinate bookkeeping block layout changed")
    repeat = engineering_gate(engineering, manifest["layers"])
    require(number(manifest["repeat_v_rms"]) >= repeat, "Repeated-query error floor understated")
    prefix_validation = prefix_gate(manifest)
    epsilon = number(manifest["metric_epsilon_v"])
    require(epsilon == max(10*number(manifest["repeat_v_rms"]), 1e-6), "Metric floor changed")
    baseline_rows = read_rows(run/"baseline.jsonl")
    baseline = indexed(baseline_rows, ("state_index", "noise_seed", "flow_index", "model", "goal"),
                       itertools.product(indices, seeds, flows, ROLES, GOALS))
    prior_rows = [r for r in read_rows(prior_baseline) if r["goal"] in GOALS]
    prior = indexed(prior_rows, ("state_index", "noise_seed", "flow_index", "model", "goal"),
                    itertools.product(indices, seeds, flows, ROLES, GOALS))
    maximum_reproduction_error = 0.0
    anchors = {}
    for key, row in baseline.items():
        expected = prior[key]
        for field in ("anchor_id", "x_sha256", "state_sha256", "reference_goal", "time"):
            require(row[field] == expected[field], f"Prior input replay differs: {field}")
        identity = tuple(row[f] for f in ("anchor_id", "x_sha256", "state_sha256", "time"))
        require(anchors.setdefault(anchor(row), identity) == identity, "Anchor inputs changed across queries")
        delta = max(abs(a-b) for a, b in zip(vector(row["velocity"], 50),
                                            vector(expected["velocity"], 50)))
        require(delta <= 1e-5, "Native velocity differs from sealed step-2 baseline")
        close(row["prior_velocity_max_abs"], delta, "Prior-baseline maximum error")
        maximum_reproduction_error = max(maximum_reproduction_error, delta)
    patch_rows = read_rows(run/"patches.jsonl")
    expected_patches = itertools.product(indices, seeds, flows, ROLES, GOALS,
                                        manifest["layers"], MODES)
    patches = indexed(patch_rows, ("state_index", "noise_seed", "flow_index", "model",
                                   "recipient", "layer", "mode"), expected_patches)
    norms = read_rows(run/"norms.jsonl.gz")
    norm_map = {}
    for row in norms:
        identity = row["norm_record_id"]
        require(identity not in norm_map, "Duplicate norm evidence ID")
        norm_map[identity] = row
    require(len(norm_map) == 864, "Incomplete source/random norm evidence")
    independent = defaultdict(list)
    random_summary = []
    self_error = 0.0
    consumed_norm_ids = set()
    for key, row in patches.items():
        i, seed, flow, role, goal, layer, mode = key
        other_role = "FT4" if role == "R0" else "R0"
        other_goal = "front" if goal == "left" else "left"
        base = baseline[i, seed, flow, role, goal]
        other = baseline[i, seed, flow, role, other_goal]
        cross = baseline[i, seed, flow, other_role, goal]
        healthy = baseline[i, seed, flow, "R0", goal]
        healthy_other = baseline[i, seed, flow, "R0", other_goal]
        for field in ("anchor_id", "x_sha256", "state_sha256", "time"):
            require(row[field] == base[field], f"Patch input changed: {field}")
        require(row["donor_model"] == (role if mode == "self" else other_role), "Wrong donor model")
        require(row["donor"] == (other_goal if mode == "cross_wrong" else goal), "Wrong donor goal")
        require(0 <= number(row["actual_patch_witness_error_max_abs"]) <= 1e-5,
                "Actual message insertion witness failed")
        references = {"cross_same": cross, "healthy_same": healthy, "same_model_other": other}
        for name, target in references.items():
            metrics = transfer_metrics(base["velocity"], target["velocity"], row["velocity"], epsilon)
            for group, fields in metrics.items():
                for field, value in fields.items():
                    close(row["metrics"][name][group][field], value,
                          f"{key}/{name}/{group}/{field}")
        distances = reference_metrics(base["velocity"], row["velocity"], healthy["velocity"],
                                      healthy_other["velocity"], epsilon)
        independent[role, goal, layer, mode].append(distances["primary"])
        if mode == "self":
            error = max(abs(a-b) for a, b in zip(vector(base["velocity"], 50),
                                                vector(row["velocity"], 50)))
            require(error <= 1e-5, "Self-message replacement changes native velocity")
            close(row["self_max_abs"], error, "Self-message recorded maximum error")
            self_error = max(self_error, error)
        if mode in ("cross_same", "random_delta"):
            norm_id = row["norm_record_id"]
            require(norm_id not in consumed_norm_ids, "Norm record reused by multiple patches")
            consumed_norm_ids.add(norm_id)
            record = norm_map[norm_id]
            require(record["block_count"] == 15 and record["block_width"] == 48,
                    "Norm record output-coordinate layout changed")
            for field in ("anchor_id", "state_index", "noise_seed", "flow_index",
                          "model", "recipient", "layer", "mode"):
                require(record[field] == row[field], f"Norm evidence identity differs: {field}")
            require(record["effective_message_sha256"] == row["effective_message_sha256"],
                    "Norm evidence message hash differs from actual patch")
            require(0 <= number(record["actual_patch_witness_error_max_abs"]) <= 1e-5,
                    "Norm evidence insertion witness failed")
            block_array(record["effective_block_l2"])
            if mode == "random_delta":
                require(record["message_dtype"] == "torch.bfloat16", "Unexpected native message precision")
                salt = manifest["configuration"]["random_seed_salt"]
                payload = f"{salt}|{row['anchor_id']}|{role}|{goal}|{layer}".encode("ascii")
                expected_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)
                require(record["random_seed"] == expected_seed, "Random direction seed changed")
                correct_patch = patches[i, seed, flow, role, goal, layer, "cross_same"]
                correct_record = norm_map[correct_patch["norm_record_id"]]
                random_summary.append(audit_random_norms(record, correct_record))
    require(consumed_norm_ids == set(norm_map), "Unreferenced/missing norm evidence")
    paired = []
    for i, seed, flow, role, layer, mode in itertools.product(
            indices, seeds, flows, ROLES, manifest["layers"], MODES):
        left, front = [patches[i, seed, flow, role, goal, layer, mode]["velocity"] for goal in GOALS]
        old_left, old_front = [baseline[i, seed, flow, "R0", goal]["velocity"] for goal in GOALS]
        paired.append({"state_index": i, "noise_seed": seed, "flow_index": flow,
                       "model": role, "layer": layer, "mode": mode,
                       "metrics": paired_response_metrics(left, front, old_left, old_front, epsilon)})
    files = {}
    for name in ("manifest.json", "COMPLETE.json", "engineering.json",
                 "baseline.jsonl", "patches.jsonl", "norms.jsonl.gz"):
        path = run/name
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if name in manifest.get("files", {}):
            require(files[name] == manifest["files"][name], f"Evidence digest mismatch: {name}")
    report = {
        "status": "passed", "audit_source_sha256": sha256(__file__),
        "scope": "independent velocity arithmetic and recorded norm bounds; no physical target claim",
        "counts": {"physical_states": len(indices), "anchors": len(anchors),
                   "baseline_queries": len(baseline), "patch_queries": len(patches),
                   "norm_records": len(norm_map), "paired_responses": len(paired)},
        "prior_baseline": {"sha256": sha256(prior_baseline),
                           "maximum_velocity_reproduction_error": maximum_reproduction_error},
        "self_replacement_max_abs": self_error, "metric_epsilon_v": epsilon,
        "prefix_validation": prefix_validation,
        "random_norm_evidence": {
            "cases": len(random_summary), "block_count": 15, "block_width": 48,
            "meaning": "output-coordinate bookkeeping blocks, not original attention heads",
            "max_relative_actual_norm_difference": max(r["max_relative_actual_norm_difference"]
                                                       for r in random_summary),
            "max_absolute_actual_norm_difference": max(r["max_absolute_actual_norm_difference"]
                                                       for r in random_summary),
            "max_recorded_cast_error_norm": max(r["max_recorded_cast_error_norm"]
                                                for r in random_summary),
            "global_effective_to_target_norm_ratio_median":
                median(r["global_effective_to_target_norm_ratio"] for r in random_summary),
            "global_effective_to_target_norm_ratio_min":
                min(r["global_effective_to_target_norm_ratio"] for r in random_summary
                    if r["global_effective_to_target_norm_ratio"] is not None),
            "global_effective_to_target_norm_ratio_max":
                max(r["global_effective_to_target_norm_ratio"] for r in random_summary
                    if r["global_effective_to_target_norm_ratio"] is not None)},
        "per_condition": [], "paired_responses": paired, "source_files": files,
        "limits": [
            "Four reused development starts; inference seeds are not independent training seeds.",
            "Reference-distance recovery and T are computational readout measures, not goal success.",
            "Hidden message norms are recorded evidence, not independently reconstructed full hidden tensors.",
            "Natural donor differences and recipient computations remain coupled.",
            "One registered random direction per cell is a development control, not a null distribution.",
        ],
    }
    for (role, goal, layer, mode), items in sorted(independent.items()):
        report["per_condition"].append({
            "model": role, "goal": goal, "layer": layer, "mode": mode, "queries": len(items),
            **{field+"_median": median(r[field] for r in items) for field in items[0]},
        })
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prior-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Audit output already exists")
    report = audit(args.run, args.prior_baseline)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                           encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"]}))
