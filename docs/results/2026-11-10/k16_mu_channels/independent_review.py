"""Independent, complete-only K16 endpoint and descriptive trajectory review.

Does not import the runner or formal summarizer. Refuses an incomplete archive.
Reads raw traces, independently reconstructs endpoints and factorial contrasts,
then verifies the formal summary. Trajectory descriptions do not prove mediation.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
from statistics import median


INITS = tuple(range(20, 40))
GROUPS = ("N", "H", "S", "HS", "P", "HP", "SP", "HSP")
FACTORS = ("H", "S", "P")
BITS = {"H": 1, "S": 2, "P": 4}
MASKS = {g: sum(BITS[f] for f in FACTORS if f in g) for g in GROUPS}
NAME = {v: k for k, v in MASKS.items()}
LABEL = {"A": "left", "B": "right"}
REGIONS = ("left", "front", "right", "back")
SINGLE = {"dir", "L_first", "R_first", "L_ever", "R_ever", "L_term", "R_term", "both_term"}
MAIN = ("dir", "total_first", "place", "term", "L_first", "R_first", "L_ever", "R_ever", "L_term", "R_term", "none_first", "front_first")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def tag(value):
    return "/".join(value) or "none"


def stats(values):
    values = list(values)
    return {"n": len(values), "min": min(values), "median": median(values), "max": max(values)} if values else {"n": 0}


def box_gap(p, site):
    excess = []
    for axis in (0, 1):
        half = sum(abs(site["rotation"][axis][j]) * site["half_size"][j] for j in range(3))
        excess.append(max(0.0, abs(p[axis] - site["centre"][axis]) - half))
    return math.hypot(*excess)


def endpoints(trace, z0, geometry):
    run = {g: 0 for g in REGIONS}
    times = {g: None for g in REGIONS}
    ever = {g: False for g in REGIONS}
    any_hit = False
    for r in trace:
        hit = [g for g in REGIONS if r["obj"][2] - z0 > .03 and box_gap(r["obj"], geometry[g]) == 0]
        assert hit == r["hits"], ("independent single-step geometry", r["step"])
        any_hit |= bool(hit)
        for g in REGIONS:
            run[g] = run[g] + 1 if g in hit else 0
            if run[g] == 5 and times[g] is None:
                times[g] = r["step"] - 4
            ever[g] |= bool(r["pred"][g])
    earliest = min((x for x in times.values() if x is not None), default=None)
    return {
        "first": [g for g in REGIONS if earliest is not None and times[g] == earliest],
        "first_step": earliest, "first_times": times, "ever": ever,
        "terminal": trace[-1]["pred"], "any_single_step_geometric_hit": any_hit,
    }


def angle(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    assert na > 0 and nb > 0
    cosine = min(1.0, abs(sum(x * y for x, y in zip(a, b))) / (na * nb))
    return 2 * math.degrees(math.acos(cosine))


def state_description(row, boundary, z0, target, geometry):
    result = {
        "step": row["step"], "both_contacts": bool(row["c1"] and row["c2"]),
        "book_lift_from_step1_m": row["obj"][2] - z0,
        "relpos_change_from_boundary_m": math.dist(row["relpos"], boundary["relpos"]),
        "relquat_change_from_boundary_degrees": angle(row["relquat"], boundary["relquat"]),
        "requested_xy_box_gap_m": box_gap(row["obj"], geometry[target]),
        "nearest_any_xy_box_gap_m": min(box_gap(row["obj"], geometry[g]) for g in REGIONS),
    }
    for name, field in (("book", "obj"), ("eef", "eef")):
        delta = [row[field][i] - boundary[field][i] for i in range(3)]
        result[name + "_displacement_from_boundary_xyz_m"] = delta
        result[name + "_xy_displacement_from_boundary_m"] = math.hypot(*delta[:2])
    return result


def trajectory_description(row, trace, derived, geometry):
    b = row["boundary"]
    initial, window = trace[b - 1], trace[b:b + 60]
    signs = [1 if r["a"][6] > 0 else -1 if r["a"][6] < 0 else 0 for r in window]
    arm = [any(abs(v) > 1 for v in r["a"][:6]) for r in window]
    grip = [abs(r["a"][6]) > 1 for r in window]
    assert all(r["clip"] == (a or g) for r, a, g in zip(window, arm, grip))
    result = {
        "init": row["init"], "condition": row["condition"], "goal": row["goal"], "requested": row["requested"],
        "first": tag(derived["first"]), "first_correct": derived["first"] == [row["requested"]],
        "ever_correct": bool(derived["ever"][row["requested"]]),
        "terminal_correct": bool(derived["terminal"][row["requested"]]),
        "any_single_step_geometric_hit": derived["any_single_step_geometric_hit"],
        "boundary": state_description(initial, initial, row["z0"], row["requested"], geometry),
        "window_end": state_description(window[-1], initial, row["z0"], row["requested"], geometry),
        "final": state_description(trace[-1], initial, row["z0"], row["requested"], geometry),
        "window_both_contacts_fraction": sum(r["c1"] and r["c2"] for r in window) / 60,
        "window_lifted_over3cm_fraction": sum(r["obj"][2] - row["z0"] > .03 for r in window) / 60,
        "window_max_relquat_change_degrees": max(angle(r["relquat"], initial["relquat"]) for r in window),
        "window_gripper_close_positive_fraction": signs.count(1) / 60,
        "window_gripper_open_negative_fraction": signs.count(-1) / 60,
        "window_gripper_zero_fraction": signs.count(0) / 60,
        "window_gripper_sign_switches": sum(a != b for a, b in zip(signs, signs[1:])),
        "window_arm_clip_rows": sum(arm), "window_gripper_clip_rows": sum(grip),
    }
    return result


def describe_subset(part):
    if not part:
        return {"n": 0}
    out = {"n": len(part), "first_distribution": dict(sorted(Counter(x["first"] for x in part).items())),
           "first_correct": sum(x["first_correct"] for x in part),
           "ever_correct": sum(x["ever_correct"] for x in part),
           "terminal_correct": sum(x["terminal_correct"] for x in part),
           "any_single_step_geometric_hit": sum(x["any_single_step_geometric_hit"] for x in part)}
    for name in ("window_both_contacts_fraction", "window_lifted_over3cm_fraction", "window_max_relquat_change_degrees",
                 "window_gripper_close_positive_fraction", "window_gripper_open_negative_fraction",
                 "window_gripper_zero_fraction", "window_gripper_sign_switches"):
        out[name] = stats(x[name] for x in part)
    out["window_arm_clip_rows_total"] = sum(x["window_arm_clip_rows"] for x in part)
    out["window_gripper_clip_rows_total"] = sum(x["window_gripper_clip_rows"] for x in part)
    for point in ("window_end", "final"):
        out[point] = {
            "both_contacts_count": sum(x[point]["both_contacts"] for x in part),
            "lifted_over3cm_count": sum(x[point]["book_lift_from_step1_m"] > .03 for x in part),
        }
        for field in ("book_lift_from_step1_m", "relpos_change_from_boundary_m", "relquat_change_from_boundary_degrees",
                      "book_xy_displacement_from_boundary_m", "eef_xy_displacement_from_boundary_m",
                      "requested_xy_box_gap_m", "nearest_any_xy_box_gap_m"):
            out[point][field] = stats(x[point][field] for x in part)
    return out


def metric_vectors(derived):
    vectors = {}
    for group in GROUPS:
        part = Counter()
        vector = {m: [] for m in (*MAIN, "front_ever", "front_term", "both_term")}
        for init in INITS:
            a, b = derived[init, group, "A"], derived[init, group, "B"]
            for side, x, target in (("L", a, "left"), ("R", b, "right")):
                part[side + "_first"] = int(x["first"] == [target])
                part[side + "_ever"] = int(x["ever"][target])
                part[side + "_term"] = int(x["terminal"][target])
            part["dir"] = part["L_first"] * part["R_first"]
            part["total_first"] = part["L_first"] + part["R_first"]
            part["place"] = part["L_ever"] + part["R_ever"]
            part["term"] = part["L_term"] + part["R_term"]
            part["both_term"] = part["L_term"] * part["R_term"]
            part["front_first"] = int("front" in a["first"]) + int("front" in b["first"])
            part["none_first"] = int(not a["first"]) + int(not b["first"])
            part["front_ever"] = int(a["ever"]["front"]) + int(b["ever"]["front"])
            part["front_term"] = int(a["terminal"]["front"]) + int(b["terminal"]["front"])
            for metric in vector:
                vector[metric].append(part[metric])
        vectors[group] = vector
    return vectors


def assert_close(a, b):
    assert abs(a - b) < 1e-12, (a, b)


def check_contrast(formal, coefficients, metric, vectors, boot_indices=None):
    assert formal["coefficients"] == coefficients, (formal["coefficients"], coefficients)
    values = [sum(coefficient * vectors[group][metric][i] for group, coefficient in coefficients.items())
              for i in range(len(INITS))]
    denominator = 20 if metric in SINGLE else 40
    assert formal["snapshot_contributions"] == dict(zip(map(str, INITS), values))
    assert formal["denominator"] == denominator
    assert_close(formal["point"], sum(values) / denominator)
    if boot_indices is not None:
        draws = sorted(sum(values[k] for k in indices) / denominator for indices in boot_indices)
        for a, b in zip(formal["ci95_cluster_bootstrap"], (draws[250], draws[9749])):
            assert_close(a, b)
    return {"coefficients": coefficients, "metric": metric, "denominator": denominator,
            "point": sum(values) / denominator, "snapshot_contributions": dict(zip(map(str, INITS), values))}


def main(archive):
    # Completeness is checked before reading any episode outcomes.
    for init in INITS:
        marker = load(archive / f"i{init}_COMPLETE.json")
        assert marker["init"] == init and marker["n_episodes"] == 16
    formal = load(archive / "summary.json")
    assert formal["episodes"] == 320 and formal["audits"]["episodes_checked"] == 320
    records, traces, paths = {}, {}, []
    for init in INITS:
        for suffix, destination in (("", records), ("_traces", traces)):
            path = archive / f"i{init}{suffix}.jsonl"
            paths.append(path)
            for row in lines(path):
                key = row["init"], row["condition"], row["goal"]
                assert key not in destination and row["init"] == init
                destination[key] = row["rows"] if suffix else row
    expected = {(i, g, q) for i in INITS for g in GROUPS for q in LABEL}
    assert set(records) == set(traces) == expected
    geometry = load(archive / "geometry.json")["sites"]
    derived, descriptions = {}, []
    for key in sorted(expected):
        row, trace = records[key], traces[key]
        init, group, goal = key
        assert len(trace) == 400 and [x["step"] for x in trace] == list(range(1, 401))
        assert row["requested"] == LABEL[goal] and row["alpha"] == 2.959 and row["window"] == 60
        base = records[init, "N", "A"]
        for field in ("boundary", "z0", "seed", "boundary_signature"):
            assert row[field] == base[field], (key, field)
        assert trace[:row["boundary"]] == traces[init, "N", "A"][:row["boundary"]]
        own = endpoints(trace, row["z0"], geometry)
        for field in ("first", "first_step", "first_times", "ever", "terminal"):
            assert row[field] == own[field], (key, field)
        derived[key] = own
        descriptions.append(trajectory_description(row, trace, own, geometry))
    vectors = metric_vectors(derived)
    assert vectors == formal["per_snapshot"], "independent per-snapshot metric mismatch"
    groups = {}
    for group in GROUPS:
        selected = [x for x in descriptions if x["condition"] == group]
        groups[group] = {
            "totals": {m: sum(values) for m, values in vectors[group].items()},
            "by_goal": {goal: describe_subset([x for x in selected if x["goal"] == goal]) for goal in LABEL},
            "none_by_goal": {goal: describe_subset([x for x in selected if x["goal"] == goal and x["first"] == "none"]) for goal in LABEL},
            "all": describe_subset(selected),
        }
        assert groups[group]["totals"] == formal["summary"][group]["totals"]
        for goal in LABEL:
            own = groups[group]["by_goal"][goal]
            source = formal["summary"][group]["per_goal"][goal]
            for field in ("first_distribution", "first_correct", "ever_correct", "terminal_correct"):
                assert own[field] == source[field], (group, goal, field)
    edges = {}
    for factor in FACTORS:
        for before in GROUPS:
            if MASKS[before] & BITS[factor]:
                continue
            after = NAME[MASKS[before] | BITS[factor]]
            name = f"add_{factor}:{before}_to_{after}"
            transitions = {}
            for goal, label in LABEL.items():
                first = Counter((tag(derived[i, before, goal]["first"]), tag(derived[i, after, goal]["first"])) for i in INITS)
                side = []
                for metric, field in (("first", "first"), ("ever", "ever"), ("term", "terminal")):
                    gain, loss = [], []
                    for init in INITS:
                        a, b = derived[init, before, goal][field], derived[init, after, goal][field]
                        a, b = (a == [label], b == [label]) if field == "first" else (bool(a[label]), bool(b[label]))
                        if a != b:
                            (gain if b else loss).append(init)
                    side.append({"metric": metric, "gained_inits": gain, "lost_inits": loss})
                transitions[goal] = {"first_destination_transitions": [{"before": a, "after": b, "count": n} for (a, b), n in sorted(first.items())], "correctness_transitions": side}
            comparison = formal["factor_added_removed_edges"][name]
            for direction, coefficients in (("added", {after: 1, before: -1}), ("removed", {before: 1, after: -1})):
                for metric in MAIN:
                    check_contrast(comparison[direction]["metrics"][metric], coefficients, metric, vectors)
            edges[name] = {"before": before, "after": after, "factor": factor, "per_goal": transitions,
                           "snapshot_metric_differences": {m: [b - a for a, b in zip(vectors[before][m], vectors[after][m])] for m in MAIN}}
    rng = random.Random(20261110)
    boot_indices = [[rng.randrange(20) for _ in range(20)] for _ in range(10000)]
    interaction_coefficients = {}
    for first, second in itertools.combinations(FACTORS, 2):
        third = next(x for x in FACTORS if x not in (first, second))
        for level in (0, 1):
            base = BITS[third] if level else 0
            coefficients = {NAME[base | BITS[first] | BITS[second]]: 1,
                            NAME[base | BITS[first]]: -1, NAME[base | BITS[second]]: -1, NAME[base]: 1}
            interaction_coefficients[f"{first}_x_{second}_given_{third}_{'old' if level else 'new'}"] = coefficients
    interaction_coefficients["H_x_S_x_P_exploratory"] = {g: (-1) ** (3 - MASKS[g].bit_count()) for g in GROUPS}
    assert set(interaction_coefficients) == set(formal["interactions"])
    interactions = {name: {m: check_contrast(formal["interactions"][name][m], coefficients, m, vectors, boot_indices)
                           for m in MAIN} for name, coefficients in interaction_coefficients.items()}
    for name, metrics in formal["comparisons_to_references"].items():
        after, before = name.split("_minus_")
        for metric in MAIN:
            check_contrast(metrics["metrics"][metric], {after: 1, before: -1}, metric, vectors)
    for factor in FACTORS:
        name = f"{factor}_averaged_over_other_factor_levels_exploratory"
        coefficients = {g: .25 if MASKS[g] & BITS[factor] else -.25 for g in GROUPS}
        for metric in MAIN:
            check_contrast(formal["marginal_factor_effects"][name][metric], coefficients, metric, vectors)
    paths.extend([archive / "summary.json", archive / "geometry.json", archive / "protocol.json", Path(__file__)])
    result = {
        "status": "complete independent descriptive review; all assertions passed", "episodes": 320,
        "trace_steps": 128000, "physical_snapshot_clusters": 20,
        "checks": {"independent_geometry_and_native_success_recomputed": 320, "formal_per_snapshot_counts_exact": True,
                   "factor_edges_points_and_snapshot_contributions_exact": 12,
                   "interaction_points_contributions_and_bootstrap_intervals_exact": 7,
                   "comparisons_and_marginal_points_and_contributions_exact": True},
        "groups": groups, "factor_edges": edges, "interactions": interactions,
        "trajectory_descriptions": descriptions,
        "interpretation_limits": [
            "H/S/P are inference-time velocity-output source interventions, not independent physical or semantic modules.",
            "S bundles height, axis-angle rotation and gripper; its effect cannot identify a single constituent.",
            "Window-end contact, relative pose, movement and sign are post-treatment descriptors, not proof of causal mediation.",
            "Two finger geometric contacts do not measure grip force or secure grasp.",
            "relquat angle describes book-relative-to-eef change; it is not world rotation or target pose error.",
            "Panda gripper uses sign: positive close, negative open. Magnitude is not force.",
            "None means no 5-step lifted region entry by step400, not no movement or no brief entry.",
            "Outcome-conditioned none subsets are exploratory selections, not treatment-effect estimates.",
            "All intervals are unadjusted development-screen intervals, without equivalence or training-root-cause inference.",
        ],
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    output = archive / "independent_review.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print("INDEPENDENT REVIEW PASS: 320 episodes; 12 edges; 7 interactions including bootstrap intervals")
    print("group L_first R_first L_ever R_ever L_term R_term paired")
    for group in GROUPS:
        t = groups[group]["totals"]
        print(group, *(t[m] for m in ("L_first", "R_first", "L_ever", "R_ever", "L_term", "R_term", "dir")))
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    main(parser.parse_args().archive.resolve())
