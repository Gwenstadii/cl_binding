"""Audit K18-C record coverage and compute scene-paired effects."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
from statistics import mean, median, quantiles

INITS = tuple(range(20, 40))
TARGET = {"A": "left", "B": "right", "N": "front"}
GROUPS = ("state", "action_time", "attention", "mlp_norm", "output")
WINDOWS = ("000030_000100", "000030_001000")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def by_scene(rows):
    key = lambda row: (row["init"], row["route"], row["denoise_index"])
    assert len(rows) == len(set(map(key, rows))) == 120
    scenes = defaultdict(list)
    for row in rows:
        scenes[row["init"]].append(row)
    assert tuple(sorted(scenes)) == INITS and {len(g) for g in scenes.values()} == {6}
    return scenes


def paired(values, stat=mean):
    assert len(values) == len(INITS)
    rng = random.Random(20261111)
    boots = sorted(stat(rng.choices(values, k=len(values))) for _ in range(5000))
    return {"value": stat(values), "positive": sum(v > 0 for v in values),
            "negative": sum(v < 0 for v in values), "bootstrap_95": [boots[125], boots[4874]]}


def probe_summary(root, window):
    probe = read(root / f"probe_{window}.json")
    audit = read(root / f"audit_{window}.json")
    selection = read(root / f"selection_{window}.json")
    data = {cond: by_scene(rows) for cond, rows in probe["conditions"].items()}
    assert len(data) == 22
    assert audit["both_endpoints_all_tensors_exact"] and audit["unchanged_other_tensors"] > 0
    assert sum(len(g) for g in audit["groups"].values()) == 155
    assert sum(t["size"] for g in audit["groups"].values() for t in g) == 99880992
    assert len(probe["random_checks"]) == 10
    for r in probe["random_checks"].values():
        assert r["max_tensor_norm_relative_error"] < 1e-4
    assert selection["top_group"] == probe["summary"]["rank"][0]

    result = {"selection": selection, "by_group": {}, "absolute": {},
              "full_horizontal_drop": probe["summary"]["full_drop"],
              "max_random_norm_error": max(x["max_tensor_norm_relative_error"] for x in probe["random_checks"].values()),
              "max_random_cosine_abs": max(abs(x["cosine_with_true_group_update"]) for x in probe["random_checks"].values()),
              "random_checks": probe["random_checks"]}
    def values(cond, horizon, metric):
        return [median(row[horizon][metric] for row in data[cond][i]) for i in INITS]
    for cond in data:
        result["absolute"][cond] = {
            horizon: {metric: median(values(cond, horizon, metric)) for metric in data[cond][20][0][horizon]}
            for horizon in ("10", "50")
        }
    for g in GROUPS:
        contrasts = {"add_loss": ("early", f"add:{g}"), "undo_rescue": (f"undo:{g}", "late"),
                     "real_vs_random_add": (f"random_add:{g}", f"add:{g}"),
                     "real_vs_random_replace": (f"random_replace:{g}", "late")}
        entry = {name: {metric: paired([a-b for a, b in zip(values(c1,"10",metric), values(c2,"10",metric))], median)
                        for metric in ("gain_H", "gain", "mu_H_projection", "mu_Z_projection", "mu_R_projection")}
                 for name, (c1, c2) in contrasts.items()}
        entry["param_tensor_count"] = len(audit["groups"][g])
        entry["param_count"] = sum(t["size"] for t in audit["groups"][g])
        entry["delta_l2"] = math.sqrt(sum(t["delta_l2"]**2 for t in audit["groups"][g]))
        entry["delta_energy_fraction"] = (entry["delta_l2"]/audit["total_delta_l2"])**2
        result["by_group"][g] = entry
    return result


def projected_progress(row, offset, centre):
    trace = {r["step"]-row["boundary"]: r for r in row["trace"]}
    start, end = trace[0]["obj"], trace[offset]["obj"]
    direction = [x-y for x, y in zip(centre[:2], start[:2])]
    return sum((x-y)*d for x, y, d in zip(end[:2], start[:2], direction)) / math.hypot(*direction)


def first_label(row):
    return "/".join(row["first"]) or "none"


def native_summary(root, window, required):
    files = sorted(root.glob(f"native_{window}_w[0-7].jsonl"))
    if not files and not required:
        return None
    selection = read(root / f"selection_{window}.json")
    top_group = selection["top_group"]
    later = window == WINDOWS[1]
    selected = ["early", "late", f"undo:{top_group}", f"random_replace:{top_group}"] if later else selection["conditions"]
    expected_worker_conditions = selected[2:] if later else selected
    observed = {}
    for lane, path in enumerate(files):
        if required:
            assert path.stem.endswith(f"w{lane}")
            assert read(path.with_name(path.stem + "_complete.json"))["conditions"] == expected_worker_conditions
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["condition"], row["goal"], row["init"])
            assert key not in observed and row["goal"] in TARGET
            if row["goal"] != "N":
                assert row["first_step"] is None or row["first_step"] > row["boundary"]
            observed[key] = row
    new_episodes = len(observed)
    if later:
        for path in sorted(root.glob(f"native_{WINDOWS[0]}_w[0-7].jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["condition"] == "early":
                    observed["early",row["goal"],row["init"]] = row
        for line in (root / "reference_1000.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert row["step"] == 1000
            observed["late",row["goal"],row["init"]] = row
    complete = set(observed) == {(c, g, i) for c in selected for g in TARGET for i in INITS}
    if required:
        assert complete, (window, len(observed))
    if not complete:
        return {"complete": False, "n": len(observed)}
    conditions = {}
    for c in selected:
        group = {}
        for g, target in TARGET.items():
            rows = [observed[c,g,i] for i in INITS]
            first = Counter(map(first_label, rows))
            group[target] = {"first": dict(first), "n": len(rows),
                             "ever": {k: sum(r["ever"][k] for r in rows) for k in TARGET.values()},
                             "terminal": {k: sum(r["terminal"][k] for r in rows) for k in TARGET.values()}}
        group["both_old_first_correct"] = sum(
            observed[c,"A",i]["first"] == ["left"] and observed[c,"B",i]["first"] == ["right"] for i in INITS
        )
        conditions[c] = group
    geometry = read(root / "geometry.json")
    right_centre = geometry["sites"]["right"]["centre"]
    comparisons = {}
    for c1,c2 in ((f"add:{top_group}", "early"), (f"undo:{top_group}", "late"),
                  (f"add:{top_group}", f"random_add:{top_group}"), (f"undo:{top_group}", f"random_replace:{top_group}"),
                  (f"random_replace:{top_group}", "late"), ("late", "early")):
        if c1 not in selected or c2 not in selected:
            continue
        by_goal = {}
        for g, target in TARGET.items():
            pairs = [(observed[c1,g,i], observed[c2,g,i]) for i in INITS]
            by_goal[target] = {"first_correct_difference": paired([int(a["first"] == [target]) - int(b["first"] == [target])
                                                                   for a,b in pairs]),
                               "terminal_difference": paired([int(a["terminal"][target]) - int(b["terminal"][target])
                                                              for a,b in pairs]),
                               "first_transitions": dict(Counter(f"{first_label(b)} -> {first_label(a)}" for a,b in pairs))}
            if g == "B" and all("trace" in a and "trace" in b for a,b in pairs):
                by_goal[target]["right_progress_60_cm"] = paired(
                    [100*(projected_progress(a,60,right_centre)-projected_progress(b,60,right_centre)) for a,b in pairs],
                    median
                )
        comparisons[f"{c1} vs {c2}"] = by_goal
    assert len(comparisons) == (4 if later else 6)
    return {"complete": True, "n": len(observed), "new_episodes": new_episodes, "conditions": conditions,
            "paired": comparisons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = {}
    for w in WINDOWS:
        if args.allow_partial and not (args.path / f"selection_{w}.json").exists():
            continue
        result[w] = {"probe": probe_summary(args.path,w),
                     "native": native_summary(args.path,w,not args.allow_partial)}
        file = args.path / f"attention_split_{w}.json"
        if file.exists():
            split = read(file)
            assert len(split["conditions"]) == 14
            for rows in split["conditions"].values():
                by_scene(rows)
            result[w]["attention_split"] = split["summary"]
        elif not args.allow_partial:
            raise FileNotFoundError(file)
    result["rotation_bridge"] = {}
    for step in (0,30,300):
        file = args.path / f"rotation_bridge_{step:06d}.jsonl"
        if args.allow_partial and not file.exists():
            continue
        data = [json.loads(line) for line in file.read_text().splitlines()]
        if not data and args.allow_partial:
            continue
        if not args.allow_partial:
            assert len(data) == len({r["offset"] for r in data}) == 8
        assert all(r["reproduced_B_losses_norms_and_probe"] for r in data)
        assert {r["group"] for r in data} == {result[WINDOWS[0]]["probe"]["selection"]["top_group"]}
        comparisons = {}
        for measure in ("gain_H","gain_all"):
            comparisons[measure] = {}
            for tag,a,b in (("full_minus_without","full","without_rotation"),
                            ("group_only","add_group_to_without","without_rotation"),
                            ("group_removal","full","remove_group_from_full")):
                values = [mean(r[measure][a])-mean(r[measure][b]) for r in data]
                comparisons[measure][tag] = {"median":median(values),"negative":sum(v < 0 for v in values),
                                               "positive":sum(v > 0 for v in values)}
        result["rotation_bridge"][step] = {"batches":len(data),"contrasts":comparisons}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
