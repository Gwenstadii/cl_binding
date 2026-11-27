"""Validate the paired K26 rollouts and summarize goal changes and readout accuracy."""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
from statistics import median


OLD = ("left", "front", "right")
ALL = (*OLD, "back")
IDS = tuple(range(600, 620))
DEV_IDS = tuple(range(630, 640))
PAIRS = tuple(combinations(OLD, 2))
WINDOW = {"reset": 150, "grip": 60}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]


def expected(phase):
    for mode in ("parent", "native"):
        yield mode, ALL
    for mode in ("readout", "oracle"):
        yield mode, OLD
    if phase == "grip":
        for mode in ("readout_step", "rotated", "readout_alt"):
            yield mode, OLD
        for pair in PAIRS:
            yield "pair", pair
        for mode in ("native_alt", "mixed_alt"):
            yield mode, ALL


def add_rows(rows, results):
    for row in rows:
        key = (row["phase"], row["mode"], tuple(row["pool"]), row["goal"], row["init"])
        assert key not in results, key
        assert set(row["terminal"]) == set(ALL) == set(row["ever"]) == set(row["times"])
        assert row["goal"] in row["pool"] and set(row["first"]) <= set(ALL)
        assert row["window_end"] == row["boundary"] + WINDOW[row["phase"]]
        assert (row["boundary"] == 0) == (row["phase"] == "reset")
        counts = Counter(d["decoded"] for d in row["decisions"])
        assert dict(counts) == row["decoded_counts"]
        mode = row["mode"]
        decisions = row["decisions"]
        n = 0 if mode in ("parent", "oracle") or mode.startswith("native") else WINDOW[row["phase"]]//10
        if mode == "readout_step":
            n *= 10
        assert len(decisions) == n, key
        for d in decisions:
            pool = row["pool"]
            idx = max(range(len(pool)), key=d["scores"].__getitem__)
            assert pool[idx] == d["decoded"]
            assert pool[(idx+int(mode == "rotated")) % len(pool)] == d["selected"]
        results[key] = row


def first(row):
    return "+".join(row["first"]) if row["first"] else "none"


def describe(rows):
    goal = rows[0]["goal"]
    return {
        "n": len(rows),
        "first": dict(Counter(first(r) for r in rows)),
        "first_correct": sum(first(r) == goal for r in rows),
        "first_correct_in_window": sum(r["first_in_window"] == [goal] for r in rows),
        "ever_correct": sum(r["ever"][goal] for r in rows),
        "terminal_correct": sum(r["terminal"][goal] for r in rows),
        "ever_by_target": {g: sum(r["ever"][g] for r in rows) for g in ALL},
        "terminal_by_target": {g: sum(r["terminal"][g] for r in rows) for g in ALL},
        "reached_by_target": {g: sum(r["times"][g] is not None for r in rows) for g in ALL},
        "front_then_back": sum(r["first"] == ["front"] and r["times"]["back"] is not None for r in rows),
        "first_decoded": dict(Counter(r["decisions"][0]["decoded"] for r in rows if r["decisions"])),
        "majority_decoded": dict(Counter(max(r["decoded_counts"], key=r["decoded_counts"].get)
                                          for r in rows if r["decoded_counts"])),
        "any_route_matches": sum(goal in r["decoded_counts"] for r in rows),
        "all_routes_match": sum(set(r["decoded_counts"]) == {goal} for r in rows),
        "first_step_median_correct": median_steps([r["first_step"] for r in rows if first(r) == goal]),
    }


def median_steps(steps):
    return median(steps) if steps else None


def changes(before, after):
    assert len(before) == len(after)
    assert all(x["goal"] == y["goal"] and x["init"] == y["init"] for x,y in zip(before, after, strict=True))
    pairs = tuple(zip(before, after, strict=True))
    goal = before[0]["goal"]

    def convert(match):
        return [x["init"] for x, y in pairs if match(first(x), first(y))]

    return {"from_back": [x["init"] for x in before if first(x) == "back"],
            "back_to_goal": convert(lambda x,y: x == "back" and y == goal),
            "back_to_other": convert(lambda x,y: x == "back" and y not in ("back", "none", goal)),
            "back_to_none": convert(lambda x,y: x == "back" and y == "none"),
            "from_wrong_goal": [x["init"] for x in before if first(x) in ALL and first(x) != goal],
            "wrong_goal_to_correct": convert(lambda x,y: x in ALL and x != goal and y == goal),
            "new_first_correct": convert(lambda x,y: x != goal and y == goal),
            "lost_first_correct": convert(lambda x,y: x == goal and y != goal),
            "first_transitions": dict(Counter(f"{first(x)}->{first(y)}" for x,y in pairs))}


def jointly(by_goal, goals, terminal=False):
    count = len(by_goal[goals[0]])
    if terminal:
        return sum(all(by_goal[g][j]["terminal"][g] for g in goals) for j in range(count))
    return sum(all(by_goal[g][j]["first"] == [g] for g in goals) for j in range(count))


def load_stage(base, stage, ids, results):
    markers = sorted(base.glob(f"{stage}_*_COMPLETE.json"))
    assert markers, stage
    entire, info = [], []
    for path in markers:
        marker = read(path)
        lane, lanes = marker["lane"], marker["lanes"]
        assert marker["stage"] == stage and len(markers) == lanes and marker["ids"] == list(ids[lane::lanes])
        entire.extend(marker["ids"])
        info.append(marker)
        if stage != "probe":
            add_rows(lines(base / f"{stage}_{lane}.jsonl"), results)
    assert sorted(entire) == list(ids), stage
    assert len({(m["cases_sha256"], tuple(sorted(m["sha256"].items()))) for m in info}) == 1
    return info


def summarize_phases(results, plans, ids):
    exact = {(p, mode, pool, g, i) for p, modes in plans.items() for mode, pool in modes for g in pool for i in ids}
    assert results.keys() == exact, (len(results), sorted(exact - results.keys())[:10])
    phases = {}
    for phase, plan in plans.items():
        modes = {(mode, pool): {g: [results[phase, mode, pool, g, i] for i in ids] for g in pool}
                 for mode, pool in plan}
        grouped = {}
        for (mode, pool), per_goal in modes.items():
            name = "pair_" + "_".join(pool) if mode == "pair" else mode
            grouped[name] = summary = {"per_goal": {g: describe(rows) for g, rows in per_goal.items()}}
            joint_pools = [s for s in (*PAIRS, OLD, ALL) if set(s) <= set(pool)]
            summary["joint_first"] = {"/".join(s): jointly(per_goal, s) for s in joint_pools}
            summary["joint_terminal"] = {"/".join(s): jointly(per_goal, s, True) for s in joint_pools}
            if pool == ALL:
                summary["old_joint_first_and_back_terminal"] = sum(
                    all(per_goal[g][i]["first"] == [g] for g in OLD) and per_goal["back"][i]["terminal"]["back"]
                    for i in range(len(ids)))
            if mode != "parent" and not mode.startswith("native"):
                baseline = "native_alt2" if mode.endswith("_alt2") else "native_alt" if mode.endswith("_alt") else "native"
                before_by_goal = modes[baseline, ALL]
                summary["changes"] = {g: changes(before_by_goal[g], rows) for g, rows in per_goal.items()}
        phases[phase] = grouped
    return phases


def summarize(base):
    results = {}
    metadata = {stage: load_stage(base, stage, IDS, results) for stage in ("parent", "main", "probe", "alt2")}
    assert len({m["cases_sha256"] for ms in metadata.values() for m in ms}) == 1
    assert len({m["sha256"]["R"] for ms in metadata.values() for m in ms}) == 1
    assert len({m["sha256"]["F"] for n, ms in metadata.items() if n != "parent" for m in ms}) == 1
    if (base / "back_READY.json").is_file():
        training = read(base / "back_READY.json")
        assert training["step"] == 3000 and training["sha256"] == metadata["main"][0]["sha256"]["F"]
    plans = {p: tuple(expected(p)) for p in WINDOW}
    plans["grip"] += (("native_alt2", ALL), ("readout_alt2", OLD), ("mixed_alt2", ALL))
    main = metadata["main"][0]
    extension = back_extension(base, main)
    diagnostic_rows = sum(r["rows"] for r in extension["development"].values())
    validation_rows = sum(r["rows"] for r in extension["verification"].values())
    return {"rows": len(results), "total_rollouts": len(results)+diagnostic_rows+validation_rows,
            "ids": IDS, "models": main["sha256"],
            "cases_sha256": main["cases_sha256"], "phases": summarize_phases(results, plans, IDS),
            "probe": probe_summary(base, metadata["probe"]), "back_extension": extension}


def back_extension(base, main):
    diagnostic, validation = {}, {}
    dev_cases = set()
    for step in (3000, 6000, 9000):
        stage = f"back_dev_{step}"
        if not any(base.glob(f"{stage}_*_COMPLETE.json")):
            continue
        results = {}
        meta = load_stage(base, stage, DEV_IDS, results)
        ready = read(base / ("back_READY.json" if step == 3000 else f"long_{step}_READY.json"))
        assert ready["step"] == step and ready["parent_sha256"] == main["sha256"]["R"]
        assert meta[0]["sha256"] == {"R": ready["parent_sha256"], "F": ready["sha256"]}
        dev_cases.add(meta[0]["cases_sha256"])
        plans = dict.fromkeys(WINDOW, (("native", ("back",)),))
        diagnostic[step] = {"rows": len(results), "checkpoint": meta[0]["sha256"],
                            "phases": summarize_phases(results, plans, DEV_IDS)}
    assert len(dev_cases) <= 1
    for step in (6000, 9000):
        stage = f"long_{step}"
        if not any(base.glob(f"{stage}_*_COMPLETE.json")):
            continue
        results = {}
        meta = load_stage(base, stage, IDS, results)
        assert meta[0]["cases_sha256"] == main["cases_sha256"]
        assert meta[0]["sha256"] == diagnostic[step]["checkpoint"]
        plans = {"reset": (("native", ("back",)),),
                 "grip": (("native", ALL), ("readout", OLD), ("oracle", OLD), ("native_alt2", ALL), ("mixed_alt2", ALL))}
        validation[step] = {"rows": len(results), "checkpoint": meta[0]["sha256"],
                           "phases": summarize_phases(results, plans, IDS)}
    return {"development": diagnostic, "verification": validation}


def probe_summary(base, markers):
    goals = {"parent3": OLD, "late3": OLD, "late4": ALL, **{"_".join(p): p for p in PAIRS}}
    output, seen = {}, set()
    for m in markers:
        for r in read(base / f"probe_{m['lane']}.json"):
            identity = (r["phase"], r["denoise_index"], r["template"], r["init"])
            assert identity not in seen and r["init"] in m["ids"]
            seen.add(identity)
            agg = output.setdefault("/".join(map(str, identity[:3])), {})
            assert r["scores"].keys() == goals.keys()
            for method, per in r["scores"].items():
                pool = goals[method]
                assert tuple(per) == pool
                per_goal = agg.setdefault(method, {})
                for actual, values in per.items():
                    selected = pool[max(range(len(pool)), key=values.__getitem__)]
                    counter = per_goal.setdefault(actual, Counter())
                    counter[selected] += 1
    assert seen == {(p,s,t,i) for p in WINDOW for s in (0,4,8) for t in range(3) for i in IDS}
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    result = summarize(args.results)
    dest = args.results / "summary.json"
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"{result['total_rollouts']} rollouts verified and summarized in {dest}")
