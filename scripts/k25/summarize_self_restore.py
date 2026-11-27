"""Validate the K25-final experimental grid and compare matched physical starts."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import median


IDS = tuple(range(600, 620))
TARGET = {"A": "left", "B": "right", "N": "front"}
SIGN = {"A": 1., "B": -1., "N": 0.}
SITES = ("left", "front", "right", "back")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def primary(row):
    return "+".join(row["first"]) or "none"


def key(row):
    return (row["phase"], row["variant"], row["arm"], row["goal"], row["init"])


def coeff(arm, method):
    return {"v_forward": method["gain"], "v_reverse": -method["gain"], "v_natural": 1.,
            "extrap_d": 1., "own_d": method["own_d_gain"]}[arm]


def validate_row(row, method):
    g = row["goal"]
    start, end = (row["boundary"] + n for n in method["control_offsets"][row["phase"]])
    assert row["window"] == [start, end]
    assert (row["phase"] == "grip") == bool(row["boundary"])
    variant = row["variant"]
    text = method["candidate_prompts"][g] if variant == "canon" else method["variants"][variant].format(goal=TARGET[g])
    assert row["text"] == text
    assert set(row["terminal"]) == set(row["ever"]) == set(row["times"]) == set(SITES)
    assert set(row["first"]) <= set(SITES)
    vals = [v for v in row["times"].values() if v is not None]
    assert row["first_step"] == min(vals, default=None)
    if row["first_step"] is None:
        assert not row["first"]
    else:
        assert set(row["first"]) == {p for p,t in row["times"].items() if t == row["first_step"]}
    reads = row["decisions"]
    arm = row["arm"]
    assert [r["block"] for r in reads] == ([] if arm == "native" else list(range(start//10,end//10)))
    assert dict(Counter(r["decoded"] for r in reads)) == row["routed"]
    for r in reads:
        assert r["decoded"] == min(TARGET, key=r["distance"].__getitem__)
        assert r["coefficient"] == SIGN[r["decoded"]]*coeff(arm, method)
        if variant == "canon":
            assert r["decoded"] == g and r["distance"][g] == 0.
        if not r["coefficient"]:
            assert r["contrast_rms"] == 0.


def read_stage(base, stage, method, cases):
    paths = sorted(base.glob(f"{stage}_*_COMPLETE.json"))
    assert paths, stage
    found, rows, gates = [], [], []
    source_sha = method["source_sha256"]
    for path in paths:
        m = read(path)
        assert m["suite"] == stage and m["lanes"] == len(paths)
        lane = m["lane"]
        assert m["ids"] == list(cases[lane::m["lanes"]])
        assert m["method_sha256"] == digest(base / "method.json") and m["source_sha256"] == source_sha
        found.extend(m["ids"])
        gate = m["gates"]
        assert gate["single_model_roles"] == ["F"] and gate["init"] == m["ids"][0]
        assert gate["canonical_native_bitwise_if_zero"] and gate["canonical_front_bypass_bitwise"]
        assert gate["canonical_pair_mean_fp64_preserved_before_cast"] and gate["pair_mean_cast_max_rms"] < .01
        assert gate["overlay_equals_v_only_weight_change"] == [1., method["gain"]]
        gates.append(gate)
        source = base / f"{stage}_{lane}.{'json' if stage == 'probe' else 'jsonl'}"
        data = read(source) if stage == "probe" else lines(source)
        assert all(r["init"] in m["ids"] for r in data)
        rows.extend(data)
    assert sorted(found) == list(cases)
    return rows, gates


def per_goal(rows):
    g = rows[0]["goal"]
    place = TARGET[g]
    docs = {"n": len(rows), "first": dict(Counter(primary(r) for r in rows)),
            "first_correct": sum(primary(r) == place for r in rows),
            "first_correct_by_window_end": sum(r["first_in_window"] == [place] for r in rows),
            "ever_success": sum(r["ever"][place] for r in rows),
            "terminal_correct": sum(r["terminal"][place] for r in rows),
            "terminal_by_goal": {p: sum(r["terminal"][p] for r in rows) for p in SITES},
            "ever_by_goal": {p: sum(r["ever"][p] for r in rows) for p in SITES},
            "grasp_confirmed": sum(r["first_grasp"] is not None for r in rows),
            "initial_read": dict(Counter(r["decisions"][0]["decoded"] for r in rows if r["decisions"])),
            "all_blocks_read_correct": sum(set(r["routed"]) == {g} for r in rows)}
    if place != "front":
        docs["first_10_lead_mm"] = median(r["first_10_approach_mm"][place]-r["first_10_approach_mm"]["front"] for r in rows)
    return docs


def compare(before, after):
    place = TARGET[after[0]["goal"]]
    assert [(r["init"], r["text"]) for r in before] == [(r["init"], r["text"]) for r in after]
    pairs = tuple(zip(before, after, strict=True))

    def matching(fn):
        return [a["init"] for a,b in pairs if fn(a,b)]

    result = {"transitions": dict(Counter(f"{primary(a)}->{primary(b)}" for a,b in pairs)),
              "old_front_cases": matching(lambda a,b: primary(a) == "front"),
              "front_to_intended": matching(lambda a,b: primary(a) == "front" and primary(b) == place),
              "new_first_correct": matching(lambda a,b: primary(a) != place and primary(b) == place),
              "lost_first_correct": matching(lambda a,b: primary(a) == place and primary(b) != place),
              "new_terminal_correct": matching(lambda a,b: not a["terminal"][place] and b["terminal"][place]),
              "lost_terminal_correct": matching(lambda a,b: a["terminal"][place] and not b["terminal"][place]),
              "same_first_and_terminal": matching(lambda a,b: a["first"] == b["first"] and a["terminal"] == b["terminal"]),
              "same_first_time_and_terminal": matching(lambda a,b: a["times"] == b["times"] and a["terminal"] == b["terminal"])}
    if place != "front" and all(a["window"][0] == b["window"][0] for a,b in pairs):
        def lead(row):
            distances = row["first_10_approach_mm"]
            return distances[place] - distances["front"]
        changes = [lead(b)-lead(a) for a,b in pairs]
        result["first_10_lead_change_median_mm"] = median(changes)
        result["first_10_lead_improved_n"] = sum(x > 0 for x in changes)
    return result


def describe_grid(rows, plan):
    results = {key(r): r for r in rows}
    assert keys(plan) == set(results) and len(results) == len(rows)
    stats = {}
    for phase, variants in plan.items():
        stats[phase] = {}
        for variant, modes in variants.items():
            stats[phase][variant] = {}
            for arm, goals in modes.items():
                by_goal = {g: [results[phase,variant,arm,g,i] for i in IDS] for g in goals}
                out = {"per_goal": {g: per_goal(rr) for g,rr in by_goal.items()}}
                for names in ("AB", "ABN"):
                    if not set(names) <= set(goals):
                        continue
                    out[f"joint_first_{names}"] = sum(all(primary(by_goal[g][j]) == TARGET[g] for g in names) for j in range(len(IDS)))
                    out[f"joint_terminal_{names}"] = sum(all(by_goal[g][j]["terminal"][TARGET[g]] for g in names) for j in range(len(IDS)))
                if arm != "native":
                    native_phase = "reset" if phase == "late" else phase
                    out["vs_native"] = {g: compare([results[native_phase,variant,"native",g,i] for i in IDS], rr)
                                        for g,rr in by_goal.items()}
                stats[phase][variant][arm] = out
            for g in set(modes["v_forward"]) & set(modes["v_reverse"]):
                forward_rows = [results[phase,variant,"v_forward",g,i] for i in IDS]
                reverse_rows = [results[phase,variant,"v_reverse",g,i] for i in IDS]
                stats[phase][variant]["v_reverse"].setdefault("vs_forward", {})[g] = compare(
                    forward_rows, reverse_rows)
                for i in IDS:
                    forward, reverse = (results[phase,variant,a,g,i]["decisions"][0] for a in ("v_forward","v_reverse"))
                    assert forward["distance"] == reverse["distance"]
                    assert forward["coefficient"] == -reverse["coefficient"]
    return stats, results


def keys(plan):
    return {(phase, variant, arm, g, i) for phase,vs in plan.items() for variant,arms in vs.items()
            for arm,goals in arms.items() for g in goals for i in IDS}


def probes(rows, method):
    expected = {(p,i,j) for p in ("grip","reset") for i in IDS for j in range(3)}
    assert len(rows) == len(expected) and {(r["phase"],r["init"],r["noise_offset"]) for r in rows} == expected
    variants = ("canon", *method["variants"])
    out = {}
    for phase in ("reset","grip"):
        block = [r for r in rows if r["phase"] == phase]
        out[phase] = {}
        for v in variants:
            out[phase][v] = {
                g: {"noise_zero": dict(Counter(r["scores"][v][g]["decoded"] for r in block if r["noise_offset"] == 0)),
                    "all_noise": dict(Counter(r["scores"][v][g]["decoded"] for r in block))} for g in TARGET}
    return out


def archive_agreement(base, results):
    if base is None:
        return {}
    previous = {(r["arm"],r["goal"],r["init"]):r for p in base.glob("third_*.jsonl") for r in lines(p)}
    arms = {"native": "baseline", "v_natural": "natural", "v_forward": "amplified", "v_reverse": "reverse",
            "own_d": "residual_d", "extrap_d": "extrap_d"}
    agreement = {}
    for new, old in arms.items():
        by = {}
        for goal in "AB":
            paired = [(results["grip","canon",new,goal,i], previous[old,goal,i]) for i in IDS if (old,goal,i) in previous]
            if paired:
                assert len(paired) == len(IDS)
                by[goal] = {"same_first": sum(a["first"] == b["first"] for a,b in paired),
                            "same_terminal": sum(a["terminal"] == b["terminal"] for a,b in paired), "n": len(paired)}
        agreement[new] = by
    return agreement


def summarize(base, archive):
    method = read(base / "method.json")
    rows, gates = [], []
    for stage in ("main", "late", "probe"):
        data, checks = read_stage(base, stage, method, IDS)
        gates.extend(checks)
        if stage == "probe":
            fixed = probes(data,method)
        else:
            for row in data:
                validate_row(row, method)
            rows.extend(data)
    phases, lookup = describe_grid(rows, method["plan"])
    return {"rollouts": len(rows), "fixed_inputs": len(data), "method_sha256": digest(base / "method.json"),
            "max_mean_cast_rms": max(x["pair_mean_cast_max_rms"] for x in gates), "gates": gates,
            "phases": phases, "fixed_readout": fixed, "archive_agreement": archive_agreement(archive,lookup)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    report = summarize(args.folder,args.archive)
    result = args.folder / "summary.json"
    result.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Verified {report['rollouts']} rollouts and {report['fixed_inputs']} fixed inputs in {result}")
