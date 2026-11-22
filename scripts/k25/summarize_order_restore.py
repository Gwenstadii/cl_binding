"""Audit raw K25 curriculum rollouts, lock dev gains, and emit paired statistics."""

import argparse
from collections import Counter
from hashlib import sha256
from math import comb
import json
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    assert not path.exists(), f"Refusing to replace locked output: {path}"
    path.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf-8")


def primary(row):
    return "+".join(row["first"]) or "none"


def validate_endpoints(row):
    sites = {"left", "front", "right", "back"}
    assert set(row["times"]) == set(row["terminal"]) == set(row["ever"]) == sites
    assert set(row["first"]) <= sites and len(row["first"]) == len(set(row["first"]))
    assert all(t is None or (type(t) is int and 1 <= t <= 400) for t in row["times"].values())
    assert all(type(row[k][g]) is bool for k in ("terminal", "ever") for g in sites)
    assert all(not row["terminal"][g] or row["ever"][g] for g in sites)
    assert row["first_grasp"] is None or 1 <= row["first_grasp"] <= 400
    assert row["first_step"] == min((t for t in row["times"].values() if t is not None), default=None)
    assert row["first"] == [g for g, t in row["times"].items() if t is not None and t == row["first_step"]]


def group_stats(rows, labels):
    by_goal = {g: sorted([r for r in rows if r["goal"] == g], key=lambda r: r["init"]) for g in labels}
    result = {}
    for g, group in by_goal.items():
        if not group:
            continue
        place = labels[g]
        result[g] = dict(n=len(group), first_correct=sum(primary(r) == place for r in group),
            terminal_correct=sum(r["terminal"][place] for r in group), first=dict(Counter(primary(r) for r in group)),
            ever_correct=sum(r["ever"][place] for r in group), grasped=sum(r["first_grasp"] is not None for r in group),
            first_read=dict(Counter(r["decisions"][0]["decoded"] for r in group if r["decisions"])),
            all_blocks_read_correct=sum(bool(r["decisions"]) and all(d["decoded"] == g for d in r["decisions"]) for r in group))
        if place != labels["N"]:
            result[g]["first_10_lead_mm"] = median(r["first_10_approach_mm"][place]-r["first_10_approach_mm"][labels["N"]] for r in group)
    joint_first = joint_terminal = None
    if by_goal["A"] and by_goal["B"]:
        assert [r["init"] for r in by_goal["A"]] == [r["init"] for r in by_goal["B"]]
        joint_first = sum(all(primary(r) == labels[g] for g, r in zip("AB", pair))
                          for pair in zip(by_goal["A"], by_goal["B"], strict=True))
        joint_terminal = sum(all(r["terminal"][labels[g]] for g, r in zip("AB", pair))
                             for pair in zip(by_goal["A"], by_goal["B"], strict=True))
    return dict(per_goal=result, joint_first=joint_first, joint_terminal=joint_terminal)


def collect(base, suite, line, protocol):
    markers = sorted(base.glob(f"{suite}_{line}_*_COMPLETE.json"))
    assert markers, (suite, line)
    expected_ids = protocol["dev_ids" if suite == "dev" else "test_ids"]
    rows, found_ids, gate_records = [], [], []
    labels = dict(zip("ABN", protocol["lines"][line]["goals"], strict=True))
    for path in markers:
        m = read(path)
        lane, lanes = m["lane"], m["lanes"]
        assert lanes == len(markers)
        assert m["ids"] == expected_ids[lane::lanes]
        assert m["suite"] == suite and m["line"] == line
        assert m["protocol_sha256"] == digest(base / "protocol.json")
        assert m["source_sha256"] == protocol["source_sha256"]
        assert m["selection_sha256"] == (None if suite == "dev" else digest(base / "selection.json"))
        data_path = base / f"{suite}_{line}_{lane}.jsonl"
        assert digest(data_path) == m["rows_sha256"]
        data = [json.loads(s) for s in data_path.read_text().splitlines()]
        assert data and all(r["init"] in m["ids"] for r in data)
        for check in m["gates"]:
            assert check["single_model_roles"] == ["F"]
            assert all(check[k] for k in ("v_only_overlay_equal", "weights_restored", "mean_preserved_fp64", "new_goal_bypass_bitwise"))
            assert check["mean_cast_rms"] < .01
        gate_records.extend(m["gates"])
        rows.extend(data)
        found_ids.extend(m["ids"])
    assert sorted(found_ids) == expected_ids
    keys = {(r["variant"], r["arm"], r["gain"], r["goal"], r["init"]) for r in rows}
    assert len(keys) == len(rows)
    if suite == "dev":
        jobs = [("canon", "native", 0., "AB")]+[("canon", "forward", a, "AB") for a in protocol["dev_gains"]]
    else:
        gain = read(base / "selection.json")["selected"][protocol["lines"][line]["dev"]]["gain"]
        jobs = [("move" if suite == "phrase" else "canon", a, 0. if a == "native" else gain,
                 "ABN" if suite == "main" and a == "native" else "AB") for a in ("native", "forward", "reverse")]
        if suite == "main" and line == "lr_front_130" and gain != protocol["standard_gain"]:
            jobs.append(("canon", "forward", protocol["standard_gain"], "AB"))
    assert keys == {(v, a, gain, g, i) for v, a, gain, goals in jobs for g in goals for i in expected_ids}
    for r in rows:
        validate_endpoints(r)
        assert r["suite"] == suite and r["line"] == line and r["target"] == labels[r["goal"]]
        expected_text = (f"pick up the book and place it in the {r['target']} compartment of the caddy" if r["variant"] == "canon"
                         else protocol["variants"][r["variant"]].format(goal=r["target"]))
        assert r["text"] == expected_text
        assert r["first_step"] == min((t for t in r["times"].values() if t is not None), default=None)
        assert r["first"] == [g for g, t in r["times"].items() if t is not None and t == r["first_step"]]
        assert dict(Counter(d["decoded"] for d in r["decisions"])) == r["routed"]
        assert [d["block"] for d in r["decisions"]] == ([] if r["arm"] == "native" else list(range(8, 18)))
        for d in r["decisions"]:
            assert d["decoded"] == min("ABN", key=d["distance"].__getitem__)
            assert d["coefficient"] == {"A": 1., "B": -1., "N": 0.}[d["decoded"]]*r["gain"]*(1. if r["arm"] == "forward" else -1.)
            if r["variant"] == "canon":
                assert d["decoded"] == r["goal"] and d["distance"][r["goal"]] == 0.
    return rows, labels, gate_records


def select(base, check=False):
    protocol = read(base / "protocol.json")
    selected = {}
    for line in protocol["dev_lines"]:
        rows, labels, _ = collect(base, "dev", line, protocol)
        stats = {str(a): group_stats([r for r in rows if r["gain"] == a], labels) for a in (0., *protocol["dev_gains"])}
        def score(a):
            st = stats[str(a)]
            return (min(st["per_goal"][g]["first_correct"] for g in "AB"), st["joint_first"],
                    sum(st["per_goal"][g]["terminal_correct"] for g in "AB"), -a)
        gain = max(protocol["dev_gains"], key=score)
        selected[line] = dict(gain=gain, dev_stats=stats, selection_score=score(gain),
            left_no_first_loss=stats[str(gain)]["per_goal"]["A"]["first_correct"] >= stats[str(0.)]["per_goal"]["A"]["first_correct"])
    if check:
        stored = read(base / "selection.json")
        assert stored["selected"] == json.loads(json.dumps(selected)) and not stored["test_results_consulted"]
        assert stored["protocol_sha256"] == digest(base / "protocol.json")
        assert digest(base / "selection_source.py") == stored["source_sha256"]
    else:
        write(base / "selection.json", dict(protocol_sha256=digest(base / "protocol.json"), selected=selected,
                                             test_results_consulted=False, source_sha256=digest(Path(__file__))))
    return selected


def matched(before, after, place, new_goal):
    aa, bb = ({r["init"]: r for r in group} for group in (before, after))
    assert aa.keys() == bb.keys() and len(aa) == len(before) == len(after)
    ids = sorted(aa)
    def both(fn):
        return [i for i in ids if fn(aa[i], bb[i])]
    gains = both(lambda a, b: primary(a) != place and primary(b) == place)
    losses = both(lambda a, b: primary(a) == place and primary(b) != place)
    n = len(gains)+len(losses)
    exact_p = min(1., 2*sum(comb(n, k) for k in range(min(len(gains), len(losses))+1))/2**n) if n else 1.
    return dict(first_transitions=dict(Counter(f"{primary(aa[i])}->{primary(bb[i])}" for i in ids)),
        new_correct=gains, lost_correct=losses, risk_difference=(len(gains)-len(losses))/len(ids),
        mcnemar_exact_p_descriptive=exact_p, original_new_goal=both(lambda a, b: primary(a) == new_goal),
        new_goal_to_intended=both(lambda a, b: primary(a) == new_goal and primary(b) == place),
        new_terminal=both(lambda a, b: not a["terminal"][place] and b["terminal"][place]),
        lost_terminal=both(lambda a, b: a["terminal"][place] and not b["terminal"][place]),
        same_boundary=both(lambda a, b: a["boundary_signature"] == b["boundary_signature"]))


def collect_standard(base, protocol):
    plan = read(base / "standard_protocol.json")
    assert plan["base_protocol_sha256"] == digest(base / "protocol.json")
    assert plan["selection_sha256"] == digest(base / "selection.json")
    assert not plan["test_results_used_to_choose_dose"]
    assert plan["gain"] == protocol["standard_gain"] and plan["line"] == "lf_right_1000"
    assert plan["ids"] == protocol["test_ids"]
    markers = sorted(base.glob("standard_*_COMPLETE.json"))
    assert markers
    rows, ids, gates = [], [], []
    labels = dict(zip("ABN", protocol["lines"][plan["line"]]["goals"], strict=True))
    for path in markers:
        m = read(path)
        assert m["lanes"] == len(markers) and m["ids"] == plan["ids"][m["lane"]::m["lanes"]]
        assert m["source_sha256"] == plan["source_sha256"]
        assert m["protocol_sha256"] == digest(base / "standard_protocol.json")
        source = base / f"standard_{m['lane']}.jsonl"
        assert digest(source) == m["rows_sha256"]
        data = [json.loads(s) for s in source.read_text().splitlines()]
        assert all(r["init"] in m["ids"] for r in data)
        check = m["gates"]
        assert check["single_model_roles"] == ["F"] and check["mean_cast_rms"] < .01
        assert all(check[k] for k in ("v_only_overlay_equal", "weights_restored", "mean_preserved_fp64", "new_goal_bypass_bitwise"))
        gates.append(check)
        rows.extend(data)
        ids.extend(m["ids"])
    assert sorted(ids) == plan["ids"]
    assert len(rows) == 2*len(ids)
    assert {(r["init"], r["goal"]) for r in rows} == {(i, g) for i in ids for g in "AB"}
    for r in rows:
        validate_endpoints(r)
        assert (r["suite"], r["line"], r["variant"], r["arm"], r["gain"]) == ("standard", plan["line"], "canon", "forward", plan["gain"])
        assert r["target"] == labels[r["goal"]]
        assert r["text"] == f"pick up the book and place it in the {r['target']} compartment of the caddy"
        assert [d["block"] for d in r["decisions"]] == list(range(8, 18))
        assert all(d["decoded"] == r["goal"] and d["coefficient"] == (1 if r["goal"] == "A" else -1)*plan["gain"] for d in r["decisions"])
        assert all(d["distance"][r["goal"]] == 0. and d["decoded"] == min("ABN", key=d["distance"].__getitem__) for d in r["decisions"])
        assert dict(Counter(d["decoded"] for d in r["decisions"])) == r["routed"]
        assert r["first_step"] == min((t for t in r["times"].values() if t is not None), default=None)
        assert r["first"] == [g for g, t in r["times"].items() if t is not None and t == r["first_step"]]
    return rows, labels, gates


def collect_parent(base, protocol):
    markers = sorted(base.glob("parent_*_COMPLETE.json"))
    assert markers
    rows, ids, sources = [], [], set()
    expected_sha = protocol["checkpoints"]["lf_right_1000"]["parent_sha256"]
    labels = dict(zip("ABN", protocol["lines"]["lf_right_1000"]["goals"], strict=True))
    for path in markers:
        m = read(path)
        assert m["lanes"] == len(markers) and m["ids"] == protocol["test_ids"][m["lane"]::m["lanes"]]
        assert m["base_protocol_sha256"] == digest(base / "protocol.json")
        assert m["parent_sha256"] == expected_sha and m["base_source_sha256"] == protocol["source_sha256"]
        source = base / f"parent_{m['lane']}.jsonl"
        assert digest(source) == m["rows_sha256"]
        data = [json.loads(s) for s in source.read_text().splitlines()]
        assert all(r["init"] in m["ids"] for r in data)
        rows.extend(data)
        ids.extend(m["ids"])
        sources.add(m["source_sha256"])
    assert len(sources) == 1 and sorted(ids) == protocol["test_ids"]
    assert len(rows) == len(ids)*2
    assert {(r["init"], r["goal"]) for r in rows} == {(i, g) for i in ids for g in "AB"}
    for r in rows:
        validate_endpoints(r)
        assert (r["suite"], r["line"], r["variant"], r["arm"], r["gain"]) == ("parent", "lf_parent", "canon", "native", 0.)
        assert r["target"] == labels[r["goal"]] and not r["decisions"] and not r["routed"]
        assert r["text"] == f"pick up the book and place it in the {r['target']} compartment of the caddy"
        assert r["first_step"] == min((t for t in r["times"].values() if t is not None), default=None)
        assert r["first"] == [g for g, t in r["times"].items() if t is not None and t == r["first_step"]]
    return rows, labels


def reserve_summary(base, protocol):
    plan = read(base / "reserve_protocol.json")
    assert plan["base_protocol_sha256"] == digest(base / "protocol.json")
    assert plan["selection_sha256"] == digest(base / "selection.json")
    assert plan["cases_sha256"] == digest(base / "reserve_cases.json")
    assert plan["base_source_sha256"] == protocol["source_sha256"]
    assert plan["designed_after_primary_set_left_losses"] and not plan["reserve_results_consulted"]
    assert plan["ids"] == list(range(900, 920)) and plan["line"] == "lr_front_1000"
    assert plan["gain"] == min(plan["eligible_development_gains"]) == 1.
    cases = read(base / "reserve_cases.json")
    assert [r["init"] for r in cases] == plan["ids"]
    assert len({r["signature"]["sim"] for r in cases}) == 20
    assert all(r["min_distance_to_dev_and_primary"] > 1e-6 for r in cases)
    markers = sorted(base.glob("reserve_*_COMPLETE.json"))
    assert markers
    rows, ids, gates = [], [], []
    labels = dict(zip("ABN", protocol["lines"][plan["line"]]["goals"], strict=True))
    for path in markers:
        m = read(path)
        assert m["lanes"] == len(markers) and m["ids"] == plan["ids"][m["lane"]::m["lanes"]]
        assert m["source_sha256"] == plan["source_sha256"] and m["protocol_sha256"] == digest(base / "reserve_protocol.json")
        source = base / f"reserve_{m['lane']}.jsonl"
        assert digest(source) == m["rows_sha256"]
        data = [json.loads(s) for s in source.read_text().splitlines()]
        assert all(r["init"] in m["ids"] for r in data)
        for check in m["gates"]:
            assert check["single_model_roles"] == ["F"] and check["mean_cast_rms"] < .01
            assert all(check[k] for k in ("v_only_overlay_equal", "weights_restored", "mean_preserved_fp64", "new_goal_bypass_bitwise"))
        gates.extend(m["gates"])
        rows.extend(data)
        ids.extend(m["ids"])
    assert sorted(ids) == plan["ids"]
    assert len(rows) == 160 and len({(r["init"], r["goal"], r["arm"], r["gain"]) for r in rows}) == 160
    assert {(r["init"], r["goal"], r["arm"], r["gain"]) for r in rows} == {
        (i, g, a, gain) for i in ids for g in "AB" for a, gain in plan["jobs"]}
    for r in rows:
        validate_endpoints(r)
        assert r["suite"] == "reserve" and r["line"] == plan["line"] and r["variant"] == "canon"
        assert r["target"] == labels[r["goal"]]
        assert r["text"] == f"pick up the book and place it in the {r['target']} compartment of the caddy"
        assert [d["block"] for d in r["decisions"]] == ([] if r["arm"] == "native" else list(range(8, 18)))
        assert dict(Counter(d["decoded"] for d in r["decisions"])) == r["routed"]
        for d in r["decisions"]:
            assert d["decoded"] == r["goal"] and d["distance"][r["goal"]] == 0.
            assert d["decoded"] == min("ABN", key=d["distance"].__getitem__)
            assert d["coefficient"] == (1 if r["goal"] == "A" else -1)*r["gain"]*(1 if r["arm"] == "forward" else -1)
        assert r["first_step"] == min((t for t in r["times"].values() if t is not None), default=None)
        assert r["first"] == [g for g, t in r["times"].items() if t is not None and t == r["first_step"]]
    subsets = {name: [r for r in rows if r["arm"] == arm and r["gain"] == gain] for name, arm, gain in (
        ("native", "native", 0.), ("forward", "forward", plan["gain"]), ("reverse", "reverse", plan["gain"]),
        ("original_gain", "forward", plan["original_gain"]))}
    out = {name: group_stats(data, labels) for name, data in subsets.items()}
    out["paired"] = {}
    for after, before in (("forward", "native"), ("reverse", "forward"), ("forward", "original_gain")):
        out["paired"][after+"_vs_"+before] = {}
        for g in "AB":
            a, b = ([r for r in subsets[name] if r["goal"] == g] for name in (before, after))
            result = matched(a, b, labels[g], labels["N"])
            assert len(result["same_boundary"]) == 20
            if after == "reverse":
                aa, bb = ({r["init"]: r for r in data} for data in (a, b))
                assert all(aa[i]["decisions"][0]["distance"] == bb[i]["decisions"][0]["distance"] for i in aa)
                assert all(aa[i]["decisions"][0]["coefficient"] == -bb[i]["decisions"][0]["coefficient"] for i in aa)
                result["same_first_read"] = 20
            out["paired"][after+"_vs_"+before][g] = result
    return out, gates


def summarize(base, check=False):
    protocol = read(base / "protocol.json")
    selection = read(base / "selection.json")
    assert selection["protocol_sha256"] == digest(base / "protocol.json")
    assert not selection["test_results_consulted"]
    report = dict(protocol_sha256=digest(base / "protocol.json"), selection_sha256=digest(base / "selection.json"),
                  analysis_source_sha256=digest(Path(__file__)),
                  lines={}, dev_episodes=0, main_episodes=0, phrase_episodes=0, standard_episodes=0, parent_episodes=0,
                  reserve_episodes=0, gates=[])
    for line in protocol["dev_lines"]:
        data, _, gates = collect(base, "dev", line, protocol)
        report["dev_episodes"] += len(data)
        report["gates"].extend(gates)
    for line in protocol["lines"]:
        outputs = {}
        gain = selection["selected"][protocol["lines"][line]["dev"]]["gain"]
        for suite in (["main", "phrase"] if line in protocol["phrase_lines"] else ["main"]):
            rows, labels, gates = collect(base, suite, line, protocol)
            report[suite+"_episodes"] += len(rows)
            report["gates"].extend(gates)
            subsets = {a: [r for r in rows if r["arm"] == a and (a == "native" or r["gain"] == gain)] for a in ("native", "forward", "reverse")}
            output = {a: group_stats(data, labels) for a, data in subsets.items()}
            if any(r["gain"] != gain for r in rows if r["arm"] == "forward"):
                output["standard_forward"] = group_stats([r for r in rows if r["arm"] == "forward" and r["gain"] == protocol["standard_gain"]], labels)
            output["paired"] = {}
            for arm, before in (("forward", "native"), ("reverse", "forward")):
                output["paired"][arm+"_vs_"+before] = {}
                for g in "AB":
                    a, b = ([r for r in subsets[name] if r["goal"] == g] for name in (before, arm))
                    result = matched(a, b, labels[g], labels["N"])
                    assert len(result["same_boundary"]) == len(protocol["test_ids"])
                    if arm == "reverse":
                        aa, bb = ({r["init"]: r for r in data} for data in (a, b))
                        assert all(aa[i]["decisions"][0]["distance"] == bb[i]["decisions"][0]["distance"] for i in aa)
                        assert all(aa[i]["decisions"][0]["coefficient"] == -bb[i]["decisions"][0]["coefficient"] for i in aa)
                        result["same_first_read"] = len(aa)
                    output["paired"][arm+"_vs_"+before][g] = result
            if "standard_forward" in output:
                std = [r for r in rows if r["arm"] == "forward" and r["gain"] == protocol["standard_gain"]]
                output["paired"]["standard_forward_vs_native"] = {
                    g: matched([r for r in subsets["native"] if r["goal"] == g], [r for r in std if r["goal"] == g], labels[g], labels["N"])
                    for g in "AB"}
            outputs[suite] = output
        report["lines"][line] = dict(goals=protocol["lines"][line]["goals"], gain=gain, outcomes=outputs)
    standard, labels, gates = collect_standard(base, protocol)
    native, _, _ = collect(base, "main", "lf_right_1000", protocol)
    native = [r for r in native if r["arm"] == "native"]
    report["standard_episodes"] = len(standard)
    report["gates"].extend(gates)
    output = report["lines"]["lf_right_1000"]["outcomes"]["main"]
    output["standard_forward"] = group_stats(standard, labels)
    output["paired"]["standard_forward_vs_native"] = {
        g: matched([r for r in native if r["goal"] == g], [r for r in standard if r["goal"] == g], labels[g], labels["N"]) for g in "AB"}
    report["standard_protocol_sha256"] = digest(base / "standard_protocol.json")
    parent, labels = collect_parent(base, protocol)
    report["parent_episodes"] = len(parent)
    report["lf_parent"] = group_stats(parent, labels)
    report["lf_adaptation_vs_parent"] = {
        g: matched([r for r in parent if r["goal"] == g], [r for r in native if r["goal"] == g], labels[g], labels["N"]) for g in "AB"}
    report["reserve_outcomes"], gates = reserve_summary(base, protocol)
    report["gates"].extend(gates)
    report["reserve_episodes"] = 160
    report["reserve_protocol_sha256"] = digest(base / "reserve_protocol.json")
    report["total_episodes"] = sum(report[k] for k in ("dev_episodes", "main_episodes", "phrase_episodes", "standard_episodes", "parent_episodes", "reserve_episodes"))
    report["max_mean_cast_rms"] = max(g["mean_cast_rms"] for g in report["gates"])
    if check:
        assert read(base / "summary.json") == report, "Recomputed statistics differ from the archived summary"
    else:
        write(base / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("select", "summary"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--check", action="store_true", help="recompute and compare without replacing locked output")
    args = parser.parse_args()
    result = select(args.folder, args.check) if args.mode == "select" else summarize(args.folder, args.check)
    print(json.dumps(result if args.mode == "select" else {k: v for k, v in result.items() if k not in ("lines", "gates")}, ensure_ascii=False, indent=2))
