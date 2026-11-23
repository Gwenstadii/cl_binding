"""Recompute every endpoint from dense traces and audit timing-only contrasts."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import median
import struct

import v_timing_protocol as spec


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def sig(obj):
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def same_prefix(a, b, blocks):
    assert a["signatures"][:blocks+1] == b["signatures"][:blocks+1]
    assert a["action_hashes"][:blocks] == b["action_hashes"][:blocks]
    for field in ("object", "eef", "contacts", "success", "regions", "actions"):
        assert a["trace"][field][:blocks*10] == b["trace"][field][:blocks*10], field


def validate(row, protocol, case):
    line, goal, window, arm = (row[k] for k in ("line", "goal", "window", "arm"))
    assert line in spec.LINES and goal in "AB" and (window, arm) in spec.jobs()
    target = spec.GOALS[line][goal]
    assert row["target"] == target and row["seed"] == case["seed"]
    assert row["text"] == f"pick up the book and place it in the {target} compartment of the caddy"
    assert row["gain"] == (0. if arm == "native" else spec.GAIN)
    offsets = (0, 400) if window == "native" else spec.WINDOWS[window]
    assert row["offsets"] == list(offsets)
    result = spec.score(row["trace"], protocol["geometry"], case["z0"])
    assert all(row[key] == value for key, value in result.items())
    interval = spec.interval_metrics(row["trace"], protocol["geometry"], case["z0"], *offsets)
    for key in ("pre", "pre_grasp", "pre_grasp_confirmed_step", "post", "by_window_end"):
        assert row[key] == interval[key], key
    assert all(math.isclose(row["first_10_approach_mm"][g], interval["first_10_approach_mm"][g],
                            rel_tol=1e-10, abs_tol=1e-7) for g in spec.SITES)
    assert len(row["signatures"]) == 41 and row["signatures"][0] == sig(case["signature"])
    assert len(row["action_hashes"]) == 40
    for k, expected in enumerate(row["action_hashes"]):
        flat = [v for a in row["trace"]["actions"][k*10:(k+1)*10] for v in a]
        assert sha256(struct.pack("<70f", *flat)).hexdigest() == expected
    decisions = row["decisions"]
    assert [d["block"] for d in decisions] == spec.active_blocks(window)
    assert dict(Counter(d["decoded"] for d in decisions)) == row["routed"]
    for d in decisions:
        assert d["decoded"] == min("ABN", key=d["distance"].__getitem__)
        assert d["distance"][goal] == 0.  # True canonical text is also a reference.
        assert all(math.isfinite(v) and v >= 0 for v in d["distance"].values())
        assert math.isfinite(d["contrast_rms"]) and d["contrast_rms"] >= 0.
        assert d["coefficient"] == {"A": 1., "B": -1., "N": 0.}[d["decoded"]]*spec.GAIN*(1 if arm == "forward" else -1)


def previous_lookup(base, protocol):
    candidates = (base.parent / "k25_order_restore", base.parent / "k25_order_restore_20261122_01")
    old = next(p for p in candidates if (p / "protocol.json").exists())
    assert digest(old / "protocol.json") == protocol["previous_protocol_sha256"]
    previous_protocol = read(old / "protocol.json")
    assert protocol["checkpoints"] == {line: previous_protocol["checkpoints"][line] for line in spec.LINES}
    assert read(base / "cases.json") == read(old / "cases.json")["test"]
    archived = old / "dependencies/cl_binding/.server"
    assert digest(archived / "k25_order_restore_20261122_01/order_restore.py") == protocol["base_source_sha256"]
    assert digest(archived / "k25_final_20261121_01/self_restore.py") == protocol["baseline_source_sha256"]
    runtime = read(base / "runtime.json")
    assert digest(base / "capture_runtime.py") == runtime["capture_source_sha256"]
    for relative, expected in runtime["validated_dependency_sha256"].items():
        assert digest(old / "dependencies" / relative) == expected, relative
    rows = {}
    for path in [*old.glob("main_*.jsonl"), *old.glob("standard_*.jsonl")]:
        for s in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(s)
            if r["line"] not in spec.LINES or r["goal"] not in "AB" or r["variant"] != "canon":
                continue
            if r["arm"] == "native":
                rows[r["line"], "native", "native", r["goal"], r["init"]] = r
            elif r["gain"] == spec.GAIN:
                rows[r["line"], "delayed_100", r["arm"], r["goal"], r["init"]] = r
    assert len(rows) == 200
    return rows


def collect(base, protocol):
    assert protocol["windows"] == {k: list(v) for k, v in spec.WINDOWS.items()}
    assert protocol["gain"] == spec.GAIN and protocol["ids"] == list(spec.IDS)
    assert protocol["lines"] == list(spec.LINES) and protocol["goals"] == spec.GOALS
    assert protocol["arms"] == list(spec.ARMS) and protocol["jobs"] == [list(x) for x in spec.jobs()]
    assert protocol["horizon"] == spec.HORIZON and protocol["expected_episodes"] == 1200 and protocol["variant"] == "canon"
    assert protocol["spec_sha256"] == digest(Path(spec.__file__))
    assert protocol["source_sha256"] == digest(Path(__file__).with_name("v_timing.py"))
    assert protocol["cases_sha256"] == digest(base / "cases.json")
    cases = {r["init"]: r for r in read(base / "cases.json")}
    assert sorted(cases) == list(spec.IDS) and len({r["signature"]["sim"] for r in cases.values()}) == 20
    prior = previous_lookup(base, protocol)
    lookup, gates, deltas, replayed = {}, [], {}, set()
    for line in spec.LINES:
        markers = sorted(base.glob(f"{line}_*_COMPLETE.json"))
        assert markers, line
        found = []
        for path in markers:
            m = read(path)
            assert m["line"] == line and m["lanes"] == len(markers)
            assert m["ids"] == list(spec.IDS)[m["lane"]::m["lanes"]]
            assert m["source_sha256"] == protocol["source_sha256"] and m["spec_sha256"] == protocol["spec_sha256"]
            assert m["protocol_sha256"] == digest(base / "protocol.json")
            gate = m["gates"]
            assert gate["single_model_roles"] == ["F"] and gate["mean_cast_rms"] < .01
            assert all(gate[k] for k in ("v_only_overlay_equal", "weights_restored", "mean_preserved_fp64", "new_goal_bypass_bitwise"))
            gates.append(gate)
            if line in deltas:
                assert deltas[line] == m["delta_sha256"]
            deltas[line] = m["delta_sha256"]
            source = base / f"{line}_{m['lane']}.jsonl.gz"
            assert digest(source) == m["rows_sha256"]
            with gzip.open(source, "rt", encoding="utf-8") as stream:
                rows = [json.loads(s) for s in stream]
            expected = {(w, a, g, i) for w, a in spec.jobs() for g in "AB" for i in m["ids"]}
            assert len(rows) == len(expected) and {spec.condition_key(r) for r in rows} == expected
            for row in rows:
                assert row["line"] == line and row["delta_sha256"] == deltas[line]
                validate(row, protocol, cases[row["init"]])
                key = line, *spec.condition_key(row)
                assert key not in lookup
                lookup[key] = row
                if key in prior:
                    old = prior[key]
                    assert all(old[k] == row[k] for k in ("text", "first", "first_step", "times", "terminal", "first_grasp"))
                    assert row["signatures"][8] == sig(old["boundary_signature"])
                    replayed.add(key)
            assert {tuple(x) for x in m["previous_replays"]} == {key[1:] for key in replayed if key[0] == line and key[-1] in m["ids"]}
            found.extend(m["ids"])
        assert sorted(found) == list(spec.IDS)
    assert len(lookup) == 1200 and replayed == set(prior)
    return lookup, gates, deltas


def group(rows, line):
    per_goal = {}
    for goal in "AB":
        rr = sorted((r for r in rows if r["goal"] == goal), key=lambda r: r["init"])
        assert [r["init"] for r in rr] == list(spec.IDS)
        target = spec.GOALS[line][goal]
        per_goal[goal] = dict(n=20, first_correct=sum(r["first"] == [target] for r in rr),
            terminal_correct=sum(r["terminal"][target] for r in rr), ever_correct=sum(r["ever"][target] for r in rr),
            first=dict(Counter(spec.primary(r) for r in rr)), grasped=sum(r["first_grasp"] is not None for r in rr),
            pre_first=dict(Counter("+".join(r["pre"]["first"]) or "none" for r in rr)),
            pre_grasp_confirmed=sum(r["pre_grasp"] is not None for r in rr),
            post_first_correct=sum(r["post"]["first"] == [target] for r in rr),
            post_reached_correct=sum(r["post"]["times"][target] is not None for r in rr),
            first_read=dict(Counter(r["decisions"][0]["decoded"] for r in rr if r["decisions"])),
            all_blocks_read_correct=sum(bool(r["decisions"]) and all(d["decoded"] == goal for d in r["decisions"]) for r in rr),
            initial_ties=sum(bool(r["decisions"]) and r["decisions"][0]["margin"] == 0 for r in rr),
            first_block_delta_rms_median=median(r["decisions"][0]["contrast_rms"] for r in rr) if rr[0]["decisions"] else None,
            first_block_applied_delta_rms_median=median(abs(r["decisions"][0]["coefficient"])*r["decisions"][0]["contrast_rms"] for r in rr) if rr[0]["decisions"] else None,
            first_10_intended_minus_new_median_mm=median(r["first_10_approach_mm"][target]-r["first_10_approach_mm"][spec.GOALS[line]["N"]] for r in rr))
    rr = {(r["goal"], r["init"]): r for r in rows}
    return dict(per_goal=per_goal,
        joint_first=sum(all(rr[g, i]["first"] == [spec.GOALS[line][g]] for g in "AB") for i in spec.IDS),
        joint_terminal=sum(all(rr[g, i]["terminal"][spec.GOALS[line][g]] for g in "AB") for i in spec.IDS))


def paired_outcomes(before, after, goal, line):
    a, b = ({r["init"]: r for r in group if r["goal"] == goal} for group in (before, after))
    assert set(a) == set(b) == set(spec.IDS)
    target, new = spec.GOALS[line][goal], spec.GOALS[line]["N"]
    def selected(fn):
        return [i for i in spec.IDS if fn(a[i], b[i])]
    result = dict(first_transitions=dict(Counter(f"{spec.primary(a[i])}->{spec.primary(b[i])}" for i in spec.IDS)),
        first_gained=selected(lambda x, y: x["first"] != [target] and y["first"] == [target]),
        first_lost=selected(lambda x, y: x["first"] == [target] and y["first"] != [target]),
        terminal_gained=selected(lambda x, y: not x["terminal"][target] and y["terminal"][target]),
        terminal_lost=selected(lambda x, y: x["terminal"][target] and not y["terminal"][target]),
        old_new_goal_first=selected(lambda x, y: x["first"] == [new]),
        new_goal_to_intended=selected(lambda x, y: x["first"] == [new] and y["first"] == [target]),
        pre_wrong_then_terminal_correct=[i for i in spec.IDS if len(b[i]["pre"]["first"]) == 1
            and b[i]["pre"]["first"] != [target] and b[i]["terminal"][target]])
    return result


def compare(before, after, goal, line, geometry):
    result = paired_outcomes(before, after, goal, line)
    a, b = ({r["init"]: r for r in group if r["goal"] == goal} for group in (before, after))
    target, new = spec.GOALS[line][goal], spec.GOALS[line]["N"]
    # Baseline approach must use the same onset as the intervention being compared.
    changes, action_rms, eef_mm, book_mm = [], [], [], []
    for i in spec.IDS:
        low = b[i]["offsets"][0]
        if a[i]["offsets"][0] == low:
            before_mm = a[i]["first_10_approach_mm"]
        else:
            point = a[i]["trace"]["start"]["object"] if low == 0 else a[i]["trace"]["object"][low-1]
            after_point = a[i]["trace"]["object"][low+9]
            before_mm = {g: 1000*(math.dist(point[:2], geometry[g]["centre"][:2])-math.dist(after_point[:2], geometry[g]["centre"][:2])) for g in (target, new)}
        after_mm = b[i]["first_10_approach_mm"]
        changes.append((after_mm[target]-after_mm[new])-(before_mm[target]-before_mm[new]))
        action_rms.append(math.sqrt(sum((x-y)**2 for aa, bb in zip(
            a[i]["trace"]["actions"][low:low+10], b[i]["trace"]["actions"][low:low+10], strict=True)
            for x, y in zip(aa, bb, strict=True))/70))
        eef_mm.append(1000*math.dist(a[i]["trace"]["eef"][low+9], b[i]["trace"]["eef"][low+9]))
        book_mm.append(1000*math.dist(a[i]["trace"]["object"][low+9], b[i]["trace"]["object"][low+9]))
    result["first_10_relative_approach_change_median_mm"] = median(changes)
    result["first_block_action_rms_median"] = median(action_rms)
    result["after_10_eef_difference_median_mm"] = median(eef_mm)
    result["after_10_book_difference_median_mm"] = median(book_mm)
    return result


def native_history(rows, line, protocol, cases):
    result = {}
    for offset in (40, 80, 160):
        result[str(offset)] = {}
        for goal in "AB":
            rr = [r for r in rows if r["goal"] == goal]
            target, new = spec.GOALS[line][goal], spec.GOALS[line]["N"]
            firsts, grasps, lead = [], [], []
            for r in rr:
                t = r["trace"]
                m = spec.interval_metrics(t, protocol["geometry"], cases[r["init"]]["z0"], offset, 400)
                firsts.append("+".join(m["pre"]["first"]) or "none")
                grasps.append(m["pre_grasp_confirmed_step"] is not None)
                if m["pre_grasp_confirmed_step"] is not None:
                    start, end = t["object"][m["pre_grasp_confirmed_step"]-1], t["object"][offset-1]
                    def approach(g):
                        center = protocol["geometry"][g]["centre"][:2]
                        return math.dist(start[:2], center)-math.dist(end[:2], center)
                    lead.append(1000*(approach(target)-approach(new)))
            result[str(offset)][goal] = dict(n=20, confirmed_first=dict(Counter(firsts)), grasp_confirmed=sum(grasps),
                grasp_to_offset_intended_minus_new_median_mm=median(lead) if lead else None,
                lead_n=len(lead))
    return result


def summarize(base):
    protocol = read(base / "protocol.json")
    geometry = protocol["geometry"]
    lookup, gates, deltas = collect(base, protocol)
    cases = {r["init"]: r for r in read(base / "cases.json")}
    report = dict(episodes=len(lookup), independent_starts=20, previous_condition_replays=200,
        protocol_sha256=digest(base / "protocol.json"), analysis_source_sha256=digest(Path(__file__)),
        verified_project_dependencies=len(read(base / "runtime.json")["validated_dependency_sha256"]),
        delta_sha256=deltas, max_mean_cast_rms=max(g["mean_cast_rms"] for g in gates), lines={})
    prefix_pairs = sign_pairs = 0
    for line in spec.LINES:
        native = [lookup[line, "native", "native", g, i] for i in spec.IDS for g in "AB"]
        output = dict(native=group(native, line), windows={}, native_history=native_history(native, line, protocol, cases),
                      continued_vs_stopped={}, fixed_length_vs_delayed={})
        for window in spec.WINDOWS:
            arms = {}
            for arm in spec.ARMS:
                rows = [lookup[line, window, arm, g, i] for i in spec.IDS for g in "AB"]
                arms[arm] = dict(**group(rows, line), vs_native={g: compare(native, rows, g, line, geometry) for g in "AB"})
                for r in rows:
                    same_prefix(lookup[line, "native", "native", r["goal"], r["init"]], r, spec.WINDOWS[window][0]//10)
                    prefix_pairs += 1
            forward = [lookup[line, window, "forward", g, i] for i in spec.IDS for g in "AB"]
            reverse = [lookup[line, window, "reverse", g, i] for i in spec.IDS for g in "AB"]
            arms["reverse"]["vs_forward"] = {g: compare(forward, reverse, g, line, geometry) for g in "AB"}
            for a, b in zip(forward, reverse, strict=True):
                same_prefix(a, b, spec.WINDOWS[window][0]//10)
                assert a["decisions"][0]["distance"] == b["decisions"][0]["distance"]
                assert a["decisions"][0]["coefficient"] == -b["decisions"][0]["coefficient"]
                sign_pairs += 1
            output["windows"][window] = arms
        # If schedules have not differed yet, no implementation detail may cause divergence.
        for a_name, b_name in combinations(spec.WINDOWS, 2):
            blocks = spec.required_common_blocks(a_name, b_name)
            for arm in spec.ARMS:
                for i in spec.IDS:
                    for goal in "AB":
                        same_prefix(lookup[line, a_name, arm, goal, i], lookup[line, b_name, arm, goal, i], blocks)
                        prefix_pairs += 1
        for stopped, continued in (("early_100", "full_400"), ("delayed_100", "delayed_tail"), ("late_100", "late_tail")):
            before = [lookup[line, stopped, "forward", g, i] for i in spec.IDS for g in "AB"]
            after = [lookup[line, continued, "forward", g, i] for i in spec.IDS for g in "AB"]
            output["continued_vs_stopped"][continued+"_vs_"+stopped] = {g: compare(before, after, g, line, geometry) for g in "AB"}
        for window in spec.FIXED_LENGTH:
            if window == "delayed_100":
                continue
            contrasts = {}
            for arm in spec.ARMS:
                before = [lookup[line, "delayed_100", arm, g, i] for i in spec.IDS for g in "AB"]
                after = [lookup[line, window, arm, g, i] for i in spec.IDS for g in "AB"]
                contrasts[arm] = {g: paired_outcomes(before, after, g, line) for g in "AB"}
            output["fixed_length_vs_delayed"][window] = contrasts
        report["lines"][line] = output
    report["verified_common_prefix_comparisons"] = prefix_pairs
    report["verified_polarity_pairs"] = sign_pairs
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = summarize(args.folder)
    path = args.folder / "summary.json"
    if args.check:
        assert read(path) == report
    else:
        assert not path.exists(), "Refusing to replace the archived summary"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in report.items() if k != "lines"}, ensure_ascii=False, indent=2))
