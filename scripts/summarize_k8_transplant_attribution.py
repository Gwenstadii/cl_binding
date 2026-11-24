"""Validate the K8 matrix and summarize paired first-goal and success results."""

import argparse
from collections import Counter
import json
from pathlib import Path

from k8_transplant_attribution import GOALS, jobs


def summarize(run, k7):
    gate = json.loads((run / "gate.json").read_bytes())
    reference = json.loads(k7.read_bytes())["runs"]
    assert gate["status"] == "passed"
    assert (run / "exit_code").read_text().strip() == "0"
    rows = [json.loads(line) for path in sorted(run.glob("worker*.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines()]
    index = {(r["init"], r["condition"], r["instruction"]): r for r in rows}
    assert len(index) == len(rows) and set(index) == set(jobs()), "Incomplete or duplicate matrix"
    out = {"episodes": len(rows), "k7_old_native_match": 0, "full_r0_reconstruction_match": 0,
           "conditions": {}, "paired_transitions": {}}
    for i in range(10):
        for g in GOALS:
            native = index[i, "r0-native", g]
            assert native["trace_sha256"] == index[i, "all16-external", g]["trace_sha256"], (i, g)
            out["full_r0_reconstruction_match"] += 1
            if g != "N":
                old = reference[f"{i}|NAT-R0|{g}"]
                assert native["first_targets"] == old["first"]
                assert native["first_step"] == old["step"]
                assert {k: v for k, v in native["success"].items() if v is not None} == old["success"]
                assert native["first_close"] == old["close"]
                out["k7_old_native_match"] += 1

    def first(row):
        return "+".join(row["first_targets"]) or "none"

    for condition in dict.fromkeys(c for _, c, _ in jobs()):
        table, transitions = {}, {}
        for g, target in GOALS.items():
            selected = [r for r in rows if r["condition"] == condition and r["instruction"] == g]
            if not selected:
                continue
            table[g] = {
                "first": dict(Counter(map(first, selected))),
                "ever": sum(r["success"][target] is not None for r in selected),
                "terminal": sum(r["terminal_pred"][target] for r in selected),
            }
            if condition.startswith("reverse-"):
                baseline = {i: first(index[i, "r0-native", g]) for i in range(10)}
            elif condition.startswith("all16-"):
                baseline = {i: "+".join(reference[f"{i}|ALL-16|{g}"]["first"]) or "none"
                            for i in range(10)}
            else:
                continue
            transitions[g] = dict(Counter(f"{baseline[r['init']]}->{first(r)}" for r in selected))
        table["D"] = sum(index[i, condition, "A"]["first_targets"] == ["left"]
                         and index[i, condition, "B"]["first_targets"] == ["front"] for i in range(10))
        out["conditions"][condition] = table
        if transitions:
            out["paired_transitions"][condition] = transitions
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--k7", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run, args.k7)
    (args.run / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
