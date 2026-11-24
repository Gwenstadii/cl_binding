"""Audit paired K9 records and assemble the K7–K9 stack/output comparison."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from k9_output_projection import CONDITIONS, GOALS, JOBS


def first(row):
    return "+".join(row["first_targets"]) or "none"


def summarize(run, k7, k8):
    rows = []
    metadata = []
    for worker in (0, 1):
        batch = [json.loads(line) for line in (run / f"worker{worker}.jsonl").read_text().splitlines()]
        expected = {(i, g, c) for i, g in JOBS[worker::2] for c in CONDITIONS}
        assert len(batch) == len(expected) and {
            (r["init"], r["instruction"], r["condition"]) for r in batch
        } == expected
        assert int((run / f"worker{worker}.done").read_text()) == len(batch)
        rows.extend(batch)
        metadata.append(json.loads((run / f"worker{worker}_meta.json").read_bytes()))
    assert metadata[0] == metadata[1]
    gate8 = json.loads(k8.with_name("gate.json").read_bytes())
    assert metadata[0]["model_sha256"] == {role: model["sha256"] for role, model in gate8["models"].items()}
    source = Path(__file__).with_name("k9_output_projection.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata[0]["source_sha256"]
    ref7 = json.loads(k7.read_bytes())["runs"]
    ref8 = json.loads(k8.read_bytes())["conditions"]["all16-output"]
    index = {(r["init"], r["instruction"], r["condition"]): r for r in rows}
    transitions = {}
    conditions = {}
    for c in CONDITIONS:
        conditions[c] = {}
        for g, target in GOALS.items():
            selected = [index[i, g, c] for i in range(10)]
            conditions[c][g] = {
                "first": dict(Counter(map(first, selected))),
                "ever": sum(r["success"][target] is not None for r in selected),
                "terminal": sum(r["terminal_pred"][target] for r in selected),
            }
            for r in selected:
                i = r["init"]
                assert r["seed"] == index[i, g, "ft4-native"]["seed"]
                if c == "ft4-native":
                    old = ref7[f"{i}|NAT-FT4|{g}"]
                    assert r["first_targets"] == old["first"] and r["first_step"] == old["step"]
                    assert r["first_close"] == old["close"]
                    assert {k: v for k, v in r["success"].items() if v is not None} == old["success"]
        conditions[c]["D"] = sum(
            first(index[i, "A", c]) == "left" and first(index[i, "B", c]) == "front" for i in range(10))
    for g in GOALS:
        transitions[g] = dict(Counter(
            f"{first(index[i, g, CONDITIONS[0]])}->{first(index[i, g, CONDITIONS[1]])}" for i in range(10)))
    return {
        "episodes": len(rows), "native_k7_matches": len(JOBS), "metadata": metadata[0],
        "conditions": conditions, "paired_first_target_transitions": transitions,
        "front_first_correct_factorial": {
            "FT4_stack_FT4_output": conditions[CONDITIONS[0]]["B"]["first"].get("front", 0),
            "FT4_stack_R0_output": conditions[CONDITIONS[1]]["B"]["first"].get("front", 0),
            "R0_stack_FT4_output": sum(ref7[f"{i}|ALL-16|B"]["first"] == ["front"] for i in range(10)),
            "R0_stack_R0_output": ref8["B"]["first"].get("front", 0),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--k7", type=Path, required=True)
    parser.add_argument("--k8", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run, args.k7, args.k8)
    (args.run / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
