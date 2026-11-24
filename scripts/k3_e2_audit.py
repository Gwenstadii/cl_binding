"""Cross-check E2 instruction A/B episodes against the archived K1 prefix rollouts.

Same (family, init, model) and the same replanning-noise rule, so the native rollouts must agree
on every goal success step and on the final predicate vector. Any disagreement means the E2
pipeline is not the audited native pipeline and the E2 numbers may not be used.
"""
import gzip
import json
from pathlib import Path

K1 = Path("/home/lzx/cl_binding/.server/k1_screen_20261101_01")
OUT = Path("/home/lzx/cl_binding/.server/k3_e2_20261102_01")
PRIMARY = {"scene2": "FT4_scene2_s1", "scene1b": "FT4_scene1b", "goal": "FT4_goal"}


def main():
    rows = []
    for path in sorted((OUT / "runs").glob("w*.jsonl")):
        rows += [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    index = {(r["family"], r["model"], r["instruction_key"], r["init"]): r for r in rows}
    checks, failures = [], []
    for family in ("scene2", "scene1b", "goal"):
        for init in range(4, 10):
            key = "A" if init % 2 == 0 else "B"
            for model, prefix_key in (("R0", "R0"), ("FT4", PRIMARY[family])):
                if family == "goal" and model == "FT4":
                    prefix_key = "FT4_goal"
                path = K1 / "prefix" / family / (prefix_key + "_i" + str(init) + ".json.gz")
                if not path.exists():
                    failures.append({"missing_prefix": str(path)})
                    continue
                with gzip.open(path, "rt") as f:
                    ref = json.load(f)
                if ref["instruction"] != key:
                    failures.append({"instruction_parity": (family, model, init, ref["instruction"], key)})
                    continue
                got = index.get((family, model, key, init))
                if got is None:
                    checks.append({"pending": (family, model, key, init)})
                    continue
                same_success = got["native_success_step"] == ref["native_success_step"]
                same_terminal = got["terminal_pred"] == ref["rows"][-1]["pred"]
                rec = {"family": family, "model": model, "instruction": key, "init": init,
                       "success_equal": same_success, "terminal_equal": same_terminal,
                       "e2_first": got["first_targets"]}
                checks.append(rec)
                if not (same_success and same_terminal):
                    failures.append(rec)
    out = {"checked": len([c for c in checks if "family" in c]), "pending": len([c for c in checks if "pending" in c]),
           "failures": failures, "passed": not failures}
    (OUT / "engineering").mkdir(exist_ok=True)
    (OUT / "engineering/k1_prefix_crosscheck.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"checked": out["checked"], "pending": out["pending"], "passed": out["passed"],
                      "failures": failures[:3]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
