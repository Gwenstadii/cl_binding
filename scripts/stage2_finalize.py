"""Finish a stage 2A run whose training and evaluation completed but whose comparison stopped on a scene-XML hash.

Nothing is retrained or re-evaluated. The run must have failed exactly at the paired comparison with a scene-XML
mismatch, every branch must hold a verified 12000-step checkpoint and 80 evaluation rows, and the alias must be
backed by a stage2_scene_equivalence.py record. The original failure stays in run_summary.json.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from stage1_dual_goal_run import read_rows
from stage2_compare import compare, report_lines, summarize
from stage2_memory_export import sha256
from stage2_run import PARENT_ROWS, dump

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scene-equivalence", type=Path, required=True)
    parser.add_argument("--schedule-reference", type=Path, help="Another run's train_steps.jsonl whose new indices must match")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    summary = json.loads((run / "run_summary.json").read_text())
    if summary["status"] != "failed" or not summary.get("error", "").endswith("scene_xml_sha256") \
            or summary.get("current_stage") != "evaluating_fixed_12000_endpoints" or (run / "COMPLETE.json").exists():
        raise ValueError("Only a run stopped by the paired scene-XML check can be finalized offline")
    evidence = json.loads(args.scene_equivalence.read_text())
    if not (evidence["identical_after_root_normalization"] and evidence["every_changed_line_mentions_a_root"]):
        raise ValueError("Scene equivalence evidence does not support an alias")
    config = json.loads((run / "config.json").read_text())
    indices = config["evaluation"]["paired_indices"]
    parent_rows = [r for path in PARENT_ROWS for r in read_rows(ROOT / path)]
    parent_hash = {r["scene_xml_sha256"] for r in parent_rows}
    branch_rows, finals, schedules = {}, {}, {}
    for branch in summary["branches"]:
        train = Path(summary["training_dirs"][branch])
        final = train / "checkpoints/012000"
        finished = json.loads((train / "training_finished_012000.json").read_text())
        marker = json.loads((final / "CHECKPOINT_COMPLETE.json").read_text())
        manifest = json.loads((train / "manifest.json").read_text())
        if finished["step"] != 12000 or not finished["frozen_parameters_unchanged"] or marker["status"] != "pass" \
                or manifest["config"] != config or manifest["git_commit"] != summary["source_commit"]:
            raise ValueError(f"{branch}: training record does not match the run")
        trace = read_rows(train / "train_steps.jsonl")
        if [r["step"] for r in trace] != list(range(1, 12001)) or not all(math.isfinite(r["loss"]) for r in trace):
            raise ValueError(f"{branch}: discontinuous or nonfinite training")
        attempts = sorted(run.glob(f"{branch}_eval_attempt*/episodes.jsonl"))
        evaluations = [json.loads((p.parent / "evaluation.json").read_text()) for p in attempts if (p.parent / "evaluation.json").exists()]
        if len(attempts) != 1 or len(evaluations) != 1 or evaluations[0]["status"] != "complete":
            raise ValueError(f"{branch}: expected exactly one completed evaluation attempt")
        branch_rows[branch] = read_rows(attempts[0])
        finals[branch] = final
        schedules[branch] = [r["indices"]["new"] for r in trace]
    hashes = {r["scene_xml_sha256"] for rows in branch_rows.values() for r in rows}
    if len(parent_hash) != 1 or len(hashes) != 1:
        raise ValueError("Each side must have exactly one scene hash")
    aliases = {next(iter(hashes)): next(iter(parent_hash))}
    if set(aliases.items()) != {(evidence["sha256_b"], evidence["sha256_a"])} and set(aliases.items()) != {(evidence["sha256_a"], evidence["sha256_b"])}:
        raise ValueError("Evidence hashes do not match the recorded scene hashes")
    if len({json.dumps(s) for s in schedules.values()}) != 1:
        raise ValueError("Branches did not consume identical new-sample schedules")
    if args.schedule_reference and [r["indices"]["new"] for r in read_rows(args.schedule_reference)] != next(iter(schedules.values())):
        raise ValueError("New-sample schedule differs from the reference run")
    comparisons = {b: compare(parent_rows, branch_rows[b], indices, config["seed"], aliases) for b in summary["branches"]}
    for b in summary["branches"]:
        folder = run / b
        folder.mkdir()
        (folder / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in branch_rows[b]))
        dump(folder / "comparison.json", comparisons[b])
    dump(run / "scene_xml_equivalence.json", evidence)
    summary.update(original_failure={"error": summary["error"], "seconds": summary.get("seconds")},
                   status="complete", current_stage="finished_offline_after_scene_alias",
                   finalized_at=datetime.now(timezone.utc).isoformat(), scene_xml_aliases=aliases,
                   schedule_reference=str(args.schedule_reference) if args.schedule_reference else None,
                   final_model_sha256={b: sha256(finals[b] / "model.safetensors") for b in summary["branches"]},
                   new_evaluation_episodes=sum(len(r) for r in branch_rows.values()), reused_parent_episodes=len(parent_rows))
    summary.pop("error")
    dump(run / "run_summary.json", summary)
    lines = report_lines(summary["branches"], summarize(parent_rows, indices, config["seed"]), comparisons)
    lines += ["", "注：比较步骤原先因场景XML哈希不同而停止；差异经核实只是仓库根目录路径（worktree软链接），"
              "物理初态/seed/步数逐条一致，见`scene_xml_equivalence.json`。未重训、未重评。"]
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump(run / "COMPLETE.json", {"status": "complete", "branches": summary["branches"], "fixed_steps": 12000,
                                 "starting_point_formally_qualified": False, "finalized_offline": True})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
