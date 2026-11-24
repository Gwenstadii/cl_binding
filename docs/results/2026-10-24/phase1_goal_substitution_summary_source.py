"""Recompute the phase-one table from existing local episode JSONL files only.

Run from the repository root:
    python docs/results/2026-10-24/phase1_goal_substitution_summary_source.py

This does not load a model, simulate an episode, or access a remote server.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = Path(__file__).with_name("phase1_goal_substitution_summary.json")

SPECS = (
    (
        "E002", "left", "right", "single_goal_chain",
        "docs/results/2026-10-22/left_then_right_v1/before_episodes.jsonl",
        "docs/results/2026-10-22/left_then_right_v1/after_episodes.jsonl",
    ),
    (
        "E003", "left", "right", "single_goal_chain",
        "docs/results/2026-10-22/left_then_right_v1/before_episodes.jsonl",
        "docs/results/2026-10-22/right_continue_v1/episodes.jsonl",
    ),
    (
        "E004", "right", "front", "single_goal_chain",
        "docs/results/2026-10-22/right_continue_v1/episodes.jsonl",
        "docs/results/2026-10-23/four_task_sequence_v1/stage3_front/episodes.jsonl",
    ),
    (
        "E005", "front", "back", "single_goal_chain",
        "docs/results/2026-10-23/four_task_sequence_v1/stage3_front/episodes.jsonl",
        "docs/results/2026-10-23/four_task_sequence_v1/stage4_back/episodes.jsonl",
    ),
    (
        "E007", "left", "right", "dual_goal_chain",
        "docs/results/2026-10-23/dual12000_then_right_v1/before/episodes.jsonl",
        "docs/results/2026-10-23/dual12000_then_right_v1/after/episodes.jsonl",
    ),
    (
        "E007", "front", "right", "dual_goal_chain",
        "docs/results/2026-10-23/dual12000_then_right_v1/before/episodes.jsonl",
        "docs/results/2026-10-23/dual12000_then_right_v1/after/episodes.jsonl",
    ),
    (
        "E009", "right", "back", "dual_goal_chain",
        "docs/results/2026-10-23/dual_right_then_back_v1/before/episodes.jsonl",
        "docs/results/2026-10-23/dual_right_then_back_v1/after/episodes.jsonl",
    ),
)


def load_source(relative_path: str) -> tuple[list[dict], dict]:
    content = (PROJECT_ROOT / relative_path).read_bytes()
    episodes = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    keys = [(row["requested_goal"], row["init_index"]) for row in episodes]
    assert len(keys) == len(set(keys)), f"Duplicate episode condition: {relative_path}"
    for row in episodes:
        goal = row["requested_goal"]
        ever = row["first_goal_times"][goal] is not None
        assert row["correct_ever"] == ever, f"Inconsistent success-ever field: {relative_path}"
        assert row["correct_then_destroyed"] == (ever and row["terminal_goal"] != goal)
    return episodes, {
        "path": relative_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "episodes_in_source": len(episodes),
    }


def summarize(spec: tuple[str, ...]) -> dict:
    experiment, old, new, chain, before_path, after_path = spec
    before_all, before_source = load_source(before_path)
    after_all, after_source = load_source(after_path)
    before = {row["init_index"]: row for row in before_all if row["requested_goal"] == old}
    after = {row["init_index"]: row for row in after_all if row["requested_goal"] == old}
    assert before.keys() == after.keys(), f"Unmatched old-goal conditions: {experiment}/{old}"
    assert len(before) == 20, f"Unexpected old-goal sample count: {experiment}/{old}"
    paired_checks = {
        "reset_state_sha256": all(
            bool(before[index]["initial_state_sha256"])
            and before[index]["initial_state_sha256"] == after[index]["initial_state_sha256"]
            for index in before
        ),
        "scene_xml_sha256": all(
            bool(before[index]["scene_xml_sha256"])
            and before[index]["scene_xml_sha256"] == after[index]["scene_xml_sha256"]
            for index in before
        ),
        "seed": all(before[index]["seed"] == after[index]["seed"] for index in before),
        "horizon": all(before[index]["steps"] == after[index]["steps"] == 400 for index in before),
    }
    assert all(paired_checks.values()), f"Pairing mismatch: {experiment}/{old}: {paired_checks}"
    eligible_indices = sorted(index for index in before if before[index]["terminal_goal"] == old)
    eligible_after = [after[index] for index in eligible_indices]
    retained = [row for row in eligible_after if row["terminal_goal"] == old]
    first_new = [row for row in eligible_after if row["first_goal"] == new]
    terminal_new = [row for row in eligible_after if row["terminal_goal"] == new]
    unfinished = [row for row in eligible_after if row["terminal_category"] == "unfinished"]
    other_terminal = [
        row for row in eligible_after
        if row["terminal_goal"] not in (old, new) and row["terminal_category"] != "unfinished"
    ]
    assert len(eligible_after) == len(retained) + len(terminal_new) + len(unfinished) + len(other_terminal)
    return {
        "experiment": experiment,
        "old": old,
        "new": new,
        "chain": chain,
        "pre_terminal_correct": len(eligible_indices),
        "total_post_correct": sum(row["terminal_goal"] == old for row in after.values()),
        "retained_terminal_correct": len(retained),
        "first_new": len(first_new),
        "terminal_new": len(terminal_new),
        "unfinished": len(unfinished),
        "other_terminal_wrong_or_ambiguous": len(other_terminal),
        "post_old_goal_success_ever": sum(row["correct_ever"] for row in after.values()),
        "post_old_goal_success_ever_within_pre_terminal_correct": sum(
            row["correct_ever"] for row in eligible_after
        ),
        "total_old_goal_conditions": len(before),
        "all_initial_indices": sorted(before),
        "pre_terminal_correct_indices": eligible_indices,
        "first_new_indices": [row["init_index"] for row in first_new],
        "terminal_new_indices": [row["init_index"] for row in terminal_new],
        "unfinished_indices": [row["init_index"] for row in unfinished],
        "paired_reset_xml_seed_horizon_verified": all(paired_checks.values()),
        "pairing_checks": paired_checks,
        "pairing_checks_applied_to_conditions": len(before),
        "before_source": before_source,
        "after_source": after_source,
    }


def main() -> None:
    rows = [summarize(spec) for spec in SPECS]
    result = {
        "schema_version": 1,
        "scope": "Derived summary of seven existing old-goal comparisons from E002-E009; no new training or evaluation.",
        "generation_command": "python docs/results/2026-10-24/phase1_goal_substitution_summary_source.py",
        "source_script": "docs/results/2026-10-24/phase1_goal_substitution_summary_source.py",
        "definitions": {
            "pre_terminal_correct": "Old-goal conditions whose before episode ended at the requested old goal; the denominator for first_new, terminal_new, and unfinished.",
            "total_post_correct": "Post episodes ending at the requested old goal among all 20 old-goal conditions, not just the previously successful subset.",
            "retained_terminal_correct": "Post episodes ending at the requested old goal within the pre_terminal_correct subset.",
            "first_new": "Within the pre_terminal_correct subset, the first reached goal in the post episode is the new goal.",
            "terminal_new": "Within the pre_terminal_correct subset, the post episode ends at the new goal.",
            "unfinished": "Within the pre_terminal_correct subset, the post episode has terminal_category unfinished.",
            "post_old_goal_success_ever": "Number of the 20 post old-goal episodes that ever satisfy the old requested-goal predicate.",
            "paired_reset_xml_seed_horizon_verified": "For every old-goal condition, before and after have equal nonempty reset-state and scene-XML SHA256, the same inference seed, and 400 control steps.",
        },
        "independence": {
            "rows_are_independent_replications": False,
            "pooled_effect_or_sample_size_computed": False,
            "shared_evidence": [
                "E002 and E003 use exactly the same original left-goal parent episodes; E003 is a continuation of E002, not a new independent seed.",
                "E003 after episodes are reused as E004 before episodes; E004 after episodes are reused as E005 before episodes.",
                "The two E007 rows share the same parent/child models and 20 physical states; they are different instructions on correlated states.",
                "E009 before left/front/right episodes reuse E007 after episodes; the right-goal row continues that same model chain.",
                "Single-goal and dual-goal chains use overlapping physical indices 30-39, one task family, and historically reused conditions; the rows cannot be pooled as independent scenes or seeds.",
            ],
            "chain_comparison_is_controlled": False,
            "chain_comparison_warning": "The chains differ in acquisition data, normalization, parent selection/training budget and evaluation indices; this table does not identify an effect of using one versus two acquired goals.",
        },
        "rows": rows,
    }
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()} ({len(rows)} comparisons).")
    for row in rows:
        print(
            f"{row['experiment']} {row['old']}->{row['new']}: "
            f"pre={row['pre_terminal_correct']}, post={row['total_post_correct']}, "
            f"first_new={row['first_new']}, terminal_new={row['terminal_new']}, "
            f"unfinished={row['unfinished']}, paired={row['paired_reset_xml_seed_horizon_verified']}"
        )


if __name__ == "__main__":
    main()
