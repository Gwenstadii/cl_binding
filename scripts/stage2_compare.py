"""Paired four-instruction comparison of a stage 2A endpoint against the sealed dual-goal parent."""

GOALS = ("left", "front", "right", "back")
OLD = ("left", "front")
NEW = "right"
FIELDS = ("initial_state_sha256", "scene_xml_sha256", "seed", "steps")


def keyed_rows(rows, indices, seed):
    expected = {(i, g) for i in indices for g in GOALS}
    keyed = {(r["init_index"], r["requested_goal"]): r for r in rows}
    if len(rows) != len(expected) or set(keyed) != expected:
        raise ValueError("Missing, repeated or unexpected evaluation conditions")
    for i in indices:
        for field in FIELDS:
            if len({keyed[i, g][field] for g in GOALS}) != 1:
                raise ValueError(f"Unpaired reset: {i} {field}")
        if keyed[i, "left"]["seed"] != seed + i or keyed[i, "left"]["steps"] != 400:
            raise ValueError("Reset seed or horizon differs from the locked protocol")
    if len({r["scene_xml_sha256"] for r in rows}) != 1:
        raise ValueError("Scene XML changed")
    return keyed


def summarize(rows, indices, seed):
    keyed = keyed_rows(rows, indices, seed)
    counts, first, terminal = {}, {}, {}
    for g in GOALS:
        selected = [keyed[i, g] for i in indices]
        counts[g] = {"total": len(selected), "success_ever": sum(r["correct_ever"] for r in selected),
                     "terminal_correct": sum(r["terminal_category"] == "correct" for r in selected),
                     "success_then_lost": sum(r["correct_then_destroyed"] for r in selected),
                     "success_then_lost_unfinished": sum(r["correct_then_destroyed"] and r["terminal_category"] == "unfinished" for r in selected)}
        first[g] = {t: sum(r["first_goal"] == t for r in selected) for t in GOALS}
        first[g]["none_or_tied"] = sum(r["first_goal"] is None for r in selected)
        terminal[g] = {t: sum(r["terminal_goal"] == t for r in selected) for t in GOALS}
        for category in ("unfinished", "ambiguous"):
            terminal[g][category] = sum(r["terminal_category"] == category for r in selected)
    joint = [i for i in indices if all(keyed[i, g]["terminal_category"] == "correct" for g in OLD)]
    return {"task_success_counts": counts, "first_goal_matrix": first, "terminal_matrix": terminal,
            "joint_old_terminal_correct_indices": joint}


def compare(parent_rows, rows, indices, seed):
    before, after = summarize(parent_rows, indices, seed), summarize(rows, indices, seed)
    pre, post = keyed_rows(parent_rows, indices, seed), keyed_rows(rows, indices, seed)
    for key in pre:
        for field in FIELDS:
            if pre[key][field] != post[key][field]:
                raise ValueError(f"Parent/branch mismatch: {key} {field}")
    old = {}
    for goal in OLD:
        old[goal] = {}
        for label, held in (("previously_successful_ever", lambda r: r["correct_ever"]),
                            ("previously_terminal_correct", lambda r: r["terminal_category"] == "correct")):
            eligible = [i for i in indices if held(pre[i, goal])]
            lost = [post[i, goal] for i in eligible if not held(post[i, goal])]
            old[goal][label] = {
                "total": len(eligible), "retained": len(eligible) - len(lost), "lost": len(lost),
                "lost_and_first_new_right": sum(r["first_goal"] == NEW for r in lost),
                "lost_and_terminal_new_right": sum(r["terminal_goal"] == NEW for r in lost),
                "lost_and_terminal_other_wrong": sum(r["terminal_category"] == "wrong_goal" and r["terminal_goal"] != NEW for r in lost),
                "lost_and_terminal_unfinished": sum(r["terminal_category"] == "unfinished" for r in lost),
                "lost_after_reaching_old_goal": sum(r["correct_ever"] for r in lost)}
    joint = before["joint_old_terminal_correct_indices"]
    new_before, new_after = before["task_success_counts"][NEW], after["task_success_counts"][NEW]
    return {"parent": before, "branch": after, "matched_pairs": len(pre), "old_goal_transitions": old,
            "previously_joint_correct_states": {
                "total": len(joint),
                "retained_joint_terminal_correct": sum(all(post[i, g]["terminal_category"] == "correct" for g in OLD) for i in joint),
                "both_old_instructions_terminal_new_right": sum(all(post[i, g]["terminal_goal"] == NEW for g in OLD) for i in joint),
                "both_old_instructions_first_new_right": sum(all(post[i, g]["first_goal"] == NEW for g in OLD) for i in joint)},
            "new_right_gain": {k: new_after[k] - new_before[k] for k in ("success_ever", "terminal_correct")},
            "new_right_descriptive_screen_16_of_20": all(new_after[k] >= 16 for k in ("success_ever", "terminal_correct")),
            "interpretation": "Exploratory pilot on reused states 20-39 from a non-qualified parent. Report wrong-new-target, "
                              "late result loss and unfinished separately on actual pre-success denominators."}
