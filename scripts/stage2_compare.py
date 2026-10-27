"""Paired four-instruction comparison of a stage 2A endpoint against the sealed dual-goal parent.

Primary metric (2026-10-24 decision): the first goal reached and ever-success, which equal LIBERO's
stop-on-success scoring because demonstrations end ~10 frames after success. Terminal state after the full
400-step horizon is kept as a secondary metric; the episodes themselves are unchanged and stay paired.
"""

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


def compare(parent_rows, rows, indices, seed, scene_aliases=None):
    """scene_aliases maps a branch scene-XML hash to the parent hash it was verified identical to (path-only diff)."""
    scene_aliases = scene_aliases or {}
    before, after = summarize(parent_rows, indices, seed), summarize(rows, indices, seed)
    pre, post = keyed_rows(parent_rows, indices, seed), keyed_rows(rows, indices, seed)
    for key in pre:
        for field in FIELDS:
            value = post[key][field]
            if field == "scene_xml_sha256":
                value = scene_aliases.get(value, value)
            if pre[key][field] != value:
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
                "lost_and_first_other_wrong": sum(r["first_goal"] not in (None, NEW, goal) for r in lost),
                "lost_and_no_goal_first": sum(r["first_goal"] is None for r in lost),
                "lost_and_terminal_new_right": sum(r["terminal_goal"] == NEW for r in lost),
                "lost_and_terminal_other_wrong": sum(r["terminal_category"] == "wrong_goal" and r["terminal_goal"] != NEW for r in lost),
                "lost_and_terminal_unfinished": sum(r["terminal_category"] == "unfinished" for r in lost),
                "lost_after_reaching_old_goal": sum(r["correct_ever"] for r in lost)}
    joint = before["joint_old_terminal_correct_indices"]
    joint_ever = [i for i in indices if all(pre[i, g]["correct_ever"] for g in OLD)]
    new_before, new_after = before["task_success_counts"][NEW], after["task_success_counts"][NEW]
    return {"scene_xml_aliases": scene_aliases, "metric_policy": {"primary": "first goal reached + ever-success (stop-on-success equivalent)",
                              "secondary": "terminal goal at 400 steps"},
            "parent": before, "branch": after, "matched_pairs": len(pre), "old_goal_transitions": old,
            "previously_joint_ever_states": {
                "total": len(joint_ever),
                "retained_joint_ever": sum(all(post[i, g]["correct_ever"] for g in OLD) for i in joint_ever),
                "both_old_instructions_first_new_right": sum(all(post[i, g]["first_goal"] == NEW for g in OLD) for i in joint_ever)},
            "previously_joint_correct_states": {
                "total": len(joint),
                "retained_joint_terminal_correct": sum(all(post[i, g]["terminal_category"] == "correct" for g in OLD) for i in joint),
                "both_old_instructions_terminal_new_right": sum(all(post[i, g]["terminal_goal"] == NEW for g in OLD) for i in joint),
                "both_old_instructions_first_new_right": sum(all(post[i, g]["first_goal"] == NEW for g in OLD) for i in joint)},
            "new_right_gain": {k: new_after[k] - new_before[k] for k in ("success_ever", "terminal_correct")},
            "new_right_descriptive_screen_16_of_20": {"ever_primary": new_after["success_ever"] >= 16,
                                                      "terminal_secondary": new_after["terminal_correct"] >= 16},
            "interpretation": "Exploratory pilot on reused states 20-39 from a non-qualified parent. Report wrong-new-target, "
                              "late result loss and unfinished separately on actual pre-success denominators."}


def report_lines(branches, parent_summary, comparisons):
    lines = ["# 2A：封存双目标父模型 → " + "／".join(branches) + "（右格12000步）", "",
             "探索性先导：父模型未通过正式资格，配对初态20—39有历史使用。每模型四指令各20回合，均跑满400步。",
             "主指标为曾成功与首次到达目标（等价成功即停）；400步终态为副指标。", "",
             "| 模型 | 左 曾成功（终态） | 前 曾成功（终态） | 右 曾成功（终态） | 后 曾成功（终态） | 左／前指令首次到右 |",
             "|---|---|---|---|---|---|"]

    def line(name, summary):
        c, f = summary["task_success_counts"], summary["first_goal_matrix"]
        cells = [f"{c[g]['success_ever']}/20（{c[g]['terminal_correct']}）" for g in GOALS]
        return f"| {name} | " + " | ".join(cells) + f" | {f['left']['right']}／{f['front']['right']} |"
    lines.append(line("父模型（复用）", parent_summary))
    lines += [line(b, comparisons[b]["branch"]) for b in branches]
    lines += ["", "原先成功分母上的保持、首次转右、从未到达任一目标，以及共同初态见各分支`comparison.json`。"]
    return lines
