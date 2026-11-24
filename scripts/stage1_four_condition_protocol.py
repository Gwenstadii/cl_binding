"""Pure paired-state analysis for left/front/right acquisition and back adaptation.

An acquisition baseline can come from a different checkpoint for each old goal.
Only episodes that actually succeeded in that baseline enter its loss denominator.
This comparison describes loss since acquisition; the immediately preceding full
snapshot is needed to attribute an additional loss to the latest training stage.
"""

from stage1_summarize_dual_unseen import GOALS, counts, validate


RETAINED_GOALS = ('left', 'front', 'right')
MATCH_FIELDS = ('initial_state_sha256', 'scene_xml_sha256', 'seed', 'steps')


def _indices(values):
    result = list(values)
    if not result or len(set(result)) != len(result):
        raise ValueError('Initial-state indices must be nonempty and unique')
    return result


def _matched(before, after):
    for key, row in before.items():
        for field in MATCH_FIELDS:
            if row[field] != after[key][field]:
                raise ValueError(f'Acquisition/after mismatch: {key} {field}')


def _first_target(row):
    if row['first_goal'] is not None:
        return row['first_goal']
    return ('unreached' if all(t is None for t in row['first_goal_times'].values())
            else 'ambiguous')


def summarize_snapshot(rows, indices, seed):
    """Validate a complete four-instruction snapshot and return exact counts.

    Matrix rows are requested instructions and columns are observed targets.
    An absent first target is either unreached or ambiguous; an absent terminal
    target is classified as unfinished or ambiguous by the evaluator.
    """
    items, indices = list(rows), _indices(indices)
    keyed = validate(items, GOALS, indices, seed)
    first = {g: {t: 0 for t in (*GOALS, 'unreached', 'ambiguous')} for g in GOALS}
    terminal = {g: {t: 0 for t in (*GOALS, 'unfinished', 'ambiguous')} for g in GOALS}
    for row in items:
        goal = row['requested_goal']
        first[goal][_first_target(row)] += 1
        terminal[goal][row['terminal_goal'] or row['terminal_category']] += 1
    joint = [i for i in indices
             if all(keyed[i, g]['terminal_category'] == 'correct' for g in ('left', 'front'))]
    return {
        'episodes': len(items),
        'initial_states': len(indices),
        'indices': indices,
        'task_success_counts': {
            g: counts([row for row in items if row['requested_goal'] == g]) for g in GOALS
        },
        'first_matrix': first,
        'terminal_matrix': terminal,
        'joint_left_front_terminal_correct': len(joint),
        'joint_left_front_terminal_correct_indices': joint,
    }


def _retention(before, after, goal, indices, predicate, new_goal):
    eligible = [i for i in indices if predicate(before[i, goal])]
    retained = [i for i in eligible if predicate(after[i, goal])]
    lost = [i for i in eligible if not predicate(after[i, goal])]
    lost_rows = [after[i, goal] for i in lost]
    return {
        'total': len(eligible),
        'retained': len(retained),
        'lost': len(lost),
        'retention_rate': len(retained) / len(eligible) if eligible else None,
        'eligible_indices': eligible,
        'retained_indices': retained,
        'lost_indices': lost,
        # First arrival is a separate diagnostic; it can differ from the
        # terminal target and must not be added to the four terminal bins.
        'lost_first_new_target': sum(row['first_goal'] == new_goal for row in lost_rows),
        'lost_terminal_new_target': sum(row['terminal_goal'] == new_goal for row in lost_rows),
        'lost_terminal_unfinished': sum(row['terminal_category'] == 'unfinished' for row in lost_rows),
        'lost_terminal_ambiguous': sum(row['terminal_category'] == 'ambiguous' for row in lost_rows),
        'lost_terminal_other_wrong': sum(
            row['terminal_category'] == 'wrong_goal' and row['terminal_goal'] != new_goal
            for row in lost_rows
        ),
    }


def compare_acquired(acquisition_rows_by_goal, after, indices, seed, new_goal='back'):
    """Compare each old goal against its actual acquired-success episodes.

    Each mapping value contains exactly one instruction over all requested
    indices. Values may come from distinct acquisition checkpoints. The final
    snapshot must contain every instruction, including the new back goal.
    No failure that already existed in a goal's baseline counts as forgetting.
    Ever-success and terminal correctness use independent success denominators.
    """
    indices = _indices(indices)
    if set(acquisition_rows_by_goal) != set(RETAINED_GOALS):
        raise ValueError('Acquisition baselines must contain exactly left/front/right')
    if new_goal not in GOALS or new_goal in RETAINED_GOALS:
        raise ValueError('The new goal must be back, separate from retained goals')
    post = validate(list(after), GOALS, indices, seed)
    retained_goals = {}
    for goal in RETAINED_GOALS:
        pre = validate(list(acquisition_rows_by_goal[goal]), (goal,), indices, seed)
        _matched(pre, post)
        retained_goals[goal] = {
            label: _retention(pre, post, goal, indices, predicate, new_goal)
            for label, predicate in (
                ('previously_successful_ever', lambda row: row['correct_ever']),
                ('previously_terminal_correct', lambda row: row['terminal_category'] == 'correct'),
            )
        }
    return {
        'new_goal': new_goal,
        'matched_pairs': len(indices) * len(RETAINED_GOALS),
        'retained_goals': retained_goals,
    }


def compare_snapshots(before, after, indices, seed):
    """Compare the full snapshots immediately before and after back training."""
    before, after, indices = list(before), list(after), _indices(indices)
    pre_summary = summarize_snapshot(before, indices, seed)
    post_summary = summarize_snapshot(after, indices, seed)
    pre = {(row['init_index'], row['requested_goal']): row for row in before}
    post = {(row['init_index'], row['requested_goal']): row for row in after}
    _matched(pre, post)
    acquired = {g: [row for row in before if row['requested_goal'] == g] for g in RETAINED_GOALS}
    transitions = compare_acquired(acquired, after, indices, seed)
    return {
        'before': pre_summary,
        'after': post_summary,
        'matched_pairs': len(pre),
        'retained_goals': transitions['retained_goals'],
    }
