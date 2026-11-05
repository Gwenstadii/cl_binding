import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from stage1_four_condition_protocol import compare_acquired, compare_snapshots, summarize_snapshot
from stage1_summarize_dual_unseen import GOALS


INDICES = [20, 21]
SEED = 1234


def row(index, goal, target):
    return {
        'init_index': index, 'requested_goal': goal, 'seed': SEED + index, 'steps': 400,
        'initial_state_sha256': f'state{index}', 'scene_xml_sha256': 'common',
        'first_goal': target,
        'first_goal_times': {g: 50 if g == target else None for g in GOALS},
        'correct_ever': target == goal, 'correct_then_destroyed': False,
        'terminal_goal': target,
        'terminal_category': 'correct' if target == goal else 'wrong_goal' if target else 'unfinished',
    }


def snapshot(target_for):
    return [row(i, g, target_for(i, g)) for i in INDICES for g in GOALS]


def acquired(items):
    return {g: [copy.deepcopy(r) for r in items if r['requested_goal'] == g]
            for g in ('left', 'front', 'right')}


def selected(items, index, goal):
    return next(r for r in items if r['init_index'] == index and r['requested_goal'] == goal)


class FourConditionTests(unittest.TestCase):
    def test_summary_has_all_counts_matrices_and_joint_indices(self):
        items = snapshot(lambda i, g: g if i == 20 or g != 'front' else None)
        result = summarize_snapshot(items, INDICES, SEED)
        self.assertEqual(result['episodes'], 8)
        self.assertEqual(result['joint_left_front_terminal_correct'], 1)
        self.assertEqual(result['joint_left_front_terminal_correct_indices'], [20])
        self.assertEqual(result['task_success_counts']['front']['terminal_correct'], 1)
        self.assertEqual(result['first_matrix']['front']['unreached'], 1)
        self.assertEqual(result['terminal_matrix']['front']['unfinished'], 1)
        for goal in GOALS:
            self.assertEqual(sum(result['first_matrix'][goal].values()), 2)
            self.assertEqual(sum(result['terminal_matrix'][goal].values()), 2)

    def test_first_and_terminal_matrices_keep_ambiguous_states_separate(self):
        items = snapshot(lambda i, g: None)
        chosen = selected(items, 20, 'right')
        chosen['first_goal_times'].update(left=50, front=50)
        chosen['terminal_category'] = 'ambiguous'
        result = summarize_snapshot(items, INDICES, SEED)
        self.assertEqual(result['first_matrix']['right']['ambiguous'], 1)
        self.assertEqual(result['terminal_matrix']['right']['ambiguous'], 1)

    def test_zero_success_immediately_before_cannot_be_new_forgetting(self):
        before = snapshot(lambda i, g: 'right')
        after = snapshot(lambda i, g: 'back')
        result = compare_snapshots(before, after, INDICES, SEED)
        self.assertEqual(result['matched_pairs'], 8)
        for goal in ('left', 'front'):
            for value in result['retained_goals'][goal].values():
                self.assertEqual((value['total'], value['lost']), (0, 0))
                self.assertIsNone(value['retention_rate'])
                self.assertEqual(value['lost_terminal_new_target'], 0)
        right = result['retained_goals']['right']['previously_terminal_correct']
        self.assertEqual((right['total'], right['lost'], right['lost_terminal_new_target']), (2, 2, 2))

    def test_right_only_terminal_successes_enter_terminal_loss_denominator(self):
        before = snapshot(lambda i, g: g)
        chosen = selected(before, 21, 'right')
        chosen.update(terminal_goal=None, terminal_category='unfinished', correct_then_destroyed=True)
        result = compare_snapshots(before, snapshot(lambda i, g: 'back'), INDICES, SEED)
        right = result['retained_goals']['right']
        self.assertEqual(right['previously_successful_ever']['total'], 2)
        self.assertEqual(right['previously_terminal_correct']['total'], 1)
        self.assertEqual(right['previously_terminal_correct']['eligible_indices'], [20])

    def test_requested_success_then_new_target_keeps_ever_but_loses_terminal(self):
        before = snapshot(lambda i, g: g)
        after = snapshot(lambda i, g: g)
        chosen = selected(after, 20, 'left')
        chosen['first_goal_times']['back'] = 100
        chosen.update(terminal_goal='back', terminal_category='wrong_goal', correct_then_destroyed=True)
        result = compare_snapshots(before, after, INDICES, SEED)['retained_goals']['left']
        self.assertEqual(result['previously_successful_ever']['lost'], 0)
        terminal = result['previously_terminal_correct']
        self.assertEqual((terminal['lost'], terminal['lost_terminal_new_target']), (1, 1))
        self.assertEqual(terminal['lost_first_new_target'], 0)
        self.assertEqual(terminal['retention_rate'], 0.5)

    def test_first_new_target_followed_by_correct_old_terminal_is_retained(self):
        before = snapshot(lambda i, g: g)
        after = snapshot(lambda i, g: g)
        chosen = selected(after, 20, 'left')
        chosen['first_goal_times']['back'] = 10
        chosen['first_goal'] = 'back'
        result = compare_snapshots(before, after, INDICES, SEED)['retained_goals']['left']
        self.assertEqual(result['previously_terminal_correct']['lost'], 0)
        self.assertEqual(result['previously_terminal_correct']['lost_first_new_target'], 0)

    def test_historical_acquisition_preserves_distinct_old_success_denominators(self):
        historical = acquired(snapshot(lambda i, g: g))
        historical['right'][1] = row(21, 'right', None)
        after = snapshot(lambda i, g: 'back')
        result = compare_acquired(historical, after, INDICES, SEED)
        self.assertEqual(result['matched_pairs'], 6)
        self.assertEqual(result['retained_goals']['left']['previously_terminal_correct']['total'], 2)
        self.assertEqual(result['retained_goals']['front']['previously_terminal_correct']['total'], 2)
        self.assertEqual(result['retained_goals']['right']['previously_terminal_correct']['total'], 1)

    def test_each_terminal_failure_bin_and_first_target_are_separate(self):
        before = snapshot(lambda i, g: g)
        cases = [('back', 'wrong_goal', 'lost_terminal_new_target'),
                 ('right', 'wrong_goal', 'lost_terminal_other_wrong'),
                 (None, 'unfinished', 'lost_terminal_unfinished'),
                 (None, 'ambiguous', 'lost_terminal_ambiguous')]
        for target, category, expected in cases:
            with self.subTest(category=category, target=target):
                after = snapshot(lambda i, g: g)
                chosen = selected(after, 20, 'left')
                replacement = row(20, 'left', 'back')
                replacement.update(terminal_goal=target, terminal_category=category)
                if target == 'right':
                    replacement['first_goal_times']['right'] = 100
                chosen.update(replacement)
                value = compare_snapshots(before, after, INDICES, SEED)['retained_goals']['left']['previously_terminal_correct']
                self.assertEqual(value['lost'], 1)
                self.assertEqual(value[expected], 1)
                self.assertEqual(value['lost_first_new_target'], 1)
                self.assertEqual(sum(value[k] for k in ('lost_terminal_new_target', 'lost_terminal_other_wrong',
                                                       'lost_terminal_unfinished', 'lost_terminal_ambiguous')), 1)

    def test_historical_acquisition_and_after_physical_states_must_match(self):
        before = acquired(snapshot(lambda i, g: g))
        after = snapshot(lambda i, g: 'back')
        for field, value in [('initial_state_sha256', 'other-state'), ('scene_xml_sha256', 'other-scene'),
                             ('seed', 999), ('steps', 399)]:
            changed = copy.deepcopy(before)
            for item in changed['left']:
                item[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_acquired(changed, after, INDICES, SEED)

    def test_individually_valid_historical_acquisition_state_mismatch_is_rejected(self):
        before = acquired(snapshot(lambda i, g: g))
        before['left'][0]['initial_state_sha256'] = 'another-unique-state'
        with self.assertRaisesRegex(ValueError, 'Acquisition/after mismatch'):
            compare_acquired(before, snapshot(lambda i, g: 'back'), INDICES, SEED)

    def test_individually_paired_snapshots_must_match_including_new_back_instruction(self):
        before = snapshot(lambda i, g: g)
        after = snapshot(lambda i, g: 'back')
        for item in after:
            if item['init_index'] == 20:
                item['initial_state_sha256'] = 'different-but-internally-paired'
        with self.assertRaisesRegex(ValueError, 'Acquisition/after mismatch'):
            compare_snapshots(before, after, INDICES, SEED)

    def test_missing_duplicate_or_unexpected_snapshot_conditions_are_rejected(self):
        good = snapshot(lambda i, g: g)
        duplicate_replacing_missing = good[:-1] + [copy.deepcopy(good[0])]
        unexpected = copy.deepcopy(good)
        unexpected[0]['init_index'] = 30
        for items in (good[:-1], good + [good[0]], duplicate_replacing_missing, unexpected):
            with self.subTest(length=len(items)), self.assertRaises(ValueError):
                summarize_snapshot(items, INDICES, SEED)

    def test_missing_duplicate_or_wrong_acquisition_instruction_is_rejected(self):
        good = acquired(snapshot(lambda i, g: g))
        cases = []
        missing = copy.deepcopy(good)
        missing['right'].pop()
        cases.append(missing)
        duplicate = copy.deepcopy(good)
        duplicate['front'][1] = copy.deepcopy(duplicate['front'][0])
        cases.append(duplicate)
        wrong = copy.deepcopy(good)
        wrong['left'][0] = row(20, 'back', 'back')
        cases.append(wrong)
        for baselines in cases:
            with self.subTest(baselines=baselines.keys()), self.assertRaises(ValueError):
                compare_acquired(baselines, snapshot(lambda i, g: 'back'), INDICES, SEED)

    def test_acquisition_mapping_must_have_exactly_three_retained_goals(self):
        good = acquired(snapshot(lambda i, g: g))
        missing = copy.deepcopy(good)
        del missing['left']
        extra = copy.deepcopy(good)
        extra['back'] = [row(i, 'back', 'back') for i in INDICES]
        for baselines in (missing, extra):
            with self.assertRaises(ValueError):
                compare_acquired(baselines, snapshot(lambda i, g: 'back'), INDICES, SEED)

    def test_duplicate_and_empty_indices_are_rejected(self):
        for indices in ([], [20, 20]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                summarize_snapshot(snapshot(lambda i, g: g), indices, SEED)

    def test_seed_horizon_inconsistent_classification_and_same_physical_state_are_rejected(self):
        good = snapshot(lambda i, g: g)
        cases = []
        for field, value in [('seed', 111), ('steps', 399), ('correct_ever', False)]:
            changed = copy.deepcopy(good)
            changed[0][field] = value
            cases.append(changed)
        duplicate_state = copy.deepcopy(good)
        for item in duplicate_state:
            item['initial_state_sha256'] = 'identical'
        cases.append(duplicate_state)
        for items in cases:
            with self.assertRaises(ValueError):
                summarize_snapshot(items, INDICES, SEED)


if __name__ == '__main__':
    unittest.main()
