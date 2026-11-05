"""Guard the R0 scientific denominators and selection/qualification boundaries."""

import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from stage1_dual_goal_run import final_qualification, summarize_dual, validate_config


def config():
    return {'old_goals': ['left', 'front'],
            'data': {'train_output': 'joint_train', 'action_alignment': 'post_action_obs_to_next_action'},
            'training': {'max_steps': 18000, 'save_every': 6000, 'samples_per_goal': 3},
            'evaluation': {'development_indices': list(range(20)), 'qualification_indices': list(range(20, 40)),
                           'paired_indices': [], 'reserve_indices': list(range(40, 50)), 'horizon': 400,
                           'development_required_correct': 16, 'qualification_required_correct': 16},
            'dual_goal': {'candidate_steps': [6000, 12000, 18000], 'development_joint_required': 16,
                          'final_required': 16, 'qualification_subset': list(range(20, 30)),
                          'qualification_required': 8}}


def episodes(indices, correct_left=None, correct_front=None):
    successes = {'left': set(indices) if correct_left is None else set(correct_left),
                 'front': set(indices) if correct_front is None else set(correct_front)}
    rows = []
    for index in indices:
        for goal in ('left', 'front'):
            good = index in successes[goal]
            rows.append({'init_index': index, 'requested_goal': goal,
                         'steps': 400, 'seed': 1234 + index,
                         'initial_state_sha256': f'state{index}', 'scene_xml_sha256': 'common_scene',
                         'correct_ever': good, 'terminal_category': 'correct' if good else 'unfinished',
                         'terminal_goal': goal if good else None})
    return rows


class DualGoalR0Tests(unittest.TestCase):
    def test_individual_eighty_percent_is_not_joint_eighty_percent(self):
        # Both marginal scores are 16/20; complementary failures leave only 12 paired successes.
        rows = episodes(range(20), range(16), range(4, 20))
        score = summarize_dual(rows, list(range(20)), 16)
        self.assertEqual(score['task_success_counts']['left']['terminal_correct'], 16)
        self.assertEqual(score['task_success_counts']['front']['terminal_correct'], 16)
        self.assertEqual(score['joint_terminal_correct'], 12)
        self.assertFalse(score['passed'])

    def test_success_lost_after_contact_does_not_pass_terminal_gate(self):
        rows = episodes(range(20), range(15), range(15))
        for row in rows:
            row['correct_ever'] = True
        score = summarize_dual(rows, list(range(20)), 16)
        self.assertEqual(score['joint_success_ever'], 20)
        self.assertEqual(score['joint_terminal_correct'], 15)
        self.assertFalse(score['passed'])

    def test_missing_duplicate_extra_and_changed_horizon_are_rejected(self):
        good = episodes(range(20))
        cases = [good[:-1], good + [copy.deepcopy(good[0])]]
        for field, value in [('init_index', 50), ('steps', 399)]:
            changed = copy.deepcopy(good); changed[0][field] = value; cases.append(changed)
        for rows in cases:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                summarize_dual(rows, list(range(20)), 16)

    def test_same_index_with_different_actual_resets_or_rng_is_rejected(self):
        for field in ('initial_state_sha256', 'scene_xml_sha256', 'seed'):
            rows = episodes(range(20)); rows[1][field] = 'mismatch'
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize_dual(rows, list(range(20)), 16)

    def test_distinct_index_labels_cannot_repeat_the_same_actual_initial_state(self):
        rows = episodes(range(20))
        for row in rows:
            if row['init_index'] == 1:
                row['initial_state_sha256'] = 'state0'
        with self.assertRaises(ValueError):
            summarize_dual(rows, list(range(20)), 16)

    def test_wrong_target_and_unfinished_remain_in_denominator(self):
        rows = episodes(range(20), range(16), range(16))
        rows[-1].update(terminal_category='wrong_goal', terminal_goal='left')
        score = summarize_dual(rows, list(range(20)), 16)
        self.assertTrue(score['passed'])
        self.assertEqual(score['episodes'], 40)
        self.assertEqual(score['terminal_matrix']['front']['left'], 1)
        self.assertEqual(score['terminal_matrix']['front']['unfinished'], 3)

    def test_final_success_cannot_retroactively_pass_failed_development(self):
        result = final_qualification(False, episodes(range(20, 40)), config())
        self.assertTrue(result['final_twenty']['passed'])
        self.assertTrue(result['qualification_subset_twenty_to_twenty_nine']['passed'])
        self.assertFalse(result['qualified_R0'])

    def test_mixed_final_twenty_cannot_hide_failure_on_fresh_ten(self):
        # Seven fresh + ten historical successes give 17/20, but fresh subset fails 8/10.
        indices = list(range(20, 40)); good = list(range(20, 27)) + list(range(30, 40))
        result = final_qualification(True, episodes(indices, good, good), config())
        self.assertTrue(result['final_twenty']['passed'])
        self.assertEqual(result['qualification_subset_twenty_to_twenty_nine']['joint_terminal_correct'], 7)
        self.assertFalse(result['qualified_R0'])

    def test_valid_joint_r0_passes_all_preregistered_gates(self):
        result = final_qualification(True, episodes(range(20, 40)), config())
        self.assertTrue(result['qualified_R0'])
        self.assertEqual(result['final_twenty']['initial_states'], 20)
        self.assertEqual(result['historically_exposed_subset_thirty_to_thirty_nine']['episodes'], 20)

    def test_budget_partition_and_inherited_normalization_changes_fail_before_training(self):
        original = config(); validate_config(original)
        cases = []
        changed = copy.deepcopy(original); changed['dual_goal']['candidate_steps'].append(24000); cases.append(changed)
        changed = copy.deepcopy(original); changed['evaluation']['paired_indices'] = list(range(20, 30)); cases.append(changed)
        changed = copy.deepcopy(original); changed['data']['normalization_source'] = 'old_left_train'; cases.append(changed)
        changed = copy.deepcopy(original); changed['dual_goal']['qualification_subset'] = list(range(30, 40)); cases.append(changed)
        for changed in cases:
            with self.assertRaises(ValueError):
                validate_config(changed)


if __name__ == '__main__':
    unittest.main()
