import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from stage1_summarize_dual_unseen import GOALS, parent_summary


def sample(index, goal, target):
    return {'init_index': index, 'requested_goal': goal, 'seed': 100+index,
            'initial_state_sha256': f'state{index}', 'scene_xml_sha256': 'same', 'steps': 400,
            'first_goal': target, 'first_goal_times': {g: 50 if g == target else None for g in GOALS},
            'correct_ever': target == goal, 'correct_then_destroyed': False,
            'terminal_goal': target,
            'terminal_category': 'correct' if goal == target else 'wrong_goal' if target else 'unfinished'}


def fixture():
    return [sample(i, g, g if g in ('left', 'front') else 'left') for i in (20, 21) for g in GOALS]


class DualUnseenTests(unittest.TestCase):
    def test_old_bias_is_counted_for_both_untrained_instructions(self):
        result = parent_summary(fixture(), [20, 21], 100)
        self.assertEqual(result['joint_old_terminal_correct'], 2)
        self.assertEqual(result['unseen_instructions']['total'], 4)
        self.assertEqual(result['unseen_instructions']['requested_success_ever'], 0)
        self.assertEqual(result['unseen_instructions']['terminal_old_target'], 4)

    def test_transient_old_arrival_survives_unfinished_terminal_state(self):
        items = fixture()
        selected = next(r for r in items if r['init_index'] == 20 and r['requested_goal'] == 'right')
        selected.update(terminal_goal=None, terminal_category='unfinished')
        result = parent_summary(items, [20, 21], 100)
        self.assertEqual(result['unseen_instructions']['first_old_target'], 4)
        self.assertEqual(result['unseen_instructions']['terminal_old_target'], 3)

    def test_old_arrival_then_requested_success_is_not_a_failed_request(self):
        items = fixture()
        selected = next(r for r in items if r['init_index'] == 20 and r['requested_goal'] == 'right')
        selected['first_goal_times']['right'] = 100
        selected.update(correct_ever=True, terminal_goal='right', terminal_category='correct')
        result = parent_summary(items, [20, 21], 100)
        self.assertEqual(result['unseen_instructions']['requested_terminal_correct'], 1)
        self.assertEqual(result['unseen_instructions']['old_target_ever_without_requested_success'], 3)

    def test_missing_duplicate_or_mismatched_back_state_is_rejected(self):
        good = fixture()
        bad_reset = copy.deepcopy(good)
        next(r for r in bad_reset if r['requested_goal'] == 'back')['initial_state_sha256'] = 'other'
        for items in (good[:-1], good+[good[0]], bad_reset):
            with self.subTest(items=len(items)), self.assertRaises(ValueError):
                parent_summary(items, [20, 21], 100)


if __name__ == '__main__':
    unittest.main()
