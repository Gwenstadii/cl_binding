import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from stage2_protocol import (BRANCHES, LOSS_DENOMINATOR, OLD_GOALS, check_budget, exposures, flow_time_from_uniform,
                             loss_weights, memory_record_bytes, select_memory, step_plan, stream_seed)

SEEDS = {'new_samples': 20261024, 'old_samples': 20261025}
POOLS = {'left': list(range(256)), 'front': list(range(256, 512))}


class MemorySelectionTests(unittest.TestCase):
    def test_memory_is_uniform_without_replacement_and_reproducible(self):
        groups = {'left': list(range(0, 5000)), 'front': list(range(5000, 10937))}
        first, second = select_memory(groups, 256, 20261024), select_memory(groups, 256, 20261024)
        self.assertEqual(first, second)
        for goal in OLD_GOALS:
            self.assertEqual(len(set(first[goal])), 256)
            self.assertTrue(set(first[goal]) <= set(groups[goal]))
        self.assertNotEqual(first, select_memory(groups, 256, 20261025))

    def test_budget_sweep_memories_are_nested(self):
        from stage2_protocol import nested_subset
        positions = {'left': list(range(256)), 'front': list(range(256, 512))}
        sizes = [nested_subset(positions, k, 20261026) for k in (1, 4, 16, 64, 256)]
        for small, large in zip(sizes, sizes[1:]):
            for goal in OLD_GOALS:
                self.assertTrue(set(small[goal]) <= set(large[goal]))
        self.assertEqual(sizes[-1]['left'], positions['left'])

    def test_memory_refuses_new_goal_or_overdraw(self):
        with self.assertRaises(ValueError):
            select_memory({'left': [1, 2], 'right': [3, 4]}, 1, 1)
        with self.assertRaises(ValueError):
            select_memory({'left': [1, 2], 'front': [3]}, 2, 1)

    def test_planned_memory_fits_budget_before_container_overhead(self):
        record = memory_record_bytes((3, 128, 128), 2, 50, 7, 8)
        self.assertEqual(record, 2 * 49152 + 1400 + 50 + 32)
        self.assertLess(512 * record, 64 * 2**20)
        self.assertFalse(check_budget({'memory.npz': 64 * 2**20, 'memory_manifest.json': 1}, 64 * 2**20)['within_budget'])


class MatchedSamplingTests(unittest.TestCase):
    def test_all_branches_consume_identical_new_records_every_step(self):
        for step in (0, 1, 2999, 11999):
            plans = {b: step_plan(b, step, SEEDS, range(5577), POOLS) for b in BRANCHES}
            self.assertEqual(plans['ft4']['new'], plans['er']['new'])
            self.assertEqual(plans['er']['new'], plans['full_replay']['new'])
            self.assertEqual(len(plans['ft4']['new']), 4)

    def test_old_draws_do_not_depend_on_or_shift_new_draws(self):
        other_pools = {'left': list(range(10)), 'front': list(range(10, 20))}
        a = step_plan('er', 7, SEEDS, range(5577), POOLS)
        b = step_plan('er', 7, SEEDS, range(5577), other_pools)
        self.assertEqual(a['new'], b['new'])
        changed = step_plan('er', 7, {**SEEDS, 'old_samples': 1}, range(5577), POOLS)
        self.assertEqual(a['new'], changed['new'])
        self.assertEqual((len(a['left']), len(a['front'])), (1, 1))
        self.assertEqual(step_plan('ft4', 7, SEEDS, range(5577), POOLS)['left'], [])

    def test_flow_streams_are_distinct_and_step_keyed(self):
        self.assertEqual(stream_seed('new-flow', 1, 5), stream_seed('new-flow', 1, 5))
        self.assertNotEqual(stream_seed('new-flow', 1, 5), stream_seed('new-flow', 1, 6))
        self.assertNotEqual(stream_seed('new-flow', 1, 5), stream_seed('left-flow', 1, 5))


class LossAndExposureTests(unittest.TestCase):
    def test_every_record_uses_the_fixed_six_denominator(self):
        self.assertEqual(LOSS_DENOMINATOR, 6)
        self.assertEqual(loss_weights('ft4'), [1 / 6] * 4)
        self.assertEqual(loss_weights('er'), [1 / 6] * 6)
        self.assertAlmostEqual(sum(loss_weights('ft4')), 4 / 6)

    def test_new_exposure_is_matched_and_old_exposure_is_reported(self):
        self.assertEqual(exposures('ft4', 12000), {'new': 48000, 'left': 0, 'front': 0})
        self.assertEqual(exposures('er', 12000), {'new': 48000, 'left': 12000, 'front': 12000})

    def test_flow_time_matches_smolvla_beta_range(self):
        self.assertAlmostEqual(flow_time_from_uniform(0.0), 0.001)
        self.assertAlmostEqual(flow_time_from_uniform(1.0), 1.0)
        # Beta(1.5, 1) median is 0.5 ** (1 / 1.5).
        self.assertAlmostEqual(flow_time_from_uniform(0.5), 0.5 ** (2 / 3) * 0.999 + 0.001)


def episode(index, goal, first, terminal, ever):
    category = 'correct' if terminal == goal else 'unfinished' if terminal is None else 'wrong_goal'
    return {'init_index': index, 'requested_goal': goal, 'seed': 20261021 + index, 'steps': 400,
            'initial_state_sha256': f's{index}', 'scene_xml_sha256': 'x', 'first_goal': first,
            'terminal_goal': terminal, 'terminal_category': category, 'correct_ever': ever,
            'correct_then_destroyed': ever and terminal != goal}


class ComparisonTests(unittest.TestCase):
    def rows(self, outcome):
        return [episode(i, g, *outcome(i, g)) for i in (20, 21) for g in ('left', 'front', 'right', 'back')]

    def test_replacement_unfinished_and_retention_are_separated_on_prior_success(self):
        from stage2_compare import compare
        parent = self.rows(lambda i, g: (g, g, True) if g in ('left', 'front') else ('left', None, False))
        def branch(i, g):
            if g == 'left':
                return ('left', 'left', True) if i == 20 else ('right', 'right', False)
            if g == 'front':
                return (None, None, False)
            return ('right', 'right', True)
        result = compare(parent, self.rows(branch), [20, 21], 20261021)
        left = result['old_goal_transitions']['left']['previously_terminal_correct']
        self.assertEqual((left['total'], left['retained'], left['lost_and_terminal_new_right']), (2, 1, 1))
        front = result['old_goal_transitions']['front']['previously_terminal_correct']
        self.assertEqual((front['lost'], front['lost_and_terminal_unfinished']), (2, 2))
        self.assertEqual(result['previously_joint_correct_states']['total'], 2)
        self.assertEqual(result['previously_joint_correct_states']['retained_joint_terminal_correct'], 0)
        self.assertEqual(result['new_right_gain'], {'success_ever': 2, 'terminal_correct': 2})
        ever = result['old_goal_transitions']['front']['previously_successful_ever']
        self.assertEqual((ever['total'], ever['lost_and_no_goal_first'], ever['lost_and_first_new_right']), (2, 2, 0))
        self.assertEqual(result['previously_joint_ever_states']['both_old_instructions_first_new_right'], 0)
        self.assertTrue(result['metric_policy']['primary'].startswith('first goal'))

    def test_verified_scene_alias_is_accepted_but_unlisted_hash_is_not(self):
        from stage2_compare import compare
        parent = self.rows(lambda i, g: (g, g, True))
        branch = self.rows(lambda i, g: (g, g, True))
        for r in branch:
            r['scene_xml_sha256'] = 'worktree'
        with self.assertRaises(ValueError):
            compare(parent, branch, [20, 21], 20261021)
        result = compare(parent, branch, [20, 21], 20261021, {'worktree': 'x'})
        self.assertEqual(result['scene_xml_aliases'], {'worktree': 'x'})

    def test_mismatched_reset_is_rejected(self):
        from stage2_compare import compare
        parent = self.rows(lambda i, g: (g, g, True))
        branch = self.rows(lambda i, g: (g, g, True))
        branch[0]['initial_state_sha256'] = 'other'
        with self.assertRaises(ValueError):
            compare(parent, branch, [20, 21], 20261021)


class ConfigTests(unittest.TestCase):
    def test_registered_config_matches_protocol(self):
        config = json.loads((ROOT / 'configs/stage2a_replay_v1.json').read_text())
        t = config['training']
        self.assertEqual((t['max_steps'], t['new_per_step'], t['old_per_goal_per_step'], t['loss_denominator']), (12000, 4, 1, 6))
        self.assertEqual(config['memory']['budget_bytes'], 67108864)
        self.assertEqual(config['memory']['per_goal'], 256)
        self.assertEqual(config['data']['normalization_source'], config['memory']['source_dataset'])
        self.assertEqual(config['parent']['model_sha256'], '15df8ed2839c281ba5d55b411b43a15177013b4d12b0e5cb98baefdcd60313f7')
        self.assertEqual(config['evaluation']['paired_indices'], list(range(20, 40)))
        self.assertIn('stop-on-success', config['metric_policy']['primary'])
        seeds = config['seeds']
        self.assertEqual((seeds['split_and_inference'], seeds['memory'], seeds['new_samples'], seeds['old_samples']),
                         (20261021, 20261024, 20261024, 20261025))


if __name__ == '__main__':
    unittest.main()
