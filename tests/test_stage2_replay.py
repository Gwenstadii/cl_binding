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
        seeds = config['seeds']
        self.assertEqual((seeds['split_and_inference'], seeds['memory'], seeds['new_samples'], seeds['old_samples']),
                         (20261021, 20261024, 20261024, 20261025))


if __name__ == '__main__':
    unittest.main()
