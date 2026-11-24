from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/k26'))
from back_training_protocol import learning_rate,phase_ranges,sample_indices


class BackContinuationProtocol(unittest.TestCase):
    def test_phase_batches_are_real_frames_with_no_cross_episode_tail_leakage(self):
        ranges,n=phase_ranges([dict(goal='back',frames=10),dict(goal='back',frames=20)],.3)
        self.assertEqual(n,30)
        for step in range(100):
            ids=sample_indices(ranges,20261124,step)
            self.assertTrue(all(0<=i<30 for i in ids[:3]))
            self.assertTrue(all(any(tail<=i<end for _,end,tail in ranges) for i in ids[3:]))
            self.assertEqual(ids,sample_indices(ranges,20261124,step))

    def test_only_back_training_episodes_enter_sampling(self):
        with self.assertRaises(AssertionError):
            phase_ranges([dict(goal='left',frames=20)])

    def test_schedule_warms_up_then_decays_without_reset(self):
        self.assertLess(learning_rate(1),learning_rate(300))
        self.assertAlmostEqual(learning_rate(300),5e-5)
        self.assertGreater(learning_rate(301),learning_rate(12000))
        self.assertGreater(learning_rate(12000),learning_rate(24000))
        self.assertAlmostEqual(learning_rate(24000),5e-6)


if __name__=='__main__':unittest.main()
