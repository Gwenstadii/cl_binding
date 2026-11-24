"""Timing is the only schedule variable, and pre-onset events are confirmed."""

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("timing", ROOT / "scripts/k25/v_timing_protocol.py")
TIMING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TIMING)


class TimingControls(unittest.TestCase):
    def test_early_vs_delayed_have_equal_exposure(self):
        for name in TIMING.FIXED_LENGTH:
            self.assertEqual(len(TIMING.active_blocks(name))*10, 100)
        self.assertEqual(TIMING.active_blocks("full_400"), list(range(40)))
        self.assertEqual(TIMING.active_blocks("delayed_100"), list(range(8, 18)))

    def test_common_prefix_until_the_first_different_injection(self):
        self.assertEqual(TIMING.required_common_blocks("native", "delayed_100"), 8)
        self.assertEqual(TIMING.required_common_blocks("early_100", "full_400"), 10)
        self.assertEqual(TIMING.required_common_blocks("delayed_100", "delayed_tail"), 18)
        self.assertEqual(TIMING.required_common_blocks("late_100", "late_tail"), 26)

    def test_three_frames_before_onset_are_not_a_confirmed_wrong_arrival(self):
        masks = [0]*400
        masks[77:82] = [2]*5  # Front at controls 78..82; cannot be known at 80.
        self.assertEqual(TIMING.movement(masks, 0, 80)["first"], [])
        self.assertEqual(TIMING.movement(masks)["first"], ["front"])
        self.assertEqual(TIMING.movement(masks)["first_step"], 78)

    def test_a_correct_post_onset_route_cannot_erase_an_earlier_wrong_first(self):
        masks = [0]*400
        masks[50:55] = [2]*5
        masks[110:115] = [4]*5
        self.assertEqual(TIMING.movement(masks)["first"], ["front"])
        self.assertEqual(TIMING.movement(masks, 80)["first"], ["right"])

    def test_grasp_before_onset_requires_the_whole_stable_run(self):
        trace = dict(object=[[0., 0., .1] for _ in range(400)], contacts=[0]*400,
                     regions=[0]*400, start=dict(object=[0., 0., 0.]))
        trace["contacts"][77:82] = [3]*5
        geometry = {g: dict(centre=[0., 0., 0.]) for g in TIMING.SITES}
        m = TIMING.interval_metrics(trace, geometry, 0., 80, 180)
        self.assertIsNone(m["pre_grasp"])
        m = TIMING.interval_metrics(trace, geometry, 0., 90, 190)
        self.assertEqual(m["pre_grasp_confirmed_step"], 82)

    def test_grid_keeps_both_signs_at_every_window(self):
        self.assertEqual(len(TIMING.jobs()), 15)
        self.assertEqual(len(TIMING.LINES)*len(TIMING.IDS)*2*len(TIMING.jobs()), 1200)
        for name in TIMING.WINDOWS:
            self.assertIn((name, "forward"), TIMING.jobs())
            self.assertIn((name, "reverse"), TIMING.jobs())


if __name__ == "__main__":
    unittest.main()
