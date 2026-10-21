import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class Stage1ProtocolTests(unittest.TestCase):
    def protocol(self):
        self.assertIsNotNone(importlib.util.find_spec("stage1_protocol"), "Stage-one protocol is not implemented")
        import stage1_protocol
        return stage1_protocol

    def test_split_is_reproducible_disjoint_and_uses_every_trajectory(self):
        p = self.protocol()
        demos = [f"demo_{i}" for i in range(50)]
        split = p.split_trajectories(demos, 40, 7)
        self.assertEqual(split, p.split_trajectories(demos, 40, 7))
        self.assertEqual(len(split["train"]), 40)
        self.assertEqual(len(split["validation"]), 10)
        self.assertFalse(set(split["train"]) & set(split["validation"]))
        self.assertEqual(set(split["train"] + split["validation"]), set(demos))

    def test_split_rejects_duplicate_trajectory_ids(self):
        with self.assertRaises(ValueError):
            self.protocol().split_trajectories(["demo_0", "demo_0", "demo_1"], 1, 0)

    def test_correct_then_wrong_is_recorded_as_late_destruction(self):
        p = self.protocol()
        r = p.summarize_events("left", [{}, {"left": True}, {"back": True}])
        self.assertEqual(r["first_goal"], "left")
        self.assertEqual(r["terminal_goal"], "back")
        self.assertTrue(r["correct_ever"])
        self.assertTrue(r["correct_then_destroyed"])
        self.assertEqual(r["terminal_category"], "wrong_goal")

    def test_multiple_targets_are_ambiguous_and_not_forced_to_one_goal(self):
        r = self.protocol().summarize_events("left", [{}, {"left": True, "front": True}])
        self.assertEqual(r["terminal_category"], "ambiguous")
        self.assertIsNone(r["terminal_goal"])

    def test_initial_success_is_rejected_as_invalid_reset(self):
        with self.assertRaises(ValueError):
            self.protocol().summarize_events("left", [{"front": True}, {"left": True}])

    def test_execution_failure_remains_in_qualification_denominator(self):
        p = self.protocol()
        rows = [{"requested_goal": "left", "terminal_category": "correct"}] * 15
        rows += [{"requested_goal": "left", "terminal_category": "unfinished"}] * 5
        self.assertFalse(p.qualification(rows, ["left"], 20, 16)["qualified"])
        rows[15] = {"requested_goal": "left", "terminal_category": "correct"}
        self.assertTrue(p.qualification(rows, ["left"], 20, 16)["qualified"])


if __name__ == "__main__":
    unittest.main()
