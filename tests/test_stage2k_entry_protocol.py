import importlib.util
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stage2k_entry_handoff import entry_eligible, summarize_handoffs


class EntryProtocolTests(unittest.TestCase):
    def eligible(self, **kwargs):
        args = dict(step=30, position=[0.05, 0.0, 0.10], initial_height=0.0,
                    centre=[0.0, 0.0, 0.0], flags={"left": False}, reached_before=False)
        args.update(kwargs)
        return entry_eligible(**args)

    def test_entry_requires_planning_boundary_before_any_goal(self):
        self.assertTrue(self.eligible())
        self.assertFalse(self.eligible(step=31))
        self.assertFalse(self.eligible(step=10))
        self.assertFalse(self.eligible(flags={"right": True}))
        self.assertFalse(self.eligible(reached_before=True))

    def test_near_but_not_lifted_or_outside_is_not_entrance(self):
        self.assertFalse(self.eligible(position=[0.05, 0, 0.02]))
        self.assertFalse(self.eligible(position=[0.11, 0, 0.10]))
        with self.assertRaises(ValueError):
            self.eligible(position=[float("nan"), 0, 0.1])

    def test_requested_entrance_must_be_unique_nearest(self):
        self.assertFalse(self.eligible(competing_centres=[[0.06, 0, 0]]))
        self.assertFalse(self.eligible(competing_centres=[[0.10, 0, 0]]))
        self.assertTrue(self.eligible(competing_centres=[[0.20, 0, 0], [0, 0.20, 0]]))

    def test_missing_entrances_stay_in_prefix_denominator(self):
        prefix = [{"source": "teacher", "goal": "left", "entry_step": 50},
                  {"source": "teacher", "goal": "left", "entry_step": None}]
        row = {"source": "teacher", "goal": "left", "continuation_policy": "student",
               "correct_ever": False, "first_goal": "right", "terminal_category": "wrong_goal",
               "correct_then_destroyed": False}
        cell = summarize_handoffs(prefix, [row])["teacher:left"]
        self.assertEqual((cell["prefixes"], cell["entrances"], cell["missing_entrance"]), (2, 1, 1))
        self.assertEqual(cell["continuations"]["student"]["total"], 1)
        self.assertEqual(cell["continuations"]["student"]["first_goal_counts"]["right"], 1)


if __name__ == "__main__":
    unittest.main()
