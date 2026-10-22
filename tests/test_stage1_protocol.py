import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class Stage1ProtocolTests(unittest.TestCase):
    def protocol(self):
        self.assertIsNotNone(importlib.util.find_spec("stage1_protocol"), "Stage-one protocol is not implemented")
        import stage1_protocol
        return stage1_protocol

    def test_post_action_observations_predict_next_action_without_crossing_trajectory(self):
        obs, actions = self.protocol().observation_action_indices(4, "post_action_obs_to_next_action")
        self.assertEqual(list(zip(obs, actions)), [(0, 1), (1, 2), (2, 3)])
        self.assertNotIn(3, obs, "The last observation has no next recorded action")

    def test_legacy_pairing_is_preserved_and_unknown_alignment_rejected(self):
        p = self.protocol()
        self.assertEqual(list(zip(*p.observation_action_indices(3, "same_index_legacy"))), [(0, 0), (1, 1), (2, 2)])
        with self.assertRaises(ValueError):p.observation_action_indices(3, "guess")
        with self.assertRaises(ValueError):p.observation_action_indices(1, "post_action_obs_to_next_action")

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

    def test_balanced_batch_is_reproducible_after_resume(self):
        p = self.protocol()
        groups = {"front": list(range(10)), "left": list(range(10, 20)), "right": list(range(20, 30))}
        batch = p.balanced_indices(groups, 2, 17, 100)
        self.assertEqual(batch, p.balanced_indices(groups, 2, 17, 100))
        self.assertEqual(len(batch), 6)
        self.assertEqual([sum(i in values for i in batch) for values in groups.values()], [2, 2, 2])

    def test_saved_demo_xml_maps_assets_without_changing_scene_geometry(self):
        p=self.protocol()
        self.assertTrue(hasattr(p, "remap_demo_xml"), "Saved-scene restoration needs portable asset mapping")
        with tempfile.TemporaryDirectory() as tmp:
            assets=Path(tmp)/"assets"; assets.mkdir(); (assets/"book.obj").write_text("mesh")
            robots=Path(tmp)/"robosuite"; robots.mkdir(); (robots/"arm.stl").write_text("mesh")
            xml='<mujoco><asset><mesh file="/old/chiliocosm/assets/book.obj"/><mesh file="/old/robosuite/arm.stl"/></asset><worldbody><body pos="1 2 3"/></worldbody></mujoco>'
            result=p.remap_demo_xml(xml,assets,robots)
            self.assertIn(str(assets/"book.obj"),result)
            self.assertIn(str(robots/"arm.stl"),result)
            self.assertIn('pos="1 2 3"',result)

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
        self.assertFalse(r["correct_then_destroyed"], "Ambiguity does not prove the correct goal was destroyed")

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

    def response_rows(self):
        p=self.protocol();rows=[]
        outcomes={0:{'front':'left','left':'left','right':'left','back':None},
                  1:{'front':'left','left':None,'right':None,'back':'back'}}
        for idx in (0,1):
            for goal in p.GOALS:
                terminal=outcomes[idx][goal]
                rows.append({'init_index':idx,'requested_goal':goal,'terminal_goal':terminal,
                             'terminal_category':'unfinished' if terminal is None else 'correct' if terminal==goal else 'wrong_goal',
                             'initial_state_sha256':f'state{idx}','scene_xml_sha256':'scene','seed':42+idx})
        return rows

    def test_instruction_bias_keeps_failed_rollouts_in_denominator(self):
        r=self.protocol().instruction_response_summary(self.response_rows(),'left')
        self.assertEqual(r['other_instructions']['total'],6)
        self.assertEqual(r['other_instructions']['trained_goal_terminal'],3)
        self.assertEqual(r['conditioned_on_trained_instruction_success']['total'],3)
        self.assertEqual(r['conditioned_on_trained_instruction_success']['trained_goal_terminal'],2)
        self.assertEqual(r['terminal_matrix']['back']['unfinished'],1)

    def test_instruction_comparison_rejects_missing_duplicate_or_mismatched_conditions(self):
        p=self.protocol();rows=self.response_rows()
        with self.assertRaises(ValueError):p.instruction_response_summary(rows[:-1],'left')
        with self.assertRaises(ValueError):p.instruction_response_summary(rows+[rows[0]],'left')
        rows[0]['initial_state_sha256']='different'
        with self.assertRaises(ValueError):p.instruction_response_summary(rows,'left')

    def test_single_task_batch_is_supported(self):
        ids=self.protocol().balanced_indices({'left':[4,5,6]},6,19,1)
        self.assertEqual(len(ids),6)
        self.assertTrue(set(ids)<={4,5,6})


if __name__ == "__main__":
    unittest.main()
