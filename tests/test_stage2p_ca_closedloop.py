"""Behavioral edge cases and budget boundaries; no GPU/model/simulator imports."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_stage2p_ca_closedloop import (
    GOALS, expected_episode_keys, group_summary, index_unique, paired_retention, paired_condition_changes,
    summarize_predicates, validate_replans, validate_engineering,
)


def trace():
    return [{goal:False for goal in GOALS} for _ in range(401)]


def replans(condition="cross_same"):
    return [{"replan_index":i, "start_step":i*10,
             "patch_flow_count":10 if condition != "native" and i < 12 else 0,
             "donor_model":"R0" if condition != "native" and i < 12 else None,
             "donor_goal":"left" if condition != "native" and i < 12 else None}
            for i in range(40)]


class ClosedLoopBehaviorAuditTests(unittest.TestCase):
    def test_exact_matrix_is_68_episodes_and_48_intervened(self):
        matrix=expected_episode_keys()
        self.assertEqual(len(matrix),68)
        self.assertEqual(sum(key[1]!="native" for key in matrix),48)
        self.assertEqual(sum(key[2]=="right" for key in matrix),4)

    def test_requested_success_then_loss_is_not_terminal_or_target_switch(self):
        flags=trace();flags[100]["left"]=True
        result=summarize_predicates("left",flags)
        self.assertTrue(result["correct_ever"])
        self.assertTrue(result["correct_then_destroyed"])
        self.assertEqual(result["terminal_category"],"unfinished")
        self.assertFalse(result["new_right_ever"])

    def test_target_substitution_is_explicit_even_if_terminal_unfinished(self):
        flags=trace();flags[150]["right"]=True
        result=summarize_predicates("left",flags)
        self.assertEqual(result["first_goal"],"right")
        self.assertTrue(result["new_right_ever"])
        self.assertFalse(result["new_right_terminal"])
        self.assertEqual(result["terminal_category"],"unfinished")

    def test_window_step_120_and_121_have_different_claims(self):
        before=trace();before[120]["left"]=True
        after=trace();after[121]["left"]=True
        a,b=[summarize_predicates("left",x) for x in (before,after)]
        self.assertTrue(a["correct_first_within_window"])
        self.assertFalse(a["correct_first_after_window"])
        self.assertTrue(b["correct_first_after_window"])
        self.assertFalse(b["correct_at_window_end"])

    def test_ambiguous_goal_events_are_not_silently_called_correct(self):
        flags=trace();flags[-1]["left"]=flags[-1]["right"]=True
        result=summarize_predicates("left",flags)
        self.assertEqual(result["terminal_category"],"ambiguous")
        self.assertFalse(result["terminal_correct"])
        self.assertTrue(result["first_goal_ambiguous"])
        self.assertIsNone(result["first_goal"])

    def test_intervention_cannot_leak_past_120_steps(self):
        rows=replans()
        self.assertEqual(validate_replans("FT4","cross_same","left",rows),120)
        changed=copy.deepcopy(rows);changed[12].update(patch_flow_count=10,donor_model="R0",donor_goal="left")
        with self.assertRaisesRegex(ValueError,"window"):
            validate_replans("FT4","cross_same","left",changed)
        changed=copy.deepcopy(rows);changed[11]["patch_flow_count"]=0
        with self.assertRaisesRegex(ValueError,"window"):
            validate_replans("FT4","cross_same","left",changed)

    def test_duplicate_cannot_replace_an_unobserved_episode(self):
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            index_unique([{"episode_id":"one"},{"episode_id":"one"}],("episode_id",))

    def test_recovery_denominator_uses_original_correct_states_only(self):
        failed=summarize_predicates("left",trace())
        good_trace=trace();good_trace[-1]["left"]=True
        good=summarize_predicates("left",good_trace)
        lookup={}
        for index in (0,1):
            for seed in (2026102801,2026102802):
                lookup["R0","native","left",index,seed]=good if index==0 and seed==2026102801 else failed
                lookup["FT4","cross_same","left",index,seed]=good
        result=paired_retention(lookup,"FT4","cross_same","left")
        self.assertEqual(result["original_terminal_correct_denominator"],1)
        self.assertEqual(result["terminal_correct"],1)
        self.assertEqual(result["terminal_correct_rate"],1)
        result=group_summary([good,failed])
        self.assertEqual(result["terminal_correct_rate"],0.5)

    def test_equal_success_totals_can_hide_different_paired_outcomes(self):
        good_trace=trace();good_trace[-1]["left"]=True
        bad_trace=trace();bad_trace[-1]["right"]=True
        good=summarize_predicates("left",good_trace)
        bad=summarize_predicates("left",bad_trace)
        lookup={}
        for index in (0,1):
            for seed in (2026102801,2026102802):
                lookup["FT4","native","left",index,seed]=good if index==0 else bad
                lookup["FT4","cross_same","left",index,seed]=bad if index==0 else good
        result=paired_condition_changes(lookup,"FT4","cross_same","left")
        self.assertEqual(result["native_incorrect_to_correct"],2)
        self.assertEqual(result["native_correct_lost"],2)
        self.assertEqual(result["native_terminal_right_to_correct"],2)

    def test_engineering_passed_boolean_cannot_hide_missing_or_bad_checks(self):
        checks=[]
        for model in ("R0","FT4"):
            for goal in ("left","front"):
                checks.append({"model":model,"goal":goal,"max_abs":0,
                               **{name:0 for name in ("native_chunk_max_abs","self_chunk_max_abs",
                                  "post_native_max_abs","post_self_max_abs","clip_max_abs","post_per_action_max_abs")},
                               "flows":[{"flow_index":i,"velocity_max_abs":0,"witness_max_abs":0} for i in range(10)]})
        fixture={"passed":True,"tolerance":1e-5,"site":9,"checks":checks,
                 "cache_isolation":{"deep_copy_per_query":True,"key_value_storage_unique":True},
                 "no_physical_actions_beyond_reset_settle":True}
        validate_engineering(fixture)
        changed=copy.deepcopy(fixture);changed["checks"][0]["flows"].pop()
        with self.assertRaisesRegex(ValueError,"matrix"):
            validate_engineering(changed)
        changed=copy.deepcopy(fixture);changed["checks"][0]["post_native_max_abs"]=1e-3
        with self.assertRaisesRegex(ValueError,"postprocess"):
            validate_engineering(changed)


if __name__ == "__main__": unittest.main()
