"""Meaningful CPU checks for step-3 evidence, with no model dependencies."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_stage2n_cross_checkpoint import (
    audit_random_norms, engineering_gate, indexed, paired_response_metrics, prefix_gate,
    reference_metrics, transfer_metrics,
)


def velocity(value=0.0, dims=7):
    return [[value]*dims for _ in range(50)]


def blocks(value):
    return [[value]*15 for _ in range(50)]


class CrossCheckpointAuditTests(unittest.TestCase):
    def test_common_shift_is_not_conditional_restoration(self):
        old_left, old_front = velocity(1), velocity(-1)
        # Both outputs restored to the same healthy left response.
        result = paired_response_metrics(velocity(1), velocity(1), old_left, old_front)
        self.assertEqual(result["primary"]["contrast_rms"], 0)
        self.assertEqual(result["primary"]["contrast_projection"], 0)
        self.assertEqual(result["primary"]["contrast_error_rms"], 2)
        self.assertEqual(result["primary"]["response_error_rms"], 2**0.5)
        self.assertEqual(result["primary"]["common_error_rms"], 1)

    def test_same_and_wrong_healthy_reference_have_opposite_margins(self):
        base, same, other = velocity(2), velocity(1), velocity(-1)
        good = reference_metrics(base, same, same, other)["primary"]
        wrong = reference_metrics(base, other, same, other)["primary"]
        self.assertEqual(good["healthy_preference_margin"], 1)
        self.assertEqual(wrong["healthy_preference_margin"], -1)
        self.assertEqual(good["healthy_distance_gain"], 1)
        self.assertLess(wrong["healthy_distance_gain"], 0)
        identity = reference_metrics(same, same, same, other)["primary"]
        self.assertIsNone(identity["healthy_distance_gain"])

    def test_padding_and_unexecuted_positions_do_not_change_primary(self):
        base, reference, patched = velocity(0, 32), velocity(1, 32), velocity(0.5, 32)
        original = transfer_metrics(base, reference, patched)
        for i, row in enumerate(patched):
            row[7:] = [1e20]*25
            if i >= 10:
                row[:7] = [-20]*7
        changed = transfer_metrics(base, reference, patched)
        self.assertEqual(original["primary"], changed["primary"])
        self.assertNotEqual(original["full50"], changed["full50"])

    def test_near_zero_reference_never_generates_huge_transfer_fraction(self):
        result = transfer_metrics(velocity(0), velocity(1e-7), velocity(1))
        self.assertIsNone(result["primary"]["T"])
        self.assertFalse(result["primary"]["valid_T"])
        self.assertEqual(result["primary"]["rms_delta"], 1)

    def test_random_norms_match_planned_difference_but_report_cast_effect(self):
        correct = {"effective_block_l2": blocks(1)}
        record = {"target_block_l2": blocks(1), "desired_block_l2": blocks(1),
                  "effective_block_l2": blocks(1.002), "cast_error_block_l2": blocks(0.002),
                  "cast_input_block_l2": blocks(1)}
        result = audit_random_norms(record, correct)
        self.assertAlmostEqual(result["max_relative_actual_norm_difference"], 0.002)
        bad = copy.deepcopy(record)
        bad["effective_block_l2"][0][0] = 1.1
        with self.assertRaisesRegex(ValueError, "triangle"):
            audit_random_norms(bad, correct)
        bad = copy.deepcopy(record)
        bad["cast_error_block_l2"][0][0] = 0.01
        with self.assertRaisesRegex(ValueError, "round-to-nearest"):
            audit_random_norms(bad, correct)

    def test_random_amplitude_cannot_match_the_whole_message_instead(self):
        correct = {"effective_block_l2": blocks(0.1)}
        record = {"target_block_l2": blocks(0.1), "desired_block_l2": blocks(2),
                  "effective_block_l2": blocks(2), "cast_error_block_l2": blocks(0),
                  "cast_input_block_l2": blocks(10)}
        with self.assertRaisesRegex(ValueError, "not matched"):
            audit_random_norms(record, correct)

    def test_duplicate_rows_cannot_cover_missing_conditions(self):
        rows = [{"goal": "left", "mode": "cross_same"}]*2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            indexed(rows, ("goal", "mode"), {("left", "cross_same"), ("front", "cross_same")})
        with self.assertRaisesRegex(ValueError, "Matrix"):
            indexed(rows[:1], ("goal", "mode"), {("left", "cross_same"), ("front", "cross_same")})

    def test_engineering_requires_all_sites_and_the_actual_readout_gate(self):
        checks = []
        for role in ("R0", "FT4"):
            for goal in ("left", "front"):
                checks.append({"model": role, "goal": goal, "record_v_max_abs": 0,
                               "prior_velocity_max_abs": 0, "self_max_abs": 0,
                               "readout_reconstruction_max_abs": 0, "repeat_v_rms": 0,
                               "site_self_max_abs": {"1": 0, "3": 0, "9": 0}})
        fixture = {"passed": True, "checks": checks,
                   "cache_isolation": {"object_unique": True, "tensor_storage_unique": True,
                                       "deep_copy_per_query": True},
                   "norm_engineering_checks": [
                       {"model": role, "recipient": goal, "layer": site,
                        "desired_norm_max_abs_error": 0, "triangle_excess_max": 0}
                       for role in ("R0", "FT4") for goal in ("left", "front")
                       for site in (1, 3, 9)]}
        self.assertEqual(engineering_gate(fixture, [1, 3, 9]), 0)
        changed = copy.deepcopy(fixture)
        del changed["checks"][0]["site_self_max_abs"]["1"]
        with self.assertRaisesRegex(ValueError, "missing"):
            engineering_gate(changed, [1, 3, 9])
        changed = copy.deepcopy(fixture)
        changed["checks"][0]["readout_reconstruction_max_abs"] = 1e-4
        with self.assertRaisesRegex(ValueError, "Engineering failed"):
            engineering_gate(changed, [1, 3, 9])

    def test_only_valid_prefix_positions_gate_frozen_conditioning(self):
        inputs = {}
        for role in ("R0", "FT4"):
            inputs[role] = {
                goal: {"images": ["a"*64], "image_masks": ["b"*64],
                       "state_before_projection": "c"*64, "image_shapes": [[1, 3, 128, 128]],
                       "state_shape": [1, 32], "language_tokens": token*64}
                for goal, token in (("left", "d"), ("front", "e"))}
        rows = []
        for goal in ("left", "front"):
            for tensor in ("keys", "values"):
                for group in ("image", "text", "state"):
                    rows.append({"goal": goal, "layer": 0, "tensor": tensor, "group": group,
                                 "max_abs": 2, "rms": 2**0.5,
                                 "active": {"count": 1, "element_count": 1, "rms": 0, "max_abs": 0},
                                 "masked": {"count": 1, "element_count": 1, "rms": 2, "max_abs": 2}})
        fixture = {"indices": [0], "prefix_input_hashes": [inputs], "prefix_audit": [rows],
                   "layer_map": {"R0": [{"vlm_layer": 0}]}}
        self.assertEqual(prefix_gate(fixture)["valid_image_text_max_abs"], 0)
        changed = copy.deepcopy(fixture)
        changed["prefix_audit"][0][0]["active"].update(max_abs=1e-4, rms=1e-4)
        with self.assertRaisesRegex(ValueError, "Frozen valid"):
            prefix_gate(changed)
        changed = copy.deepcopy(fixture)
        changed["prefix_input_hashes"][0]["FT4"]["left"]["images"] = ["f"*64]
        with self.assertRaisesRegex(ValueError, "Cross-model"):
            prefix_gate(changed)


if __name__ == "__main__":
    unittest.main()
