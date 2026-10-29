"""CPU-only evidence checks without model dependencies."""

import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_stage2m_cross_attention import active_vector, audit_prefix_inputs, indexed, transfer_metrics


def velocity(dimensions=32):
    return [[0.0] * dimensions for _ in range(50)]


class CrossAttentionAuditTests(unittest.TestCase):
    def test_ignore_padding_and_unexecuted_future_for_primary(self):
        before, target, after = velocity(), velocity(), velocity()
        for index in range(50):
            target[index][0] = 1.0
            after[index][0] = 0.5
        original = transfer_metrics(before, target, after)
        changed = copy.deepcopy(after)
        for index, row in enumerate(changed):
            row[7:] = [1e12] * 25
            if index >= 10:
                row[0] = 20.0
        result = transfer_metrics(before, target, changed)
        self.assertEqual(result["primary"], original["primary"])
        self.assertEqual(result["motion"], original["motion"])
        self.assertNotEqual(result["full50"], original["full50"])
        self.assertEqual(len(active_vector(changed)), 70)

    def test_direction_sign_and_orthogonal_change(self):
        before, target, after = velocity(7), velocity(7), velocity(7)
        for index in range(50):
            target[index][0] = 1.0
            after[index][0] = -0.5
            after[index][1] = 0.25
        result = transfer_metrics(before, target, after)
        self.assertAlmostEqual(result["primary"]["T"], -0.5)
        self.assertAlmostEqual(result["primary"]["aligned_rms"], 0.5 / math.sqrt(7))
        self.assertAlmostEqual(result["primary"]["perp_rms"], 0.25 / math.sqrt(7))
        self.assertFalse(result["gripper"]["valid_T"])
        self.assertIsNone(result["gripper"]["T"])

    def test_near_zero_gate_uses_rms_not_squared_norm(self):
        before, target, after = velocity(7), velocity(7), velocity(7)
        for index in range(50):
            target[index] = [5e-6] * 7
            after[index] = [1e-3] * 7
        result = transfer_metrics(before, target, after, repeat_rms=1e-6)
        self.assertFalse(result["primary"]["valid_T"])
        self.assertIsNone(result["primary"]["T"])
        self.assertAlmostEqual(result["primary"]["epsilon"], 1e-5)
        self.assertGreater(result["primary"]["rms_delta"], 0)
        result = transfer_metrics(before, target, after, repeat_rms=0)
        self.assertTrue(result["primary"]["valid_T"])
        self.assertAlmostEqual(result["primary"]["T"], 200)

    def test_duplicate_cannot_mask_missing_matrix_cell(self):
        expected = {(0, "left"), (0, "front")}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            indexed([{"i": 0, "goal": "left"}, {"i": 0, "goal": "left"}],
                    ("i", "goal"), expected)
        with self.assertRaisesRegex(ValueError, "Matrix mismatch"):
            indexed([{"i": 0, "goal": "left"}], ("i", "goal"), expected)

    def test_invalid_effective_coordinates_and_shape_rejected(self):
        values = velocity(7)
        values[0][0] = float("nan")
        with self.assertRaises(ValueError):
            active_vector(values)
        with self.assertRaises(ValueError):
            active_vector([[0.0] * 7] * 49)
        with self.assertRaises(ValueError):
            transfer_metrics(velocity(), velocity(), velocity(), repeat_rms=-1)

    def test_prefix_gate_excludes_masked_queries_but_requires_matching_inputs(self):
        tokens = {goal: digit * 64 for goal, digit in
                  (("left", "1"), ("front", "2"), ("right", "3"))}
        inputs = {}
        for role in ("R0", "FT4"):
            inputs[role] = {}
            for goal in tokens:
                inputs[role][goal] = {"images": ["a"*64], "image_masks": ["b"*64],
                    "language_tokens": tokens[goal], "language_mask": "c"*64,
                    "state_before_projection": "d"*64, "image_shapes": [[1, 3, 128, 128]],
                    "language_shape": [1, 2], "state_shape": [1, 7]}
        rows = []
        for goal in tokens:
            for tensor in ("keys", "values"):
                for group in ("image", "text", "state"):
                    rows.append({"goal": goal, "layer": 0, "tensor": tensor, "group": group,
                        "max_abs": 2.0, "rms": math.sqrt(2),
                        "active": {"count": 1, "element_count": 1, "max_abs": 0.0, "rms": 0.0},
                        "masked": {"count": 1, "element_count": 1, "max_abs": 2.0, "rms": 2.0}})
        fixture = {"indices": [0], "prefix_input_hashes": [inputs], "prefix_audit": [rows],
                   "layer_map": {"R0": [{"vlm_layer": 0, "expert_layer": 0, "kind": "SA"}]}}
        result = audit_prefix_inputs(fixture)
        self.assertEqual(result["max_abs_by_position_subset"]["all"]["text"], 2)
        self.assertEqual(result["max_abs_by_position_subset"]["active"]["text"], 0)
        changed = copy.deepcopy(fixture)
        changed["prefix_audit"][0][0]["active"].update(max_abs=1e-4, rms=1e-4)
        with self.assertRaisesRegex(ValueError, "Frozen valid"):
            audit_prefix_inputs(changed)
        changed = copy.deepcopy(fixture)
        changed["prefix_input_hashes"][0]["FT4"]["left"]["language_tokens"] = "e"*64
        with self.assertRaisesRegex(ValueError, "Preprocessed input mismatch"):
            audit_prefix_inputs(changed)


if __name__ == "__main__":
    unittest.main()
