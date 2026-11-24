"""Protocol and evidence checks without torch, GPU, or simulator imports."""

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_stage2o_pc_heads import expected_cells, relative_reduction, validate_head_norm_support
from audit_stage2n_cross_checkpoint import indexed
from stage2o_pc_heads import head_span


class PCHeadProtocolTests(unittest.TestCase):
    def test_matrix_is_fixed_and_head_screen_never_includes_extra_noise(self):
        cells = expected_cells()
        self.assertEqual(len(cells), 5400)
        self.assertEqual(sum(k[-1] is None for k in cells), 2160)
        self.assertEqual(sum(k[-1] is not None for k in cells), 3240)
        self.assertTrue(all(k[1] == 2026102801 for k in cells if k[-1] is not None))

    def test_actual_head_width_is_not_post_projection_hidden_width(self):
        # Actual readout width supplied at runtime, no inference from hidden=720.
        self.assertEqual(head_span(3, 15, 960), slice(192, 256))
        with self.assertRaises(ValueError):
            head_span(15, 15, 960)
        with self.assertRaises(ValueError):
            head_span(0, 15, 961)

    def test_head_norm_evidence_rejects_cross_head_leak(self):
        rows = [[0.0]*15 for _ in range(50)]
        for row in rows: row[2] = 1
        record = {"head":2,"mode":"head_random", "effective_block_l2":rows,
                  "desired_block_l2":rows,"target_block_l2":rows}
        validate_head_norm_support(record)
        changed = copy.deepcopy(record)
        changed["effective_block_l2"][0][4] = 0.001
        with self.assertRaisesRegex(ValueError,"untouched"):
            validate_head_norm_support(changed)

    def test_duplicate_head_cannot_replace_missing_head(self):
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            indexed([{"head":0},{"head":0}],("head",),{(0,),(1,)})

    def test_per_input_relative_gain_is_not_ratio_of_medians(self):
        self.assertAlmostEqual(relative_reduction(1,2),0.5)
        self.assertIsNone(relative_reduction(1,0))
        self.assertIsNone(relative_reduction(1,1e-7))
        self.assertLess(relative_reduction(3,2),0)

    def test_registered_config_agrees_with_independent_matrix(self):
        config=json.loads((Path(__file__).resolve().parents[1]/"configs/stage2o_pc_heads_20261029.json").read_text())
        self.assertEqual(config["expected_patch_count"],len(expected_cells()))
        self.assertEqual(config["expected_heads"],15)
        self.assertEqual(config["expected_norm_records"],2592)
        self.assertEqual(config["sites"],[1,3,9])


if __name__ == "__main__": unittest.main()
