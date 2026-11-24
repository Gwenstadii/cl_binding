"""Physical-pair scoring and adversarial archive-integrity checks (no CUDA)."""

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/results/2026-11-22/k25_order_restore"
SPEC = importlib.util.spec_from_file_location("order_statistics", ROOT / "scripts/k25/summarize_order_restore.py")
STATS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATS)


class PhysicalPairs(unittest.TestCase):
    def test_leaving_wrong_target_does_not_count_as_recovery(self):
        before = dict(init=1, first=["front"], terminal={"right": False}, boundary_signature="same")
        after = copy.deepcopy(before)
        after["first"] = []
        result = STATS.matched([before], [after], "right", "front")
        self.assertEqual(result["new_goal_to_intended"], [])
        self.assertEqual(result["new_correct"], [])

    def test_same_physical_state_cannot_be_counted_twice(self):
        row = dict(init=1, first=["right"], terminal={"right": True}, boundary_signature="same")
        with self.assertRaises(AssertionError):
            STATS.matched([row, row], [row, row], "right", "front")


class ArchiveIntegrity(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="k25_stats_", dir=scratch)
        self.base = Path(self.directory.name).resolve()
        # Verify the managed cleanup target before registering recursive cleanup.
        assert self.base.is_relative_to(scratch.resolve())
        self.addCleanup(self.directory.cleanup)
        for name in ("protocol.json", *[p.name for p in BASE.glob("dev_lr_front_130_*.jsonl")],
                     *[p.name for p in BASE.glob("dev_lr_front_130_*_COMPLETE.json")]):
            shutil.copyfile(BASE / name, self.base / name)
        self.protocol = STATS.read(self.base / "protocol.json")

    def collect(self):
        return STATS.collect(self.base, "dev", "lr_front_130", self.protocol)

    def replace_with_valid_hash(self, rows):
        path = self.base / "dev_lr_front_130_0.jsonl"
        path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
        marker_path = self.base / "dev_lr_front_130_0_COMPLETE.json"
        marker = STATS.read(marker_path)
        marker["rows_sha256"] = STATS.digest(path)
        marker_path.write_text(json.dumps(marker), encoding="utf-8")

    def original_rows(self):
        return [json.loads(s) for s in (self.base / "dev_lr_front_130_0.jsonl").read_text().splitlines()]

    def test_complete_development_grid_passes(self):
        self.assertEqual(len(self.collect()[0]), 64)

    def test_duplicate_trial_is_rejected_even_if_hash_matches(self):
        rows = self.original_rows()
        self.replace_with_valid_hash(rows+[rows[0]])
        with self.assertRaises(AssertionError):
            self.collect()

    def test_missing_trial_is_rejected_even_if_hash_matches(self):
        self.replace_with_valid_hash(self.original_rows()[1:])
        with self.assertRaises(AssertionError):
            self.collect()

    def test_wrong_correction_sign_is_rejected(self):
        rows = self.original_rows()
        rows[2]["decisions"][0]["coefficient"] *= -1
        self.replace_with_valid_hash(rows)
        with self.assertRaises(AssertionError):
            self.collect()


if __name__ == "__main__":
    unittest.main()
