from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stage2l_entry_confirm import jobs_for, summarize_rows


class EntryConfirmationProtocolTests(unittest.TestCase):
    def test_full_matrix_shares_unoffset_seeds(self):
        ids = [f"state_{i}" for i in range(12)]
        jobs = jobs_for(ids, [2026102801, 2026102802, 2026102803])
        self.assertEqual(len(jobs), 72)
        for ident in ids:
            self.assertEqual({seed for snapshot, _, seed in jobs if snapshot == ident},
                             {2026102801, 2026102802, 2026102803})

    def test_duplicate_or_missing_protocol_values_are_rejected(self):
        for ids, seeds in [([], [1]), (["a", "a"], [1]), (["a"], []), (["a"], [1, 1]), (["a"], [-1])]:
            with self.assertRaises(ValueError):
                jobs_for(ids, seeds)

    def test_noise_repeats_do_not_become_distinct_states(self):
        rows = []
        for seed in [11, 12, 13]:
            for policy in ("teacher", "student"):
                success = policy == "teacher" or seed == 12
                rows.append({"snapshot_id": "teacher_front_00", "source": "teacher", "goal": "front",
                    "continuation_policy": policy, "noise_seed": seed, "correct_ever": success,
                    "terminal_category": "correct" if success else "unfinished", "first_goal": "front" if success else None})
        summary = summarize_rows(rows, ["teacher_front_00"], [11, 12, 13])
        self.assertEqual(summary["distinct_snapshots"], 1)
        self.assertEqual(summary["continuations"], 6)
        self.assertEqual(summary["cells"]["teacher:front"]["teacher"]["distinct_snapshots"], 1)
        self.assertEqual(summary["per_snapshot"]["teacher_front_00"]["paired_outcomes"],
                         {"teacher_only": 2, "both": 1})
        with self.assertRaises(ValueError):
            summarize_rows(rows[:-1], ["teacher_front_00"], [11, 12, 13])
        with self.assertRaises(ValueError):
            summarize_rows(rows + [rows[0]], ["teacher_front_00"], [11, 12, 13])


if __name__ == "__main__":
    unittest.main()
