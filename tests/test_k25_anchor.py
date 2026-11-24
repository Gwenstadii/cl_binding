"""Guard against confounding donor competence with the restricted V repair."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/k25"))
import anchor_protocol as p


class AnchorAudit(unittest.TestCase):
    def test_complete_anchor_gain_window_polarity_factorial(self):
        got = {(j["anchor"],j["gain"],j["window"],j["arm"]) for j in p.jobs(True) if j["family"] == "delta"}
        expected = {(a,g,w,s) for a in (0,30) for g in p.GAINS for w in p.WINDOWS for s in p.ARMS}
        self.assertEqual(got,expected)

    def test_imported_episodes_are_only_the_previously_measured_controls(self):
        actual = p.expected_keys(p.IDS)
        all_conditions = p.expected_keys(p.IDS,True)
        reused = all_conditions-actual
        self.assertEqual(len(actual)*2,1360)
        self.assertEqual(len(all_conditions)*2,1680)
        self.assertEqual(len(reused)*2,320)
        self.assertTrue(all(k[:3] == ("delta",30,p.temporal.GAIN) for k in reused))

    def test_policy_and_v_weights_qualifications_are_separate(self):
        native = [j for j in p.jobs() if j["family"] == "policy"]
        rollback = [j for j in p.jobs() if j["family"] == "v_weights"]
        self.assertEqual({j["anchor"] for j in native},{0,30})
        self.assertTrue(all(j["goals"] == "ABN" and j["gain"] == 0 and j["arm"] == "native" for j in native))
        self.assertTrue(all(j["goals"] == "AB" and j["window"] == "full_400" and p.offsets(j) == (0,400) for j in rollback))


if __name__ == "__main__":
    unittest.main()
