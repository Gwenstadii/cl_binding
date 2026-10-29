import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage2q_all_ca_overnight import jobs,matched_seed
from audit_stage2q_all_ca import expected_keys,FIELDS

class OvernightProtocol(unittest.TestCase):
    def test_all_sites_windows_and_controls_are_in_the_independent_matrix(self):
        cases=jobs();keys=[tuple(r[k] for k in FIELDS) for r in cases]
        self.assertEqual(len(keys),420);self.assertEqual(len(set(keys)),420)
        self.assertEqual(set(keys),expected_keys())
    def test_positive_controls_include_both_windows_and_do_not_use_right_teacher(self):
        cases=[r for r in jobs() if r['condition']=='full_old_response']
        self.assertEqual(len(cases),16)
        self.assertEqual({r['window'] for r in cases},{120,400})
        self.assertTrue(all(r['requested_goal'] in ('left','front') for r in cases))
    def test_registered_compute_budget_includes_random_calibration(self):
        cases=jobs();patched=sum(r['window'] for r in cases if r['site'] is not None)
        random=sum(r['window'] for r in cases if r['condition']=='random_delta')
        self.assertEqual(patched,99840);self.assertEqual(random,33280)
        self.assertEqual(420*400+patched+random,301120)
    def test_no_multi_layer_or_adaptive_combination_is_scheduled(self):
        cases=[r for r in jobs() if r['site'] is not None]
        self.assertEqual({r['site'] for r in cases},{1,3,5,7,9,11,13,15})
        self.assertTrue(all(type(r['site']) is int for r in cases))
    def test_random_controls_share_directions_across_the_two_windows(self):
        a=matched_seed('j002|FT4|random_delta|site3|w120|left|i0|s1|r0|f0','FT4','left',3,5)
        b=matched_seed('j122|FT4|random_delta|site3|w400|left|i0|s1|r0|f0','FT4','left',3,5)
        c=matched_seed('j122|FT4|random_delta|site3|w400|left|i0|s1|r0|f1','FT4','left',3,5)
        self.assertEqual(a,b);self.assertNotEqual(a,c)

if __name__=='__main__':unittest.main()
