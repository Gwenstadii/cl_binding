from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/k25"))
import mu_swap_protocol as p
from mu_swap_math import compose


class CommonResponseControl(unittest.TestCase):
    def test_background_change_is_shared_and_does_not_change_contrast(self):
        for beta in (1.,2.959):
            late = [compose(q,2.,9.,.4,"late",beta,s) for q,s in ((3.,5.294),(1.,-5.294))]
            parent = [compose(q,2.,9.,.4,"parent",beta,s) for q,s in ((3.,5.294),(1.,-5.294))]
            self.assertAlmostEqual(parent[0]-late[0],7.)
            self.assertAlmostEqual(parent[1]-late[1],7.)
            self.assertAlmostEqual(parent[0]-parent[1],late[0]-late[1])

    def test_k14_gain_acts_on_actual_late_residual(self):
        self.assertAlmostEqual(compose(4.,2.,10.,0.,"parent",2.959,0.),15.918)
        self.assertEqual(compose(4.,2.,10.,0.,"late",1.,0.),4.)
        self.assertEqual(compose(4.,2.,10.,0.,"parent",1.,0.),12.)

    def test_reverse_only_changes_v_addition(self):
        positive = compose(4.,2.,10.,.5,"parent",1.,5.294)
        reverse = compose(4.,2.,10.,.5,"parent",1.,-5.294)
        self.assertAlmostEqual((positive+reverse)/2,12.)
        self.assertAlmostEqual(positive-reverse,5.294)

    def test_complete_grid_and_reference_replays_are_not_new_samples(self):
        self.assertEqual(len(p.expected_keys()),600)
        self.assertEqual(sum(p.reference_job(j) for j in p.jobs())*40,200)
        union = set()
        for lane in range(40):
            init,goal = p.unit(lane)
            part = p.expected_keys([init],goal)
            self.assertFalse(union.intersection(part))
            union.update(part)
        self.assertEqual(union,p.expected_keys())
        self.assertEqual(len(p.IDS),20)


if __name__ == "__main__":
    unittest.main()
