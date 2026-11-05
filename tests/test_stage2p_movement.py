import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_stage2p_movement import first_run,inside_xy

class MovementTests(unittest.TestCase):
    def test_disjoint_visits_cannot_fake_sustained_entry(self):
        self.assertIsNone(first_run([1,2,4,5,7],5))
        self.assertEqual(first_run([1,2,4,5,6,7,8],5),4)
    def test_front_rotation_swaps_local_y_z_extents(self):
        site={'centre':[0,0,0],'rotation':[[1,0,0],[0,0,1],[0,-1,0]],'half_size':[.02775,.03595,.06216]}
        self.assertTrue(inside_xy([0,.05,10],site))
        self.assertFalse(inside_xy([.03,0,10],site))
        self.assertFalse(inside_xy([0,.07,10],site))
    def test_tilted_geometry_is_not_silently_treated_as_axis_aligned(self):
        with self.assertRaises(ValueError):inside_xy([0,0,0],{'centre':[0,0,0],'rotation':[[.7,.7,0],[.7,-.7,0],[0,0,1]],'half_size':[1,1,1]})

if __name__=='__main__':unittest.main()
