import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage2s_same_instruction80 import jobs
from audit_stage2s_trajectories import direction_metrics


class SameInstructionProtocol(unittest.TestCase):
    def test_no_switch_and_four_paired_runs_per_goal(self):
        cases=jobs();self.assertEqual(len(cases),24)
        for goal in ('left','front','right'):
            for mode in ('native','handoff80'):
                self.assertEqual({(j['init_index'],j['noise_seed']) for j in cases if j['requested_goal']==goal and j['mode']==mode},
                    {(i,s) for i in (0,1) for s in (2026102801,2026102802)})
        self.assertTrue(all('prefix_goal' not in j for j in cases))

    def test_command_and_object_motion_are_not_conflated(self):
        ev=[{'eef_position':[0,0,0],'object_positions':{'black_book_1_pos':[0,0,0]},
             'controller':{'world_delta':[-.01,0,0]},'native_grasp':False,'relative_book_eef':[0,0,0]} for _ in range(401)]
        ev[90]={**ev[90],'object_positions':{'black_book_1_pos':[.1,0,0]}}
        q=direction_metrics(ev,'left',{'left':{'centre':[1,0,0]},'right':{'centre':[-1,0,0]}},10)
        self.assertLess(q['command_projection_requested_cm'],0)
        self.assertGreater(q['book_distance_reduction_requested_cm'],0)
        self.assertEqual(q['native_grasp_steps'],0)

    def test_zero_command_has_no_invented_cosine(self):
        ev=[{'eef_position':[0,0,0],'object_positions':{'black_book_1_pos':[0,0,0]},
             'controller':{'world_delta':[0,0,0]},'native_grasp':None,'relative_book_eef':[0,0,0]} for _ in range(401)]
        q=direction_metrics(ev,'left',{'left':{'centre':[1,0,0]},'right':{'centre':[-1,0,0]}},10)
        self.assertIsNone(q['command_cosine_requested'])
        self.assertEqual(q['native_grasp_available_steps'],0)


if __name__=='__main__':unittest.main()
