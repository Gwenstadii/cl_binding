import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage1_dual_then_right_run import compare_snapshots, summarize_requests, validate_config

def row(index,goal,target):
    correct=target==goal
    return {'init_index':index,'requested_goal':goal,'seed':1234+index,'steps':400,
        'initial_state_sha256':f'state{index}','scene_xml_sha256':'common',
        'first_goal':target,'first_goal_times':{g:100 if g==target else None for g in ('front','left','right','back')},
        'correct_ever':correct,'terminal_goal':target,'terminal_category':'correct' if correct else 'wrong_goal' if target else 'unfinished',
        'correct_then_destroyed':False}

def before_rows():return [row(i,g,g if g!='right' else None) for i in (20,21) for g in ('left','front','right')]
def after_rows():return [row(i,g,'right') for i in (20,21) for g in ('left','front','right')]

class DualThenRightTests(unittest.TestCase):
    def test_both_previously_acquired_instructions_can_be_replaced_by_new_target(self):
        result=compare_snapshots(before_rows(),after_rows(),[20,21],1234)
        self.assertEqual(result['matched_pairs'],6)
        self.assertEqual(result['previously_joint_correct_states']['both_old_instructions_terminal_new_right'],2)
        for goal in ('left','front'):
            old=result['old_goal_transitions'][goal]['previously_terminal_correct']
            self.assertEqual((old['total'],old['lost_and_terminal_new_right']),(2,2))

    def test_preexisting_failure_is_excluded_from_old_skill_loss_denominator(self):
        before=before_rows();before[1]=row(20,'front',None)
        result=compare_snapshots(before,after_rows(),[20,21],1234)
        self.assertEqual(result['old_goal_transitions']['front']['previously_successful_ever']['total'],1)
        self.assertEqual(result['previously_joint_correct_states']['total'],1)

    def test_old_success_then_new_target_is_not_counted_as_ever_skill_lost(self):
        after=after_rows();after[0].update(first_goal='left',correct_ever=True,correct_then_destroyed=True)
        after[0]['first_goal_times']['left']=80
        result=compare_snapshots(before_rows(),after,[20,21],1234)
        old=result['old_goal_transitions']['left']
        self.assertEqual(old['previously_successful_ever']['lost'],1)
        self.assertEqual(old['previously_terminal_correct']['lost_and_terminal_new_right'],2)

    def test_missing_duplicate_reset_and_seed_errors_are_rejected(self):
        good=before_rows();cases=[good[:-1],good+[good[0]]]
        for field,value in [('initial_state_sha256','different'),('scene_xml_sha256','other'),('seed',999),('steps',399)]:
            changed=copy.deepcopy(good);changed[0][field]=value;cases.append(changed)
        for items in cases:
            with self.subTest(items=len(items)),self.assertRaises(ValueError):summarize_requests(items,[20,21],1234)

    def test_before_after_must_match_even_when_each_snapshot_is_internally_paired(self):
        changed=after_rows()
        for r in changed:
            if r['init_index']==20:r['initial_state_sha256']='different-but-shared'
        with self.assertRaises(ValueError):compare_snapshots(before_rows(),changed,[20,21],1234)

    def test_plan_rejects_replay_changed_budget_and_wrong_initial_checkpoint(self):
        path=Path(__file__).resolve().parents[1]/'configs/stage1_dual12000_then_right_v1.json'
        config=json.loads(path.read_text());validate_config(config)
        for section,key,value in [('sequential','replay',True),('training','max_steps',18000),
                                  ('sequential','initial_checkpoint','outputs/r0/dual_left_front_20261023_01/checkpoints/018000')]:
            changed=copy.deepcopy(config);changed[section][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_config(changed)

if __name__=='__main__':unittest.main()
