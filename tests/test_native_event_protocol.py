import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/k26'))
from native_event_protocol import NativeEventEpisode, execute_action_chunk, rescore_native_flags


class NativeEventProtocolTests(unittest.TestCase):
    def test_first_success_stays_success_when_later_state_was_destroyed(self):
        flags=[{'back':False}]*80+[{'back':True}]+[{'back':False}]*319
        result=rescore_native_flags('back',flags)
        self.assertTrue(result['success'])
        self.assertEqual(result['stop_step'],81)

    def test_another_goal_does_not_end_requested_back_task(self):
        result=rescore_native_flags('back',[{'front':True,'back':False},{'front':False,'back':True}])
        self.assertEqual(result['first_success_step'],2)

    def test_mid_chunk_success_neither_executes_nor_requests_an_extra_action(self):
        episode=NativeEventEpisode('back');episode.steps=80
        requested=[];executed=[]
        def actions():
            for n in range(10):requested.append(n);yield n
        def step(action):executed.append(action);return {'back':True}
        _,count=execute_action_chunk(actions(),episode,step,lambda outcome:outcome)
        self.assertEqual((requested,executed,count),([0],[0],1))
        self.assertEqual(episode.first_success_step,81)

    def test_timeout_and_incomplete_records_are_distinct(self):
        result=rescore_native_flags('back',[{'back':False}]*400)
        self.assertFalse(result['success']);self.assertEqual(result['stop_step'],400)
        with self.assertRaises(ValueError):rescore_native_flags('back',[{'back':False}]*8)

    def test_done_episode_cannot_receive_more_physical_steps(self):
        episode=NativeEventEpisode('back');episode.observe({'back':True})
        with self.assertRaises(RuntimeError):episode.observe({'back':False})
        calls=[]
        self.assertEqual(execute_action_chunk([1,2],episode,lambda a:calls.append(a),lambda _:{}),(None,0))
        self.assertEqual(calls,[])

    def test_missing_requested_native_predicate_rejected(self):
        with self.assertRaises(KeyError):NativeEventEpisode('back').observe({'front':True})


if __name__=='__main__':unittest.main()
