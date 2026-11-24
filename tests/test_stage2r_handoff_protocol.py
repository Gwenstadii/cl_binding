import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from stage2r_handoff_protocol import jobs, FIELDS, summarize_episode
from audit_stage2r_handoff import summarize


class HandoffProtocol(unittest.TestCase):
    def test_following_prefix_does_not_pass_two_instruction_selection(self):
        rows=[]
        for i in (0,1):
            for seed in (2026102801,2026102802):
                for goal in ('left','front'):
                    rows.append({'source_model':'R0','prefix_goal':'left','boundary':40,
                        'continuation_model':'FT4','requested_goal':goal,'init_index':i,'noise_seed':seed,
                        'boundary_signature':{'same_state':str((i,seed))},'pretarget_eligible':True,
                        'suffix_first_correct':goal=='left','suffix_first_right':False,
                        'suffix_correct_ever':goal=='left','terminal_correct':goal=='left',
                        'suffix_movement':{'first_targets':['left']}})
        _,pairs,_=summarize(rows)
        self.assertEqual(pairs[0]['eligible_both_instructed'],0)
        self.assertEqual(pairs[0]['eligible_both_follow_prefix'],4)

    def test_fixed_matrix_has_four_branches_per_identical_boundary(self):
        js = jobs()
        self.assertEqual(len(js), 192)
        self.assertEqual(len({tuple(j[k] for k in FIELDS) for j in js}), 192)
        for source in ('R0', 'FT4'):
            selected = [j for j in js if j['source_model'] == source and j['prefix_goal'] == 'left'
                        and j['boundary'] == 40 and j['init_index'] == 0 and j['noise_seed'] == 2026102801]
            self.assertEqual({(j['continuation_model'], j['requested_goal']) for j in selected},
                             {('R0', 'left'), ('R0', 'front'), ('FT4', 'left'), ('FT4', 'front')})

    def test_original_global_horizon_and_noise_boundaries(self):
        js = jobs()
        self.assertEqual(sum(400-j['boundary'] for j in js), 61440)
        self.assertEqual(sum(j['boundary'] for j in js), 15360)
        self.assertTrue(all(j['boundary'] % 10 == 0 for j in js))

    def test_already_reached_prefix_does_not_count_as_fresh_selection(self):
        geometry = {g: {'centre': [i, 0, 0], 'half_size': [.1, .1, .1],
                        'rotation': [[1,0,0],[0,1,0],[0,0,1]]}
                    for i, g in enumerate(('left','front','right','back'))}
        events = [{'step': step, 'object_positions': {'black_book_1_pos': [0, 0, 1 if 20 <= step <= 50 else 0]},
                   'predicates': {g: False for g in geometry}} for step in range(1, 401)]
        a = summarize_episode({'boundary': 40, 'requested_goal': 'left'}, events, geometry)
        self.assertFalse(a['pretarget_eligible'])
        self.assertTrue(a['suffix_first_correct'])
        # Prefix/suffix may both say Left, but this is explicitly ineligible.
        b = summarize_episode({'boundary': 80, 'requested_goal': 'left'}, events, geometry)
        self.assertTrue(b['full_correct_ever'])
        self.assertFalse(b['suffix_correct_ever'])
        self.assertFalse(b['pretarget_eligible'])

    def test_changed_instruction_is_scored_against_new_request(self):
        geometry = {g: {'centre': [i,0,0], 'half_size': [.1,.1,.1],
                        'rotation': [[1,0,0],[0,1,0],[0,0,1]]}
                    for i, g in enumerate(('left','front','right','back'))}
        events = [{'step': t, 'object_positions': {'black_book_1_pos': [1 if t > 80 else 5, 0, 1 if t > 80 else 0]},
                   'predicates': {g: False for g in geometry}} for t in range(1, 401)]
        a = summarize_episode({'boundary': 80, 'requested_goal': 'front'}, events, geometry)
        self.assertTrue(a['pretarget_eligible'])
        self.assertTrue(a['suffix_first_correct'])
        b = summarize_episode({'boundary': 80, 'requested_goal': 'left'}, events, geometry)
        self.assertFalse(b['suffix_first_correct'])


if __name__ == '__main__': unittest.main()
