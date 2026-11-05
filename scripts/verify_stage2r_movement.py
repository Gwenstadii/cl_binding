"""Independent sliding-window recount of primary prefix/suffix movement fields."""
import argparse
from collections import defaultdict, deque
import gzip
import itertools
import json
from pathlib import Path


def verify(root):
    manifest=json.loads((root/'evaluation/manifest.json').read_text())
    sites=manifest['configuration']['movement_geometry']['sites']
    eps={r['episode_id']:r for r in map(json.loads,(root/'evaluation/episodes.jsonl').read_text().splitlines())}
    groups=defaultdict(dict);count=0
    with gzip.open(root/'evaluation/events.jsonl.gz','rt') as f:
        iterator=(json.loads(line) for line in f)
        for eid,events in itertools.groupby(iterator,lambda r:r['episode_id']):
            events=list(events)[1:];ep=eps[eid];h=ep['boundary']
            assert [r['step'] for r in events]==list(range(1,401))
            z0=events[0]['object_positions']['black_book_1_pos'][2]
            times={}
            for name,part in [('prefix',events[:h]),('suffix',events[h:]),('full',events)]:
                history={g:deque(maxlen=5) for g in sites};first={g:None for g in sites}
                for r in part:
                    p=r['object_positions']['black_book_1_pos']
                    for g,s in sites.items():
                        ext=[sum(abs(s['rotation'][d][j])*s['half_size'][j] for j in range(3)) for d in (0,1)]
                        hit=p[2]-z0>.03 and all(abs(p[d]-s['centre'][d])<=ext[d] for d in (0,1))
                        history[g].append(hit)
                        if first[g] is None and len(history[g])==5 and all(history[g]):first[g]=r['step']-4
                start=min((t for t in first.values() if t is not None),default=None)
                targets=[g for g,t in first.items() if start is not None and t==start]
                assert first==ep[name+'_movement']['times'] and targets==ep[name+'_movement']['first_targets'],(eid,name)
                times[name]=(first,targets)
            p=events[h-1]['object_positions']['black_book_1_pos']
            inside=[]
            for g,s in sites.items():
                ext=[sum(abs(s['rotation'][d][j])*s['half_size'][j] for j in range(3)) for d in (0,1)]
                if p[2]-z0>.03 and all(abs(p[d]-s['centre'][d])<=ext[d] for d in (0,1)):inside.append(g)
            eligible=not times['prefix'][1] and not inside and not any(any(r['predicates'].values()) for r in events[:h])
            assert eligible==ep['pretarget_eligible']
            correct=times['suffix'][1]==[ep['requested_goal']]
            assert correct==ep['suffix_first_correct']
            key=(ep['source_model'],ep['prefix_goal'],h,ep['continuation_model'],ep['init_index'],ep['noise_seed'])
            groups[key][ep['requested_goal']]=(eligible,correct,times['suffix'][1])
            count+=1
    audit=json.loads((root/'audit.json').read_text());expected={
        (r['source_model'],r['prefix_goal'],r['boundary'],r['continuation_model'],r['init_index'],r['noise_seed']):r
        for r in audit['instruction_pair_results']}
    assert count==192 and len(groups)==96
    for k,rs in groups.items():
        left,front=rs['left'],rs['front'];pair=expected[k]
        assert pair['pretarget_eligible']==(left[0] and front[0])
        assert pair['both_instructed']==(left[1] and front[1])
        assert pair['left_first']==left[2] and pair['front_first']==front[2]
    return {'status':'passed','episodes':count,'instruction_pairs':len(groups),
        'scope':'Independent sliding-window prefix/suffix/full box-entry times, pretarget eligibility and dual-instruction choices',
        'near8_and_norms':'Auxiliary near8 not independently recounted here; no norm intervention in this experiment.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    out=a.root/'movement_independent_v1.json';assert not out.exists()
    result=verify(a.root);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
