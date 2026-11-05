"""Audit saved paths and distinguish commanded direction from observed motion."""
import argparse
from collections import defaultdict
import csv
import gzip
import itertools
import json
import math
from pathlib import Path
from audit_stage2r_handoff import info
from stage2r_handoff_protocol import summarize_episode
from stage2s_same_instruction80 import references


def unit(v):
    n=math.hypot(*v)
    return [x/n for x in v] if n>1e-9 else None


def direction_metrics(events,goal,sites,window):
    origin=events[80];end=events[80+window]
    book=origin['object_positions']['black_book_1_pos'];eef=origin['eef_position']
    cmd=[sum(r['controller']['world_delta'][d] for r in events[81:81+window]) for d in (0,1)]
    out={'window':window,'command_sum_xy_m':cmd,'command_sum_norm_cm':100*math.hypot(*cmd),
        'native_grasp_steps':sum(r['native_grasp'] is True for r in events[81:81+window]),
        'native_grasp_available_steps':sum(r['native_grasp'] is not None for r in events[81:81+window])}
    for label,g in [('requested',goal),('right','right')]:
        centre=sites[g]['centre'];bearing=unit([centre[d]-book[d] for d in (0,1)])
        out['command_projection_'+label+'_cm']=100*sum(a*b for a,b in zip(cmd,bearing)) if bearing else None
        out['command_cosine_'+label]=sum(a*b for a,b in zip(unit(cmd),bearing)) if unit(cmd) and bearing else None
        for name,start,finish in [('book',book,end['object_positions']['black_book_1_pos']),('eef',eef,end['eef_position'])]:
            out[name+'_distance_reduction_'+label+'_cm']=100*(math.dist(start[:2],centre[:2])-math.dist(finish[:2],centre[:2]))
    out['relative_book_eef_change_cm']=100*math.dist(origin['relative_book_eef'],end['relative_book_eef'])
    return out


def audit(root,local_reference_base=None):
    load=lambda p:json.loads(p.read_text())
    data=root/'evaluation';m=load(data/'manifest.json');cfg=m['configuration']
    assert m['status']=='complete' and m==load(data/'COMPLETE.json')
    for name,h in m['files'].items():assert info(data/name)==h,name
    assert m['counts']['integrator_velocity_calls']==1600 and m['counts']['patch_flows']==0
    gate=load(data/'engineering.json');assert gate['passed'] and len(gate['checks'])==6
    assert max(r['max_abs'] for r in gate['checks'])<=1e-5
    refcfg=json.loads(json.dumps(cfg))
    if local_reference_base:
        for name,folder in [('q','stage2q_all_ca'),('r','stage2r_early_handoff')]:refcfg['references'][name]['server_root']=str(local_reference_base/folder)
    refs=references(refcfg)
    eps=list(map(json.loads,(data/'episodes.jsonl').read_text().splitlines()));byid={e['episode_id']:e for e in eps}
    fields=('mode','requested_goal','init_index','noise_seed')
    expected=set(itertools.product(('native','handoff80'),('left','front','right'),(0,1),(2026102801,2026102802)))
    assert len(eps)==len(byid)==24 and {tuple(e[k] for k in fields) for e in eps}==expected
    seen=set();directions=[];max_error=0.;paths={};replay_count=0
    with gzip.open(data/'events.jsonl.gz','rt') as f:
        for eid,group in itertools.groupby((json.loads(line) for line in f),lambda r:r['episode_id']):
            ev=list(group);assert [r['step'] for r in ev]==list(range(401));assert eid not in seen;seen.add(eid)
            ep=byid[eid];key=tuple(ep[k] for k in fields)
            computed=summarize_episode({'boundary':80,'requested_goal':ep['requested_goal']},ev[1:],cfg['movement_geometry']['sites'])
            assert all(ep[k]==v for k,v in computed.items())
            if key in refs:
                original=refs[key][1];replay_count+=1
                for actual,ref in zip(ev[1:],original[1:]):
                    assert actual['predicates']==ref['predicates'] and actual['action_unclipped']==ref['action_unclipped']
                    err=max(abs(a-b) for k,p in actual['object_positions'].items() for a,b in zip(p,ref['object_positions'][k]))
                    max_error=max(max_error,err);assert err<=1e-7
            for r in ev[1:]:
                assert r['action_executed']==[max(-1,min(1,v)) for v in r['action_unclipped']]
                c=r['controller'];assert max(abs(c['world_delta'][d]-(c['goal_pos'][d]-c['eef_at_set_goal'][d])) for d in range(3))<1e-12
                assert all(math.isfinite(v) for v in c['world_delta'])
            for window in (10,20,40):directions.append({**{k:ep[k] for k in fields},'episode_id':eid,**direction_metrics(ev,ep['requested_goal'],cfg['movement_geometry']['sites'],window)})
            paths[eid]=ev
    assert len(seen)==24 and replay_count==20
    replans=[json.loads(x) for x in gzip.open(data/'replans.jsonl.gz','rt')];assert len(replans)==160
    assert {(r['episode_id'],r['replan_index']) for r in replans}=={(e['episode_id'],j) for e in eps if e['reference_episode_id'] is None for j in range(40)}
    for r in replans:
        e=byid[r['episode_id']];assert e['requested_goal']=='right' and e['mode']=='handoff80'
        assert r['noise_generator_seed']==e['noise_seed']+1000003*r['replan_index']
        assert r['effective_model']==('R0' if r['replan_index']<8 else 'FT4') and r['requested_goal']=='right'
    # Decode all videos, including replay and newly generated episodes.
    try:
        import av
        video_checks=[]
        for ep in eps:
            p=data/'videos'/ep['video'];assert info(p)['sha256']==ep['video_sha256']
            with av.open(str(p)) as v:count=sum(1 for _ in v.decode(video=0))
            assert count==400;video_checks.append({'file':ep['video'],'frames':count,'sha256':ep['video_sha256']})
    except ImportError:video_checks=None
    table=[]
    for mode,goal in itertools.product(('native','handoff80'),('left','front','right')):
        xs=[e for e in eps if e['mode']==mode and e['requested_goal']==goal]
        table.append({'mode':mode,'requested_goal':goal,'episodes':4,'whole_first_correct':sum(e['full_movement']['first_targets']==[goal] for e in xs),
            'whole_first_right':sum(e['full_movement']['first_targets']==['right'] for e in xs),
            'whole_correct_ever':sum(e['full_correct_ever'] for e in xs),'terminal_correct':sum(e['terminal_correct'] for e in xs),
            'pretarget_at80':sum(e['pretarget_eligible'] for e in xs)})
    return {'status':'passed','episodes':24,'replayed_episodes':20,'new_right_handoffs':4,'replay_position_max_abs':max_error,
            'per_condition':table,'direction_rows':directions,'video_checks':video_checks,
            'limits':['Two development starts and two noises, no parameter updates.',
                'Same instruction throughout; this does not compare three instructions from the SAME handoff state.',
                'R0 was not trained on Right; its Right prefix is exploratory.',
                'Command projections are sums of native Cartesian goal offsets, NOT integrated displacement or latent intentions.']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();assert not (a.root/'audit.json').exists()
    result=audit(a.root);(a.root/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,key in [('results.csv','per_condition'),('directions.csv','direction_rows')]:
        with (a.root/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(result[key][0]));w.writeheader();w.writerows(result[key])
    (a.root/'report.md').write_text('# Same-instruction 80-step comparison\n\nComplete; see results.csv and directions.csv.\n')
    print(json.dumps({'status':result['status'],'episodes':24}))
