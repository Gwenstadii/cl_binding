"""Streaming, model-free audit and movement report for the fixed night batch."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from audit_stage2p_ca_closedloop import index_unique,validate_actions
from audit_stage2n_cross_checkpoint import audit_random_norms
from analyze_stage2p_movement import inside_xy,first_run

def rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as f:
        for line in f:yield json.loads(line)

def info(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return dict(bytes=path.stat().st_size,sha256=h.hexdigest())

def expected_keys():
    common=list(itertools.product(('left','front'),(0,1),(2026102801,2026102802)))
    keys={(r,'native',None,0,g,i,s) for r in ('R0','FT4') for g,i,s in common}
    keys|={('FT4','native',None,0,'right',i,s) for i in (0,1) for s in (2026102801,2026102802)}
    keys|={('FT4','full_old_response',None,w,g,i,s) for w in (120,400) for g,i,s in common}
    keys|={('FT4',m,j,w,g,i,s) for j in (1,3,5,7,9,11,13,15) for w in (120,400)
           for m in ('cross_same','cross_wrong','random_delta') for g,i,s in common}
    return keys

FIELDS=('model','condition','site','window','requested_goal','init_index','noise_seed')
def audit(root):
    data=root/'evaluation';m=json.loads((data/'manifest.json').read_text());cfg=m['configuration']
    assert m['status']=='complete' and m==json.loads((data/'COMPLETE.json').read_text())
    assert m['driver_sha256']==info(Path(__file__).with_name('stage2q_all_ca_overnight.py'))['sha256']
    assert m['model_sha256']=={'R0':cfg['teacher_model_sha256'],'FT4':cfg['student_model_sha256']}
    assert m['scene_xml_sha256']==cfg['movement_geometry']['canonical_scene_xml_sha256']
    for name,expected in m['files'].items():assert info(data/name)==expected,name
    gate=json.loads((data/'engineering.json').read_text())
    assert gate['passed'] and len(gate['checks'])==32
    assert {(r['site'],r['model'],r['goal']) for r in gate['checks']}==set(itertools.product((1,3,5,7,9,11,13,15),('R0','FT4'),('left','front')))
    assert all(r['max_abs']<=1e-5 and len(r['flows'])==10 for r in gate['checks'])
    for field in ('script_sha256','driver_sha256','config_sha256','model_sha256','sources'):assert gate[field]==m[field]
    eps=list(rows(data/'episodes.jsonl'));lookup=index_unique(eps,FIELDS,expected_keys())
    byid=index_unique(eps,('episode_id',));reset=defaultdict(set);outputs={};groups=defaultdict(list)
    identity={};initial_books=defaultdict(list);raw_event_count=0
    for eid,iterator in itertools.groupby(rows(data/'events.jsonl.gz'),lambda r:r['episode_id']):
        ep=byid[eid,];events=list(iterator);assert [r['step'] for r in events]==list(range(401))
        assert eid not in outputs;raw_event_count+=len(events)
        assert not any(events[0]['predicates'].values())
        for r in events:
            assert set(r['predicates'])=={'left','front','right','back'}
            assert all(type(v) is bool for v in r['predicates'].values())
        reset[ep['init_index']].add(ep['initial_state_sha256']);assert ep['scene_xml_sha256']==m['scene_xml_sha256']
        initial=events[1]['object_positions']['black_book_1_pos'];initial_books[ep['init_index']].append(initial)
        hits={metric:{g:[] for g in ('left','front','right','back')} for metric in ('box','near8')}
        for r in events[1:]:
            validate_actions(r);p=r['object_positions']['black_book_1_pos'];assert all(math.isfinite(v) for v in p)
            if p[2]-initial[2]<=.03:continue
            distances={g:math.dist(p[:2],s['centre'][:2]) for g,s in cfg['movement_geometry']['sites'].items()}
            for g,s in cfg['movement_geometry']['sites'].items():
                if inside_xy(p,s):hits['box'][g].append(r['step'])
                if distances[g]<.08 and all(distances[g]+1e-9<v for k,v in distances.items() if k!=g):hits['near8'][g].append(r['step'])
        result={**{k:ep[k] for k in FIELDS},'episode_id':eid}
        for metric in hits:
            times={g:first_run(v,5) for g,v in hits[metric].items()};first=min((t for t in times.values() if t is not None),default=None)
            targets=[g for g,t in times.items() if first is not None and t==first]
            result.update({metric+'_correct':times[ep['requested_goal']] is not None,
                metric+'_first_correct':targets==[ep['requested_goal']],metric+'_first_right':targets==['right'],
                metric+'_times':times,metric+'_first_targets':targets})
        active=[g for g,v in events[-1]['predicates'].items() if v]
        result['terminal_correct']=active==[ep['requested_goal']]
        result['native_ever']=any(r['predicates'][ep['requested_goal']] for r in events)
        outputs[eid]=result;groups[tuple(ep[k] for k in FIELDS[:5])].append(result)
        if ep['model']=='R0' and ep['condition']=='native' or ep['condition']=='full_old_response' and ep['window']==400:
            identity[ep['condition'],ep['requested_goal'],ep['init_index'],ep['noise_seed']]=events
    assert len(outputs)==420 and raw_event_count==168420 and all(len(v)==1 for v in reset.values())
    assert all(max(math.dist(v[0],x) for x in v)<=1e-6 for v in initial_books.values())
    for g,i,s in itertools.product(('left','front'),(0,1),(2026102801,2026102802)):
        a=identity['native',g,i,s];b=identity['full_old_response',g,i,s]
        assert all(x['predicates']==y['predicates'] for x,y in zip(a,b))
        assert max(abs(x-y) for aa,bb in zip(a[1:],b[1:]) for x,y in zip(aa['action_executed'],bb['action_executed']))<=1e-5
    rseen=set();noise=defaultdict(set);calls=defaultdict(int);norm_expected=set();max_witness=0.
    for r in rows(data/'replans.jsonl.gz'):
        eid=r['episode_id'];ep=byid[eid,];j=r['replan_index'];assert 0<=j<40 and (eid,j) not in rseen
        rseen.add((eid,j));assert r['start_step']==10*j and r['noise_generator_seed']==ep['noise_seed']+1000003*j
        noise[ep['noise_seed'],j].add(r['noise_sha256'])
        active=ep['site'] is not None and 10*j<ep['window'];expected_patch=10 if active else 0
        assert r['patch_flow_count']==expected_patch
        expected_role='R0' if ep['condition']=='full_old_response' and 10*j<ep['window'] else ep['model']
        assert r['effective_model']==expected_role and r['site']==ep['site'] and r['window']==ep['window']
        if active:
            expected_goal=('front' if ep['requested_goal']=='left' else 'left') if ep['condition']=='cross_wrong' else ep['requested_goal']
            assert r['donor_model']=='R0' and r['donor_goal']==expected_goal
        else:assert r['donor_model'] is None and r['donor_goal'] is None
        assert r['donor_denoise_calls']==expected_patch
        assert r['recipient_denoise_calls']==(20 if active and ep['condition']=='random_delta' else 10)
        for k in ('patch_flow_count','recipient_denoise_calls','donor_denoise_calls'):calls[k]+=r[k]
        max_witness=max(max_witness,r['actual_message_witness_max_abs'])
        if active and ep['condition']=='random_delta':norm_expected|={(eid,j,f) for f in range(10)}
    assert len(rseen)==16800 and all(len(v)==1 for v in noise.values()) and max_witness<=1e-5
    assert dict(calls)=={'patch_flow_count':99840,'recipient_denoise_calls':201280,'donor_denoise_calls':99840}
    norm_seen=set();max_norm=0.
    for r in rows(data/'norms.jsonl.gz'):
        key=r['episode_id'],r['replan_index'],r['flow_index'];assert key not in norm_seen;norm_seen.add(key)
        ep=byid[r['episode_id'],]
        anchor=r['episode_id']+f"|r{r['replan_index']}|f{r['flow_index']}"
        anchor=re.sub(r'^j\d+\|','',anchor);anchor=re.sub(r'\|w\d+\|','|',anchor)
        payload=f"{cfg['random_seed_salt']}|{anchor}|FT4|{ep['requested_goal']}|{ep['site']}".encode('ascii')
        assert r['random_seed']==int.from_bytes(hashlib.sha256(payload).digest()[:8],'big')%(2**63)
        q=audit_random_norms(r,{'effective_block_l2':r['correct_delta_block_l2']})
        max_norm=max(max_norm,q['max_relative_actual_norm_difference'])
    assert norm_seen==norm_expected and len(norm_seen)==33280
    table=[]
    for key,rs in sorted(groups.items(),key=lambda item:str(item[0])):
        assert len(rs)==4
        table.append(dict(zip(FIELDS[:5],key))|{'episodes':4}|{field:sum(r[field] for r in rs) for field in
            ('box_correct','box_first_correct','box_first_right','near8_correct','near8_first_correct','terminal_correct','native_ever')}|
            {'first_'+goal:sum(r['box_first_targets']==[goal] for r in rs) for goal in ('left','front','right','back')}|
            {'no_movement_entry':sum(not r['box_first_targets'] for r in rs)})
    result={'status':'passed','episodes':420,'physical_states':2,'per_condition':table,'episode_results':list(outputs.values()),
        'counts':dict(calls),'random_norm_records':len(norm_seen),'max_norm_relative_difference':max_norm,
        'limitations':['Development screen; 4 episodes per goal, two physical starts, multiple comparisons.',
                       'Primary endpoint is book-centre transport, no release/contact/final requirement.',
                       'M interventions only; no individual P/C intervention, no weight update.',
                       'Norm/witness claims cannot reconstruct hidden tensors from logged evidence.']}
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# Overnight CA movement screen','', 'Status: complete and audited. No training. Two development states; no population claim.','',
           '|model|condition|site|window|goal|correct transport|first correct|first right|near correct|',
           '|---|---|---|---|---|---|---|---|---|']
    lines += [f"|{r['model']}|{r['condition']}|{r['site']}|{r['window']}|{r['requested_goal']}|{r['box_correct']}/4|{r['box_first_correct']}/4|{r['box_first_right']}/4|{r['near8_correct']}/4|" for r in table]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    assert not (args.root/'audit.json').exists(),'Existing audit must be retained'
    r=audit(args.root);print(json.dumps({'status':r['status'],'episodes':r['episodes']}))
