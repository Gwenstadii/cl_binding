"""Model-free evidence audit and full condition/pair summary for early handoff."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
from stage2r_handoff_protocol import FIELDS, summarize_episode


def info(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}


def rows(p):
    opener=gzip.open if p.suffix=='.gz' else open
    with opener(p,'rt') as f:
        for line in f:yield json.loads(line)


def summarize(episodes):
    groups=defaultdict(list);pairs=defaultdict(dict)
    for r in episodes:
        groups[tuple(r[k] for k in FIELDS[:5])].append(r)
        pairs[tuple(r[k] for k in FIELDS if k!='requested_goal')][r['requested_goal']]=r
    table=[]
    for key,rs in sorted(groups.items()):
        assert len(rs)==4
        eligible=[r for r in rs if r['pretarget_eligible']]
        table.append(dict(zip(FIELDS[:5],key))|{'episodes':4,'pretarget_eligible':len(eligible),
            'suffix_first_correct':sum(r['suffix_first_correct'] for r in rs),
            'suffix_first_right':sum(r['suffix_first_right'] for r in rs),
            'suffix_correct_ever':sum(r['suffix_correct_ever'] for r in rs),
            'terminal_correct':sum(r['terminal_correct'] for r in rs),
            'eligible_first_correct':sum(r['suffix_first_correct'] for r in eligible),
            'eligible_first_right':sum(r['suffix_first_right'] for r in eligible),
            'eligible_terminal_correct':sum(r['terminal_correct'] for r in eligible),
            'suffix_first_left':sum(r['suffix_movement']['first_targets']==['left'] for r in rs),
            'suffix_first_front':sum(r['suffix_movement']['first_targets']==['front'] for r in rs),
            'suffix_no_entry':sum(not r['suffix_movement']['first_targets'] for r in rs)})
    pairrows=[]
    pfields=[k for k in FIELDS if k!='requested_goal']
    for key,rs in sorted(pairs.items()):
        assert set(rs)=={'left','front'}
        left,front=rs['left'],rs['front']
        assert left['boundary_signature']==front['boundary_signature']
        eligible=left['pretarget_eligible'] and front['pretarget_eligible']
        lf,ff=left['suffix_movement']['first_targets'],front['suffix_movement']['first_targets']
        source_goal=left['prefix_goal']
        pairrows.append(dict(zip(pfields,key))|{'pretarget_eligible':eligible,'left_first':lf,'front_first':ff,
            'both_instructed':lf==['left'] and ff==['front'],
            'both_right':lf==ff==['right'],'both_follow_prefix':lf==ff==[source_goal],
            'both_no_entry':not lf and not ff})
    pgroups=defaultdict(list)
    for r in pairrows:pgroups[tuple(r[k] for k in FIELDS[:4])].append(r)
    pt=[]
    for key,rs in sorted(pgroups.items()):
        eligible=[r for r in rs if r['pretarget_eligible']]
        pt.append(dict(zip(FIELDS[:4],key))|{'pairs':len(rs),'pretarget_eligible_pairs':len(eligible),
            **{'eligible_'+field:sum(r[field] for r in eligible) for field in
               ('both_instructed','both_right','both_follow_prefix','both_no_entry')}})
    return table,pt,pairrows


def audit(root,reference_root):
    data=root/'evaluation';load=lambda p:json.loads(p.read_text())
    m=load(data/'manifest.json');cfg=m['configuration']
    assert m['status']=='complete' and m==load(data/'COMPLETE.json')
    for name,expected in m['files'].items():assert info(data/name)==expected,name
    assert m['counts']['integrator_velocity_calls']==m['counts']['recipient_denoise_calls']==61440
    assert m['counts']['donor_denoise_calls']==m['counts']['patch_flows']==0
    gate=load(data/'engineering.json');assert gate['passed'] and len(gate['checks'])==48
    assert all(r['prefix_position_max_abs']<=1e-7 and r['first_chunk_action_max_abs']<=1e-5 for r in gate['checks'])
    for k in ('config_sha256','driver_sha256','model_sha256','sources'):assert gate[k]==m[k]
    src=reference_root/'evaluation'
    for name,expected in cfg['source_reference_hashes'].items():assert info(src/name)['sha256']==expected
    source_eps=list(rows(src/'episodes.jsonl'))
    needed={e['episode_id'] for e in source_eps if e['condition']=='native' and e['requested_goal'] in ('left','front')
            or e['condition']=='full_old_response' and e['window']==120}
    originals={eid:[] for eid in needed}
    for r in rows(src/'events.jsonl.gz'):
        if r['episode_id'] in originals:originals[r['episode_id']].append(r)
    hybrids={(e['requested_goal'],e['init_index'],e['noise_seed']):e['episode_id'] for e in source_eps if e['condition']=='full_old_response' and e['window']==120}
    source_noise={}
    for r in rows(src/'replans.jsonl.gz'):
        key=(r['noise_generator_seed'],r['replan_index'])
        if key in source_noise:assert source_noise[key]==r['noise_sha256']
        source_noise[key]=r['noise_sha256']
    eps=list(rows(data/'episodes.jsonl'));byid={r['episode_id']:r for r in eps}
    expected=set(itertools.product(('R0','FT4'),('left','front'),(40,80,120),('R0','FT4'),('left','front'),(0,1),(2026102801,2026102802)))
    assert len(byid)==len(eps)==192 and {tuple(r[k] for k in FIELDS) for r in eps}==expected
    signatures=defaultdict(set);event_ids=set();total=0;native_checks=0;hybrid_checks=0;max_action=0.;max_position=0.
    for eid,group in itertools.groupby(rows(data/'events.jsonl.gz'),lambda r:r['episode_id']):
        assert eid not in event_ids;event_ids.add(eid);ev=list(group);ep=byid[eid];h=ep['boundary']
        assert [r['step'] for r in ev]==list(range(401));total+=len(ev)
        recomputed=summarize_episode(ep,ev[1:],cfg['movement_geometry']['sites'])
        assert all(ep[k]==v for k,v in recomputed.items()),eid
        original=originals[ep['source_episode_id']]
        assert ev[0]['predicates']==original[0]['predicates']
        for r in ev[1:]:
            assert r['phase']==('replayed_prefix' if r['step']<=h else 'continuation')
            assert r['action_executed']==[max(-1,min(1,v)) for v in r['action_unclipped']]
        reference=None
        native=ep['continuation_model']==ep['source_model'] and ep['requested_goal']==ep['prefix_goal']
        hybrid=ep['source_model']=='R0' and ep['continuation_model']=='FT4' and ep['requested_goal']==ep['prefix_goal'] and h==120
        if native:reference=original;native_checks+=1
        if hybrid:reference=originals[hybrids[ep['requested_goal'],ep['init_index'],ep['noise_seed']]];hybrid_checks+=1
        for row in ev[1:]:
            step=row['step'];ref=original[step] if step<=h else reference[step] if reference is not None else None
            if ref is None:continue
            assert row['predicates']==ref['predicates']
            ae=max(abs(a-b) for a,b in zip(row['action_unclipped'],ref['action_unclipped']))
            pe=max(abs(a-b) for k,p in row['object_positions'].items() for a,b in zip(p,ref['object_positions'][k]))
            max_action=max(max_action,ae);max_position=max(max_position,pe)
            assert ae<=1e-5 and pe<=1e-7,(eid,step,ae,pe)
        signatures[ep['snapshot_id']].add(json.dumps(ep['boundary_signature'],sort_keys=True))
    assert total==76992 and len(event_ids)==192 and native_checks==48 and hybrid_checks==8
    assert len(signatures)==48 and all(len(v)==1 for v in signatures.values())
    seen=set();n=0
    for r in rows(data/'replans.jsonl.gz'):
        ep=byid[r['episode_id']];j=r['replan_index'];key=r['episode_id'],j
        assert key not in seen;seen.add(key)
        assert ep['boundary']//10<=j<40 and r['noise_generator_seed']==ep['noise_seed']+1000003*j
        assert r['noise_sha256']==source_noise[r['noise_generator_seed'],j]
        assert r['patch_flow_count']==r['donor_denoise_calls']==0 and r['recipient_denoise_calls']==10
        assert r['effective_model']==ep['continuation_model'] and r['requested_goal']==ep['requested_goal']
        n+=1
    assert n==6144
    table,pt,pairrows=summarize(eps)
    assert len(table)==48 and len(pt)==24 and len(pairrows)==96
    return {'status':'passed','episodes':192,'physical_steps_including_replayed_prefix':76800,
        'continuation_steps':61440,'replayed_prefix_steps':15360,'event_rows':total,'replans':n,
        'boundary_states':48,'instruction_pairs':96,'native_identity_checks':native_checks,
        'hybrid120_identity_checks':hybrid_checks,'max_identity_action_error':max_action,
        'max_replayed_position_error':max_position,'per_condition':table,'paired_summary':pt,'instruction_pair_results':pairrows,
        'limitations':['2 physical initial states x 2 shared noise seeds; repeated developmental trajectories.',
            '120-step boundaries can already have achieved a target; not fresh selection evidence.',
            'Source prefixes reconstructed by original controls, not policy inference.',
            'Snapshot image/controller signatures are worker witnesses, not independently reconstructed from hidden tensors.',
            'Changed instructions test conditional behavior from fixed states; require R0 reference before interpreting inability to redirect.']}


def export(root,result):
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,key in (('results.csv','per_condition'),('instruction_pairs.csv','paired_summary')):
        with (root/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(result[key][0]));w.writeheader();w.writerows(result[key])
    lines=['# Early handoff x instruction: complete and audited','',
           '|source|prefix|h|receiver|request|pretarget eligible|suffix first correct|suffix first right|terminal correct|',
           '|---|---|---|---|---|---|---|---|---|']
    for r in result['per_condition']:
        lines.append(f"|{r['source_model']}|{r['prefix_goal']}|{r['boundary']}|{r['continuation_model']}|{r['requested_goal']}|{r['pretarget_eligible']}/4|{r['suffix_first_correct']}/4|{r['suffix_first_right']}/4|{r['terminal_correct']}/4|")
    (root/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--reference-root',type=Path,required=True)
    args=parser.parse_args();assert not (args.root/'audit.json').exists()
    r=audit(args.root,args.reference_root);export(args.root,r)
    print(json.dumps({'status':r['status'],'episodes':r['episodes']}))
