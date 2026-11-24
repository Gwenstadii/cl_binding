"""Full trace audit of timing-only replication and three-old-goal extension."""
import argparse
import ast
from collections import Counter
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
from statistics import median
import struct

import v_timing_protocol as scoring

PROFILES=('lr_front_1000','lf_right_1000','lr_front_12000','lf_right_12000')
IDS=tuple(range(800,820))
OLD=('left','front','right')
ALL=(*OLD,'back')
ARMS=('parent','native','parallel150','parallel400')


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return sha256(path.read_bytes()).hexdigest()
def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:yield from (json.loads(line) for line in f)


def dense(r,p,case):
    assert r['seed']==case['seed'] and r['text']==f'pick up the book and place it in the {r["target"]} compartment of the caddy'
    result=scoring.score(r['trace'],p['geometry'],case['z0'])
    assert all(r[k]==v for k,v in result.items())
    assert len(r['signatures'])==41 and len(r['action_hashes'])==40
    assert r['signatures'][0]==sha256(json.dumps(case['signature'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    for block,h in enumerate(r['action_hashes']):
        values=[v for a in r['trace']['actions'][block*10:(block+1)*10] for v in a]
        assert sha256(struct.pack('<70f',*values)).hexdigest()==h


def metric(r,key):
    if key=='first':return r['first']==[r['target']]
    if key=='first150':return scoring.movement(r['trace']['regions'],0,150)['first']==[r['target']]
    if key=='placed150':return bool(r['trace']['success'][149] & (1<<scoring.SITES.index(r['target'])))
    if key=='terminal':return r['terminal'][r['target']]
    if key=='ever':return r['ever'][r['target']]
    if key=='grasp':return r['first_grasp'] is not None
    raise ValueError(key)


METRICS=('first','first150','placed150','terminal','ever','grasp')


def counts(selected,goals):
    keyed={(r['goal'],r['init']):r for r in selected}
    assert set(keyed)=={(g,i) for g in goals for i in IDS}
    per={}
    for g in goals:
        rs=[keyed[g,i] for i in IDS]
        per[g]=dict(n=20,target=rs[0]['target'],**{m:sum(metric(r,m) for r in rs) for m in METRICS},
            first_distribution=dict(Counter('+'.join(r['first']) or 'none' for r in rs)),
            first_correct_final_failed=sum(metric(r,'first') and not metric(r,'terminal') for r in rs),
            placed150_final_failed=sum(metric(r,'placed150') and not metric(r,'terminal') for r in rs),
            wrong_Back_ever=sum(r['target']!='back' and r['ever']['back'] for r in rs),
            wrong_Back_terminal=sum(r['target']!='back' and r['terminal']['back'] for r in rs),
            Front_then_Back=sum(r['times']['front'] is not None and r['times']['back'] is not None and r['times']['front']<r['times']['back'] for r in rs))
    def joint(group):
        indices={m:[i for i in IDS if all(metric(keyed[g,i],m) for g in group)] for m in METRICS}
        return dict(n=20,**{m:len(v) for m,v in indices.items()},ids=indices)
    out=dict(per_goal=per,joint=joint(goals))
    if set(goals)==set(ALL):
        out['old_joint']=joint(OLD)
        out['old_first_and_Back_final']=dict(n=20,ids=[i for i in IDS if all(metric(keyed[g,i],'first') for g in OLD) and metric(keyed['back',i],'terminal')])
        out['old_first_and_Back_final']['count']=len(out['old_first_and_Back_final']['ids'])
    return out


def changes(before,after,goals):
    a={(r['goal'],r['init']):r for r in before};b={(r['goal'],r['init']):r for r in after}
    assert set(a)==set(b)=={(g,i) for g in goals for i in IDS}
    def compare(group):
        result={}
        for m in METRICS:
            before_ids={i for i in IDS if all(metric(a[g,i],m) for g in group)}
            after_ids={i for i in IDS if all(metric(b[g,i],m) for g in group)}
            result[m]=dict(gained=sorted(after_ids-before_ids),lost=sorted(before_ids-after_ids),kept=sorted(before_ids&after_ids),neither=sorted(set(IDS)-(before_ids|after_ids)))
        return result
    return dict(per_goal={g:compare((g,)) for g in goals},joint=compare(goals))


def same_prefix(a,b):
    assert a['trace']['start']==b['trace']['start']
    assert a['signatures'][:16]==b['signatures'][:16]
    assert a['action_hashes'][:15]==b['action_hashes'][:15]
    for k in ('object','eef','contacts','success','regions','actions'):assert a['trace'][k][:150]==b['trace'][k][:150],k
    assert a['decisions'][:15]==b['decisions'][:15]


def describe(values):
    values=sorted(values);assert values and all(math.isfinite(x) for x in values)
    return dict(n=len(values),median=median(values),q25=values[len(values)//4],q75=values[len(values)*3//4],min=values[0],max=values[-1])


def collect(base):
    fp,qp,runtime=(read(base/n) for n in ('full_protocol.json','four_protocol.json','runtime.json'))
    plan=read(base/'analysis_plan.json');cal=read(base/'four_calibration.json');prep=read(base/'four_preparation.json')
    assert fp['expected']==160 and qp['expected']==320 and plan['total_new_evaluation_episodes']==480
    assert fp['window']==[0,400] and fp['horizon']==qp['horizon']==400
    assert digest(base/'analysis_plan.json')==fp['plan_sha256']==qp['plan_sha256']==runtime['plan_sha256']
    assert digest(base/'full_protocol.json')==qp['full_protocol_sha256']==runtime['full_protocol_sha256']
    assert digest(base/'four_protocol.json')==runtime['four_protocol_sha256']
    for n,h in runtime['sources'].items():assert digest(base/n)==h,n
    assert runtime['capture_source_sha256']==digest(base/'capture_runtime.py')
    assert fp['sources']['parallel_full400.py']==runtime['sources']['parallel_full400.py']
    assert all(qp['sources'][n]==runtime['sources'][n] for n in qp['sources'])
    previous=base.parent.parent/'2026-11-23/k25_parallel'
    op=read(previous/'protocol.json')
    assert digest(previous/'protocol.json')==fp['previous_protocol_sha256']
    assert op['models']==fp['models'] and op['gains']==fp['gains']
    assert fp['previous_sources']==runtime['previous_sources']==op['sources']
    assert runtime['packages']==op['versions']
    assert runtime['original_sources']==op['original_sources']
    actual_hashes=set(runtime['actual_modules'].values())
    assert all(op['original_sources'][name] in actual_hashes for name in ['k21.py','k22.py'])
    for name,h in op['original_sources'].items():assert digest(previous/'history'/name)==h,name
    assert op['sources']['parallel_reproduction.py'] in actual_hashes
    assert digest(base.parent.parent/'2026-11-22/k25_v_timing/v_timing.py') in actual_hashes
    assert digest(base/'parallel_subspace.py') in actual_hashes
    for n,h in op['sources'].items():assert digest(previous/n)==h,n
    dependency_root=base.parent.parent/'2026-11-22/k25_order_restore/dependencies'
    assert runtime['dependency_sha256']==op['validated_dependency_sha256']
    for name,h in runtime['dependency_sha256'].items():assert digest(dependency_root/name)==h,name
    original=(previous/'parallel_reproduction.py').read_text(encoding='utf-8')
    node=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='episode')
    function='\n'.join(original.splitlines()[node.lineno-1:node.end_lineno])+'\n'
    assert sha256(function.encode()).hexdigest()==fp['patch']['original_function_sha256']
    for before,after in fp['patch']['edits'].items():assert function.count(before)==1;function=function.replace(before,after)
    assert sha256(function.encode()).hexdigest()==fp['patch']['modified_function_sha256']
    assert set(fp['patch']['edits'])=={'if block < 15:','window=[0,150]','case["z0"],0,150)'}
    assert digest(base/'four_preparation.json')==qp['preparation_sha256']==cal['preparation_sha256']
    assert digest(base/'four_calibration.json')==qp['calibration_sha256']
    assert cal['sources']==prep['sources']==qp['sources'] and not cal['behaviour_used_for_selection']
    assert {(s['init'],s['step']) for s in cal['stats']}=={(i,t) for i in range(10) for t in range(30,151,10)}
    assert cal['n']==130 and qp['alpha']==cal['alpha']==min(10.,sorted(s['ratio'] for s in cal['stats'])[65])
    for name,key in (('P_READY.json','parent'),('F_READY.json','executor')):
        marker=read(base/'history'/name)
        assert digest(base/'history'/name)==qp['files'][str(PurePosixPath(qp['models'][key]).with_name('READY.json'))]
        assert marker['sha256']==qp['files'][qp['models'][key]] and marker['count']==155
    assert read(base/'history/F_READY.json')['parent_sha256']==read(base/'history/P_READY.json')['sha256']
    training=read(base/'history/k26_training_manifest.json')
    assert training['parent_sha256']==read(base/'history/P_READY.json')['sha256'] and training['steps']==3000
    assert read(base/'history/k26_training_COMPLETE.json')['frozen_unchanged']
    cases={r['init']:r for r in read(base/'cases.json')}
    assert list(cases)==list(IDS) and read(base/'cases.json')==read(previous/'cases.json')
    assert digest(base/'cases.json')==qp['cases_sha256']
    full=[];reference=[];weights={};gates=0
    for profile in PROFILES:
        for lane in range(8):
            key=f'full_{profile}_{lane}';path=base/(key+'.jsonl.gz');marker=read(base/(key+'_COMPLETE.json'))
            assert marker['rows_sha256']==digest(path) and marker['protocol_sha256']==digest(base/'full_protocol.json')
            assert marker['source_sha256']==fp['sources']['parallel_full400.py'] and marker['weights_unchanged']
            if profile in weights:assert weights[profile]==marker['weights_fingerprints']
            weights[profile]=marker['weights_fingerprints']
            old_marker=read(previous/f'{profile}_{lane}_COMPLETE.json')
            old_path=previous/f'{profile}_{lane}.jsonl.gz'
            assert digest(old_path)==old_marker['rows_sha256'] and marker['weights_fingerprints']==old_marker['weights_fingerprints']
            old_rows={(r['init'],r['goal']):r for r in rows(old_path)}
            new=list(rows(path));assert len(new)==marker['episodes']==len(old_rows)
            for gate in marker['gates']:
                assert all(gate[k] for k in ['original_output_bitwise','formula_full32_bitwise','coefficient_from_50x7','padding_from_parent_axis']);gates+=1
            seen=set()
            for r in new:
                k=(r['init'],r['goal']);assert k not in seen and k in old_rows;seen.add(k)
                ref=old_rows[k];dense(r,fp,cases[r['init']]);dense(ref,fp,cases[r['init']]);same_prefix(r,ref)
                assert r['profile']==profile and r['alpha']==fp['gains'][profile]['alpha'] and r['window']==[0,400]
                assert [d['block'] for d in r['decisions']]==list(range(40))
                for d in r['decisions']:assert all(len(v)==10 and all(math.isfinite(x) for x in v) for v in d['denoising'].values())
                full.append(r);reference.append(ref)
    assert len(full)==160 and gates==64
    four=[];four_weights=None;four_gates=0
    for lane in range(20):
        key=f'four_{lane}';path=base/(key+'.jsonl.gz');marker=read(base/(key+'_COMPLETE.json'))
        assert marker['ids']==[800+lane] and marker['episodes']==16
        assert marker['rows_sha256']==digest(path) and marker['protocol_sha256']==digest(base/'four_protocol.json')
        assert marker['sources']==qp['sources'] and marker['weights_unchanged']
        if four_weights is not None:assert four_weights==marker['weights_fingerprints']
        four_weights=marker['weights_fingerprints']
        for gate in marker['gates']:
            assert gate['rank'] in (0,1,2)
            if gate['rank']==2:assert gate['self_reconstruction_checked']
            if gate['rank']==2 and gate['condition']<1e4:assert gate['normal_equations_checked']
            assert gate['actual_query_exact']==(gate['goal'] in OLD);four_gates+=1
        new=list(rows(path));assert {(r['arm'],r['goal']) for r in new}=={(a,g) for a in ARMS for g in ALL} and len(new)==16
        keyed={(r['arm'],r['goal']):r for r in new}
        for r in new:
            assert r['init']==800+lane and r['goal']==r['target'] and r['alpha']==qp['alpha'];dense(r,qp,cases[r['init']])
            blocks=40 if r['arm']=='parallel400' else 15 if r['arm']=='parallel150' else 0
            assert [d['block'] for d in r['decisions']]==list(range(blocks))
            for d in r['decisions']:
                assert len(d['denoising'])==10
                for item in d['denoising']:
                    assert item['rank'] in (0,1,2) and item['orthogonality_error']<5e-6
                    assert 0<=item['off_axis_fraction']<=1.000001
                    assert all(math.isfinite(v) for name,v in item.items() if not isinstance(v,list))
                    assert all(math.isfinite(v) for name,values in item.items() if isinstance(values,list) for v in values)
            four.append(r)
        for g in ALL:same_prefix(keyed['parallel150',g],keyed['parallel400',g])
    assert len(four)==320 and four_gates==80
    queue=read(base/'queue_COMPLETE.json');assert queue==dict(episodes=480,units=52,failures=[])
    inventory=read(base/'final_inventory.json')
    assert inventory['queue']==queue and inventory['source_sha256']==digest(base/'final_inventory.py')
    assert inventory['full_protocol_sha256']==digest(base/'full_protocol.json') and inventory['four_protocol_sha256']==digest(base/'four_protocol.json')
    for path,h in qp['files'].items():assert inventory['verified_files'][path]==h,path
    for path,h in runtime['actual_modules'].items():assert inventory['verified_files'][path]==h,path
    timings={}
    for profile in PROFILES:
        rs=[r for r in full if r['profile']==profile];refs=[r for r in reference if r['profile']==profile]
        diagnostic={}
        for goal in 'AB':
            diagnostic[goal]={}
            sign=1 if goal=='A' else -1
            for name,low,high in [('first150',0,15),('after150',15,40)]:
                values=[sign*c for r in rs if r['goal']==goal for d in r['decisions'][low:high] for c in d['denoising']['coefficient']]
                diagnostic[goal][name]=dict(n_inputs=len(values),correct_sign=sum(v>0 for v in values),correct_sign_fraction=sum(v>0 for v in values)/len(values),coefficient=describe(values))
        timings[profile]=dict(alpha=fp['gains'][profile]['alpha'],parallel150=counts(refs,('A','B')),parallel400=counts(rs,('A','B')),paired=changes(refs,rs,('A','B')),diagnostics=diagnostic)
    groups={arm:counts([r for r in four if r['arm']==arm],ALL) for arm in ARMS}
    diagnostic={}
    for arm in ARMS[2:]:
        diagnostic[arm]={}
        for g in ALL:
            entries=[v for r in four if r['arm']==arm and r['goal']==g for d in r['decisions'] for v in d['denoising']]
            diagnostic[arm][g]=dict(n_inputs=len(entries),rank={str(k):v for k,v in Counter(v['rank'] for v in entries).items()},
                fields={k:describe([v[k] for v in entries]) for k in ('condition','scaled_projection_over_parent','off_axis_fraction','orthogonality_error')},
                nearest_old_axis_correct=(sum(max(range(3),key=lambda i:v['old_axis_cosines'][i])==OLD.index(g) for v in entries) if g in OLD else None))
    # Reuse the logged first Euler input, before goal-dependent trajectories diverge.
    initial={(r['init'],r['goal']):r['decisions'][0]['denoising'][0] for r in four if r['arm']=='parallel150'}
    for i in IDS:
        assert max(initial[i,g]['parent_norm'] for g in ALL)-min(initial[i,g]['parent_norm'] for g in ALL)<1e-12
        assert max(initial[i,g]['late_norm'] for g in ALL)-min(initial[i,g]['late_norm'] for g in ALL)<1e-12
    first_input={g:dict(n=20,
        residual_over_parent=describe([initial[i,g]['residual_norm']/max(initial[i,g]['parent_norm'],1e-30) for i in IDS]),
        scaled_projection_over_parent=describe([initial[i,g]['scaled_projection_over_parent'] for i in IDS]),
        off_axis_fraction=describe([initial[i,g]['off_axis_fraction'] for i in IDS])) for g in ALL}
    return dict(episodes=480,full400_episodes=160,four_episodes=320,development_prefixes=10,independent_initial_states=20,reused_states=True,
        bitwise_first150_prefixes=240,full_original_formula_gates=gates,four_real_tensor_gates=four_gates,
        alpha_four=qp['alpha'],timing=timings,four=groups,four_diagnostics=diagnostic,four_fixed_first_input=first_input,
        four_duration_changes=changes([r for r in four if r['arm']=='parallel150'],[r for r in four if r['arm']=='parallel400'],ALL),
        weights_fingerprints_155_trainable=dict(two_old=weights,four=four_weights),
        source_sha256=digest(Path(__file__)),full_protocol_sha256=digest(base/'full_protocol.json'),four_protocol_sha256=digest(base/'four_protocol.json'),
        runtime_sha256=digest(base/'runtime.json'),final_inventory_sha256=digest(base/'final_inventory.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('--check',action='store_true');args=p.parse_args()
    result=collect(args.base.resolve());path=args.base/'summary.json'
    if args.check:assert read(path)==result;print('PARALLEL4_SUMMARY_CHECKED',result['episodes'])
    else:path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n');print('PARALLEL4_SUMMARY_WRITTEN',result['episodes'])
