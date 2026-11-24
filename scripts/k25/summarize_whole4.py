"""Paired full-residual vs frozen parallel4 audit, without policy inference."""
import argparse
import ast
from collections import Counter
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path

ALL=('left','front','right','back')
OLD=ALL[:3]
ARMS=dict(whole150='parallel150',whole400='parallel400')
EXTRA={'executed_residual_norm7','scaled_full_over_parent','scaled_removed_over_parent','removed_norm_full32'}


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return sha256(path.read_bytes()).hexdigest()


def load_analysis(folder):
    spec=importlib.util.spec_from_file_location('frozen_parallel4_audit',folder/'summarize_parallel4.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert digest(folder/'summarize_parallel4.py')==read(folder/'summary.json')['source_sha256']
    return module


def close_values(a,b):
    if isinstance(a,list):
        assert len(a)==len(b)
        return max((close_values(x,y) for x,y in zip(a,b)),default=0.)
    assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
    return abs(a-b)


def stage_counts(a,selected):
    out=a.counts(selected,ALL)
    for g in ALL:
        bit=1<<a.scoring.SITES.index(g)
        group=[r for r in selected if r['goal']==g]
        early=[r['init'] for r in group if any(m&bit for m in r['trace']['success'][:150])]
        first=[dict(init=r['init'],step=next((i+1 for i,m in enumerate(r['trace']['success']) if m&bit),None)) for r in group]
        out['per_goal'][g].update(ever150=len(early),ever150_ids=sorted(early),first_placement_steps=sorted(first,key=lambda r:r['init']))
    return out


def collect(base):
    reference=base.parent/'k25_parallel4';a=load_analysis(reference)
    p=read(base/'protocol.json');op=read(reference/'four_protocol.json');runtime=read(base/'runtime.json')
    manifest=read(reference/'artifact_manifest.json')
    used=['four_protocol.json','four_calibration.json','cases.json','runtime.json','summary.json','summarize_parallel4.py',*op['sources']]
    used += [f'four_{lane}{suffix}' for lane in range(20) for suffix in ['.jsonl.gz','_COMPLETE.json']]
    for name in used:assert digest(reference/name)==manifest['files']['docs/results/2026-11-24/k25_parallel4/'+name]['sha256'],name
    assert p['expected']==160 and p['horizon']==400 and p['arms']==ARMS
    assert p['ids']==op['ids']==list(range(800,820))
    assert p['models']==op['models'] and p['files']==op['files'] and p['reference_sources']==op['sources']
    assert p['alpha']==op['alpha']==read(reference/'four_calibration.json')['alpha']
    assert p['calibration_sha256']==op['calibration_sha256']==digest(reference/'four_calibration.json')
    cal=read(reference/'four_calibration.json');assert cal['n']==130 and not cal['behaviour_used_for_selection']
    assert p['alpha']==min(10.,sorted(s['ratio'] for s in cal['stats'])[65])
    assert p['reference_protocol_sha256']==digest(reference/'four_protocol.json')
    assert p['cases_sha256']==digest(base/'cases.json')==op['cases_sha256']
    assert read(base/'cases.json')==read(reference/'cases.json')
    assert p['plan_sha256']==runtime['plan_sha256']==digest(base/'analysis_plan.json')
    assert runtime['protocol_sha256']==digest(base/'protocol.json')
    assert runtime['capture_source_sha256']==digest(base/'capture_runtime.py')
    for n,h in p['sources'].items():assert digest(base/n)==h,n
    for n,h in op['sources'].items():assert digest(reference/n)==h,n
    text=(reference/'parallel_four.py').read_text(encoding='utf-8')
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='episode')
    function='\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n'
    assert sha256(function.encode()).hexdigest()==p['original_episode_sha256']
    assert runtime['reference_runtime_sha256']==digest(reference/'runtime.json')
    rr=read(reference/'runtime.json');assert runtime['packages']==rr['packages']
    assert set(rr['actual_modules'].values())<=set(runtime['actual_modules'].values())
    assert runtime['dependency_sha256']==rr['dependency_sha256']
    dependencies=base.parent.parent/'2026-11-22/k25_order_restore/dependencies'
    for path,h in runtime['dependency_sha256'].items():assert digest(dependencies/path)==h,path
    cases={r['init']:r for r in read(base/'cases.json')}
    refs=[];new=[];fingerprint=None;gates=0;max_input_difference=0.
    for lane in range(20):
        rpath=reference/f'four_{lane}.jsonl.gz';rm=read(reference/f'four_{lane}_COMPLETE.json')
        assert digest(rpath)==rm['rows_sha256']
        ref_rows=list(a.rows(rpath));assert len(ref_rows)==16
        reference_keys={(r['arm'],r['goal']):r for r in ref_rows}
        for r in ref_rows:a.dense(r,op,cases[r['init']])
        refs.extend(ref_rows)
        path=base/f'whole_{lane}.jsonl.gz';marker=read(base/f'whole_{lane}_COMPLETE.json')
        assert marker['rows_sha256']==digest(path) and marker['protocol_sha256']==digest(base/'protocol.json')
        assert marker['ids']==[800+lane] and marker['episodes']==8 and marker['sources']==p['sources']
        assert marker['weights_unchanged'] and marker['weights_fingerprints']==rm['weights_fingerprints']
        if fingerprint is not None:assert marker['weights_fingerprints']==fingerprint
        fingerprint=marker['weights_fingerprints']
        for gate in marker['gates']:
            assert gate['whole_formula_full32_bitwise'] and gate['original_projection_callback_bitwise'] and gate['whole_callback_bitwise']
            assert gate['removed_orthogonality_error']<5e-6;gates+=1
        selected=list(a.rows(path));assert len(selected)==8
        keyed={(r['arm'],r['goal']):r for r in selected}
        assert set(keyed)=={(arm,g) for arm in ARMS for g in ALL}
        for r in selected:
            assert r['init']==800+lane and r['method']=='full_residual' and r['alpha']==p['alpha']
            assert r['target']==r['goal'] and r['goal'] in ALL
            a.dense(r,p,cases[r['init']])
            high=150 if r['arm']=='whole150' else 400
            assert r['window']==[0,high] and [d['block'] for d in r['decisions']]==list(range(high//10))
            paired=reference_keys[ARMS[r['arm']],r['goal']]
            assert r['signatures'][0]==paired['signatures'][0] and r['seed']==paired['seed']
            initial=r['decisions'][0]['denoising'][0];original=paired['decisions'][0]['denoising'][0]
            assert set(initial)==set(original)|EXTRA
            for key,value in original.items():max_input_difference=max(max_input_difference,close_values(initial[key],value))
            for block in r['decisions']:
                assert len(block['denoising'])==10
                for d in block['denoising']:
                    assert d['rank'] in (0,1,2) and d['orthogonality_error']<5e-6
                    assert all(math.isfinite(v) for value in d.values() for v in (value if isinstance(value,list) else [value]))
                    assert math.isclose(d['executed_residual_norm7'],p['alpha']*d['residual_norm'],rel_tol=1e-12,abs_tol=1e-12)
                    assert math.isclose(d['scaled_full_over_parent'],d['executed_residual_norm7']/max(d['parent_norm'],1e-30),rel_tol=1e-12,abs_tol=1e-12)
                    assert math.isclose(d['scaled_removed_over_parent'],d['scaled_full_over_parent']*d['off_axis_fraction'],rel_tol=1e-9,abs_tol=1e-11)
                    projected=p['alpha']*d['projected_norm']/max(d['parent_norm'],1e-30)
                    assert math.isclose(d['scaled_full_over_parent']**2,projected**2+d['scaled_removed_over_parent']**2,rel_tol=1e-9,abs_tol=1e-11)
            new.append(r)
        for g in ALL:a.same_prefix(keyed['whole150',g],keyed['whole400',g])
    assert len(refs)==320 and len(new)==160 and gates==80
    queue=read(base/'queue_COMPLETE.json');assert queue==dict(episodes=160,units=20,failures=[])
    inventory=read(base/'final_inventory.json')
    assert inventory['queue']==queue and inventory['protocol_sha256']==digest(base/'protocol.json')
    assert inventory['source_sha256']==digest(base/'final_inventory.py')
    for path,h in p['files'].items():assert inventory['files'][path]==h,path
    for path,h in runtime['actual_modules'].items():assert inventory['files'][path]==h,path
    comparisons={}
    for name,ref_name in ARMS.items():
        old=[r for r in refs if r['arm']==ref_name];whole=[r for r in new if r['arm']==name]
        old_counts,whole_counts=stage_counts(a,old),stage_counts(a,whole)
        early={g:dict(gained=sorted(set(whole_counts['per_goal'][g]['ever150_ids'])-set(old_counts['per_goal'][g]['ever150_ids'])),
            lost=sorted(set(old_counts['per_goal'][g]['ever150_ids'])-set(whole_counts['per_goal'][g]['ever150_ids']))) for g in ALL}
        comparisons[name]=dict(reference_arm=ref_name,parallel=old_counts,whole=whole_counts,paired=a.changes(old,whole,ALL),paired_ever150=early,
            paired_old=a.changes([r for r in old if r['goal'] in OLD],[r for r in whole if r['goal'] in OLD],OLD))
    description={}
    for name in ARMS:
        description[name]={}
        for goal in ALL:
            values=[d for r in new if r['arm']==name and r['goal']==goal for b in r['decisions'] for d in b['denoising']]
            description[name][goal]=dict(n_inputs=len(values),rank={str(k):v for k,v in Counter(d['rank'] for d in values).items()},
                fields={k:a.describe([d[k] for d in values]) for k in ['scaled_full_over_parent','scaled_projection_over_parent','scaled_removed_over_parent','off_axis_fraction']})
    initial={g:dict(n=20,fields={k:a.describe([r['decisions'][0]['denoising'][0][k] for r in new if r['arm']=='whole150' and r['goal']==g]) for k in ['scaled_full_over_parent','scaled_projection_over_parent','scaled_removed_over_parent','off_axis_fraction']}) for g in ALL}
    return dict(new_episodes=160,reference_parallel_episodes=160,reference_context_episodes=160,reused_physical_starts=20,new_training=False,new_calibration=False,
        alpha=p['alpha'],windows=[[0,150],[0,400]],matched_initial_inputs=80,initial_input_max_absolute_difference=max_input_difference,bitwise_whole_first150_pairs=80,real_tensor_gates=gates,
        comparisons=comparisons,context={arm:stage_counts(a,[r for r in refs if r['arm']==arm]) for arm in ['parent','native']},
        whole_duration_changes=a.changes([r for r in new if r['arm']=='whole150'],[r for r in new if r['arm']=='whole400'],ALL),
        component_diagnostics=description,fixed_first_input=initial,weights_fingerprints_155_trainable=fingerprint,
        source_sha256=digest(Path(__file__)),reference_analysis_sha256=digest(reference/'summarize_parallel4.py'),protocol_sha256=digest(base/'protocol.json'),
        runtime_sha256=digest(base/'runtime.json'),final_inventory_sha256=digest(base/'final_inventory.json'),reference_manifest_sha256=digest(reference/'artifact_manifest.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('--check',action='store_true');args=p.parse_args()
    result=collect(args.base.resolve());path=args.base/'summary.json'
    if args.check:assert read(path)==result;print('WHOLE4_SUMMARY_CHECKED',result['new_episodes'])
    else:path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n');print('WHOLE4_SUMMARY_WRITTEN',result['new_episodes'])
