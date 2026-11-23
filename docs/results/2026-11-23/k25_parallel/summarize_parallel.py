"""Audit and rescore the one-method K22 reproduction; no model inference."""
import argparse
from collections import Counter
from hashlib import sha256
import gzip
import json
import math
from pathlib import Path
from statistics import median
import struct

import parallel_protocol as p


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        yield from (json.loads(line) for line in f)


def describe(values):
    values=sorted(values)
    assert values and all(math.isfinite(v) for v in values)
    return dict(n=len(values),median=median(values),q25=values[len(values)//4],q75=values[3*len(values)//4],min=values[0],max=values[-1])


def dense(row,protocol,case):
    goal=p.GOALS[row.get('profile',row.get('line'))][row['goal']]
    assert row['target']==goal and row['seed']==case['seed']
    assert row['text']==f'pick up the book and place it in the {goal} compartment of the caddy'
    scored=p.scoring.score(row['trace'],protocol['geometry'],case['z0'])
    assert all(row[k]==v for k,v in scored.items())
    assert len(row['signatures'])==41
    assert row['signatures'][0]==sha256(json.dumps(case['signature'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert len(row['action_hashes'])==40
    for block,h in enumerate(row['action_hashes']):
        values=[v for action in row['trace']['actions'][block*10:(block+1)*10] for v in action]
        assert sha256(struct.pack('<70f',*values)).hexdigest()==h


def counts(selected):
    assert len(selected)==40
    bykey={(r['goal'],r['init']):r for r in selected}
    assert len(bykey)==40 and {i for _,i in bykey}==set(p.IDS)
    out={}
    for g in 'AB':
        group=[r for r in selected if r['goal']==g]
        out[g]=dict(target=group[0]['target'],n=20,
            first=sum(r['first']==[r['target']] for r in group),
            terminal=sum(r['terminal'][r['target']] for r in group),
            ever=sum(r['ever'][r['target']] for r in group),
            grasp=sum(r['first_grasp'] is not None for r in group),
            first_distribution=dict(Counter('+'.join(r['first']) or 'none' for r in group)),
            first_by150=sum(p.scoring.movement(r['trace']['regions'],0,150)['first']==[r['target']] for r in group),
            first_then_lost=sum(r['first']==[r['target']] and not r['terminal'][r['target']] for r in group))
    joint={k:[] for k in ('first','terminal','ever','grasp','first_by150')}
    for i in p.IDS:
        pair=[bykey[g,i] for g in 'AB']
        for metric,fn in (
            ('first',lambda r:r['first']==[r['target']]),
            ('terminal',lambda r:r['terminal'][r['target']]),
            ('ever',lambda r:r['ever'][r['target']]),
            ('grasp',lambda r:r['first_grasp'] is not None),
            ('first_by150',lambda r:p.scoring.movement(r['trace']['regions'],0,150)['first']==[r['target']])):
            if all(fn(r) for r in pair): joint[metric].append(i)
    out['joint']=dict(n=20,**{k:len(v) for k,v in joint.items()},ids=joint)
    return out


def diagnostics(selected,alpha):
    out={}
    for g in 'AB':
        group=[r for r in selected if r['goal']==g]
        sign=1 if g=='A' else -1
        vals={k:[] for k in ('signed_cos','signed_c1','scaled_coefficient','late_over_parent','scaled_residual_over_parent','mu_shift_over_parent','scaled_perp_over_parent','projection_energy_fraction')}
        trajectory=[]
        correct=zero=0
        for r in group:
            local=[]
            for decision in r['decisions']:
                d=decision['denoising']
                for j in range(10):
                    old=d['parent_norm'][j]
                    resid=d['residual_norm'][j]
                    c1=d['c1'][j]
                    signed=sign*c1
                    vals['signed_cos'].append(sign*d['cos'][j])
                    vals['signed_c1'].append(signed)
                    vals['scaled_coefficient'].append(sign*d['coefficient'][j])
                    vals['late_over_parent'].append(d['late_norm'][j]/max(old,1e-30))
                    vals['scaled_residual_over_parent'].append(alpha*resid/max(old,1e-30))
                    vals['mu_shift_over_parent'].append(d['mean_gap_norm'][j]/max(old,1e-30))
                    vals['scaled_perp_over_parent'].append(d['perp_over_old'][j])
                    vals['projection_energy_fraction'].append((c1*old/max(resid,1e-30))**2)
                    correct+=signed>0
                    zero+=d['zero_axis'][j]
                    local.append(signed)
            trajectory.append(dict(init=r['init'],signed_c1_median=median(local),correct_sign=sum(v>0 for v in local),n=len(local)))
        assert len(vals['signed_cos'])==3000
        out[g]=dict(n_inputs=3000,correct_sign=correct,correct_sign_fraction=correct/3000,zero_axis=zero,
            fields={k:describe(v) for k,v in vals.items()},trajectory=trajectory)
    return out


def references(base,protocol,cases):
    anchor=base.parent/'k25_anchor'
    timing=base.parent.parent/'2026-11-22/k25_v_timing'
    out={}
    for profile in p.PROFILES[:2]:
        out[profile]={}
        for source,folder in (('parent',anchor),('native',timing)):
            assert read(folder/'cases.json')==list(cases.values())
            rp=read(folder/'protocol.json')
            assert rp['geometry']==protocol['geometry']
            cp=rp['checkpoints'][profile+':0'] if source=='parent' else rp['checkpoints'][profile]
            expected=cp['sha256'] if source=='parent' else cp['parent_sha256']
            assert expected==protocol['models'][profile]['parent_sha256']
            if source=='native': assert cp['late_sha256']==protocol['models'][profile]['executor_sha256']
            selected=[]
            files={}
            for lane in range(8):
                path=folder/f'{profile}_{lane}.jsonl.gz'
                marker=read(folder/f'{profile}_{lane}_COMPLETE.json')
                assert digest(path)==marker['rows_sha256']
                files[path.name]=marker['rows_sha256']
                for row in rows(path):
                    pick=(row.get('family')=='policy' and row.get('anchor')==0) if source=='parent' else row['window']=='native'
                    if pick and row['goal'] in 'AB':
                        dense(row,protocol,cases[row['init']])
                        selected.append(row)
            out[profile][source]=dict(counts=counts(selected),archive=folder.name,protocol_sha256=digest(folder/'protocol.json'),row_files=files,executed_this_round=False)
    return out


def collect(base):
    protocol=read(base/'protocol.json')
    assert protocol['expected']==160 and protocol['horizon']==400 and protocol['window']==[0,150]
    assert protocol['profiles']==list(p.PROFILES) and protocol['ids']==list(p.IDS)
    assert digest(base/'cases.json')==protocol['cases_sha256']
    assert digest(base/'preparation.json')==protocol['preparation_sha256']
    preparation=read(base/'preparation.json')
    amendment=read(base/'source_amendment.json')
    assert digest(base/'source_amendment.json')==protocol['technical_amendment_sha256']
    assert not amendment['method_or_gain_or_window_changed'] and amendment['completed_shards']==0
    old_source=(base/'technical_attempt_01/parallel_reproduction.py').read_bytes()
    new_source=(base/'parallel_reproduction.py').read_bytes()
    assert sha256(old_source).hexdigest()==amendment['old_source_sha256']==preparation['sources']['parallel_reproduction.py']
    assert sha256(new_source).hexdigest()==amendment['new_source_sha256']==protocol['sources']['parallel_reproduction.py']
    assert old_source.replace(amendment['replacement']['before'].encode(),amendment['replacement']['after'].encode())==new_source
    assert digest(base/'technical_attempt_01/protocol.json')==amendment['old_protocol_sha256']
    for item in amendment['interrupted']:
        assert digest(base/'technical_attempt_01'/item['file'])==item['sha256']
    for name,h in protocol['sources'].items(): assert digest(base/name)==h,name
    for name,h in protocol['original_sources'].items(): assert digest(base/'history'/name)==h,name
    runtime=read(base/'runtime.json')
    assert runtime['capture_source_sha256']==digest(base/'capture_runtime.py')
    assert runtime['analysis_plan_sha256']==digest(base/'analysis_plan.json')
    assert runtime['queue_source_sha256']==digest(base/'run_parallel_reproduction.py')
    assert runtime['packages']==protocol['versions']
    assert runtime['dependency_sha256']==protocol['validated_dependency_sha256']
    dependency_root=base.parent.parent/'2026-11-22/k25_order_restore/dependencies'
    for relative,h in protocol['validated_dependency_sha256'].items(): assert digest(dependency_root/relative)==h,relative
    timing_source=base.parent.parent/'2026-11-22/k25_v_timing/v_timing.py'
    assert digest(timing_source) in runtime['actual_shared_modules'].values()
    assert digest(base/'v_timing_protocol.py') in runtime['actual_shared_modules'].values()
    cases={r['init']:r for r in read(base/'cases.json')}
    assert set(cases)==set(p.IDS)
    gains={}
    for profile in p.PROFILES:
        gain=protocol['gains'][profile]
        gains[profile]=gain['alpha']
        if profile.endswith('1000'):
            path=base/f'calibration_{profile}.json'
            cal=read(path)
            assert digest(path)==gain['sha256'] and cal['n']==130
            assert cal['preparation_sha256']==protocol['preparation_sha256']
            assert cal['source_sha256']==preparation['sources']['parallel_reproduction.py']
            assert {(s['init'],s['step']) for s in cal['stats']}=={(i,t) for i in range(10) for t in range(30,151,10)}
            assert cal['alpha']==min(10.,p.upper_median([s['ratio'] for s in cal['stats']]))==gain['alpha']
            assert not cal['behaviour_used_for_selection']
        elif profile.startswith('lf_'):
            assert digest(base/'history/original_lf_alpha.json')==gain['sha256']
            assert read(base/'history/original_lf_alpha.json')['alpha']==gain['alpha']
        else:
            assert gain['alpha']==2.959 and gain['sha256']==protocol['original_sources']['k21.py']
    result=[]
    keys=set()
    fingerprints={}
    gate_count=0
    for profile in p.PROFILES:
        for lane in range(8):
            path=base/f'{profile}_{lane}.jsonl.gz'
            complete=read(base/f'{profile}_{lane}_COMPLETE.json')
            assert complete['profile']==profile and complete['ids']==p.ids(lane)
            assert complete['weights_unchanged'] and len(complete['weights_fingerprints'])==2
            if profile in fingerprints: assert fingerprints[profile]==complete['weights_fingerprints']
            fingerprints[profile]=complete['weights_fingerprints']
            assert complete['protocol_sha256']==digest(base/'protocol.json')
            assert complete['sources']==protocol['sources'] and complete['original_sources']==protocol['original_sources']
            assert digest(path)==complete['rows_sha256']
            assert all(n['values_equal'] for n in complete['normalizers'].values())
            for gate in complete['gates']:
                assert all(gate[k] for k in ('original_output_bitwise','formula_full32_bitwise','coefficient_from_50x7','padding_from_parent_axis'))
                gate_count+=1
            shard=list(rows(path))
            assert len(shard)==complete['episodes']==len(p.expected(profile,lane))
            for row in shard:
                key=(row['profile'],row['goal'],row['init'])
                assert key not in keys and key in p.expected(profile,lane)
                keys.add(key)
                dense(row,protocol,cases[row['init']])
                assert row['alpha']==gains[profile] and row['window']==[0,150]
                intervals=p.scoring.interval_metrics(row['trace'],protocol['geometry'],cases[row['init']]['z0'],0,150)
                for k,v in intervals.items(): assert row[k]==v,k
                assert [d['block'] for d in row['decisions']]==list(range(15))
                for decision in row['decisions']:
                    d=decision['denoising']
                    assert set(d)=={'c1','cos','perp_over_old','zero_axis','parent_norm','late_norm','residual_norm','mean_gap_norm','coefficient'}
                    assert all(len(v)==10 and all(math.isfinite(x) for x in v) for v in d.values())
                    assert all(abs(c)<=1.000001 for c in d['cos'])
                    assert all(math.isclose(gains[profile]*a,b,rel_tol=1e-12) for a,b in zip(d['c1'],d['coefficient']))
                result.append(row)
    assert len(result)==160 and gate_count==64
    anchor=base.parent/'k25_anchor'
    anchor_summary=read(anchor/'summary.json')
    anchor_manifest=read(anchor/'artifact_manifest.json')
    assert digest(anchor/'summary.json')==anchor_manifest['files']['docs/results/2026-11-23/k25_anchor/summary.json']['sha256']
    for profile in p.PROFILES[:2]:
        previous=anchor_summary['weights_fingerprints'][profile]
        assert fingerprints[profile]==[previous['1000'],previous['0']]
        assert fingerprints[profile][1]==fingerprints[profile.replace('1000','12000')][1]
    queue=read(base/'queue_COMPLETE.json')
    assert queue==dict(episodes=160,units=32,failures=[])
    inventory=read(base/'final_inventory.json')
    assert inventory['queue']==queue and inventory['protocol_sha256']==digest(base/'protocol.json')
    assert inventory['source_sha256']==digest(base/'final_inventory.py')
    for profile,cfg in protocol['models'].items():
        assert inventory['sha256'][cfg['parent']+'/model.safetensors']==cfg['parent_sha256']
        key=cfg['executor']+('/model.safetensors' if profile.endswith('12000') else '')
        assert inventory['sha256'][key]==cfg['executor_sha256']
    out=dict(episodes=160,calibration_prefixes=20,calibration_controls=3200,reused_initial_states=20,completed_shards=32,original_formula_bitwise_gates=64,
        technical_interrupted_shards=len(amendment['interrupted']),technical_complete_rows_readable=sum(r['complete_rows_readable'] for r in amendment['interrupted']),
        source_sha256=digest(Path(__file__)),protocol_sha256=digest(base/'protocol.json'),runtime_sha256=digest(base/'runtime.json'),final_inventory_sha256=digest(base/'final_inventory.json'),
        gains=gains,weights_fingerprints_155_trainable=fingerprints,
        profiles={profile:dict(counts=counts([r for r in result if r['profile']==profile]),diagnostics=diagnostics([r for r in result if r['profile']==profile],gains[profile])) for profile in p.PROFILES},
        references=references(base,protocol,cases))
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('base',type=Path)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    summary=collect(args.base.resolve())
    path=args.base/'summary.json'
    if args.check:
        assert read(path)==summary
        print('PARALLEL_SUMMARY_CHECKED',summary['episodes'],flush=True)
    else:
        path.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print('PARALLEL_SUMMARY_WRITTEN',summary['episodes'],flush=True)
