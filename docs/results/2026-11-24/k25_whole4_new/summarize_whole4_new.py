"""Rescore complete new-model rollouts with frozen K25 scoring definitions."""
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from statistics import median

ALL=('left','front','right','back')
EXTRA={'executed_residual_norm7','scaled_full_over_parent','scaled_removed_over_parent','removed_norm_full32'}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return sha256(p.read_bytes()).hexdigest()
def stage_counts(a,rows):
    result=a.counts(rows,ALL)
    for g in ALL:
        selected=[r for r in rows if r['goal']==g];bit=1<<a.scoring.SITES.index(g)
        result['per_goal'][g].update(ever150=sum(any(m&bit for m in r['trace']['success'][:150]) for r in selected),
            final50_fraction=[dict(init=r['init'],fraction=sum(bool(m&bit) for m in r['trace']['success'][-50:])/50) for r in selected])
    return result

def collect(base):
    ref=base.parent/'k25_parallel4'
    spec=importlib.util.spec_from_file_location('reference_audit',ref/'summarize_parallel4.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    assert digest(ref/'summarize_parallel4.py')==read(ref/'summary.json')['source_sha256']
    p=read(base/'protocol.json');prep=read(base/'preparation.json');selection=read(base/'selection.json')
    start,end=read(base/'start_identity.json'),read(base/'final_identity.json')
    assert start['files']==end['files'] and start['packages']==end['packages']
    assert start['source_sha256']==end['source_sha256']==digest(base/'capture_identity.py')
    for n,h in prep['sources'].items():assert start['files'][str(Path('/home/lzx/cl_binding/.server/k25_whole4_new_20261124_01')/n).replace('\\','/')]==h
    assert p['preparation_sha256']==digest(base/'preparation.json') and prep['plan_sha256']==digest(base/'plan.json')
    for name,value in p['sources'].items():assert digest(base/name)==value,name
    assert p['calibration_sha256']==digest(base/'calibration.json')
    cal=read(base/'calibration.json');assert len(cal['stats'])==130 and p['alpha']==min(10.,sorted(r['ratio'] for r in cal['stats'])[65])
    assert cal['step']==p['selected_step']==selection['selected_step']
    assert start['files'][p['models']['executor']]==prep['candidates'][str(p['selected_step'])]['sha256']
    assert p['cases_sha256']==digest(base/'cases.json')==digest(ref/'cases.json')
    cases={r['init']:r for r in read(base/'cases.json')};dev_cases={r['init']:r for r in read(base/'dev_cases.json')}
    assert digest(base/'dev_cases.json')==prep['dev_cases_sha256'] and set(cases).isdisjoint(dev_cases)
    oldp=read(ref/'four_protocol.json');oldmanifest=read(ref/'artifact_manifest.json')
    parent=[];rows=[];fingerprints=None;prefix_pairs=0;initial_pairs=0
    for lane in range(20):
        path=base/f'new_{lane}.jsonl.gz';marker=read(base/f'new_{lane}_COMPLETE.json')
        assert digest(path)==marker['rows_sha256'] and marker['protocol_sha256']==digest(base/'protocol.json')
        assert marker['weights_unchanged'] and marker['episodes']==20
        if fingerprints is not None:assert marker['weights_fingerprints']==fingerprints
        fingerprints=marker['weights_fingerprints']
        assert all(g['whole_full32_formula_bitwise'] and g['whole_callback_bitwise'] for g in marker['whole_gates'])
        assert len(marker['whole_gates'])==len(marker['projection_gates'])==4
        selected=list(a.rows(path));assert len(selected)==20
        keyed={(r['arm'],r['goal']):r for r in selected}
        assert set(keyed)=={(arm,g) for arm in p['arms'] for g in ALL}
        for r in selected:
            assert r['init']==800+lane and r['alpha']==p['alpha'] and r['checkpoint_step']==p['selected_step']
            a.dense(r,p,cases[r['init']])
            high=150 if r['arm'].endswith('150') else 400
            expected=list(range(high//10)) if r['arm']!='native' else []
            assert [d['block'] for d in r['decisions']]==expected
        for method in ['parallel','whole']:
            for g in ALL:a.same_prefix(keyed[method+'150',g],keyed[method+'400',g]);prefix_pairs+=1
        for suffix in ['150','400']:
            for g in ALL:
                x=keyed['parallel'+suffix,g]['decisions'][0]['denoising'][0]
                y=keyed['whole'+suffix,g]['decisions'][0]['denoising'][0]
                assert x=={k:v for k,v in y.items() if k not in EXTRA};initial_pairs+=1
        rows.extend(selected)
        rp=ref/f'four_{lane}.jsonl.gz';rm=read(ref/f'four_{lane}_COMPLETE.json')
        assert fingerprints[1]==rm['weights_fingerprints'][1] and fingerprints[0]!=rm['weights_fingerprints'][0]
        assert digest(rp)==rm['rows_sha256']==oldmanifest['files']['docs/results/2026-11-24/k25_parallel4/'+rp.name]['sha256']
        reused=[r for r in a.rows(rp) if r['arm']=='parent'];assert len(reused)==4
        for r in reused:a.dense(r,oldp,cases[r['init']])
        parent.extend(reused)
    qualification=[]
    for selected in selection['results']:
        step=selected['step'];dev=[]
        for lane in range(4):
            path=base/f'dev_{step}_{lane}.jsonl.gz';m=read(base/f'dev_{step}_{lane}_COMPLETE.json')
            assert m['weights_unchanged'] and digest(path)==m['rows_sha256']
            dev.extend(a.rows(path))
        assert len(dev)==20 and {r['init'] for r in dev}==set(dev_cases)
        for r in dev:a.dense(r,p,dev_cases[r['init']])
        counts=dict(step=step,first=sum(r['first']==['back'] for r in dev),ever=sum(r['ever']['back'] for r in dev),terminal=sum(r['terminal']['back'] for r in dev))
        assert counts==selected
        qualification.append(dict(**counts,placed150=sum(bool(r['trace']['success'][149]&8) for r in dev),
            final50_fraction=[dict(init=r['init'],fraction=sum(bool(m&8) for m in r['trace']['success'][-50:])/50) for r in dev]))
    passed=[r['step'] for r in qualification if r['terminal']>=16 and r['ever']>=16]
    assert selection['qualified']==bool(passed) and p['selected_step']==(passed[0] if passed else 24000)
    groups={arm:[r for r in rows if r['arm']==arm] for arm in p['arms']}
    comparisons={}
    for suffix in ['150','400']:
        x,y=groups['parallel'+suffix],groups['whole'+suffix]
        comparisons['whole'+suffix]=dict(parallel=stage_counts(a,x),whole=stage_counts(a,y),paired=a.changes(x,y,ALL),
            paired_old=a.changes([r for r in x if r['goal']!='back'],[r for r in y if r['goal']!='back'],ALL[:3]))
    complete=read(base/'COMPLETE.json');assert complete['episodes']==400 and complete['qualification_episodes']==20*len(qualification) and not complete['failures']
    initial={g:{key:median(r['decisions'][0]['denoising'][0][key] for r in groups['whole150'] if r['goal']==g)
        for key in ['scaled_full_over_parent','scaled_projection_over_parent','scaled_removed_over_parent','off_axis_fraction']} for g in ALL}
    return dict(new_episodes=400,qualification_episodes=20*len(qualification),reused_parent_episodes=80,alpha=p['alpha'],
        selected_step=p['selected_step'],qualified=selection['qualified'],qualification=qualification,comparisons=comparisons,
        context=dict(native=stage_counts(a,groups['native']),parent=stage_counts(a,parent)),
        bitwise_duration_prefix_pairs=prefix_pairs,identical_initial_diagnostics_pairs=initial_pairs,
        source_sha256=digest(Path(__file__)),protocol_sha256=digest(base/'protocol.json'),
        whole_gates=80,projection_gates=80,weights_fingerprints=fingerprints,fixed_first_input=initial)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('base',type=Path);a.add_argument('--check',action='store_true');v=a.parse_args()
    result=collect(v.base.resolve());dest=v.base/'summary.json'
    if v.check:assert read(dest)==result;print('WHOLE4_NEW_CHECKED',result['new_episodes'])
    else:dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n');print('WHOLE4_NEW_WRITTEN',result['new_episodes'])
