"""Audit K17 and rank component effects separately by H background and endpoint.

The statistical unit is20 physical snapshots, not640 independent rollouts.
All comparisons are paired across snapshots; unadjusted exploratory intervals.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
from statistics import median

INITS=tuple(range(20,40))
GROUPS=('N','Z','R','ZR','G','ZG','RG','ZRG','H','HZ','HR','HZR','HG','HZG','HRG','HZRG')
FACTORS=('H','Z','R','G')
BLOCKS={'H':(0,1),'Z':(2,),'R':(3,4,5),'G':(6,)}
REF={'N':'N','H':'H','ZRG':'S','HZRG':'HS'}
FRESH=tuple(c for c in GROUPS if c not in REF)
LABEL={'A':'left','B':'right'}
METRICS=('L_first','R_first','L_ever','R_ever','L_term','R_term','total_first','dir','place','term','front_first','none_first')
SINGLE={'L_first','R_first','L_ever','R_ever','L_term','R_term','dir','both_term'}
SHA={'R':'225322ba28d0ad152fcced19950c43e3e40a435f1835cf64a664226da7830003','F':'8ff1bf8f372d728a3218101a3cc93fb6fb9d484e8fefb39c48d81a0c8ab2c515'}

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def selected(c):return [i for b in FACTORS if b in c for i in BLOCKS[b]]
def name(factors):return ''.join(f for f in FACTORS if f in factors) or 'N'
def tag(x):return '/'.join(x) or 'none'

def metrics(left,right):
    out={'dir':[int(a['first']==['left'] and b['first']==['right']) for a,b in zip(left,right)],
         'both_term':[int(a['terminal']['left'] and b['terminal']['right']) for a,b in zip(left,right)]}
    for side,rows,target in (('L',left,'left'),('R',right,'right')):
        out[side+'_first']=[int(r['first']==[target]) for r in rows]
        out[side+'_ever']=[int(r['ever'][target]) for r in rows]
        out[side+'_term']=[int(r['terminal'][target]) for r in rows]
    for k,a,b in (('total_first','L_first','R_first'),('place','L_ever','R_ever'),('term','L_term','R_term')):
        out[k]=[x+y for x,y in zip(out[a],out[b])]
    for k,target,field in (('front_first','front','first'),('none_first',None,'first'),('front_ever','front','ever'),('front_term','front','terminal')):
        def hit(r):return int(not r[field]) if target is None else int(target in r[field]) if field=='first' else int(r[field][target])
        out[k]=[hit(a)+hit(b) for a,b in zip(left,right)]
    return out

class Statistics:
    def __init__(self,per):
        self.per=per;self.cache={};rng=random.Random(20261110)
        self.boot=[[rng.randrange(20) for _ in range(20)] for _ in range(10000)]
    def contrast(self,coef,metric):
        coef={g:w for g,w in coef.items() if w};key=(tuple(sorted(coef.items())),metric)
        if key in self.cache:return self.cache[key]
        assert sum(coef.values())==0 and set(coef)<=set(GROUPS)
        delta=[sum(w*self.per[g][metric][k] for g,w in coef.items()) for k in range(20)]
        denom=20 if metric in SINGLE else 40
        samples=sorted(sum(delta[k] for k in ix)/denom for ix in self.boot)
        result={'coefficients':coef,'metric':metric,'point':sum(delta)/denom,'ci95':[samples[250],samples[9749]],'n_clusters':20,'denominator':denom,'snapshot_delta':dict(zip(map(str,INITS),delta)),'unadjusted_development_interval':True}
        self.cache[key]=result;return result
    def compare(self,after,before):return {m:self.contrast({after:1,before:-1},m) for m in METRICS}

def transitions(records,after,before):
    result={}
    for goal,target in LABEL.items():
        result[goal]={}
        for metric,field in (('first','first'),('ever','ever'),('terminal','terminal')):
            gained=[];lost=[]
            for init in INITS:
                a=records[init,before,goal];b=records[init,after,goal]
                old=a[field]==[target] if field=='first' else a[field][target]
                new=b[field]==[target] if field=='first' else b[field][target]
                if new and not old:gained.append(init)
                if old and not new:lost.append(init)
            result[goal][metric]={'gained':gained,'lost':lost,'net':len(gained)-len(lost)}
        counts=Counter((tag(records[i,before,goal]['first']),tag(records[i,after,goal]['first'])) for i in INITS)
        result[goal]['first_transitions']=[{'before':a,'after':b,'n':n} for (a,b),n in sorted(counts.items())]
    return result

def audit(archive):
    protocol=read(archive/'protocol.json')
    assert protocol['inits']==list(INITS) and protocol['alpha']==2.959
    assert protocol['blocks']=={f:list(BLOCKS[f]) for f in FACTORS}
    assert protocol['conditions']=={c:[f for f in FACTORS if f in c] for c in GROUPS}
    assert protocol['episodes']['new_component_episodes']==480 and protocol['episodes']['reused_k16_episodes']==160
    assert protocol['reference_mapping']==REF
    for source,key in (('reference_k16.py','k16_script_sha256'),('reference_k15.py','k15_script_sha256')):assert digest(archive/source)==protocol[key]
    for source,key in (('source_code/k14.py','code/k14.py'),('source_code/k14s5.py','code/k14s5.py'),('snapshots.json','snapshots/snapshots.json')):
        assert digest(archive/source)==protocol['source_sha256'][key]
    for source,expected in read(archive/'reference_source_hashes.json').items():assert digest(archive/source)==expected,source
    gates=read(archive/'gates.json');assert gates['pass'] and gates['padding_always_new']
    assert gates['executed_partition']==protocol['blocks']
    assert [r['init'] for r in gates['checks']]==[20,21,22]
    n_gate=0
    for check in gates['checks']:
        assert check['same_signature'] and check['alpha0_identity'] and check['alpha1_N_native']
        assert {(r['goal'],r['condition']) for r in check['items']}==set(itertools.product('AB',GROUPS))
        for item in check['items']:
            assert item['selected_indices']==selected(item['condition'])
            assert item['mask_bitwise'] and item['dispatch_bitwise'] and item['finite32']
            assert math.isfinite(item['roundoff_rms_real']) and item['roundoff_rms_real']>=0
            n_gate+=1
    assert n_gate==96
    records={};traces={};paths=[]
    def load(path,target,source):
        paths.append(path)
        for row in map(json.loads,path.read_text().splitlines()):
            key=(row['init'],row['condition'],row['goal']);assert key not in target
            if target is records:assert row['evaluation_source']==source
            target[key]=row['rows'] if target is traces else row
    load(archive/'reference_episodes.jsonl',records,'reused_k16')
    load(archive/'reference_traces.jsonl',traces,None)
    assert len(records)==len(traces)==160
    runner_sha=digest(archive/'k17.py');protocol_sha=digest(archive/'protocol.json')
    for init in INITS:
        marker=read(archive/f'i{init}_COMPLETE.json')
        assert marker['init']==init and marker['n_episodes']==24
        assert marker['n_audit_reference_episodes']==(8 if init==20 else 0)
        assert marker['model_sha256']==SHA and marker['script_sha256']==runner_sha
        assert marker['protocol_sha256']==protocol_sha and marker['source_hashes']==protocol['source_sha256']
        assert marker['k16_script_sha256']==protocol['k16_script_sha256'] and marker['k15_script_sha256']==protocol['k15_script_sha256']
        for check in marker['processor_check'].values():assert check['identical_across']==['F','R'] and check['compared_keys']==40 and not check['bookkeeping_keys_differing_ignored']
        load(archive/f'i{init}.jsonl',records,'new_k17');load(archive/f'i{init}_traces.jsonl',traces,None)
        paths.append(archive/f'i{init}_COMPLETE.json')
    expected=set(itertools.product(INITS,GROUPS,'AB'))
    assert set(records)==set(traces)==expected and len(records)==640
    spec=importlib.util.spec_from_file_location('k17_geometry',archive/'dependencies/k1_geometry.py')
    geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
    geometry=read(archive/'geometry.json')['sites']
    for init in INITS:
        base=records[init,'N','A'];boundary=base['boundary'];assert 70<=boundary<=110 and boundary%10==0
        prefix=traces[init,'N','A'][:boundary]
        assert not any(r['hits'] or any(r['pred'].values()) for r in prefix)
        for c in GROUPS:
            for goal,target in LABEL.items():
                key=init,c,goal;r=records[key];trace=traces[key]
                assert len(trace)==400 and [v['step'] for v in trace]==list(range(1,401)) and trace[:boundary]==prefix,key
                assert r['requested']==target and r['alpha']==2.959 and r['window']==60 and r['n_window_queries']==60
                assert r['mu_old_indices']==selected(c) and r['mu_old_blocks']==[b for b in FACTORS if b in c]
                assert r['prefix_model']=='Front-12000' and r['prefix_goal']=='Front'
                for f in ('boundary','seed','z0','boundary_signature'):assert r[f]==base[f],(key,f)
                move=geo.movement('scene2',geometry,trace,r['z0'],None)
                assert r['first']==move['first_targets'] and r['first_step']==move['first_step'] and r['first_times']==move['times'],key
                assert r['ever']=={g:any(v['pred'][g] for v in trace) for g in geo.GOALS} and r['terminal']==trace[-1]['pred'],key
                assert abs(r['clip_rate_window']-sum(v['clip'] for v in trace[boundary:boundary+60])/60)<1e-12
    audit_rows=list(map(json.loads,(archive/'pilot_references.jsonl').read_text().splitlines()))
    audit_traces=list(map(json.loads,(archive/'pilot_reference_traces.jsonl').read_text().splitlines()))
    assert len(audit_rows)==len(audit_traces)==8
    for r,t in zip(audit_rows,audit_traces):
        key=r['init'],r['condition'],r['goal'];assert r['init']==20 and key==(t['init'],t['condition'],t['goal'])
        old=records[key]
        for f in ('first','first_step','first_times','ever','terminal','seed','boundary','z0','boundary_signature'):assert r[f]==old[f]
        assert t['rows']==traces[key]
    old_summary=read(archive/'reference_k16_summary.json');assert old_summary['model_sha256']==SHA
    per={c:metrics([records[i,c,'A'] for i in INITS],[records[i,c,'B'] for i in INITS]) for c in GROUPS}
    for c,source in REF.items():assert per[c]==old_summary['per_snapshot'][source],c
    paths.extend(archive/p for p in ('protocol.json','k17.py','gates.json','pilot_audit.json','pilot_references.jsonl','pilot_reference_traces.jsonl','reference_k16_summary.json','reference_source_hashes.json','geometry.json','live_action_metadata.json','reference_k16.py','reference_k15.py','snapshots.json'))
    return records,traces,per,geometry,geo,paths,{'unique_regime_cases':640,'new_component_rollouts':480,'reused_k16_rollouts':160,'audit_reference_repeats':8,'new_physical_rollouts':488,'unique_trace_steps_checked':256000,'audit_trace_steps_exact':3200,'same_input_mask_gates':96,'all32_branches_common_prefix_per_init':True,'reused_per_snapshot_metrics_exact_k16':True,'source_model_and_processor_hashes_passed':True}

def diagnostics(records,traces,geometry,geo,edges):
    site=geometry['right'];half=geo.extents(site,(0,1,2));top=site['centre'][2]+half[2]
    details={}
    for c in GROUPS:
        sides={}
        for goal,target in LABEL.items():
            vals=[]
            for init in INITS:
                r=records[init,c,goal];trace=traces[init,c,goal];boundary=r['boundary'];window=trace[boundary:boundary+60];last=window[-1]
                right_xy=[v for v in trace[boundary:] if geo.inside(v['obj'],site)]
                q0=trace[boundary-1]['relquat'];q1=last['relquat'];den=math.sqrt(sum(x*x for x in q0)*sum(x*x for x in q1))
                angle=2*math.degrees(math.acos(min(1,abs(sum(x*y for x,y in zip(q0,q1)))/max(den,1e-12))))
                vals.append({'init':init,'first':r['first'],'window_lift_m':last['obj'][2]-r['z0'],'window_both_contact':bool(last['c1'] and last['c2']),'relative_pose_change_deg':angle,'window_gripper_positive_fraction':sum(v['a'][6]>0 for v in window)/60,'arm_clip_fraction':sum(any(abs(x)>1 for x in v['a'][:6]) for v in window)/60,'right_xy_min_center_above_top_m':min((v['obj'][2]-top for v in right_xy),default=None),'right_xy_all_center_above_top':bool(right_xy) and all(v['obj'][2]>top for v in right_xy),'ever_correct':r['ever'][target],'terminal_correct':r['terminal'][target]})
            sides[goal]={'window_both_contacts':sum(v['window_both_contact'] for v in vals),'window_lift_m_median':median(v['window_lift_m'] for v in vals),'relative_pose_change_deg_median':median(v['relative_pose_change_deg'] for v in vals),'window_all_positive_gripper_cases':sum(v['window_gripper_positive_fraction']==1 for v in vals),'arm_clip_fraction_mean':sum(v['arm_clip_fraction'] for v in vals)/20,'right_first_but_never_success':[v for v in vals if goal=='B' and v['first']==['right'] and not v['ever_correct']],'per_episode':vals}
        details[c]=sides
    cross={}
    for label,edge in edges.items():
        after,before,factor=edge['after'],edge['before'],edge['factor'];outside=[j for j in range(7) if j not in BLOCKS[factor]]
        parts=[]
        for init in INITS:
            for goal in 'AB':
                start=records[init,before,goal]['boundary']
                a=traces[init,before,goal][start:start+10];b=traces[init,after,goal][start:start+10]
                delta=[[y['a'][j]-x['a'][j] for j in range(7)] for x,y in zip(a,b)]
                sgn=lambda value:1 if value>0 else -1 if value<0 else 0
                parts.append({'init':init,'goal':goal,'outside_factor_coordinates_changed':any(d[j]!=0 for d in delta for j in outside),'command_rms_difference_by_coordinate':[math.sqrt(sum(d[j]**2 for d in delta)/10) for j in range(7)],'gripper_sign_changes':sum(sgn(x['a'][6])!=sgn(y['a'][6]) for x,y in zip(a,b))})
        cross[label]={'factor':factor,'after':after,'before':before,'outside_coordinates':outside,'outside_coordinates_changed_cases':sum(v['outside_factor_coordinates_changed'] for v in parts),'per_episode':parts,'interpretation':'same initial physical input, generated first action chunk before new observations; cross-coordinate differences may arise inside denoising; difference alone does not prove behavioral mediation'}
    return {'right_region_upper_z_m':top,'post_treatment_states':details,'same_initial_input_first_chunk_coupling':cross,'state_correlations_are_not_causal_mediation':True}

def main(archive):
    records,traces,per,geometry,geo,paths,audits=audit(archive)
    stats=Statistics(per);summary={};edges={};primary={};interactions={}
    for c in GROUPS:
        sides={}
        for g,target in LABEL.items():
            rows=[records[i,c,g] for i in INITS]
            sides[g]={'n':20,'requested':target,'first_distribution':dict(Counter(tag(r['first']) for r in rows)),'first_correct':sum(r['first']==[target] for r in rows),'ever_correct':sum(r['ever'][target] for r in rows),'terminal_correct':sum(r['terminal'][target] for r in rows)}
        summary[c]={'totals':{k:sum(v) for k,v in per[c].items()},'per_goal':sides,'old_mean_indices':selected(c)}
    def comparison(after,before):return {'after':after,'before':before,'metrics':stats.compare(after,before),'transitions':transitions(records,after,before)}
    for factor in FACTORS:
        for before in GROUPS:
            if factor in before:continue
            after=name(set(before)-{'N'}|{factor});key=f'{factor}:{before}_to_{after}'
            edges[key]={'factor':factor,**comparison(after,before)}
    assert len(edges)==32
    for h in (False,True):
        base='H' if h else 'N';full='HZRG' if h else 'ZRG';level='H_old' if h else 'H_new'
        primary[level]={}
        for factor in 'ZRG':
            only=name(({'H'} if h else set())|{factor})
            without=name(set(full)-{factor})
            primary[level][factor]={'single_addition':comparison(only,base),'removal_loss_full_minus_without':comparison(full,without)}
    for a,b in itertools.combinations(FACTORS,2):
        rest=[f for f in FACTORS if f not in (a,b)]
        for bits in itertools.product((False,True),repeat=2):
            fixed={f for f,present in zip(rest,bits) if present}
            n0=name(fixed);n1=name(fixed|{a});n2=name(fixed|{b});n12=name(fixed|{a,b})
            key=f'{a}_x_{b}_given_{name(fixed)}'
            interactions[key]={m:stats.contrast({n12:1,n1:-1,n2:-1,n0:1},m) for m in METRICS}
    report={'experiment':'K17 full H x height x rotation x gripper source factorial','audits':audits,'protocol':read(archive/'protocol.json'),'summary':summary,'per_snapshot':per,'primary_component_contrasts':primary,'factor_switch_edges':edges,'conditional_pair_interactions':interactions,'diagnostics':diagnostics(records,traces,geometry,geo,edges),'statistics':{'n_clusters':20,'bootstrap':10000,'seed':20261110,'unit':'snapshot, both instructions andall16 regimes together','unadjusted_development_screen':True,'paired_split_product_not_internal_multiplication':True,'reused160_not_new_validation':True}}
    paths.append(Path(__file__).resolve())
    report['source_sha256']={str(p.resolve().relative_to(archive)) if p.resolve().is_relative_to(archive) else str(p.resolve()):digest(p) for p in paths}
    (archive/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# K17高度、旋转、夹爪共同响应拆分','', '640个唯一条件×回合：480新增、160复用K16；另8条参照审计精确复现3200步。','所有组指令区别均来自续训模型并放大2.959倍，填充平均固定新。','', '|组|左先左/20|右先右/20|成对/20|左曾成功/20|右曾成功/20|左终态/20|右终态/20|前/40|无区域/40|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    cols=('L_first','R_first','dir','L_ever','R_ever','L_term','R_term','front_first','none_first')
    for c in GROUPS:lines.append('|'+c+'|'+'|'.join(str(summary[c]['totals'][k]) for k in cols)+'|')
    lines+=['','## 主要单项添加与移除对照','','|H背景|因素|对照|右首次差|右曾成功差|右终态差|左终态差|','|---|---|---|---|---|---|---|']
    def fmt(x):return f"{x['point']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]"
    for level,parts in primary.items():
        for factor,tests in parts.items():
            for kind,test in tests.items():lines.append('|'+level+'|'+factor+'|'+kind+':'+test['after']+'−'+test['before']+'|'+'|'.join(fmt(test['metrics'][k]) for k in ('R_first','R_ever','R_term','L_term'))+'|')
    lines+=['','区间按20物理快照聚类、10000次bootstrap；未校正多重比较，供开发机制分析，不作等效或普遍必要性结论。S拆为Z高度、R三轴旋转、G夹爪；H水平。字母代表μ取旧来源，其他μ取新。','单项来源效果可以经后续去噪跨坐标传播，不能直接解释为实际高度、姿态或夹爪力的独立中介。']
    (archive/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('AUDITED: 640 unique cases /256000 steps,480 new+160 reused;8 audit repeats/3200 exact steps;96 mask gates')
    print('\n'.join(lines[5:24]))
    return report

def selftest():
    # Test denominators, source names and a known conditional interaction without data.
    assert len({name(set(c)-{'N'}) for c in GROUPS})==16
    per={c:{m:[0]*20 for m in METRICS} for c in GROUPS}
    per['HZRG']['R_term']=[1]*20
    stat=Statistics(per)
    x=stat.contrast({'HZRG':1,'HZR':-1},'R_term');assert x['point']==1 and x['ci95']==[1,1]
    x=stat.contrast({'HZR':1,'HZ':-1,'HR':-1,'H':1},'R_term');assert x['point']==0
    assert all(selected(c)==sorted(selected(c)) and all(i<7 for i in selected(c)) for c in GROUPS)
    print('synthetic statistics and factorial design checks passed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path);p.add_argument('--selftest',action='store_true');args=p.parse_args()
    if args.selftest:selftest()
    else:
        assert args.archive is not None
        main(args.archive.resolve())
