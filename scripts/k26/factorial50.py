"""Launch paired 50-start mean/gain/projection evaluations; never train models.

Server deployment expects this file beside protocol/case/reuse artifacts.
Archived complete400 traces may be reused only after identity and SHA checks.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RUN = Path(__file__).resolve().parent
SERVER = Path('/home/lzx/cl_binding/.server')
THREE = SERVER/'k25_parallel_20261123_01'
FOUR = SERVER/'k25_parallel4_20261124_01'
NEW = SERVER/'k25_whole4_new_20261124_01'
sys.path.insert(0, str(FOUR))
import parallel_four as ref
sys.path.insert(0, str(NEW))
import whole_residual
import numpy as np
import torch

GOALS = ('left', 'front', 'right', 'back')
ARMS = ('native', 'mu150', 'mu400', 'scale150', 'scale400',
        'whole150', 'whole400', 'parallel150', 'parallel400')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def rows(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def identity_files(protocol):
    files = {}
    for p in protocol['profiles'].values():
        files.update(p['files'])
    for path, expected in files.items():
        assert sha(path) == expected, path
    return files


def metrics(trace, case, geometry, goal):
    scored = ref.old.spec.scoring.score(trace, geometry, case['z0'])
    bit = 1 << GOALS.index(goal)
    final = bool(trace['success'][-1] & bit)
    last50 = sum(bool(v & bit) for v in trace['success'][-50:])
    return dict(**scored, geometric_correct=scored['first'] == [goal],
                final400_success=final, hold45of50=final and last50 >= 45,
                final50_count=last50, final50_fraction=last50/50,
                final10_both_contact_count=sum(v == 3 for v in trace['contacts'][-10:]))


def prepare():
    assert not (RUN/'protocol.json').exists()
    tp, fp, np_ = read(THREE/'protocol.json'), read(FOUR/'four_protocol.json'), read(NEW/'protocol.json')
    profiles = {}
    for name, m in tp['models'].items():
        parent = Path(m['parent'])/'model.safetensors'
        executor = Path(m['executor']) if name.endswith('1000') else Path(m['executor'])/'model.safetensors'
        old = ('left','right') if name.startswith('lr_') else ('left','front')
        profiles[name] = dict(kind='three', old=list(old), goals=[*old, 'front' if name.startswith('lr_') else 'right'],
                              models=m, alpha=tp['gains'][name]['alpha'],
                              gain_source=tp['gains'][name],
                              files={str(parent):m['parent_sha256'], str(executor):m['executor_sha256']})
    for name, source in [('back3000', fp), ('F4-B24', np_)]:
        m = source['models']
        files = {str(Path(m['base'])/'model.safetensors'):tp['models']['lr_front_1000']['parent_sha256'],
                 m['parent']:fp['files'][m['parent']]}
        files[m['executor']] = fp['files'][m['executor']] if name == 'back3000' else read(NEW/'preparation.json')['candidates']['24000']['sha256']
        profiles[name] = dict(kind='four', old=list(ref.OLD), goals=list(ref.ALL), models=m,
                              alpha=source['alpha'], files=files,
                              gain_source=dict(path=str((FOUR/'four_calibration.json') if name=='back3000' else NEW/'calibration.json'),
                                               sha256=source['calibration_sha256']))
    protocol = dict(name='K25-factorial50', created_utc=datetime.now(timezone.utc).isoformat(), profiles=profiles,
                    ids=list(range(800,850)), arms=list(ARMS), horizon=400, stop_on_success=False,
                    instructions='pick up the book and place it in the {goal} compartment of the caddy',
                    primary=['geometric_correct','final400_success'], auxiliary=['hold45of50','final50_fraction','ever'],
                    terminal_definition='native requested predicate at400; additionally report native predicate>=45/last50 AND at400; no release gate',
                    intervention_windows=[150,400], native_shared_between_windows=True,
                    geometry=tp['geometry'], source_sha256=sha(__file__),
                    historical_protocols={str(p):sha(p) for p in [THREE/'protocol.json',FOUR/'four_protocol.json',NEW/'protocol.json']},
                    missing_combinations=['three-task sets LRB/LFB/RFB and other unverified training orders',
                                          'other four-task orders outside the six explicitly selected checkpoints'],
                    legacy_inventory_note='A separate old LF->Right->Back12000 branch exists; it is not one of the six approved checkpoint identities and is not silently substituted.',
                    new_training_steps=0, reuse_policy='only exact model, normalization, gain, windows, initial state, seed and complete400 trace; no early-stop traces or K15 snapshots')
    protocol['expected_episodes'] = sum(len(p['goals'])*50*len(ARMS) for p in profiles.values())
    assert protocol['expected_episodes'] == 9000
    identity_files(protocol)
    for name, p in profiles.items():
        if p['kind']=='four':
            base = Path(p['models']['base'])
            for n in ['policy_preprocessor_step_5_normalizer_processor.safetensors', 'policy_postprocessor_step_0_unnormalizer_processor.safetensors']:
                p['files'][str(base/n)] = sha(base/n)
        else:
            for n in ['policy_preprocessor_step_5_normalizer_processor.safetensors', 'policy_postprocessor_step_0_unnormalizer_processor.safetensors']:
                base=Path(p['models']['parent'])
                p['files'][str(base/n)] = sha(base/n)
    cases = read(THREE/'cases.json')
    assert len(cases)==20 and [c['init'] for c in cases]==list(range(800,820))
    ctx = ref.old.original.k21.km.Ctx('scene2')
    known=set()
    try:
        assert ctx.geometry==protocol['geometry']
        for c in cases:
            obs=ref.old.base.sr.reset(ctx,c)
            assert ctx.signature(obs)==c['signature']
            known.add(c['signature']['sim'])
        for i in range(20,50):
            seed=202611260+i
            np.random.seed(seed);ctx.env.seed(seed);ctx.env.reset()
            c=dict(init=800+i, state=np.asarray(ctx.env.get_sim_state(),dtype=np.float64).tolist(), reset_seed=seed, seed=15960041+107*i)
            obs=ref.old.base.sr.reset(ctx,c)
            assert not any(ctx.flags().values())
            c.update(signature=ctx.signature(obs),z0=float(obs[ctx.obj][2]),book_start=obs[ctx.obj].tolist())
            assert c['signature']['sim'] not in known
            known.add(c['signature']['sim']);cases.append(c)
    finally:
        ctx.env.close()
    assert len(known)==50
    write(RUN/'cases.json',cases)
    protocol['cases_sha256']=sha(RUN/'cases.json')
    write(RUN/'protocol.json',protocol)
    print('PREPARED',protocol['expected_episodes'],flush=True)


def reuse():
    p=read(RUN/'protocol.json');cases={c['init']:c for c in read(RUN/'cases.json')}
    sources=[]
    for profile in ref.old.spec.PROFILES:
        for path in sorted(THREE.glob(profile+'_*.jsonl.gz')):
            sources.append((profile,path,'parallel150',THREE/'protocol.json'))
        for path in sorted(FOUR.glob('full_'+profile+'_*.jsonl.gz')):
            sources.append((profile,path,'parallel400',FOUR/'full_protocol.json'))
    timing=SERVER/'k25_v_timing_20261122_01'
    for profile in ['lr_front_1000','lf_right_1000']:
        for path in sorted(timing.glob(profile+'_*.jsonl.gz')):
            sources.append((profile,path,'native',timing/'protocol.json'))
    for path in sorted(FOUR.glob('four_*.jsonl.gz')):
        sources.append(('back3000',path,None,FOUR/'four_protocol.json'))
    for path in sorted((SERVER/'k25_whole4_20261124_01').glob('whole_*.jsonl.gz')):
        sources.append(('back3000',path,None,SERVER/'k25_whole4_20261124_01/protocol.json'))
    for path in sorted(NEW.glob('new_*.jsonl.gz')):
        sources.append(('F4-B24',path,None,NEW/'protocol.json'))
    selected={};bindings={}
    for profile,path,fixed_arm,original_protocol in sources:
        marker=read(path.with_name(path.name.replace('.jsonl.gz','_COMPLETE.json')))
        assert marker['rows_sha256']==sha(path)
        assert marker['protocol_sha256']==sha(original_protocol)
        bindings[str(path)]=dict(sha256=sha(path),marker_sha256=sha(path.with_name(path.name.replace('.jsonl.gz','_COMPLETE.json'))),protocol_sha256=sha(original_protocol))
        for line_no,r in enumerate(rows(path),1):
            arm=fixed_arm or r.get('arm')
            if fixed_arm=='native' and r.get('window')!='native':continue
            if arm not in ARMS:continue
            goal=r.get('target') or r.get('goal')
            if goal not in p['profiles'][profile]['goals']:continue
            c=cases[r['init']]
            assert r['seed']==c['seed'] and len(r['trace']['actions'])==400
            assert r['text']==p['instructions'].format(goal=goal)
            if arm!='native':assert r['alpha']==p['profiles'][profile]['alpha']
            if profile=='F4-B24':assert r['checkpoint_step']==24000
            sig=ref.old.timing.signature
            # Stored signature is a SHA of the common reset signature, not a new reset.
            expected=hashlib.sha256(json.dumps(c['signature'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            assert r['signatures'][0]==expected
            trace=r['trace']
            result=metrics(trace,c,p['geometry'],goal)
            key=profile,r['init'],arm,goal
            assert key not in selected,key
            selected[key]=dict(profile=profile,init=r['init'],goal=goal,arm=arm,seed=r['seed'],alpha=p['profiles'][profile]['alpha'],
                provenance=dict(mode='reused_complete400',path=str(path),sha256=bindings[str(path)]['sha256'],line=line_no),
                trace=trace,signatures=r['signatures'],action_hashes=r['action_hashes'],seconds=r['seconds'],**result)
    for profile in p['profiles']:
        for i in range(800,820):
            selected_rows=[r for (pr,init,_,_),r in selected.items() if pr==profile and init==i]
            if selected_rows:
                dest=RUN/f'reuse_{profile}_{i}.jsonl.gz'
                assert not dest.exists()
                with gzip.open(dest,'xt') as f:
                    for r in selected_rows:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
    assert len(selected)==1200,len(selected)
    write(RUN/'reuse_manifest.json',dict(episodes=len(selected),by_profile={pr:sum(k[0]==pr for k in selected) for pr in p['profiles']},
                                        sources=bindings,files={f.name:sha(f) for f in RUN.glob('reuse_*.jsonl.gz')}))
    print('REUSE_VALIDATED',len(selected),'NEW_REQUIRED',p['expected_episodes']-len(selected),flush=True)


def load(profile,p):
    cp=p['profiles'][profile]
    if cp['kind']=='three':
        lab=ref.old.load(profile,dict(models={profile:cp['models']}))
    else:
        lab=ref.load(dict(models=cp['models'],geometry=p['geometry']))
    lab.alpha=cp['alpha'];lab.spec=cp
    lab.real_text={g:lab.ctx.all_texts[g] for g in cp['goals']}
    return lab


def fingerprint(lab):
    names=set(ref.old.base.names())
    return [ref.old.tensors_hash({n:v.detach().cpu() for n,v in rt.models['F'].named_parameters() if n in names}) for rt in (lab.rt.executer,lab.rt.parent)]


def prefixes(lab,obs,text,arm):
    old=lab.spec['old']
    keys=('A','B') if lab.spec['kind']=='three' else old
    texts=[lab.ctx.all_texts[g] for g in old]
    roles='F' if arm.startswith('scale') else 'RF'
    pre={(r,k):lab.rt.prefix(r,obs,t) for r in roles for k,t in zip(keys,texts)}
    pre['F','Q']=lab.rt.prefix('F',obs,text)
    return pre


def field(lab,pre,text,arm,x,t,diag):
    method=''.join(c for c in arm if not c.isdigit())
    if lab.spec['kind']=='three':
        if method=='parallel':
            return ref.old.projection(lab,pre,text,x,t,diag)
        f=[lab.rt.query('F',pre['F',k],x,t).float() for k in 'AB']
        value=lab.rt.query('F',pre['F','Q'],x,t)
        mf=(f[0]+f[1])/2
        residual=value.float()-mf
        if method=='scale':out=mf+lab.alpha*residual
        else:
            parent=[lab.rt.query('R',pre['R',k],x,t).float() for k in 'AB']
            mp=(parent[0]+parent[1])/2
            out=mp+(residual if method=='mu' else lab.alpha*residual)
        return out.to(value.dtype)
    late=np.stack([lab.rt.query('F',pre['F',g],x,t)[0].float().cpu().numpy() for g in lab.spec['old']])
    value=lab.rt.query('F',pre['F','Q'],x,t)
    actual=value[0].float().cpu().numpy()
    if method=='scale':
        mf=late.astype(np.float64).mean(0)
        out=mf+lab.alpha*(actual.astype(np.float64)-mf)
    else:
        parent=np.stack([lab.rt.query('R',pre['R',g],x,t)[0].float().cpu().numpy() for g in lab.spec['old']])
        if method=='parallel':out,_=ref.math.project(parent,late,actual,lab.alpha)
        elif method=='whole':out,_=whole_residual.whole(parent,late,actual,lab.alpha)
        else:out=parent.astype(np.float64).mean(0)+(actual.astype(np.float64)-late.astype(np.float64).mean(0))
    return torch.as_tensor(out,device=value.device,dtype=value.dtype).unsqueeze(0)


def episode(lab,case,goal,arm,max_blocks=40):
    started=time.monotonic();text=lab.real_text[goal]
    obs=ref.old.base.sr.reset(lab.ctx,case)
    assert lab.ctx.signature(obs)==case['signature']
    trace=dict(start=dict(object=obs[lab.ctx.obj].tolist(),eef=obs['robot0_eef_pos'].tolist()),
               object=[],eef=[],contacts=[],success=[],regions=[],actions=[])
    signatures=[ref.old.timing.signature(lab,obs)];hashes=[]
    for block in range(max_blocks):
        active=arm!='native' and block<(15 if arm.endswith('150') else 40)
        if active:
            pre=prefixes(lab,obs,text,arm)
            chunk=lab.rt.integrate(lambda x,t:field(lab,pre,text,arm,x,t,defaultdict(list)),lab.rt.noise(case['seed'],block),10)[...,:7]
        else:chunk=lab.native('F',text,obs,case['seed'],block)
        physical=lab.rt.physical(chunk)
        actual=np.clip(physical[:10],-1,1).astype(np.float32)
        hashes.append(hashlib.sha256(actual.tobytes()).hexdigest())
        obs,events=lab.execute(obs,physical,block*10,case['z0'],None)
        for row,action in zip(events,actual,strict=True):
            trace['object'].append(row['obj']);trace['eef'].append(row['eef'])
            trace['contacts'].append(int(row['c1'])+2*int(row['c2']))
            trace['success'].append(sum(1<<j for j,g in enumerate(GOALS) if row['pred'][g]))
            trace['regions'].append(sum(1<<j for j,g in enumerate(GOALS) if g in row['hits']))
            trace['actions'].append(action.tolist())
        signatures.append(ref.old.timing.signature(lab,obs))
    r=dict(profile=lab.profile,init=case['init'],goal=goal,arm=arm,seed=case['seed'],alpha=lab.alpha,
           provenance=dict(mode='new_online_complete400'),trace=trace,signatures=signatures,action_hashes=hashes,seconds=time.monotonic()-started)
    if max_blocks==40:r.update(metrics(trace,case,lab.ctx.geometry,goal))
    return r


def gate(lab,p,cases):
    case=cases[0];profile=lab.profile
    reusefile=RUN/f'reuse_{profile}_{case["init"]}.jsonl.gz'
    archived={(r['arm'],r['goal']):r for r in rows(reusefile)}
    goal=lab.spec['old'][0]
    checked=[]
    for arm in ['parallel150','whole150','native']:
        if (arm,goal) not in archived:continue
        new=episode(lab,case,goal,arm,1);old=archived[arm,goal]
        assert new['action_hashes'][0]==old['action_hashes'][0],(profile,arm,'action')
        assert new['signatures'][:2]==old['signatures'][:2],(profile,arm,'state')
        checked.append(arm)
    # Structural four-cell gate on real model outputs at one common denoising input.
    obs=ref.old.base.sr.reset(lab.ctx,case);text=lab.real_text[goal]
    pre=prefixes(lab,obs,text,'whole400');x=lab.rt.noise(case['seed'],0);t=torch.ones(1,device=x.device)
    vals={a:field(lab,pre,text,a,x,t,defaultdict(list)) for a in ['mu400','scale400','whole400','parallel400']}
    assert all(torch.isfinite(v).all() and tuple(v.shape)==(1,50,32) for v in vals.values())
    previous=lab.alpha;lab.alpha=1.
    scale1=field(lab,pre,text,'scale400',x,t,defaultdict(list))
    own=lab.rt.query('F',pre['F','Q'],x,t)
    assert torch.equal(scale1,own)
    assert torch.equal(field(lab,pre,text,'whole400',x,t,defaultdict(list)),field(lab,pre,text,'mu400',x,t,defaultdict(list)))
    lab.alpha=previous
    return dict(archive_prefix_bitwise=checked,alpha1_native_identity=True,alpha1_whole_equals_mu=True,finite_full32=True)


def worker(profile,lane):
    torch.set_num_threads(2)
    p=read(RUN/'protocol.json');assert sha(__file__)==p['source_sha256']
    allcases=read(RUN/'cases.json');lab=load(profile,p);lab.profile=profile
    before=fingerprint(lab);checks=gate(lab,p,allcases)
    selected=allcases[lane::10]
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            # Gate above also runs under inference/autocast via the worker entry context.
            for case in selected:
                stem=f'case_{profile}_{case["init"]}';dest=RUN/(stem+'.jsonl.gz');marker=RUN/(stem+'_COMPLETE.json')
                if marker.exists():
                    assert read(marker)['sha256']==sha(dest);continue
                if dest.exists():dest.rename(RUN/(stem+f'_interrupted_{time.time_ns()}.jsonl.gz'))
                reusefile=RUN/f'reuse_{profile}_{case["init"]}.jsonl.gz'
                reuse_rows=list(rows(reusefile)) if reusefile.exists() else []
                reused={(r['arm'],r['goal']):r for r in reuse_rows};written=0;fresh=0
                with gzip.open(dest,'xt',compresslevel=5) as f:
                    for arm in ARMS:
                        for goal in lab.spec['goals']:
                            r=reused.get((arm,goal))
                            if r is None:r=episode(lab,case,goal,arm);fresh+=1
                            f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n');f.flush();written+=1
                            print('RESULT',profile,case['init'],arm,goal,int(r['geometric_correct']),int(r['final400_success']),r['provenance']['mode'],round(r['seconds'],2),flush=True)
                assert written==len(ARMS)*len(lab.spec['goals']) and fingerprint(lab)==before
                write(marker,dict(profile=profile,init=case['init'],episodes=written,new_episodes=fresh,sha256=sha(dest),weights_unchanged=True,
                                  protocol_sha256=sha(RUN/'protocol.json'),gates=checks))
    finally:lab.ctx.env.close()
    write(RUN/f'worker_{profile}_{lane}_COMPLETE.json',dict(profile=profile,lane=lane,weights_unchanged=fingerprint(lab)==before))


def summarize():
    p=read(RUN/'protocol.json');allrows=[]
    for profile in p['profiles']:
        for init in p['ids']:
            stem=f'case_{profile}_{init}';m=read(RUN/(stem+'_COMPLETE.json'));path=RUN/(stem+'.jsonl.gz')
            assert sha(path)==m['sha256'] and m['weights_unchanged']
            rs=list(rows(path));assert len(rs)==m['episodes'];allrows.extend(rs)
    keys={(r['profile'],r['init'],r['arm'],r['goal']):r for r in allrows}
    assert len(keys)==len(allrows)==p['expected_episodes']
    summary={};metrics_=('geometric_correct','final400_success','hold45of50')
    report=['# K25-factorial50-result：五组响应对照，每句50次','',
            '全部完成。每种修复同时评1–150和1–400窗口，原生基线共享。主列G/T/H分别为几何首达、第400步原生放置、且最后50步至少45步满足的保持计数，各/50；不附加松手门槛。',
            '每个模型所有已训练指令均测，包括新指令；同一50个初态跨方法配对。旧完整记录仅在身份及控制条件一致时复用，早停轨迹不复用。','']
    for profile,cp in p['profiles'].items():
        report += [f'## {profile}，α={cp["alpha"]}', '', '| 条件 | '+' | '.join(cp['goals'])+' | 旧句成组G/T/H | 全部已训句成组G/T/H |',
                   '| --- | '+' | '.join(['---']*(len(cp['goals'])+2))+' |']
        summary[profile]={}
        for arm in ARMS:
            per={}
            for g in cp['goals']:
                rs=[keys[profile,i,arm,g] for i in p['ids']]
                per[g]={k:sum(r[k] for r in rs) for k in metrics_}
                per[g]['n']=len(rs);assert len(rs)==50
            joint={}
            for label,gs in [('old',cp['old']),('all',cp['goals'])]:
                joint[label]={k:sum(all(keys[profile,i,arm,g][k] for g in gs) for i in p['ids']) for k in metrics_}
            summary[profile][arm]=dict(per_goal=per,joint=joint)
            fmt=lambda counts:'/'.join(str(counts[k]) for k in metrics_)
            report.append('| '+arm+' | '+' | '.join(fmt(per[g]) for g in cp['goals'])+' | '+fmt(joint['old'])+' | '+fmt(joint['all'])+' |')
        report.append('')
    counts=dict(total=len(allrows),new=sum(r['provenance']['mode']=='new_online_complete400' for r in allrows),reused=sum(r['provenance']['mode']=='reused_complete400' for r in allrows))
    write(RUN/'summary.json',dict(counts=counts,results=summary,protocol_sha256=sha(RUN/'protocol.json')))
    report += ['## 配置和证据','',f'实际记录：{counts}。模型身份、固定α、缺失组合及源码SHA见protocol.json；每初态完整轨迹和完成标记在同目录。',
               '1–150从151步交还原生F；全程组持续修复400步。成功不提前终止，才能计算终态。μ-only=μP+rF；scale=μF+αrF；whole=μP+αrF；parallel沿用旧μ与父旧方向投影。',
               '没有新训练，没有用本轮成功率选α，没有新增未授权训练组合；见protocol.json的missing_combinations。']
    (RUN/'K25-factorial50-result.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    write(RUN/'COMPLETE.json',dict(**counts,completed_utc=datetime.now(timezone.utc).isoformat(),summary_sha256=sha(RUN/'summary.json')))


def controller():
    p=read(RUN/'protocol.json');identity_files(p)
    pending=[(profile,lane) for lane in range(10) for profile in p['profiles'] if not (RUN/f'worker_{profile}_{lane}_COMPLETE.json').exists()]
    live=[];began=time.time()
    write(RUN/'STARTED.json',dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),expected=p['expected_episodes']))
    try:
        while pending or live:
            for gpu in (0,1):
                while pending and sum(item[3]==gpu for item in live)<4:
                    profile,lane=pending.pop(0);log=(RUN/f'worker_{profile}_{lane}.log').open('a')
                    proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'worker','--profile',profile,'--lane',str(lane)],
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)})
                    live.append((proc,log,(profile,lane),gpu));print('START',profile,lane,gpu,proc.pid,flush=True)
            for item in list(live):
                proc,log,key,gpu=item
                if proc.poll() is None:continue
                live.remove(item);log.close()
                assert proc.returncode==0 and (RUN/f'worker_{key[0]}_{key[1]}_COMPLETE.json').exists(),(key,proc.returncode)
                print('FINISH',key,flush=True)
            markers=[read(f) for f in RUN.glob('case_*_COMPLETE.json')]
            write(RUN/'status.json',dict(running=True,pid=os.getpid(),elapsed_seconds=time.time()-began,
                pending_workers=len(pending),live_workers=len(live),completed_cases=len(markers),
                validated_episodes=sum(m['episodes'] for m in markers),new_episodes=sum(m['new_episodes'] for m in markers),expected=p['expected_episodes']))
            time.sleep(3)
        summarize()
        write(RUN/'status.json',dict(running=False,complete=True,**read(RUN/'COMPLETE.json')))
    finally:
        for proc,log,_,_ in live:
            if proc.poll() is None:proc.terminate();proc.wait()
            log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','reuse','worker','controller','summarize']);parser.add_argument('--profile');parser.add_argument('--lane',type=int)
    args=parser.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='reuse':reuse()
        elif args.mode=='worker':
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):worker(args.profile,args.lane)
        elif args.mode=='controller':controller()
        else:summarize()
    except BaseException as error:
        import traceback
        write(RUN/f'FAILED_{args.mode}_{args.profile}_{args.lane}.json',dict(error=str(error),traceback=traceback.format_exc()))
        raise
