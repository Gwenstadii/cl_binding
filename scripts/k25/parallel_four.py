"""Four-instruction history: three old goals + Back, one continuous projector."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch

RUN=Path(__file__).resolve().parent
PREVIOUS=RUN.parent/'k25_parallel_20261123_01'
sys.path.insert(0,str(PREVIOUS))
import parallel_reproduction as old
sys.path.insert(0,str(RUN))
import parallel_subspace as math

OLD=('left','front','right')
ALL=(*OLD,'back')
ARMS=('parent','native','parallel150','parallel400')
K26=RUN.parent/'k26_multigoal_20261121_01'
P=RUN.parent/'k19_retention_20261111_01/runs/20261121_replay_full/001000'
F=K26/'back/003000'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def prepare():
    assert not (RUN/'four_preparation.json').exists()
    fp=old.base.read(RUN/'full_protocol.json')
    base=Path(fp['models']['lr_front_1000']['parent'])
    mp,mf=old.base.read(P/'READY.json'),old.base.read(F/'READY.json')
    assert mp['sha256']=='f8bd197dc20886ee14873429c90a7b51ea00ccc2656c28fc542b625b1261772c'
    assert mf['sha256']=='ccee300a0cfc2ae9755eb6a149e0e1dee1d1f57e6a1677584162bab61788ad83'
    assert mf['parent_sha256']==mp['sha256'] and mf['frozen_unchanged'] and mf['step']==3000
    files={str(P/'trainable.safetensors'):mp['sha256'],str(F/'trainable.safetensors'):mf['sha256'],str(base/'model.safetensors'):fp['models']['lr_front_1000']['parent_sha256']}
    files.update({str(folder/'READY.json'):digest(folder/'READY.json') for folder in (P,F)})
    files.update({str(base/n):digest(base/n) for n in ('policy_preprocessor_step_5_normalizer_processor.safetensors','policy_postprocessor_step_0_unnormalizer_processor.safetensors')})
    assert all(digest(Path(path))==h for path,h in files.items())
    p=dict(models=dict(base=str(base),parent=str(P/'trainable.safetensors'),executor=str(F/'trainable.safetensors')),
        files=files,sources={n:digest(RUN/n) for n in ('parallel_four.py','parallel_subspace.py')},
        old_sources=fp['previous_sources'],full_protocol_sha256=digest(RUN/'full_protocol.json'),
        cases_sha256=digest(RUN/'cases.json'),plan_sha256=digest(RUN/'analysis_plan.json'),geometry=fp['geometry'],
        ids=list(range(800,820)),old_goals=OLD,all_goals=ALL,arms=ARMS,expected=320,horizon=400,
        solver=dict(contrast='Helmert, three old prompts',dtype='float64',rcond=math.RCOND,fit='50x7',lift='full32'),
        checkpoint_choice='original selected Back3000; no choice using new results',
        source_parent_commit='e61358e0d1c203aea7d8cb57f0dfcadb986a8711')
    old.base.write(RUN/'four_preparation.json',p)
    print('FOUR_PREPARED',320,flush=True)


def load(config):
    lab=old.original.k21.Lab(family='scene2',models=False)
    cp=config['models']
    frt=old.original.k21.km.make_runtime(lab.ctx,{'FT4':cp['base']},['F'])
    prt=old.original.k21.km.make_runtime(lab.ctx,{'FT4':cp['base']},['F'])
    for rt,key in ((frt,'executor'),(prt,'parent')):
        old.base.sr.assign(type('Holder',(),{'rt':rt})(),'F',old.base.selected_weights(Path(cp[key])))
    lab.rt=old.DualRuntime(frt,prt)
    lab.goal={g:g for g in ALL};lab.text={g:lab.ctx.all_texts[g] for g in ALL}
    assert all(lab.text[g]==f'pick up the book and place it in the {g} compartment of the caddy' for g in ALL)
    assert lab.ctx.geometry==config['geometry']
    return lab


def velocities(lab,pre,x,t):
    parent=np.stack([lab.rt.query('R',pre['R',g],x,t)[0].float().cpu().numpy() for g in OLD])
    late=np.stack([lab.rt.query('F',pre['F',g],x,t)[0].float().cpu().numpy() for g in OLD])
    value=lab.rt.query('F',pre['F','Q'],x,t)[0]
    return parent,late,value.float().cpu().numpy(),value


def prefixes(lab,obs,text):
    pre={(role,g):lab.rt.prefix(role,obs,lab.text[g]) for role in 'RF' for g in OLD}
    pre['F','Q']=lab.rt.prefix('F',obs,text)
    return pre


def field(lab,pre,x,t,diagnostics):
    parent,late,actual,value=velocities(lab,pre,x,t)
    out,diag=math.project(parent,late,actual,lab.alpha)
    diagnostics.append(diag)
    return torch.from_numpy(out).to(device=value.device,dtype=value.dtype)[None]


def calibrate():
    assert not (RUN/'four_calibration.json').exists()
    p=old.base.read(RUN/'four_preparation.json');lab=load(p);stats=[]
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for init in range(10):
                seed=old.original.k21.km.noise_seed('scene2',init)
                obs,_=lab.ctx.reset(init);z0=float(obs[lab.ctx.obj][2])
                for block in range(16):
                    if block>=3:
                        pre=prefixes(lab,obs,lab.text['back'])
                        x=lab.rt.noise(seed,block);t=torch.ones(1,device=x.device)
                        parent,late,_,_=velocities(lab,pre,x,t)
                        stats.append(dict(init=init,step=block*10,**math.fields(parent,late)))
                    chunk=lab.native('F',lab.text['back'],obs,seed,block)
                    obs,_=lab.execute(obs,lab.rt.physical(chunk),block*10,z0,None)
                print('FOUR_CALIBRATION',init,flush=True)
    finally:lab.ctx.env.close()
    assert len(stats)==130
    alpha=min(10.,old.spec.upper_median([r['ratio'] for r in stats]))
    old.base.write(RUN/'four_calibration.json',dict(alpha=alpha,n=130,stats=stats,behaviour_used_for_selection=False,preparation_sha256=digest(RUN/'four_preparation.json'),sources=p['sources']))
    print('FOUR_CALIBRATED',alpha,flush=True)


def lock():
    assert not (RUN/'four_protocol.json').exists()
    p=old.base.read(RUN/'four_preparation.json');c=old.base.read(RUN/'four_calibration.json')
    assert c['preparation_sha256']==digest(RUN/'four_preparation.json')
    old.base.write(RUN/'four_protocol.json',dict(**p,alpha=c['alpha'],calibration_sha256=digest(RUN/'four_calibration.json'),preparation_sha256=digest(RUN/'four_preparation.json')))
    print('FOUR_LOCKED',c['alpha'],flush=True)


def gate(lab,case):
    obs=old.base.sr.reset(lab.ctx,case);x=lab.rt.noise(case['seed'],0);t=torch.ones(1,device=x.device)
    checks=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for g in ALL:
            parent,late,q,_=velocities(lab,prefixes(lab,obs,lab.text[g]),x,t)
            out,diag=math.project(parent,late,q,lab.alpha)
            # Independent full-rank normal-equations solve, on the same recorded values.
            pc=parent.astype(np.float64)-parent.astype(np.float64).mean(0)
            b=np.einsum('ntd,nk->tdk',pc,math.helmert(3));a=b[:,:7,:].reshape(-1,2)
            r=(q-late.astype(np.float64).mean(0))[:,:7].reshape(-1)
            if diag['rank']==2 and diag['condition']<1e4:
                coeff=np.linalg.solve(a.T@a,a.T@r)
                expected=parent.astype(np.float64).mean(0)+lab.alpha*np.einsum('tdk,k->td',b,coeff)
                np.testing.assert_allclose(out,expected,rtol=1e-7,atol=1e-9)
            assert diag['orthogonality_error']<1e-6
            if g in OLD:
                assert np.array_equal(q,late[OLD.index(g)])
            if diag['rank']==2:
                for i in range(3):
                    same,_=math.project(parent,parent,parent[i],1.)
                    np.testing.assert_allclose(same,parent[i],rtol=1e-7,atol=1e-8)
            checks.append(dict(goal=g,normal_equations_checked=diag['rank']==2 and diag['condition']<1e4,rank=diag['rank'],condition=diag['condition'],self_reconstruction_checked=diag['rank']==2,actual_query_exact=(g in OLD)))
    return checks


def episode(lab,case,goal,arm):
    started=monotonic();text=lab.text[goal];obs=old.base.sr.reset(lab.ctx,case)
    trace=dict(start=dict(object=obs[lab.ctx.obj].tolist(),eef=obs['robot0_eef_pos'].tolist()),object=[],eef=[],contacts=[],success=[],regions=[],actions=[])
    signatures=[old.timing.signature(lab,obs)];hashes=[];decisions=[]
    for block in range(40):
        active=arm=='parallel400' or (arm=='parallel150' and block<15)
        if active:
            pre=prefixes(lab,obs,text);diags=[]
            chunk=lab.rt.integrate(lambda x,t:field(lab,pre,x,t,diags),lab.rt.noise(case['seed'],block),10)[...,:7]
            assert len(diags)==10
            decisions.append(dict(block=block,denoising=diags))
        else:chunk=lab.native('R' if arm=='parent' else 'F',text,obs,case['seed'],block)
        physical=lab.rt.physical(chunk);actual=np.clip(physical[:10],-1,1).astype(np.float32)
        hashes.append(sha256(actual.tobytes()).hexdigest())
        obs,events=lab.execute(obs,physical,block*10,case['z0'],None)
        for row,action in zip(events,actual,strict=True):
            trace['object'].append(row['obj']);trace['eef'].append(row['eef'])
            trace['contacts'].append(int(row['c1'])+2*int(row['c2']))
            trace['success'].append(sum(1<<j for j,g in enumerate(old.spec.scoring.SITES) if row['pred'][g]))
            trace['regions'].append(sum(1<<j for j,g in enumerate(old.spec.scoring.SITES) if g in row['hits']))
            trace['actions'].append(action.tolist())
        signatures.append(old.timing.signature(lab,obs))
    result=old.spec.scoring.score(trace,lab.ctx.geometry,case['z0'])
    return dict(experiment='four',profile='lfr_back3000',arm=arm,init=case['init'],goal=goal,target=goal,text=text,seed=case['seed'],alpha=lab.alpha,
        **result,trace=trace,signatures=signatures,action_hashes=hashes,decisions=decisions,seconds=monotonic()-started)


def worker(lane):
    p=old.base.read(RUN/'four_protocol.json')
    for n,h in p['sources'].items():assert digest(RUN/n)==h,n
    lab=load(p);lab.alpha=p['alpha'];case=old.base.read(RUN/'cases.json')[lane]
    assert case['init']==800+lane
    checks=gate(lab,case);selected=set(old.base.names())
    def fingerprint():return [old.tensors_hash({n:v.detach().cpu() for n,v in rt.models['F'].named_parameters() if n in selected}) for rt in (lab.rt.executer,lab.rt.parent)]
    before=fingerprint();dest=RUN/f'four_{lane}.jsonl.gz';assert not dest.exists();count=0
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            with old.gzip.open(dest,'xt',encoding='utf-8',compresslevel=5) as f:
                for arm in ARMS:
                    for g in ALL:
                        row=episode(lab,case,g,arm)
                        f.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');f.flush();count+=1
                        print('FOUR',lane,case['init'],arm,g,row['first'],row['terminal'][g],round(row['seconds'],2),flush=True)
    finally:lab.ctx.env.close()
    assert count==16 and before==fingerprint()
    old.base.write(RUN/f'four_{lane}_COMPLETE.json',dict(lane=lane,ids=[case['init']],episodes=count,gates=checks,
        weights_fingerprints=before,weights_unchanged=True,rows_sha256=digest(dest),protocol_sha256=digest(RUN/'four_protocol.json'),sources=p['sources']))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','calibrate','lock','worker']);a.add_argument('--lane',type=int)
    args=a.parse_args();torch.set_num_threads(2)
    if args.mode=='worker':worker(args.lane)
    else:globals()[args.mode]()
