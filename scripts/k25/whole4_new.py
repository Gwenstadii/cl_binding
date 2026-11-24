"""Frozen whole4 controller comparison with a fresh Back continuation."""
import argparse
from hashlib import file_digest,sha256
import inspect
import json
from pathlib import Path
import sys
from time import monotonic
import numpy as np
import torch

RUN=Path(__file__).resolve().parent
REFERENCE=RUN.parent/'k25_parallel4_20261124_01'
TRAIN=RUN.parent/'k25_back_retrain_20261124_01'
sys.path.insert(0,str(REFERENCE))
import parallel_four as ref
sys.path.insert(0,str(RUN))
import whole_residual as full

ARMS=('native','parallel150','whole150','parallel400','whole400')

def read(p):return json.loads(Path(p).read_text())
def write(p,v):ref.old.base.write(Path(p),v)
def digest(p):
    with Path(p).open('rb') as f:return file_digest(f,'sha256').hexdigest()

def setup(step):
    p=read(REFERENCE/'four_protocol.json')
    p['models']['executor']=str(TRAIN/f'training/{step:06d}/trainable.safetensors')
    return p

def fingerprints(lab):
    names=set(ref.old.base.names())
    return [ref.old.tensors_hash({n:v.detach().cpu() for n,v in rt.models['F'].named_parameters() if n in names}) for rt in (lab.rt.executer,lab.rt.parent)]

def prepare():
    assert not (RUN/'preparation.json').exists()
    complete=read(TRAIN/'training/COMPLETE.json');assert complete['steps']==24000 and complete['frozen_unchanged']
    config=read(TRAIN/'config.json');assert digest(TRAIN/'config.json')==complete['config_sha256']
    p=read(REFERENCE/'four_protocol.json')
    for name,value in p['sources'].items():assert digest(REFERENCE/name)==value
    candidates={}
    for step in sorted(complete['saved_steps'],reverse=True):
        folder=TRAIN/f'training/{step:06d}';m=read(folder/'READY.json')
        assert m['step']==step and m['count']==155 and m['frozen_unchanged'] and m['parent_sha256']==config['parent_sha256']
        assert digest(folder/'trainable.safetensors')==m['sha256']
        candidates[str(step)]=dict(weights=str(folder/'trainable.safetensors'),sha256=m['sha256'],ready_sha256=digest(folder/'READY.json'))
    cases=read(REFERENCE/'cases.json');write(RUN/'cases.json',cases)
    ctx=ref.old.original.k21.km.Ctx('scene2');dev=[]
    try:
        known={r['signature']['sim'] for r in cases}
        for i,init in enumerate(range(640,660)):
            seed=202611240+i;np.random.seed(seed);ctx.env.seed(seed);ctx.env.reset()
            r=dict(init=init,state=np.asarray(ctx.env.get_sim_state(),dtype=np.float64).tolist(),reset_seed=seed,seed=17964041+107*i)
            obs=ref.old.base.sr.reset(ctx,r)
            assert not any(ctx.flags().values())
            r.update(signature=ctx.signature(obs),z0=float(obs[ctx.obj][2]),book_start=obs[ctx.obj].tolist())
            assert r['signature']['sim'] not in known;known.add(r['signature']['sim']);dev.append(r)
    finally:ctx.env.close()
    write(RUN/'dev_cases.json',dev)
    write(RUN/'preparation.json',dict(training_complete=complete,training_config_sha256=digest(TRAIN/'config.json'),
        candidates=candidates,reference_protocol_sha256=digest(REFERENCE/'four_protocol.json'),reference_sources=p['sources'],
        cases_sha256=digest(RUN/'cases.json'),dev_cases_sha256=digest(RUN/'dev_cases.json'),plan_sha256=digest(RUN/'plan.json'),
        sources={n:digest(RUN/n) for n in ['whole4_new.py','run_whole4_new.py','whole_residual.py','parallel_subspace.py']},
        geometry=p['geometry'],models=p['models']))
    print('NEW_PREPARED',len(candidates),flush=True)

def dev(step,lane):
    prep=read(RUN/'preparation.json');assert digest(setup(step)['models']['executor'])==prep['candidates'][str(step)]['sha256']
    lab=ref.load(setup(step));lab.alpha=1.;before=fingerprints(lab)
    cases=read(RUN/'dev_cases.json')[lane::4];path=RUN/f'dev_{step}_{lane}.jsonl.gz'
    assert not path.exists()
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),ref.old.gzip.open(path,'xt') as f:
            for case in cases:
                r=ref.episode(lab,case,'back','native');r.update(experiment='qualification',checkpoint_step=step)
                f.write(json.dumps(r,allow_nan=False,separators=(',',':'))+'\n');f.flush()
                print('DEV',step,case['init'],r['first'],r['ever']['back'],r['terminal']['back'],flush=True)
    finally:lab.ctx.env.close()
    assert before==fingerprints(lab)
    write(RUN/f'dev_{step}_{lane}_COMPLETE.json',dict(step=step,lane=lane,episodes=len(cases),rows_sha256=digest(path),weights_fingerprints=before,weights_unchanged=True))

def calibrate(step):
    assert not (RUN/'calibration.json').exists()
    lab=ref.load(setup(step));stats=[]
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for init in range(10):
                seed=ref.old.original.k21.km.noise_seed('scene2',init)
                obs,_=lab.ctx.reset(init);z0=float(obs[lab.ctx.obj][2])
                for block in range(16):
                    if block>=3:
                        pre=ref.prefixes(lab,obs,lab.text['back']);x=lab.rt.noise(seed,block);t=torch.ones(1,device=x.device)
                        p,f,_,_=ref.velocities(lab,pre,x,t)
                        stats.append(dict(init=init,step=block*10,**ref.math.fields(p,f)))
                    chunk=lab.native('F',lab.text['back'],obs,seed,block)
                    obs,_=lab.execute(obs,lab.rt.physical(chunk),block*10,z0,None)
                print('NEW_CALIBRATE',init,flush=True)
    finally:lab.ctx.env.close()
    assert len(stats)==130
    alpha=min(10.,sorted(r['ratio'] for r in stats)[65])
    write(RUN/'calibration.json',dict(alpha=alpha,stats=stats,n=130,step=step,behaviour_used_for_selection=False))
    prep=read(RUN/'preparation.json');selection=read(RUN/'selection.json')
    write(RUN/'protocol.json',dict(models=setup(step)['models'],geometry=prep['geometry'],alpha=alpha,selected_step=step,
        selection=selection,calibration_sha256=digest(RUN/'calibration.json'),cases_sha256=prep['cases_sha256'],
        sources=prep['sources'],reference_sources=prep['reference_sources'],preparation_sha256=digest(RUN/'preparation.json'),
        arms=ARMS,ids=list(range(800,820)),horizon=400,expected=400))
    print('NEW_LOCKED',alpha,flush=True)

def field(lab,pre,x,t,diags):
    parent,late,q,value=ref.velocities(lab,pre,x,t)
    out,d=full.whole(parent,late,q,lab.alpha);diags.append(d)
    return torch.from_numpy(out).to(device=value.device,dtype=value.dtype)[None]

def whole_episode():
    source=inspect.getsource(ref.episode);ns=dict(ref.__dict__);ns['field']=field
    exec(compile(source,'<frozen_episode_whole>','exec'),ns)
    return ns['episode']

def worker(lane):
    p=read(RUN/'protocol.json')
    for n,v in p['sources'].items():assert digest(RUN/n)==v,n
    for n,v in p['reference_sources'].items():assert digest(REFERENCE/n)==v,n
    lab=ref.load(p);lab.alpha=p['alpha'];case=read(RUN/'cases.json')[lane];before=fingerprints(lab)
    gate=ref.gate(lab,case);whole=whole_episode();path=RUN/f'new_{lane}.jsonl.gz';assert not path.exists()
    checks=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        obs=ref.old.base.sr.reset(lab.ctx,case);x=lab.rt.noise(case['seed'],0);t=torch.ones(1,device=x.device)
        for goal in ref.ALL:
            pre=ref.prefixes(lab,obs,lab.text[goal]);pa,la,q,value=ref.velocities(lab,pre,x,t)
            out,_=full.whole(pa,la,q,lab.alpha)
            expected=pa.astype(np.float64).mean(0)+lab.alpha*(q.astype(np.float64)-la.astype(np.float64).mean(0))
            assert np.array_equal(out,expected)
            actual=field(lab,pre,x,t,[])
            assert torch.equal(actual,torch.from_numpy(expected).to(device=value.device,dtype=value.dtype)[None])
            checks.append(dict(goal=goal,whole_full32_formula_bitwise=True,whole_callback_bitwise=True))
    count=0
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),ref.old.gzip.open(path,'xt',compresslevel=5) as f:
            for arm in ARMS:
                for goal in ref.ALL:
                    actual_arm=arm.replace('whole','parallel');fn=whole if arm.startswith('whole') else ref.episode
                    row=fn(lab,case,goal,actual_arm)
                    row.update(experiment='whole4_new',arm=arm,checkpoint_step=p['selected_step'])
                    f.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');f.flush();count+=1
                    print('NEW',lane,arm,goal,row['first'],row['ever'][goal],row['terminal'][goal],round(row['seconds'],2),flush=True)
    finally:lab.ctx.env.close()
    assert count==20 and before==fingerprints(lab)
    write(RUN/f'new_{lane}_COMPLETE.json',dict(lane=lane,episodes=count,rows_sha256=digest(path),weights_fingerprints=before,
        weights_unchanged=True,projection_gates=gate,whole_gates=checks,protocol_sha256=digest(RUN/'protocol.json')))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','dev','calibrate','worker']);a.add_argument('--step',type=int);a.add_argument('--lane',type=int);v=a.parse_args();torch.set_num_threads(2)
    if v.mode=='prepare':prepare()
    elif v.mode=='dev':dev(v.step,v.lane)
    elif v.mode=='calibrate':calibrate(v.step)
    else:worker(v.lane)
