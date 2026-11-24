"""Matched four-command full-residual arm; original episode source is unchanged."""
import argparse
from hashlib import sha256
import inspect
from pathlib import Path
import sys

import numpy as np
import torch

RUN=Path(__file__).resolve().parent
REFERENCE=RUN.parent/'k25_parallel4_20261124_01'
sys.path.insert(0,str(REFERENCE))
import parallel_four as ref
sys.path.insert(0,str(RUN))
import whole_residual as math

ARMS=dict(whole150='parallel150',whole400='parallel400')


def digest(path):return sha256(path.read_bytes()).hexdigest()


def field(lab,pre,x,t,diagnostics):
    parent,late,actual,value=ref.velocities(lab,pre,x,t)
    output,diag=math.whole(parent,late,actual,lab.alpha)
    diagnostics.append(diag)
    return torch.from_numpy(output).to(device=value.device,dtype=value.dtype)[None]


def episode_function():
    source=inspect.getsource(ref.episode)
    namespace=dict(ref.__dict__);namespace['field']=field
    exec(compile(source,'<original_parallel4_episode_full_residual>','exec'),namespace)
    episode=namespace['episode']
    assert episode.__globals__['field'] is field
    return episode,sha256(source.encode()).hexdigest()


def prepare():
    assert not (RUN/'protocol.json').exists()
    p=ref.old.base.read(REFERENCE/'four_protocol.json')
    cal=ref.old.base.read(REFERENCE/'four_calibration.json')
    assert digest(REFERENCE/'four_calibration.json')==p['calibration_sha256']
    assert p['alpha']==cal['alpha']==min(10.,sorted(r['ratio'] for r in cal['stats'])[65])
    for n,h in p['sources'].items():assert digest(REFERENCE/n)==h,n
    assert digest(RUN/'parallel_subspace.py')==p['sources']['parallel_subspace.py']
    for name,h in p['files'].items():assert digest(Path(name))==h,name
    _,episode_sha=episode_function()
    ref.old.base.write(RUN/'cases.json',ref.old.base.read(REFERENCE/'cases.json'))
    out=dict(reference_protocol_sha256=digest(REFERENCE/'four_protocol.json'),reference_sources=p['sources'],
        models=p['models'],files=p['files'],alpha=p['alpha'],calibration_sha256=p['calibration_sha256'],
        calibration_rule='Reuse exact independent-dev norm-ratio calibration,130 inputs; no behavioural tuning or projected-norm recalibration.',
        geometry=p['geometry'],ids=p['ids'],old_goals=p['old_goals'],all_goals=p['all_goals'],arms=ARMS,
        expected=160,horizon=400,cases_sha256=digest(RUN/'cases.json'),plan_sha256=digest(RUN/'analysis_plan.json'),
        sources={n:digest(RUN/n) for n in ('whole_four.py','whole_residual.py','run_whole_four.py','parallel_subspace.py')},
        original_episode_sha256=episode_sha,episode_edit='Source unchanged; only global field callable replaced. Arm metadata renamed after execution.',
        precision='float64 construction and native output dtype, identical to parallel4',
        source_parent_commit='1c7ac293bafd0fa691d1d128f985095e599ae85e')
    ref.old.base.write(RUN/'protocol.json',out)
    print('WHOLE4_LOCKED',160,p['alpha'],flush=True)


def gates(lab,case):
    obs=ref.old.base.sr.reset(lab.ctx,case);x=lab.rt.noise(case['seed'],0);t=torch.ones(1,device=x.device)
    checks=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for g in ref.ALL:
            pre=ref.prefixes(lab,obs,lab.text[g]);p,f,q,value=ref.velocities(lab,pre,x,t)
            output,d=math.whole(p,f,q,lab.alpha)
            expected=p.astype(np.float64).mean(0)+lab.alpha*(q.astype(np.float64)-f.astype(np.float64).mean(0))
            assert np.array_equal(output,expected)
            projected,_=ref.math.project(p,f,q,lab.alpha)
            other=[];native=ref.field(lab,pre,x,t,other)
            assert torch.equal(native,torch.from_numpy(projected).to(device=value.device,dtype=value.dtype)[None])
            executed=[];actual=field(lab,pre,x,t,executed)
            assert torch.equal(actual,torch.from_numpy(expected).to(device=value.device,dtype=value.dtype)[None])
            pc=p.astype(np.float64)-p.astype(np.float64).mean(0)
            b=np.einsum('ntd,nk->tdk',pc,ref.math.helmert(3))[:,:7,:].reshape(-1,2)
            removed=(output-projected)[:,:7].reshape(-1)
            error=float(np.linalg.norm(b.T@removed)/max(np.linalg.norm(b)*np.linalg.norm(removed),1e-30))
            assert error<5e-6
            checks.append(dict(goal=g,whole_formula_full32_bitwise=True,original_projection_callback_bitwise=True,
                whole_callback_bitwise=True,removed_orthogonality_error=error,full_over_parent=d['scaled_full_over_parent'],removed_over_parent=d['scaled_removed_over_parent']))
    return checks


def worker(lane):
    p=ref.old.base.read(RUN/'protocol.json')
    for n,h in p['sources'].items():assert digest(RUN/n)==h,n
    for n,h in p['reference_sources'].items():assert digest(REFERENCE/n)==h,n
    assert digest(REFERENCE/'four_protocol.json')==p['reference_protocol_sha256']
    lab=ref.load(p);lab.alpha=p['alpha'];case=ref.old.base.read(RUN/'cases.json')[lane]
    assert case['init']==800+lane
    checks=gates(lab,case);episode,h=episode_function();assert h==p['original_episode_sha256']
    selected=set(ref.old.base.names())
    def fingerprints():return [ref.old.tensors_hash({n:v.detach().cpu() for n,v in rt.models['F'].named_parameters() if n in selected}) for rt in (lab.rt.executer,lab.rt.parent)]
    before=fingerprints();dest=RUN/f'whole_{lane}.jsonl.gz';assert not dest.exists();count=0
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            with ref.old.gzip.open(dest,'xt',encoding='utf-8',compresslevel=5) as f:
                for name,arm in ARMS.items():
                    for goal in ref.ALL:
                        row=episode(lab,case,goal,arm)
                        row.update(experiment='whole4',method='full_residual',arm=name,window=[0,150 if name=='whole150' else 400])
                        f.write(ref.json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');f.flush();count+=1
                        print('WHOLE4',lane,case['init'],name,goal,row['first'],row['terminal'][goal],round(row['seconds'],2),flush=True)
    finally:lab.ctx.env.close()
    assert count==8 and before==fingerprints()
    ref.old.base.write(RUN/f'whole_{lane}_COMPLETE.json',dict(lane=lane,ids=[case['init']],episodes=count,gates=checks,
        weights_fingerprints=before,weights_unchanged=True,rows_sha256=digest(dest),protocol_sha256=digest(RUN/'protocol.json'),sources=p['sources']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker']);p.add_argument('--lane',type=int);a=p.parse_args()
    torch.set_num_threads(2)
    if a.mode=='prepare':prepare()
    else:worker(a.lane)
