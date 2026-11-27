"""K17: full H x Z x rotation x gripper mean-source factorial, new padding.

Four K16 endpoints are reused; init20 reruns all four as exact audit controls.
All actual-query conditional responses remain adapted and amplified2.959.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
OLD=Path('/home/lzx/cl_binding/.server/k16_mu_channels_20261110_01')
OLD_SHA='e0bdcc875cb1981a5625ee916c3a19295150cad02d3d2536ef637e255d4364cd'
assert hashlib.sha256((OLD/'k16.py').read_bytes()).hexdigest()==OLD_SHA
sys.path.insert(0,str(OLD))
import k16
Lab5,ALPHA,INSTR,LABEL,HORIZON,WINDOW=k16.Lab5,k16.ALPHA,k16.INSTR,k16.LABEL,k16.HORIZON,k16.WINDOW
BLOCKS={'H':(0,1),'Z':(2,),'R':(3,4,5),'G':(6,)}
CONDITIONS=('N','Z','R','ZR','G','ZG','RG','ZRG','H','HZ','HR','HZR','HG','HZG','HRG','HZRG')
REFERENCES={'N':'N','H':'H','ZRG':'S','HZRG':'HS'}
FRESH=tuple(c for c in CONDITIONS if c not in REFERENCES)
INDICES={c:tuple(d for b in BLOCKS if b in c for d in BLOCKS[b]) for c in CONDITIONS}
assert len(FRESH)==12 and sorted(sum(BLOCKS.values(),()))==list(range(7))
assert len(set(INDICES.values()))==16


def combine(rt,condition,old,new,actual,alpha=ALPHA):
    mu=new.clone()
    ix=list(INDICES[condition])
    if ix:mu[...,ix]=old[...,ix]
    raw=mu+alpha*(actual-new)
    out=raw.to(rt.vdtype)
    assert rt.torch.isfinite(out).all()
    return out,mu,raw


def velocity(lab,condition,prefixes,qpre,x,t,diag):
    if condition in REFERENCES:
        return k16.dispatch(lab,REFERENCES[condition],prefixes,qpre,x,t,diag)
    old,new,actual=k16.get_fields(lab.rt,prefixes,qpre,x,t)
    out,_,raw=combine(lab.rt,condition,old,new,actual)
    for b,dims in BLOCKS.items():
        diag['mean_shift_rms_'+b].append(float((old-new)[...,list(dims)].square().mean().sqrt()))
    diag['roundoff_real'].append(float((out.float()-raw)[...,:7].square().mean().sqrt()))
    return out


def gates(lab,snapshots):
    torch=lab.torch
    checks=[]
    for init in (20,21,22):
        snap=snapshots[init]
        obs,_=lab.rebuild(init,[lab.np.asarray(a,dtype=lab.np.float32) for a in snap['controls']],snap['z0'])
        assert lab.ctx.signature(obs)==snap['signature']
        prefixes=k16.make_prefixes(lab,obs)
        x=lab.rt.noise(snap['seed'],snap['boundary']//10)
        t=torch.ones(1,dtype=torch.float32,device=x.device)
        zeros={};items=[]
        for goal in 'AB':
            qpre=lab.rt.prefix('F',obs,INSTR[goal])
            old,new,actual=k16.get_fields(lab.rt,prefixes,qpre,x,t)
            n,_,_=combine(lab.rt,'N',old,new,actual)
            full,_,_=combine(lab.rt,'HZRG',old,new,actual)
            native,_,_=combine(lab.rt,'N',old,new,actual,alpha=1.)
            assert torch.equal(native,actual.to(lab.rt.vdtype))
            zeros[goal]={}
            for condition in CONDITIONS:
                out,_,raw=combine(lab.rt,condition,old,new,actual)
                selected=list(INDICES[condition])
                expected=n.clone()
                if selected:expected[...,selected]=full[...,selected]
                assert torch.equal(out,expected),(init,goal,condition)
                assert torch.equal(out,velocity(lab,condition,prefixes,qpre,x,t,defaultdict(list)))
                zero,_,_=combine(lab.rt,condition,old,new,actual,alpha=0.)
                zeros[goal][condition]=zero
                items.append({'goal':goal,'condition':condition,'selected_indices':selected,'mask_bitwise':True,'dispatch_bitwise':True,'finite32':True,'roundoff_rms_real':float((out.float()-raw)[...,:7].square().mean().sqrt())})
        for c in CONDITIONS:assert torch.equal(zeros['A'][c],zeros['B'][c])
        checks.append({'init':init,'same_signature':True,'alpha0_identity':True,'alpha1_N_native':True,'items':items})
    (HERE/'gates.json').write_text(json.dumps({'pass':True,'checks':checks,'executed_partition':{b:list(v) for b,v in BLOCKS.items()},'padding_always_new':True},indent=2,allow_nan=False))


def run(init,run_gates,audit_references):
    assert not audit_references or init==20
    snapshots={s['init']:s for s in json.loads((k16.SOURCE/'snapshots/snapshots.json').read_text())['records']}
    lab=Lab5();torch=lab.torch;snap=snapshots[init]
    def episode(condition,goal,audit=False):
        started=time.monotonic();boundary=snap['boundary'];z0=snap['z0'];seed=snap['seed']
        obs,rows=lab.rebuild(init,[lab.np.asarray(a,dtype=lab.np.float32) for a in snap['controls']],z0)
        signature=lab.ctx.signature(obs);assert signature==snap['signature']
        diag=defaultdict(list);calls=0;q=INSTR[goal]
        for r in range(boundary//10,HORIZON//10):
            if r*10<boundary+WINDOW:
                qpre=lab.rt.prefix('F',obs,q)
                prefixes=k16.make_prefixes(lab,obs)
                def field(x,t):
                    nonlocal calls
                    calls+=1
                    return velocity(lab,condition,prefixes,qpre,x,t,diag)
                chunk=lab.rt.integrate(field,lab.rt.noise(seed,r),10)[...,:7]
            else:chunk=lab.native('F',q,obs,seed,r)
            obs,events=lab.execute(obs,lab.rt.physical(chunk),r*10,z0);rows.extend(events)
        assert len(rows)==400 and calls==60
        move=lab.ctx.geo.movement('scene2',lab.ctx.geometry,rows,z0,None)
        win=rows[boundary:boundary+WINDOW]
        record={'init':init,'condition':condition,'goal':goal,'requested':LABEL[goal],'seed':seed,'boundary':boundary,'window':WINDOW,'alpha':ALPHA,'prefix_model':'Front-12000','prefix_goal':'Front','z0':z0,'boundary_signature':signature,'mu_old_blocks':[b for b in BLOCKS if b in condition],'mu_old_indices':list(INDICES[condition]),'evaluation_source':'k17_audit_repeat' if audit else 'new_k17','first':move['first_targets'],'first_step':move['first_step'],'first_times':move['times'],'ever':{g:any(a['pred'][g] for a in rows) for g in lab.ctx.geo.GOALS},'terminal':rows[-1]['pred'],'clip_rate_window':sum(a['clip'] for a in win)/60,'clip_rate_all':sum(a['clip'] for a in rows[boundary:])/(400-boundary),'diagnostics':{k:[sum(v)/len(v),min(v),max(v)] for k,v in diag.items()},'n_window_queries':calls,'seconds':time.monotonic()-started}
        return record,rows
    def execute(conditions,record_name,trace_name,audit=False):
        with (HERE/record_name).open('x') as records,(HERE/trace_name).open('x') as traces:
            for condition in conditions:
                for goal in 'AB':
                    record,rows=episode(condition,goal,audit)
                    records.write(json.dumps(record,allow_nan=False)+'\n');records.flush()
                    traces.write(json.dumps({'init':init,'condition':condition,'goal':goal,'rows':rows},allow_nan=False)+'\n');traces.flush()
                    print(init,condition,goal,record['first'],record['terminal'][LABEL[goal]],f"{record['seconds']:.1f}s",flush=True)
    try:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            if run_gates:gates(lab,snapshots)
            if audit_references:execute(REFERENCES,'pilot_references.jsonl','pilot_reference_traces.jsonl',True)
            execute(FRESH,f'i{init}.jsonl',f'i{init}_traces.jsonl')
        (HERE/f'i{init}_COMPLETE.json').write_text(json.dumps({'init':init,'n_episodes':24,'n_audit_reference_episodes':8 if audit_references else 0,'model_sha256':lab.rt.sha,'processor_check':lab.rt.processor_check,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),'k16_script_sha256':OLD_SHA,'k15_script_sha256':k16.K15_SHA256,'source_hashes':k16.LOCKED},indent=2))
    finally:lab.ctx.env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init',type=int,choices=range(20,40),required=True)
    p.add_argument('--gates',action='store_true')
    p.add_argument('--audit-references',action='store_true')
    args=p.parse_args();run(args.init,args.gates,args.audit_references)
