"""K15: complete the background x fixed-gain comparison on K14 snapshots.

New condition: mu_new + 2.959 * (v_new(actual q) - mu_new).
Reproduce old_gain and native reference episodes without changing K14.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/lzx/cl_binding/.server/k14_snapshot_batch1_20261108_01')
LOCKED = {'code/k14.py':'89e7a80342adc99cba7374696cf925139143141a10e7c518dcef6a5fd9a0bf2c',
          'code/k14s5.py':'de5565c3bc6d971bd97f8d4d792251e73ae07c57b6c59b97c146ed6a40d8ba0e',
          'snapshots/snapshots.json':'0ee8b429c5098aeb30ea2404968c1db068b894fe57407a47c5c92ac78329db5f'}
for relative,digest in LOCKED.items():
    assert hashlib.sha256((SOURCE/relative).read_bytes()).hexdigest()==digest,relative
sys.path.insert(0,str(SOURCE/'code'))
from k14s5 import Lab5, ALPHA
from k14 import INSTR, LABEL, HORIZON, WINDOW
assert ALPHA == 2.959 and HORIZON == 400 and WINDOW == 60
CONDITIONS=('new_gain','old_gain','native')


def new_velocity(rt, prefixes, qpre, x, t, diag, alpha=ALPHA):
    # The old model is not queried by, or used in, this condition's output.
    left=rt.query('F',prefixes['A'],x,t).float()
    right=rt.query('F',prefixes['B'],x,t).float()
    actual=rt.query('F',qpre,x,t).float()
    mu=(left+right)/2
    out=(mu+alpha*(actual-mu)).to(rt.vdtype)
    source=(actual-mu)[...,:7]
    effective=(out.float()-mu)[...,:7]
    diag['gain_ratio_real'].append(float(effective.norm()/source.norm().clamp_min(1e-12)))
    return out


def gates(lab, snapshots):
    checks=[]
    torch=lab.torch
    for init in (20,21,22):
        snap=snapshots[init]
        obs,_=lab.rebuild(init,[lab.np.asarray(x,dtype=lab.np.float32) for x in snap['controls']],snap['z0'])
        assert lab.ctx.signature(obs)==snap['signature']
        prefixes={g:lab.rt.prefix('F',obs,INSTR[g]) for g in 'AB'}
        x=lab.rt.noise(snap['seed'],snap['boundary']//10)
        t=torch.ones(1,dtype=torch.float32,device=x.device)
        native={g:lab.rt.query('F',prefixes[g],x,t) for g in 'AB'}
        zero={}; amplified={}
        for g in 'AB':
            one=new_velocity(lab.rt,prefixes,prefixes[g],x,t,defaultdict(list),alpha=1.)
            assert torch.equal(one,native[g]),'alpha=1 must reproduce native exactly'
            zero[g]=new_velocity(lab.rt,prefixes,prefixes[g],x,t,defaultdict(list),alpha=0.)
            amplified[g]=new_velocity(lab.rt,prefixes,prefixes[g],x,t,defaultdict(list))
        assert torch.equal(zero['A'],zero['B'])
        mu=(native['A'].float()+native['B'].float())/2
        actual_mean=(amplified['A'].float()+amplified['B'].float())/2
        rms=float((actual_mean-mu)[...,:7].square().mean().sqrt())
        assert rms<.03
        checks.append({'init':init,'alpha1_native_bitwise':True,'alpha0_instruction_identity':True,
                       'amplified_mean_roundoff_rms':rms,'same_snapshot_signature':True})
    (HERE/'gates.json').write_text(json.dumps({'pass':True,'checks':checks},indent=2))
    (HERE/'geometry.json').write_text(json.dumps({'sites':lab.ctx.geometry,'canonical_scene_xml_sha256':lab.ctx.xml_sha},indent=2))


def run(init,run_gates):
    snapshots={s['init']:s for s in json.loads((SOURCE/'snapshots/snapshots.json').read_text())['records']}
    assert set(snapshots)==set(range(20,40))
    lab=Lab5()
    torch=lab.torch
    snap=snapshots[init]
    def episode(condition,goal):
        started=time.monotonic()
        boundary,z0,seed=snap['boundary'],snap['z0'],snap['seed']
        controls=[lab.np.asarray(c,dtype=lab.np.float32) for c in snap['controls']]
        obs,rows=lab.rebuild(init,controls,z0)
        signature=lab.ctx.signature(obs)
        assert signature==snap['signature']
        diag=defaultdict(list)
        q=INSTR[goal]
        calls=0
        for r in range(boundary//10,HORIZON//10):
            if condition!='native' and r*10<boundary+WINDOW:
                qpre=lab.rt.prefix('F',obs,q)
                if condition=='old_gain':
                    prefixes={(role,g):lab.rt.prefix(role,obs,INSTR[g]) for role in 'RF' for g in 'AB'}
                    def velocity(x,t):
                        nonlocal calls
                        calls+=1
                        return lab.corrected(prefixes,qpre,x,t,diag)
                else:
                    prefixes={g:lab.rt.prefix('F',obs,INSTR[g]) for g in 'AB'}
                    def velocity(x,t):
                        nonlocal calls
                        calls+=1
                        return new_velocity(lab.rt,prefixes,qpre,x,t,diag)
                chunk=lab.rt.integrate(velocity,lab.rt.noise(seed,r),10)[...,:7]
            else:
                chunk=lab.native('F',q,obs,seed,r)
            obs,events=lab.execute(obs,lab.rt.physical(chunk),r*10,z0)
            rows.extend(events)
        assert len(rows)==400 and calls==(0 if condition=='native' else 60)
        move=lab.ctx.geo.movement('scene2',lab.ctx.geometry,rows,z0,None)
        win=rows[boundary:boundary+WINDOW]
        record={'init':init,'condition':condition,'goal':goal,'requested':LABEL[goal],
            'seed':seed,'boundary':boundary,'window':WINDOW,'alpha':ALPHA,
            'prefix_model':'Front-12000','prefix_goal':'Front','z0':z0,'boundary_signature':signature,
            'first':move['first_targets'],'first_step':move['first_step'],'first_times':move['times'],
            'ever':{g:any(x['pred'][g] for x in rows) for g in lab.ctx.geo.GOALS},'terminal':rows[-1]['pred'],
            'clip_rate_window':sum(x['clip'] for x in win)/len(win),
            'clip_rate_all':sum(x['clip'] for x in rows[boundary:])/(400-boundary),
            'diagnostics':{k:[sum(v)/len(v),min(v),max(v)] for k,v in diag.items()},
            'n_window_queries':calls,'seconds':time.monotonic()-started}
        return record,rows
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        if run_gates:gates(lab,snapshots)
        with (HERE/f'i{init}.jsonl').open('x') as records,(HERE/f'i{init}_traces.jsonl').open('x') as traces:
            for condition in CONDITIONS:
                for goal in 'AB':
                    rec,rows=episode(condition,goal)
                    traces.write(json.dumps({'init':init,'condition':condition,'goal':goal,'rows':rows},allow_nan=False)+'\n');traces.flush()
                    records.write(json.dumps(rec,allow_nan=False)+'\n');records.flush()
                    print(init,condition,goal,rec['first'],rec['terminal'][LABEL[goal]],f"{rec['seconds']:.1f}s",flush=True)
    (HERE/f'i{init}_COMPLETE.json').write_text(json.dumps({'init':init,'n_episodes':6,'model_sha256':lab.rt.sha,
        'model_paths':lab.rt.paths,'processor_check':lab.rt.processor_check,'alpha':ALPHA,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_hashes':LOCKED},indent=2))
    lab.ctx.env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init',type=int,choices=range(20,40),required=True)
    parser.add_argument('--gates',action='store_true')
    args=parser.parse_args();run(args.init,args.gates)
