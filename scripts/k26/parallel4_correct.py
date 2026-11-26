"""Fresh F4-B24 parallel4 rollouts, native requested-goal termination per action."""
import argparse,gzip,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
RUN=Path(__file__).resolve().parent
sys.path.insert(0,'/home/lzx/cl_binding/.server/k25_parallel4_20261124_01')
import parallel_four as ref
import numpy as np
import torch
from native_event_protocol import NativeEventEpisode
OLD_RUN=Path('/home/lzx/cl_binding/.server/k25_whole4_new_20261124_01')
ARMS=('parent','native','parallel150','parallel400')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def load():
    registry=read(RUN/'F4-B24.json');protocol=read(OLD_RUN/'protocol.json')
    assert sha(Path(registry['checkpoint_dir'])/'model.safetensors')==registry['full_checkpoint_model_sha256']
    lab=ref.old.original.k21.Lab(family='scene2',models=False)
    frt=ref.old.original.k21.km.make_runtime(lab.ctx,{'FT4':registry['checkpoint_dir']},['F'])
    prt=ref.old.original.k21.km.make_runtime(lab.ctx,{'FT4':registry['frozen_model_and_processor_base']},['F'])
    ref.old.base.sr.assign(type('Holder',(),{'rt':prt})(),'F',ref.old.base.selected_weights(Path(registry['three_task_parent'])))
    lab.rt=ref.old.DualRuntime(frt,prt);lab.goal={g:g for g in ref.ALL};lab.text={g:lab.ctx.all_texts[g] for g in ref.ALL}
    lab.alpha=protocol['alpha'];assert abs(lab.alpha-.781400847920)<1e-12
    assert lab.ctx.geometry==protocol['geometry']
    return lab
def fingerprint(lab):
    names=set(ref.old.base.names())
    return [ref.old.tensors_hash({n:v.detach().cpu() for n,v in rt.models['F'].named_parameters() if n in names}) for rt in (lab.rt.executer,lab.rt.parent)]
def episode(lab,case,goal,arm,reference):
    started=time.monotonic();obs=ref.old.base.sr.reset(lab.ctx,case);monitor=NativeEventEpisode(goal)
    assert not any(lab.ctx.flags().values())
    trace={k:[] for k in ('object','eef','contacts','success','actions')};decisions=[];prefix_equal=True
    for block in range(40):
        active=arm=='parallel400' or (arm=='parallel150' and block<15)
        if active:
            pre=ref.prefixes(lab,obs,lab.text[goal]);diags=[]
            chunk=lab.rt.integrate(lambda x,t:ref.field(lab,pre,x,t,diags),lab.rt.noise(case['seed'],block),10)[...,:7]
            decisions.append(dict(block=block,denoising=diags))
        else:chunk=lab.native('R' if arm=='parent' else 'F',lab.text[goal],obs,case['seed'],block)
        controls=lab.rt.physical(chunk)
        for k in range(10):
            obs,executed=lab.ctx.step(controls[k])
            if k==9:obs=lab.ctx.fresh(lab.ctx.env)
            row=lab.ctx.row(block*10+k+1,obs,controls[k],executed);c1,c2=lab.contacts()
            trace['object'].append(row['obj']);trace['eef'].append(row['eef']);trace['contacts'].append(int(c1)+2*int(c2))
            trace['success'].append(sum(1<<i for i,g in enumerate(ref.old.spec.scoring.SITES) if row['pred'][g]))
            action=np.asarray(executed,dtype=np.float32).tolist();trace['actions'].append(action)
            j=monitor.steps
            # Verify the actually executed fresh prefix, never use archived actions for control.
            same=(np.array_equal(np.asarray(action,dtype=np.float32),np.asarray(reference['trace']['actions'][j],dtype=np.float32)) and
                  np.allclose(row['obj'],reference['trace']['object'][j],rtol=0,atol=1e-7) and
                  trace['success'][-1]==reference['trace']['success'][j])
            prefix_equal=prefix_equal and same
            if monitor.observe(row['pred']):break
        if monitor.done:break
    result=monitor.result()
    return dict(experiment='K25-parallel4-correct',model='F4-B24',init=case['init'],goal=goal,arm=arm,
        seed=case['seed'],alpha=lab.alpha,**{k:v for k,v in result.items() if k!='goal'},
        stop_contacts=trace['contacts'][-1],trace=trace,decisions=decisions,
        archived_executed_prefix_equal=prefix_equal,seconds=time.monotonic()-started)
def worker(lane):
    torch.set_num_threads(2);lab=load();before=fingerprint(lab);case=read(OLD_RUN/'cases.json')[lane]
    refs={}
    for folder,pattern in [(OLD_RUN,f'new_{lane}.jsonl.gz'),(Path('/home/lzx/cl_binding/.server/k25_parallel4_20261124_01'),f'four_{lane}.jsonl.gz')]:
        with gzip.open(folder/pattern,'rt') as f:
            for line in f:
                r=json.loads(line)
                if r['arm'] in ARMS and (folder==OLD_RUN or r['arm']=='parent'):refs[r['arm'],r['goal']]=r
    dest=RUN/f'case_{lane}.jsonl.gz';n=0
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),gzip.open(dest,'xt',compresslevel=5) as log:
            for arm in ARMS:
                for goal in ref.ALL:
                    row=episode(lab,case,goal,arm,refs[arm,goal]);log.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');log.flush();n+=1
                    print('RESULT',lane,arm,goal,row['success'],row['stop_step'],row['archived_executed_prefix_equal'],flush=True)
    finally:lab.ctx.env.close()
    assert n==16 and fingerprint(lab)==before
    write(RUN/f'case_{lane}_COMPLETE.json',dict(lane=lane,episodes=n,sha256=sha(dest),weights_unchanged=True))
def controller():
    assert not (RUN/'manifest.json').exists();registry=read(RUN/'F4-B24.json');p=read(OLD_RUN/'protocol.json')
    write(RUN/'manifest.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),model=registry,arms=ARMS,
        goals=ref.ALL,ids=list(range(800,820)),expected=320,maximum_actions=400,stop_on_requested_native_success=True,
        alpha=p['alpha'],calibration='reuse fixed model-specific original independent130-input calibration; not tuned on results',
        calibration_sha256=sha(OLD_RUN/'calibration.json'),cases_sha256=sha(OLD_RUN/'cases.json'),
        source_sha256=sha(Path(__file__)),protocol_helper_sha256=sha(RUN/'native_event_protocol.py'),
        old_parallel_source_sha256=sha(Path(ref.__file__)),new_training_steps=0))
    pending=list(range(20));live=[];completed=[]
    try:
        while pending or live:
            for gpu in (0,1):
                while pending and sum(x[2]==gpu for x in live)<4:
                    lane=pending.pop(0);log=(RUN/f'worker_{lane}.log').open('x')
                    proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'worker','--lane',str(lane)],cwd=RUN,stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL)
                    live.append((proc,log,gpu,lane))
            for item in list(live):
                proc,log,gpu,lane=item
                if proc.poll() is None:continue
                live.remove(item);log.close();assert proc.returncode==0,(lane,proc.returncode)
                completed.append(lane)
            write(RUN/'status.json',dict(status='running',completed=sorted(completed),active=[x[3] for x in live],pending=pending))
            time.sleep(3)
        summarize()
        write(RUN/'COMPLETE.json',dict(episodes=320,completed_at_utc=datetime.now(timezone.utc).isoformat(),shards=20))
        write(RUN/'status.json',dict(status='complete',episodes=320))
    finally:
        for proc,log,_,_ in live:
            if proc.poll() is None:proc.terminate();proc.wait()
            log.close()
def summarize():
    rows=[]
    for lane in range(20):
        path=RUN/f'case_{lane}.jsonl.gz';m=read(RUN/f'case_{lane}_COMPLETE.json');assert sha(path)==m['sha256']
        with gzip.open(path,'rt') as f:rows.extend(json.loads(line) for line in f)
    assert len(rows)==320
    summary={};paired={}
    for arm in ARMS:
        subset=[r for r in rows if r['arm']==arm];keyed={(r['init'],r['goal']):r for r in subset}
        assert len(keyed)==80
        summary[arm]={g:dict(n=20,success=sum(keyed[i,g]['success'] for i in range(800,820)),steps=[keyed[i,g]['stop_step'] for i in range(800,820)]) for g in ref.ALL}
        summary[arm]['old_joint']=sum(all(keyed[i,g]['success'] for g in ref.OLD) for i in range(800,820))
        summary[arm]['four_joint']=sum(all(keyed[i,g]['success'] for g in ref.ALL) for i in range(800,820))
        if arm.startswith('parallel'):
            paired[arm]={g:dict(gained=[i for i in range(800,820) if keyed[i,g]['success'] and not next(r for r in rows if r['arm']=='native' and r['goal']==g and r['init']==i)['success']],
                               lost=[i for i in range(800,820) if not keyed[i,g]['success'] and next(r for r in rows if r['arm']=='native' and r['goal']==g and r['init']==i)['success']]) for g in ref.ALL}
    write(RUN/'summary.json',dict(episodes=320,results=summary,paired_vs_native=paired,
        archived_prefix_equal_count=sum(r['archived_executed_prefix_equal'] for r in rows),
        success=sum(r['success'] for r in rows),sources={p.name:sha(p) for p in RUN.glob('case_*.jsonl.gz')}))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['controller','worker','summarize']);a.add_argument('--lane',type=int);v=a.parse_args()
    try:
        if v.mode=='worker':worker(v.lane)
        elif v.mode=='controller':controller()
        else:summarize()
    except BaseException as e:
        import traceback
        write(RUN/f'FAILED_{v.mode}_{v.lane}.json',dict(error=str(e),traceback=traceback.format_exc()));raise
