"""Rank existing checkpoints and independently rescore native Back; NO optimizer."""
import json,gzip,sys,os,subprocess,time
from pathlib import Path
import torch
ROOT=Path('/home/lzx/cl_binding/.server/k25_back_stable_20261124_01')
sys.path.insert(0,str(ROOT))
import back_stable_train as original
RUN=Path(__file__).resolve().parent
def write(p,v):original.original_eval.write(p,v)
def read(p):return json.loads(p.read_text())

def worker(step):
    original.RUN=RUN
    torch.set_num_threads(2)
    cases=read(ROOT/'confirmation_cases.json')
    assert len(cases)==20 and {r['init'] for r in cases}==set(range(700,720))
    folder=ROOT/f'{step:06d}';marker=read(folder/'READY.json')
    assert original.sha(folder/'trainable.safetensors')==marker['sha256'] and marker['step']==step
    result=original.evaluate(folder/'trainable.safetensors',cases,f'confirm_{step}')
    print('EXISTING_CHECKPOINT_CONFIRMATION',step,result,flush=True)

def prepare():
    assert not (RUN/'plan.json').exists()
    rows=read(ROOT/'attempts.json')
    ranked=sorted(rows,key=lambda r:(r['development']['terminal'],r['development']['stable_released_final'],r['step']),reverse=True)
    choices=[r['step'] for r in ranked[:3]]
    assert choices==[56000,72000,64000]
    audit=[]
    for record in rows:
        step=record['step'];marker=read(ROOT/f'dev_{step}_COMPLETE.json');path=ROOT/f'dev_{step}.jsonl.gz'
        assert original.sha(path)==marker['rows_sha256']
        assert original.sha(ROOT/f'{step:06d}/trainable.safetensors')==marker['weights_sha256']
        with gzip.open(path,'rt') as f:raw=list(map(json.loads,f))
        assert len(raw)==20 and {r['init'] for r in raw}==set(range(640,660))
        counts=dict(n=20,terminal=sum(r['terminal']['back'] for r in raw),ever=sum(r['ever']['back'] for r in raw),stable_released_final=sum(r['stable_released_final'] for r in raw))
        assert counts==record['development'];audit.append(record)
    write(RUN/'plan.json',dict(request='select from existing original chain; no new training',
        selection='top3 by dev terminal, then stable release, then later step; confirmation comparison on same20 states700-719',
        selected_candidates=choices,development_audited=audit,qualification='do not call qualified unless BOTH dev and confirmation terminal>=16 and stable_released_final>=16',
        confirmation_cases_sha256=original.sha(ROOT/'confirmation_cases.json'),source_sha256=original.sha(Path(__file__))))
    print('SELECTION_PREPARED',choices,flush=True)

def run():
    p=read(RUN/'plan.json');live=[];pending=list(p['selected_candidates'])
    try:
        while pending or live:
            while pending and len(live)<2:
                step=pending.pop(0);log=(RUN/f'worker_{step}.log').open('x')
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'worker',str(step)],stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'1'})
                live.append((proc,log,step));print('START',step,proc.pid,flush=True)
            for item in list(live):
                proc,log,step=item;code=proc.poll()
                if code is None:continue
                live.remove(item);log.close();assert code==0,(step,code);print('DONE',step,flush=True)
            time.sleep(1)
    finally:
        for proc,log,_ in live:
            if proc.poll() is None:proc.terminate();proc.wait()
            log.close()
    results=[]
    for step in p['selected_candidates']:
        dev=next(r['development'] for r in p['development_audited'] if r['step']==step)
        conf=read(RUN/f'confirm_{step}_COMPLETE.json')
        results.append(dict(step=step,development=dev,confirmation=conf,
            qualified=all(d['terminal']>=16 and d['stable_released_final']>=16 for d in (dev,conf))))
    write(RUN/'COMPLETE.json',dict(results=results,episodes=60,new_training=False,
        candidate_best_by_confirmation=sorted(results,key=lambda r:(r['confirmation']['terminal'],r['confirmation']['stable_released_final']),reverse=True)[0]['step']))

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='worker':worker(int(sys.argv[2]))
    else:run()
