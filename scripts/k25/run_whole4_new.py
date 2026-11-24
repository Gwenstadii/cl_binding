"""Qualification, independent calibration, then requested matched comparison."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RUN=Path(__file__).resolve().parent
def write(name,value):(RUN/name).write_text(json.dumps(value,indent=2)+'\n')
def run_jobs(jobs,limit):
    pending=list(jobs);live=[]
    try:
        while pending or live:
            counts={g:sum(gpu==g for _,_,gpu,_ in live) for g in (0,1)}
            while pending:
                gpu=min(counts,key=counts.get)
                if counts[gpu]>=limit:break
                name,args=pending.pop(0);log=(RUN/(name+'.log')).open('a')
                proc=subprocess.Popen([sys.executable,'-u',str(RUN/'whole4_new.py'),*args],stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)})
                live.append((proc,log,gpu,name));counts[gpu]+=1;print('START',name,gpu,proc.pid,flush=True)
            for item in list(live):
                proc,log,gpu,name=item;code=proc.poll()
                if code is None:continue
                live.remove(item);log.close();assert code==0,(name,code)
                print('DONE',name,flush=True)
            time.sleep(1)
    finally:
        for proc,log,_,_ in live:
            if proc.poll() is None:proc.terminate();proc.wait()
            log.close()

try:
    prep=json.loads((RUN/'preparation.json').read_text());qualification=[];selected=None
    for step in map(int,prep['candidates']):
        run_jobs([(f'dev_{step}_{lane}',['dev','--step',str(step),'--lane',str(lane)]) for lane in range(4)],2)
        rows=[]
        for lane in range(4):
            with gzip.open(RUN/f'dev_{step}_{lane}.jsonl.gz','rt') as f:rows.extend(map(json.loads,f))
        assert len(rows)==20 and {r['init'] for r in rows}==set(range(640,660))
        result=dict(step=step,first=sum(r['first']==['back'] for r in rows),ever=sum(r['ever']['back'] for r in rows),terminal=sum(r['terminal']['back'] for r in rows))
        qualification.append(result);print('QUALIFICATION',result,flush=True)
        write('qualification_progress.json',qualification)
        if result['terminal']>=16 and result['ever']>=16:selected=step;break
    qualified=selected is not None
    if not qualified:selected=24000
    write('selection.json',dict(selected_step=selected,qualified=qualified,results=qualification,
        rule='latest-to-earliest, first terminal>=16/20 and ever>=16/20 on Back-only independent dev; if none, terminal24000 diagnostic only'))
    subprocess.run([sys.executable,'-u',str(RUN/'whole4_new.py'),'calibrate','--step',str(selected)],check=True)
    run_jobs([(f'new_{lane}',['worker','--lane',str(lane)]) for lane in range(20)],4)
    write('COMPLETE.json',dict(episodes=400,shards=20,qualification_episodes=20*len(qualification),selected_step=selected,qualified=qualified,failures=[]))
except BaseException as error:
    write('FAILED.json',dict(type=type(error).__name__,message=str(error)));raise
