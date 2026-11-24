"""Bounded queue; four-command work waits for its calibrated protocol."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

run=Path(__file__).resolve().parent
profiles=['lr_front_1000','lf_right_1000','lr_front_12000','lf_right_12000']
pending=[dict(key=f'full_{p}_{l}',args=['parallel_full400.py','worker','--profile',p,'--lane',str(l)],ready='full_protocol.json',gpu=(i+l)%2) for l in range(8) for i,p in enumerate(profiles)]
pending+=[dict(key=f'four_{l}',args=['parallel_four.py','worker','--lane',str(l)],ready='four_protocol.json',gpu=l%2) for l in range(20)]
pending=[j for j in pending if not (run/(j['key']+'_COMPLETE.json')).exists()]
live=[]
failures=[]
previous_limit=None
try:
    while pending or live:
        capacity=json.loads((run/'capacity.json').read_text())
        limit=capacity['workers_per_gpu'];assert limit in (1,2,3,4)
        if limit!=previous_limit:
            print('CAPACITY',capacity,flush=True);previous_limit=limit
        counts={g:sum(j['gpu']==g for _,_,j in live) for g in (0,1)}
        while True:
            at=next((i for i,j in enumerate(pending) if counts[j['gpu']]<limit and (run/j['ready']).exists()),None)
            if at is None:break
            j=pending.pop(at)
            log=(run/(j['key']+'.log')).open('a',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-u',*[str(run/j['args'][0]),*j['args'][1:]]],env={**os.environ,'CUDA_VISIBLE_DEVICES':str(j['gpu'])},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            live.append((proc,log,j));counts[j['gpu']]+=1
            print('START',j['key'],j['gpu'],proc.pid,flush=True)
        for item in list(live):
            proc,log,j=item
            code=proc.poll()
            if code is None:continue
            live.remove(item);log.close()
            if code or not (run/(j['key']+'_COMPLETE.json')).exists(): failures.append(dict(key=j['key'],code=code))
            print('FINISH',j['key'],code,flush=True)
        if failures:raise RuntimeError(failures)
        time.sleep(1)
finally:
    for proc,log,_ in live:
        if proc.poll() is None:proc.terminate();proc.wait()
        log.close()
(run/'queue_COMPLETE.json').write_text(json.dumps(dict(episodes=480,units=52,failures=[]))+'\n')
