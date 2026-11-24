"""Twenty matched starts, bounded to four workers per GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

run=Path(__file__).resolve().parent
pending=[i for i in range(20) if not (run/f'whole_{i}_COMPLETE.json').exists()]
live=[];failures=[]
try:
    while pending or live:
        counts={g:sum(lane%2==g for _,_,lane in live) for g in (0,1)}
        while True:
            at=next((j for j,lane in enumerate(pending) if counts[lane%2]<4),None)
            if at is None:break
            lane=pending.pop(at);log=(run/f'whole_{lane}.log').open('a',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-u',str(run/'whole_four.py'),'worker','--lane',str(lane)],
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(lane%2)},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            live.append((proc,log,lane));counts[lane%2]+=1;print('START',lane,lane%2,proc.pid,flush=True)
        for item in list(live):
            proc,log,lane=item;code=proc.poll()
            if code is None:continue
            live.remove(item);log.close()
            if code or not (run/f'whole_{lane}_COMPLETE.json').exists():failures.append(dict(lane=lane,code=code))
            print('FINISH',lane,code,flush=True)
        if failures:raise RuntimeError(failures)
        time.sleep(1)
finally:
    for proc,log,_ in live:
        if proc.poll() is None:proc.terminate();proc.wait()
        log.close()
(run/'queue_COMPLETE.json').write_text(json.dumps(dict(episodes=160,units=20,failures=[]))+'\n')
