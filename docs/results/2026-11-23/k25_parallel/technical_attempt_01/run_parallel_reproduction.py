"""Only schedule the one predeclared method, bounded to four workers per GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from parallel_protocol import PROFILES

run=Path(__file__).resolve().parent
pending=[(p,l,(j+l)%2) for l in range(8) for j,p in enumerate(PROFILES) if not (run/f"{p}_{l}_COMPLETE.json").exists()]
live=[]
failures=[]
try:
    while pending or live:
        counts={g:sum(x[4] == g for x in live) for g in (0,1)}
        while len(live) < 8:
            index=next((i for i,(_,_,g) in enumerate(pending) if counts[g] < 4),None)
            if index is None:
                break
            profile,lane,gpu=pending.pop(index)
            log=(run/f"{profile}_{lane}.log").open("a",encoding="utf-8")
            proc=subprocess.Popen([sys.executable,"-u",str(run/"parallel_reproduction.py"),"worker","--profile",profile,"--lane",str(lane)],
                env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu)},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            live.append((proc,log,profile,lane,gpu))
            counts[gpu]+=1
            print("START",profile,lane,gpu,proc.pid,flush=True)
        for item in list(live):
            proc,log,profile,lane,_=item
            code=proc.poll()
            if code is None:
                continue
            live.remove(item)
            log.close()
            if code or not (run/f"{profile}_{lane}_COMPLETE.json").exists():
                failures.append(dict(profile=profile,lane=lane,code=code))
            print("FINISH",profile,lane,code,flush=True)
        if failures:
            raise RuntimeError(failures)
        time.sleep(1)
finally:
    for proc,log,_,_,_ in live:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
        log.close()
(run/"queue_COMPLETE.json").write_text(json.dumps(dict(episodes=160,units=32,failures=failures))+"\n")
