"""Bounded GPU queue for the predeclared mean-source contrasts."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

RUN = Path(__file__).resolve().parent


def run(workers):
    pending=[(lane,(lane//2+lane%2)%2) for lane in range(40) if not (RUN/f"unit_{lane:02d}_COMPLETE.json").exists()]
    live=[]
    failures=[]
    last=None
    try:
        while live or pending:
            capfile=RUN/"capacity.json"
            cap=json.loads(capfile.read_text()) if capfile.exists() else dict(workers=workers,per_gpu=max(1,workers//2))
            assert 1 <= cap["workers"] <= 8 and 1 <= cap["per_gpu"] <= 4
            if cap != last:
                print("CAPACITY",cap,flush=True)
                last=cap
            counts={g:sum(job[3] == g for job in live) for g in (0,1)}
            while len(live) < cap["workers"] and shutil.disk_usage(RUN).free > 128*1024**2:
                index=next((i for i,(_,g) in enumerate(pending) if counts[g] < cap["per_gpu"]),None)
                if index is None:
                    break
                lane,gpu=pending.pop(index)
                log=(RUN/f"unit_{lane:02d}.log").open("a",encoding="utf-8")
                process=subprocess.Popen([sys.executable,"-u",str(RUN/"mu_swap.py"),"worker","--lane",str(lane)],
                    env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu)},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                live.append((process,log,lane,gpu))
                counts[gpu]+=1
                print("START",lane,gpu,process.pid,flush=True)
            for job in list(live):
                process,log,lane,_=job
                code=process.poll()
                if code is None:
                    continue
                live.remove(job)
                log.close()
                if code or not (RUN/f"unit_{lane:02d}_COMPLETE.json").exists():
                    failures.append(dict(lane=lane,code=code))
                print("FINISH",lane,code,flush=True)
            if failures:
                raise RuntimeError(failures)
            if pending and not live and shutil.disk_usage(RUN).free <= 128*1024**2:
                raise RuntimeError("Insufficient free space; unrelated data is preserved")
            time.sleep(1)
    finally:
        for process,log,_,_ in live:
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()
    (RUN/"queue_COMPLETE.json").write_text(json.dumps(dict(workers=workers,units=40,episodes=600,failures=failures))+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    assert 1 <= args.workers <= 8
    run(args.workers)
