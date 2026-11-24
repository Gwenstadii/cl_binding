"""Run the locked anchor audit with bounded workers and no model training."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from anchor_protocol import LINES

RUN = Path(__file__).resolve().parent


def run(workers):
    pending = [(line,lane,(lane+j)%2) for lane in range(8) for j,line in enumerate(LINES)
               if not (RUN/f"{line}_{lane}_COMPLETE.json").exists()]
    live, failures = [], []
    last = None
    try:
        while live or pending:
            capfile = RUN/"capacity.json"
            cap = json.loads(capfile.read_text()) if capfile.exists() else dict(workers=workers,per_gpu=4)
            assert 1 <= cap["workers"] <= 8 and 1 <= cap["per_gpu"] <= 4
            if cap != last:
                print("CAPACITY",cap,flush=True)
                last = cap
            counts = {g:sum(j[3] == g for j in live) for g in (0,1)}
            while len(live) < cap["workers"] and shutil.disk_usage(RUN).free > 128*1024**2:
                k = next((k for k,(_,_,g) in enumerate(pending) if counts[g] < cap["per_gpu"]),None)
                if k is None:
                    break
                line,lane,gpu = pending.pop(k)
                stem = f"{line}_{lane}"
                log = (RUN/f"{stem}.log").open("a",encoding="utf-8")
                command = [sys.executable,"-u",str(RUN/"anchor_check.py"),"worker","--line",line,"--lane",str(lane),"--lanes","8"]
                proc = subprocess.Popen(command,env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu)},
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                live.append((proc,log,stem,gpu))
                counts[gpu] += 1
                print("START",proc.pid,gpu,stem,flush=True)
            for job in list(live):
                proc,log,stem,_ = job
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                live.remove(job)
                if code or not (RUN/f"{stem}_COMPLETE.json").exists():
                    failures.append(dict(job=stem,code=code))
                print("FINISH",stem,code,flush=True)
            if failures:
                raise RuntimeError(failures)
            if pending and not live and shutil.disk_usage(RUN).free <= 128*1024**2:
                raise RuntimeError("Insufficient space for additional logs; no unrelated files are removed")
            time.sleep(1)
    finally:
        for proc,log,_,_ in live:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            log.close()
    (RUN/"queue_COMPLETE.json").write_text(json.dumps(dict(workers=workers,lanes=8,failures=failures))+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=8)
    args = parser.parse_args()
    assert 1 <= args.workers <= 8
    run(args.workers)
