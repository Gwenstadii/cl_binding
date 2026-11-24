"""Bounded timing experiment queue; at most two new workers per GPU."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from v_timing_protocol import LINES

RUN = Path(__file__).resolve().parent


def run(workers, lanes):
    pending = [(line, lane, (lane+j) % 2) for lane in range(lanes) for j, line in enumerate(LINES)
               if not (RUN / f"{line}_{lane}_COMPLETE.json").exists()]
    live, failed = [], []
    last_capacity = None
    try:
        while pending or live:
            capacity_file = RUN / "capacity.json"
            capacity = json.loads(capacity_file.read_text()) if capacity_file.exists() else dict(workers=workers, per_gpu=2)
            limit, per_gpu = capacity["workers"], capacity["per_gpu"]
            assert 1 <= limit <= 8 and 1 <= per_gpu <= 4
            if capacity != last_capacity:
                print("CAPACITY", capacity, flush=True)
                last_capacity = capacity
            counts = {gpu: sum(job[3] == gpu for job in live) for gpu in (0, 1)}
            while len(live) < limit:
                index = next((i for i, (_, _, gpu) in enumerate(pending) if counts[gpu] < per_gpu), None)
                if index is None:
                    break
                line, lane, gpu = pending.pop(index)
                stem = f"{line}_{lane}"
                log = (RUN / f"{stem}.log").open("a", encoding="utf-8")
                command = [sys.executable, "-u", str(RUN / "v_timing.py"), "worker", "--line", line,
                           "--lane", str(lane), "--lanes", str(lanes)]
                proc = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                live.append((proc, log, stem, gpu))
                counts[gpu] += 1
                print("START", proc.pid, gpu, stem, flush=True)
            for job in list(live):
                proc, log, stem, _ = job
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                live.remove(job)
                if code or not (RUN / f"{stem}_COMPLETE.json").exists():
                    failed.append(dict(job=stem, code=code))
                print("FINISH", stem, code, flush=True)
            if failed:
                # Do not spend the full grid after an engineering identity fails.
                raise RuntimeError(failed)
            time.sleep(1)
    finally:
        for proc, log, _, _ in live:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            log.close()
    (RUN / "queue_COMPLETE.json").write_text(json.dumps(dict(workers=workers, lanes=lanes, failures=failed))+"\n")
    assert not failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lanes", type=int, default=8)
    args = parser.parse_args()
    assert 1 <= args.workers <= 8 and 1 <= args.lanes <= 20
    run(args.workers, args.lanes)
