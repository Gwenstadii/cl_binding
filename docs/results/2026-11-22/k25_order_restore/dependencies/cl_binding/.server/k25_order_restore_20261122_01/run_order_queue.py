"""Bounded local worker queue for the K25 curriculum experiment server."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RUN = Path(__file__).resolve().parent
DEV = ("lr_front_130", "lr_front_1000", "lf_right_1000")
MAIN = ("lr_front_130", "lr_front_300", "lr_front_1000", "lf_right_1000")


def run(stage, workers):
    if stage in ("standard", "parent"):
        prerequisite = "main" if stage == "standard" else "standard"
        marker = RUN / f"queue_{prerequisite}_COMPLETE.json"
        while not marker.exists():
            time.sleep(1)
        assert not json.loads(marker.read_text())["failures"]
        suites = [(stage, "lf_right_1000")]
    else:
        suites = [("dev", line) for line in DEV] if stage == "dev" else (
            [("main", line) for line in MAIN]+[("phrase", line) for line in ("lr_front_1000", "lf_right_1000")])
    def stem_for(suite, line, lane):
        return f"{suite}_{lane}" if suite in ("standard", "parent") else f"{suite}_{line}_{lane}"
    pending = [(suite, line, lane) for suite, line in suites for lane in range(workers)
               if not (RUN / f"{stem_for(suite, line, lane)}_COMPLETE.json").exists()]
    live, failures = {}, []
    try:
        while pending or live:
            for slot in range(workers):
                if slot in live or not pending:
                    continue
                suite, line, lane = pending.pop(0)
                stem = stem_for(suite, line, lane)
                handle = (RUN / f"{stem}.log").open("a", encoding="utf-8")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(slot % 2)}
                if suite == "standard":
                    cmd = [sys.executable, "-u", str(RUN / "order_restore_standard.py"), "worker"]
                elif suite == "parent":
                    cmd = [sys.executable, "-u", str(RUN / "order_restore_parent.py")]
                else:
                    cmd = [sys.executable, "-u", str(RUN / "order_restore.py"), suite, "--line", line]
                cmd += ["--lane", str(lane), "--lanes", str(workers)]
                process = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
                live[slot] = process, handle, stem
                print("START", slot, process.pid, stem, flush=True)
            for slot, (process, handle, stem) in list(live.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del live[slot]
                if code or not (RUN / f"{stem}_COMPLETE.json").exists():
                    failures.append(dict(job=stem, exit_code=code))
                print("FINISH", slot, stem, code, flush=True)
            time.sleep(1)
    finally:
        for process, handle, _ in live.values():
            if process.poll() is None:
                process.terminate()
            handle.close()
    (RUN / f"queue_{stage}_COMPLETE.json").write_text(json.dumps(dict(stage=stage, workers=workers, failures=failures))+"\n")
    assert not failures, failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("dev", "main", "standard", "parent"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    assert 1 <= args.workers <= 8
    run(args.stage, args.workers)
