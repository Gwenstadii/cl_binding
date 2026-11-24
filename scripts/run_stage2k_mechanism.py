"""Launch two bounded diagnostic arms, each gated by a runtime smoke check.

This only writes under the specified fresh output subdirectories. Existing
training processes, source checkouts and model files are never changed.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(args):
    cfg = json.loads(args.config.read_text())
    root = Path(cfg["runtime_root"])
    out = Path(cfg["output_root"])
    code = args.code_dir.resolve()
    if (out / "launcher.json").exists():
        raise FileExistsError("Launcher already registered; no implicit restart")
    if shutil.disk_usage(out).free < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB free; refusing additional work")
    current = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                       "--format=csv,noheader,nounits"], text=True)
    gpu_rows = [tuple(int(v.strip()) for v in line.split(",")) for line in current.splitlines()]
    requested = {cfg[name + "_gpu"] for name in args.arms}
    if len(requested) != len(args.arms) or not requested <= {row[0] for row in gpu_rows}:
        raise ValueError("Require one distinct existing GPU per requested arm")
    if any(mem > 1536 or util > 5 for gpu, mem, util in gpu_rows if gpu in requested):
        raise RuntimeError("GPU busy; no process is stopped or preempted")
    files = ["stage2k_response_swap.py", "stage2k_entry_handoff.py", "run_stage2k_mechanism.py"]
    registry = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                "configuration": cfg, "arms": args.arms, "python": sys.executable, "gpu_before": current,
                "code_sha256": {name: hashlib.sha256((code / name).read_bytes()).hexdigest() for name in files}}
    dump(out / "registered_config.json", cfg)
    dump(out / "launcher.json", registry)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "MUJOCO_GL": "egl",
           "LIBERO_CONFIG_PATH": str(root / ".server/libero"), "TOKENIZERS_PARALLELISM": "false",
           "OMP_NUM_THREADS": "4", "PYTHONPATH": str(root / "scripts")}
    python = str(root / ".venv/bin/python")

    def run(arm, label, arguments, gpu):
        command = [python, "-u", str(code / ("stage2k_" + arm + ".py")), *map(str, arguments)]
        record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                  "command": command, "gpu": gpu}
        status = out / f"{arm}_{label}_process.json"
        dump(status, record)
        started = time.monotonic()
        try:
            with (out / f"{arm}_{label}.log").open("w") as log:
                result = subprocess.run(command, cwd=root, env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                        stdout=log, stderr=subprocess.STDOUT, timeout=110 * 60)
            record.update(status="complete" if result.returncode == 0 else "failed", returncode=result.returncode)
        except Exception as error:
            record.update(status="failed", error=repr(error))
        record["seconds"] = time.monotonic() - started
        dump(status, record)
        if record["status"] != "complete":
            raise RuntimeError(f"{arm} {label} failed; see preserved log")

    def response():
        base = ["--runtime-scripts", root / "scripts", "--config", cfg["base_config"],
                "--teacher", root / cfg["teacher"], "--student", root / cfg["ft4"],
                "--identity-tolerance", cfg["sampling_identity_max_normalized_action_error"]]
        run("response_swap", "smoke", [*base, "--output", out / "response_smoke", "--indices", 0,
                                        "--horizon", 20, "--modes", *cfg["response_modes"]], cfg["response_gpu"])
        run("response_swap", "pilot", [*base, "--output", out / "response_pilot", "--indices",
                                        *cfg["development_indices"], "--horizon", cfg["horizon"],
                                        "--modes", *cfg["response_modes"]], cfg["response_gpu"])

    def entry():
        base = ["--runtime-scripts", root / "scripts", "--config", cfg["base_config"],
                "--teacher", root / cfg["teacher"], "--student", root / cfg["cid"],
                "--centres", out / "fixed_centres.json", "--entry-radius", cfg["entry_radius_m"],
                "--min-lift", cfg["entry_min_lift_m"], "--min-step", cfg["entry_min_step"],
                "--horizon", cfg["horizon"], "--continuation", cfg["continuation_horizon"],
                "--self-action-tolerance", cfg["self_handoff_first_action_max_error"]]
        run("entry_handoff", "smoke", [*base, "--output", out / "entry_smoke", "--indices", 0], cfg["entry_gpu"])
        smoke = json.loads((out / "entry_smoke/manifest.json").read_text())
        if not smoke["continuation_rows"] or any(not r["self_first_step_passed"] for r in smoke["continuation_rows"]):
            raise RuntimeError("No validated handoff in smoke; no pilot launched")
        run("entry_handoff", "pilot", [*base, "--output", out / "entry_pilot", "--indices",
                                        *cfg["development_indices"]], cfg["entry_gpu"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        functions = {"response": response, "entry": entry}
        futures = {name: pool.submit(functions[name]) for name in args.arms}
        outcomes = {}
        for name, future in futures.items():
            try:
                future.result()
                outcomes[name] = {"status": "complete"}
            except Exception as error:
                outcomes[name] = {"status": "failed", "error": repr(error)}
    registry.update(status="complete" if all(r["status"] == "complete" for r in outcomes.values()) else "partial_failure",
                    outcomes=outcomes, finished_at=datetime.now(timezone.utc).isoformat())
    dump(out / "launcher.json", registry)
    print(json.dumps(registry), flush=True)
    if registry["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=["response", "entry"], default=["response", "entry"])
    main(parser.parse_args())
