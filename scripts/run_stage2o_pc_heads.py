"""Launch registered P/C and true-head screening after engineering verification."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main(args):
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    root, output, code = Path(cfg["runtime_root"]), Path(cfg["output_root"]), args.code_dir.resolve()
    if (output / "launcher.json").exists():
        raise FileExistsError("Existing launch is not restarted")
    if not output.is_dir() or not code.is_dir():
        raise FileNotFoundError("Prepare an isolated output/code directory")
    for role in ("teacher", "student"):
        if sha(Path(cfg[role]) / "model.safetensors") != cfg[role+"_model_sha256"]:
            raise ValueError(f"Changed checkpoint: {role}")
    for name in ("prior_baseline", "prior_candidates", "prior_patch"):
        if sha(cfg[name+"_path"]) != cfg[name+"_sha256"]:
            raise ValueError(f"Changed source evidence: {name}")
    if json.loads(Path(cfg["prior_candidates_path"]).read_text())["selected"] != cfg["candidate_sites"]:
        raise ValueError("Candidate sites do not match prior registered ranking")
    for name, expected in cfg["expected_model_sources"].items():
        if sha(name) != expected:
            raise ValueError(f"Changed model implementation: {name}")
    if shutil.disk_usage(output).free < cfg["minimum_disk_free_bytes"]:
        raise RuntimeError("Insufficient disk; no cleanup is performed")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    rows = {int(a[0]): (int(a[1]), int(a[2])) for line in gpu.splitlines()
            if (a := [p.strip() for p in line.split(",")])}
    memory, utilization = rows[cfg["gpu"]]
    if memory > 1536 or utilization > 5:
        raise RuntimeError("Requested GPU is occupied; no preemption")
    names = ("run_stage2o_pc_heads.py", "stage2o_pc_heads.py", "stage2n_cross_checkpoint.py",
             "audit_stage2o_pc_heads.py", "audit_stage2n_cross_checkpoint.py",
             "stage2m_cross_attention.py", "stage2k_entry_handoff.py", "stage2k_response_swap.py")
    report = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "registered_commit": args.registered_commit, "configuration": cfg,
        "code_sha256": {name: sha(code / name) for name in names}, "gpu_before": gpu,
        "runtime_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "runtime_dirty_before": subprocess.check_output(
            ["git", "-C", str(root), "status", "--short"], text=True).splitlines(),
        "phases": [],
    }
    write(output / "registered_config.json", cfg)
    write(output / "launcher.json", report)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(cfg["gpu"]),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "MUJOCO_GL": "egl",
           "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4",
           "LIBERO_CONFIG_PATH": str(root / ".server/libero"),
           "PYTHONPATH": os.pathsep.join((str(code), str(root / "scripts")))}
    base = [str(root / ".venv/bin/python"), "-u", str(code / "stage2o_pc_heads.py"),
            "--config", str(args.config.resolve()), "--runtime-scripts", str(root / "scripts"),
            "--max-minutes", str(cfg["maximum_worker_minutes"]),
            "--max-output-bytes", str(cfg["maximum_output_bytes"])]

    def run(phase):
        command = [*base, "--phase", phase, "--output", str(output / phase)]
        row = {"name": phase, "status": "running", "command": command,
               "started_at": datetime.now(timezone.utc).isoformat()}
        report["phases"].append(row)
        write(output / "launcher.json", report)
        started = time.monotonic()
        with (output / (phase+".log")).open("w") as log:
            result = subprocess.run(
                command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=(cfg["maximum_worker_minutes"]+5)*60)
        row.update(status="complete" if result.returncode == 0 else "failed",
                   returncode=result.returncode, seconds=time.monotonic()-started)
        write(output / "launcher.json", report)
        if result.returncode or not (output / phase / "COMPLETE.json").is_file():
            raise RuntimeError(f"{phase} failed; evidence retained, no automatic retry")

    try:
        run("engineering")
        run("analysis")
        report.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        report.update(status="failed", error=repr(error), finished_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        write(output / "launcher.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--registered-commit", required=True)
    main(parser.parse_args())
