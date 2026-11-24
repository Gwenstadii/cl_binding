"""Launch the registered engineering gate and steps 1-2 in an isolated directory.

Read-only checkpoints and runtime worktree; no training, closed-loop actions,
downloads, cleanup or implicit retries. GPU occupancy is checked before dispatch.
"""

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
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(args):
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    root, output = Path(cfg["runtime_root"]), Path(cfg["output_root"])
    code = args.code_dir.resolve()
    if (output / "launcher.json").exists():
        raise FileExistsError("An existing dispatch is never automatically restarted")
    if not (output.is_dir() and code.is_dir()):
        raise FileNotFoundError("Prepare a fresh isolated output/code directory first")
    for role in ("teacher", "student"):
        checkpoint = Path(cfg[role])
        if sha(checkpoint / "model.safetensors") != cfg[role + "_model_sha256"]:
            raise ValueError(f"{role} model identity changed")
    for source, expected in cfg["expected_model_sources"].items():
        if sha(source) != expected:
            raise ValueError(f"Model implementation changed: {source}")
    if shutil.disk_usage(output).free < cfg["minimum_disk_free_bytes"]:
        raise RuntimeError("Insufficient free disk; no files are deleted")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True
    )
    rows = {
        int(parts[0]): (int(parts[1]), int(parts[2]))
        for line in gpu.splitlines()
        if (parts := [part.strip() for part in line.split(",")])
    }
    memory, utilization = rows[cfg["gpu"]]
    if memory > 1536 or utilization > 5:
        raise RuntimeError("Requested GPU is occupied; no process is stopped")
    scripts = ("run_stage2m_cross_attention.py", "stage2m_cross_attention.py")
    report = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "registered_commit": args.registered_commit,
        "configuration": cfg,
        "code_sha256": {name: sha(code / name) for name in scripts},
        "gpu_before": gpu,
        "runtime_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "runtime_dirty_before": subprocess.check_output(
            ["git", "-C", str(root), "status", "--short"], text=True
        ).splitlines(),
        "phases": [],
    }
    write(output / "registered_config.json", cfg)
    write(output / "launcher.json", report)
    env = {
        **os.environ, "CUDA_VISIBLE_DEVICES": str(cfg["gpu"]),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "MUJOCO_GL": "egl", "TOKENIZERS_PARALLELISM": "false",
        "LIBERO_CONFIG_PATH": str(root / ".server/libero"),
        "OMP_NUM_THREADS": "4",
        "PYTHONPATH": os.pathsep.join((str(code), str(root / "scripts"))),
    }
    base = [
        str(root / ".venv/bin/python"), "-u", str(code / "stage2m_cross_attention.py"),
        "--runtime-scripts", str(root / "scripts"), "--config", cfg["base_config"],
        "--teacher", cfg["teacher"], "--student", cfg["student"],
        "--max-minutes", str(cfg["maximum_worker_minutes"]),
        "--max-output-bytes", str(cfg["maximum_output_bytes"]),
        "--identity-tolerance", str(cfg["identity_max_absolute_error"]),
    ]

    def run(phase, indices, seeds):
        command = [
            *base, "--phase", phase, "--output", str(output / phase),
            "--indices", *map(str, indices), "--seeds", *map(str, seeds),
        ]
        record = {
            "name": phase, "status": "running", "command": command,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        report["phases"].append(record)
        write(output / "launcher.json", report)
        started = time.monotonic()
        with (output / (phase + ".log")).open("w") as log:
            result = subprocess.run(
                command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=(cfg["maximum_worker_minutes"] + 5) * 60,
            )
        record.update(
            returncode=result.returncode, seconds=time.monotonic() - started,
            status="complete" if result.returncode == 0 else "failed",
        )
        write(output / "launcher.json", report)
        if result.returncode:
            raise RuntimeError(f"{phase} failed; output preserved and no automatic retry")
        marker = output / phase / "COMPLETE.json"
        if not marker.is_file():
            raise RuntimeError(f"Missing completion marker: {phase}")

    try:
        run("engineering", cfg["engineering_indices"], cfg["engineering_seeds"])
        run("analysis", cfg["development_indices"], cfg["noise_seeds"])
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
