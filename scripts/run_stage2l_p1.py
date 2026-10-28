"""Run a two-continuation engineering check, then the sealed P1 72-cell matrix.

Only inference is launched. The existing simulator snapshots, checkpoints and
teammate source checkout are read-only. Each invocation requires a fresh output.
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(args):
    cfg = json.loads(args.config.read_text())
    root, output, source = map(Path, (cfg["runtime_root"], cfg["output_root"], cfg["source_snapshots"]))
    code = args.code_dir.resolve()
    if (output / "launcher.json").exists():
        raise FileExistsError("Existing launch; no implicit restart")
    if not output.is_dir():
        raise FileNotFoundError("Prepare an independent output/code directory before dispatch")
    if sha(source / "manifest.json") != cfg["source_manifest_sha256"]:
        raise ValueError("Sealed source manifest changed")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Snapshot source is incomplete")
    snapshots = [f"{row['source']}_{row['goal']}_{row['init_index']:02d}"
                 for row in manifest["prefix_rows"] if row["entry_step"] is not None]
    if len(snapshots) != cfg["source_snapshot_count"] or len(set(snapshots)) != len(snapshots):
        raise ValueError("Unexpected snapshot source matrix")
    for sid in snapshots:
        if not (source / (sid + "_snapshot.npz")).is_file() or not (source / (sid + "_snapshot.json")).is_file():
            raise FileNotFoundError(sid)
    for role in ("teacher", "student"):
        if sha(Path(cfg[role]) / "model.safetensors") != cfg[role + "_model_sha256"]:
            raise ValueError(f"Changed {role} checkpoint")
    if shutil.disk_usage(output).free < cfg["minimum_disk_free_bytes"]:
        raise RuntimeError("Insufficient disk space; no data are deleted")
    gpu_text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                        "--format=csv,noheader,nounits"], text=True)
    gpu_rows = {int(p[0]): (int(p[1]), int(p[2])) for line in gpu_text.splitlines()
                if (p := [value.strip() for value in line.split(",")])}
    mem, util = gpu_rows[cfg["gpu"]]
    if mem > 1536 or util > 5:
        raise RuntimeError("Requested GPU is busy; nothing is stopped or preempted")
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "configuration": cfg, "gpu_before": gpu_text, "snapshot_ids": snapshots,
              "code_sha256": {name: sha(code / name) for name in
                  ("stage2l_entry_confirm.py", "stage2k_entry_handoff.py", "run_stage2l_p1.py")}, "phases": []}
    write(output / "launcher.json", report)
    write(output / "registered_config.json", cfg)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(cfg["gpu"]), "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "MUJOCO_GL": "egl", "TOKENIZERS_PARALLELISM": "false",
           "LIBERO_CONFIG_PATH": str(root / ".server/libero"), "OMP_NUM_THREADS": "4",
           "PYTHONPATH": os.pathsep.join((str(code), str(root / "scripts")))}
    base = [str(root / ".venv/bin/python"), "-u", str(code / "stage2l_entry_confirm.py"),
            "--runtime-scripts", str(root / "scripts"), "--config", cfg["base_config"],
            "--snapshots", str(source), "--horizon", str(cfg["horizon"]),
            "--teacher", cfg["teacher"], "--student", cfg["student"],
            "--max-minutes", str(cfg["maximum_worker_minutes"]),
            "--max-output-bytes", str(cfg["maximum_output_bytes"])]
    if cfg["self_check"]:
        base.append("--self-check")

    def run(name, seeds, selection=None):
        command = [*base, "--output", str(output / name), "--seeds", *map(str, seeds)]
        if selection:
            command += ["--snapshot-ids", *selection]
        record = {"name": name, "status": "running", "command": command,
                  "started_at": datetime.now(timezone.utc).isoformat()}
        report["phases"].append(record)
        write(output / "launcher.json", report)
        start = time.monotonic()
        with (output / (name + ".log")).open("w") as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=(cfg["maximum_worker_minutes"] + 5) * 60)
        record.update(returncode=result.returncode, seconds=time.monotonic() - start,
                      status="complete" if result.returncode == 0 else "failed")
        write(output / "launcher.json", report)
        if result.returncode:
            raise RuntimeError(f"{name} failed; preserved output and no automatic retry")

    try:
        run("engineering", cfg["smoke_seeds"], cfg["smoke_snapshot_ids"])
        if not (output / "engineering/COMPLETE.json").is_file():
            raise RuntimeError("Engineering worker did not create its completion marker")
        run("confirmation", cfg["seeds"])
        if not (output / "confirmation/COMPLETE.json").is_file():
            raise RuntimeError("Confirmation worker did not create its completion marker")
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
    main(parser.parse_args())
