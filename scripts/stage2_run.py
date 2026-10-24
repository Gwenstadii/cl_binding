"""Run the registered stage 2A pilot: FT4 and ER trained in parallel, fixed 12000 endpoints, paired evaluation.

Parent rows (80) are reused from E007 before (left/front/right) and E008 (back) after an exact condition check.
Each branch is trained once on its own GPU; evaluation never reruns low scores, and a runtime error is retried
at most once with the failed attempt kept.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from stage1_dual_goal_run import read_rows
from stage2_compare import GOALS, compare, summarize
from stage2_memory_export import sha256

ROOT = Path(__file__).resolve().parents[1]
PARENT_ROWS = (".server/dual12000_then_right_20261023_01/before/episodes.jsonl",
               ".server/dual_parent12000_unseen_back_20261023_01/episodes.jsonl")


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage2a_replay_v1.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--branches", nargs="+", default=["ft4", "er"], choices=["ft4", "er", "full_replay"])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run = args.run_dir.resolve()
    if run.exists():
        raise FileExistsError("Use a new run directory; no implicit restart")
    if len(args.branches) > 2:
        raise ValueError("Two 3090s: launch at most two branches at once")
    e, indices = config["evaluation"], config["evaluation"]["paired_indices"]
    parent_rows = [r for path in PARENT_ROWS for r in read_rows(ROOT / path)]
    parent_summary = summarize(parent_rows, indices, config["seed"])
    trains = {b: ROOT / "outputs/stage2a" / f"{run.name}_{b}" for b in args.branches}
    if any(path.exists() for path in trains.values()):
        raise FileExistsError("Training directory already exists")
    run.mkdir(parents=True)
    cfg = run / "config.json"
    dump(cfg, config)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "source_commit": commit,
              "branches": args.branches, "training_dirs": {b: str(p) for b, p in trains.items()},
              "parent_rows_reused": [str(ROOT / p) for p in PARENT_ROWS], "parent_summary": parent_summary,
              "phases": []}
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "MUJOCO_GL": "egl",
           "LIBERO_CONFIG_PATH": str(ROOT / ".server/libero"), "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4"}
    started = time.monotonic()
    workers, handles = [], []

    def save():
        dump(run / "run_summary.json", report)

    def launch(name, script, arguments, gpu):
        command = [sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, arguments)]
        handle = (run / f"{name}.log").open("w"); handles.append(handle)
        worker = subprocess.Popen(command, cwd=ROOT, env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                  stdout=handle, stderr=subprocess.STDOUT)
        workers.append(worker)
        report["phases"].append({"name": name, "command": command, "gpu": gpu, "pid": worker.pid,
                                 "started_elapsed_s": time.monotonic() - started}); save()
        return worker

    def finish(name, worker, timeout):
        code = worker.wait(timeout=timeout)
        next(p for p in report["phases"] if p["name"] == name).update(returncode=code, finished_elapsed_s=time.monotonic() - started)
        save()
        if code:
            raise RuntimeError(f"{name} failed with exit {code}")

    def check_training(branch):
        train = trains[branch]
        final = train / "checkpoints/012000"
        finished = json.loads((train / "training_finished_012000.json").read_text())
        marker = json.loads((final / "CHECKPOINT_COMPLETE.json").read_text())
        manifest = json.loads((train / "manifest.json").read_text())
        if finished["step"] != 12000 or finished["target"] != 12000 or not finished["frozen_parameters_unchanged"]:
            raise ValueError(f"{branch}: training did not finish exactly 12000 with frozen VLM")
        if marker["status"] != "pass" or manifest["config"] != config or manifest["git_commit"] != commit:
            raise ValueError(f"{branch}: checkpoint, config or source commit does not match the run")
        trace = read_rows(train / "train_steps.jsonl")
        if [r["step"] for r in trace] != list(range(1, 12001)) or not all(math.isfinite(r["loss"]) for r in trace):
            raise ValueError(f"{branch}: discontinuous or nonfinite training")
        return final, trace

    def evaluate(branch, checkpoint, gpu):
        for attempt in (1, 2):
            name = f"{branch}_eval_attempt{attempt}"
            worker = launch(name, "stage1_r0_eval.py", ["evaluate", "--config", cfg, "--checkpoint", checkpoint,
                            "--partition", "paired", "--instruction-goals", *GOALS, "--output", run / name, "--trace"], gpu)
            try:
                finish(name, worker, 80 * 60)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                if worker.poll() is None:
                    worker.kill(); worker.wait()
                report.setdefault("evaluation_runtime_failures", []).append({"branch": branch, "attempt": attempt, "error": str(error)})
                save()
                if attempt == 2:
                    raise
                continue
            return read_rows(run / name / "episodes.jsonl")

    try:
        save()
        trainers = {b: launch(f"{b}_train", "stage2_replay_train.py", ["--config", cfg, "--branch", b, "--run-dir", trains[b]], gpu)
                    for gpu, b in enumerate(args.branches)}
        for b, worker in trainers.items():
            finish(f"{b}_train", worker, 3 * 3600 + 10 * 60)
        finals = {b: check_training(b)[0] for b in args.branches}
        if [r["indices"]["new"] for r in read_rows(trains[args.branches[0]] / "train_steps.jsonl")] != \
           [r["indices"]["new"] for r in read_rows(trains[args.branches[-1]] / "train_steps.jsonl")]:
            raise ValueError("Branches did not consume identical new-sample schedules")
        report["current_stage"] = "evaluating_fixed_12000_endpoints"; save()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(len(args.branches)) as pool:
            futures = {b: pool.submit(evaluate, b, finals[b], gpu) for gpu, b in enumerate(args.branches)}
            rows = {b: f.result() for b, f in futures.items()}
        comparisons = {b: compare(parent_rows, rows[b], indices, config["seed"]) for b in args.branches}
        for b in args.branches:
            folder = run / b; folder.mkdir()
            (folder / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[b]))
            dump(folder / "comparison.json", comparisons[b])
        report.update(status="complete", current_stage="finished", finished_at=datetime.now(timezone.utc).isoformat(),
                      seconds=time.monotonic() - started,
                      final_model_sha256={b: sha256(finals[b] / "model.safetensors") for b in args.branches},
                      new_evaluation_episodes=80 * len(args.branches), reused_parent_episodes=len(parent_rows))
        save()
        lines = ["# 2A：封存双目标父模型 → 匹配FT4／普通ER（右格12000步）", "",
                 "探索性先导：父模型未通过正式资格，配对初态20—39有历史使用。每模型四指令各20回合。", "",
                 "| 模型 | 左 曾/终 | 前 曾/终 | 右 曾/终 | 后 曾/终 |", "|---|---|---|---|---|"]
        def line(name, summary):
            c = summary["task_success_counts"]
            return f"| {name} | " + " | ".join(f"{c[g]['success_ever']}/20、{c[g]['terminal_correct']}/20" for g in GOALS) + " |"
        lines.append(line("父模型（复用）", parent_summary))
        lines += [line(b, comparisons[b]["branch"]) for b in args.branches]
        lines += ["", "原先成功分母上的保持、首次／最终转右、未完成，以及13个共同初态见各分支`comparison.json`。"]
        (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        dump(run / "COMPLETE.json", {"status": "complete", "branches": args.branches, "fixed_steps": 12000,
                                     "starting_point_formally_qualified": False})
    except BaseException as error:
        report.update(status="failed", error=str(error), seconds=time.monotonic() - started); save(); raise
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
