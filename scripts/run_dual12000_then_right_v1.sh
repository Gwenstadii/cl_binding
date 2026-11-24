#!/usr/bin/env bash
# Start once, preserve the original joint acquisition, and survive SSH disconnects.
set -euo pipefail
cd "$(dirname "$0")/.."
run_name=dual12000_then_right_20261023_01
session_name=dual12000-right-20261023
test ! -e ".server/$run_name"
test ! -e "outputs/sequential/$run_name"
if tmux has-session -t "$session_name" 2>/dev/null; then
  printf '%s\n' 'The experiment tmux session already exists; refusing duplicate launch.' >&2
  exit 1
fi
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import subprocess
old=Path('.server/dual_left_front_20261023_01/run_summary.json')
if old.exists() and json.loads(old.read_text())['status']=='running':
    raise RuntimeError('The original joint acquisition is still running; check it first')
free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
values=[int(line.strip()) for line in free.splitlines() if line.strip()]
if len(values)<2 or any(value<8192 for value in values[:2]):
    raise RuntimeError('Both3090s need at least8GiB free before this planned launch')
PY
.venv/bin/python -m unittest discover -s tests -p 'test_stage1*.py' -q
tmux new-session -d -s "$session_name" ".venv/bin/python -u scripts/stage1_dual_then_right_run.py --config configs/stage1_dual12000_then_right_v1.json --run-dir .server/$run_name > .server/$run_name.launcher.log 2>&1"
date -Is
git rev-parse HEAD
tmux list-panes -t "$session_name" -F '#{pane_pid} #{pane_current_command}'
