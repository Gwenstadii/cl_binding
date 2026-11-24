#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_name=dual_right_then_back_20261023_01
session=dual-right-back-20261023
test ! -e ".server/$run_name"
test ! -e "outputs/sequential/$run_name"
if tmux has-session -t "$session" 2>/dev/null; then
  printf '%s\n' 'Same experiment session already exists; refusing duplicate launch.' >&2
  exit 1
fi
.venv/bin/python - <<'PY'
import json,subprocess
from pathlib import Path
source=Path('.server/dual12000_then_right_20261023_01')
assert json.loads((source/'run_summary.json').read_text())['status']=='complete'
assert json.loads((source/'COMPLETE.json').read_text())['status']=='complete'
free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
values=[int(x.strip()) for x in free.splitlines()]
assert len(values)>=2 and all(x>=8192 for x in values[:2]),'Both GPUs need at least8GiB free'
PY
.venv/bin/python -m unittest discover -s tests -p 'test_stage1*.py' -q
tmux new-session -d -s "$session" ".venv/bin/python -u scripts/stage1_dual_then_back_run.py --config configs/stage1_dual_right_then_back_v1.json --run-dir .server/$run_name > .server/$run_name.launcher.log 2>&1"
date -Is
git rev-parse HEAD
tmux list-panes -t "$session" -F '#{pane_pid} #{pane_current_command}'
