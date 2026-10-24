#!/usr/bin/env bash
# Launch the registered 2A pilot once (FT4 on GPU0, ER on GPU1); survive SSH disconnects. Requires researcher sign-off.
set -euo pipefail
cd "$(dirname "$0")/.."
run_name=stage2a_ft4_er_20261024_01
session_name=stage2a-ft4-er-20261024
test ! -e ".server/$run_name"
test ! -e "outputs/stage2a/${run_name}_ft4"
test ! -e "outputs/stage2a/${run_name}_er"
test -s outputs/stage2a/memory/left_front_uniform256_v1/memory.npz
if tmux has-session -t "$session_name" 2>/dev/null; then
  printf '%s\n' 'The experiment tmux session already exists; refusing duplicate launch.' >&2
  exit 1
fi
.venv/bin/python - <<'PY'
import subprocess
free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
values=[int(line.strip()) for line in free.splitlines() if line.strip()]
if len(values)<2 or any(value<8192 for value in values[:2]):
    raise RuntimeError('Both 3090s need at least 8GiB free before this planned launch')
PY
.venv/bin/python -m unittest discover -s tests -p 'test_stage*.py' -q
tmux new-session -d -s "$session_name" ".venv/bin/python -u scripts/stage2_run.py --config configs/stage2a_replay_v1.json --run-dir .server/$run_name --branches ft4 er > .server/$run_name.launcher.log 2>&1"
date -Is
git rev-parse HEAD
tmux list-panes -t "$session_name" -F '#{pane_pid} #{pane_current_command}'
