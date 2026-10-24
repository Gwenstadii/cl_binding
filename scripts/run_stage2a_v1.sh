#!/usr/bin/env bash
# Launch the registered 2A pilot once: FT4 on GPU0 and ER on GPU1 in one run, the full-replay ceiling on GPU0
# in a second run. Survives SSH disconnects. Approved by the researcher on 2026-10-24.
set -euo pipefail
cd "$(dirname "$0")/.."
run_name=stage2a_ft4_er_20261024_01
session_name=stage2a-ft4-er-20261024
ceiling_name=stage2a_full_replay_20261024_01
ceiling_session=stage2a-full-replay-20261024
for path in ".server/$run_name" "outputs/stage2a/${run_name}_ft4" "outputs/stage2a/${run_name}_er" \
            ".server/$ceiling_name" "outputs/stage2a/${ceiling_name}_full_replay"; do
  test ! -e "$path"
done
test -s outputs/stage2a/memory/left_front_uniform256_v1/memory.npz
if tmux has-session -t "$session_name" 2>/dev/null || tmux has-session -t "$ceiling_session" 2>/dev/null; then
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
tmux new-session -d -s "$session_name" ".venv/bin/python -u scripts/stage2_run.py --config configs/stage2a_replay_v1.json --run-dir .server/$run_name --branches ft4 er --gpus 0 1 > .server/$run_name.launcher.log 2>&1"
tmux new-session -d -s "$ceiling_session" ".venv/bin/python -u scripts/stage2_run.py --config configs/stage2a_replay_v1.json --run-dir .server/$ceiling_name --branches full_replay --gpus 0 > .server/$ceiling_name.launcher.log 2>&1"
date -Is
git rev-parse HEAD
tmux list-panes -t "$session_name" -F '#{pane_pid} #{pane_current_command}'
tmux list-panes -t "$ceiling_session" -F '#{pane_pid} #{pane_current_command}'
