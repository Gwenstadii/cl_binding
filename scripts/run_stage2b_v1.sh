#!/usr/bin/env bash
# Stage 2B on the same sealed parent, approved by the researcher on 2026-10-24:
#   wave 1: two adaptation replicates of FT4/ER (seeds 20261124, 20261224), one FT4 and one ER per GPU;
#   wave 2: ER budget sweep with nested memories of 64/16/4/1 records per old goal, two runs per GPU.
# Scene-XML alias: worktree symlink paths only, see .server/stage2a_scene_equivalence_20261024/equivalence.json.
set -euo pipefail
cd "$(dirname "$0")/.."
session_name=stage2b-20261024
runs=(stage2b_seed2_20261024_01 stage2b_seed3_20261024_01 stage2b_mem064_20261024_01 stage2b_mem016_20261024_01
      stage2b_mem004_20261024_01 stage2b_mem001_20261024_01)
for run in "${runs[@]}"; do test ! -e ".server/$run"; done
for k in 064 016 004 001; do test -s "outputs/stage2a/memory/left_front_nested${k}_v1/memory.npz"; done
for s in 20261124 20261224; do test -s "outputs/stage2a/memory/left_front_uniform256_seed${s}_v1/memory.npz"; done
if tmux has-session -t "$session_name" 2>/dev/null; then
  printf '%s\n' 'The stage 2B tmux session already exists; refusing duplicate launch.' >&2
  exit 1
fi
.venv/bin/python -m unittest discover -s tests -p 'test_stage*.py' -q
test ! -e .server/stage2b_20261024.waves.log
tmux new-session -d -s "$session_name" "bash scripts/stage2b_waves.sh"
date -Is
git rev-parse HEAD
tmux list-panes -t "$session_name" -F '#{pane_pid} #{pane_current_command}'
