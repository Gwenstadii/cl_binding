#!/usr/bin/env bash
# Stage 2B on the same sealed parent, approved by the researcher on 2026-10-24:
#   wave 1: two adaptation replicates of FT4/ER (seeds 20261124, 20261224), one FT4 and one ER per GPU;
#   wave 2: ER budget sweep with nested memories of 64/16/4/1 records per old goal, two runs per GPU.
# Scene-XML alias: worktree symlink paths only, see .server/stage2a_scene_equivalence_20261024/equivalence.json.
set -euo pipefail
cd "$(dirname "$0")/.."
session_name=stage2b-20261024
alias_arg="d09ebc89fc415015efa908a281c06e7b42499227cdcb752b31e55a0e1cf78050=642f54014f818e592cd23a80545850deefe88cf6b787d4f5e9b626edee2607bc"
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
run() { .venv/bin/python -u scripts/stage2_run.py --config "configs/$1.json" --run-dir ".server/$2" --branches $3 --gpus $4 --scene-alias "$alias_arg" > ".server/$2.launcher.log" 2>&1; }
export -f run; export alias_arg
tmux new-session -d -s "$session_name" bash -c '
  run stage2b_seed2_v1 stage2b_seed2_20261024_01 "ft4 er" "0 1" & run stage2b_seed3_v1 stage2b_seed3_20261024_01 "ft4 er" "1 0" & wait
  run stage2b_er_mem064_v1 stage2b_mem064_20261024_01 er 0 & run stage2b_er_mem016_v1 stage2b_mem016_20261024_01 er 1 &
  run stage2b_er_mem004_v1 stage2b_mem004_20261024_01 er 0 & run stage2b_er_mem001_v1 stage2b_mem001_20261024_01 er 1 & wait
  date -Is > .server/stage2b_20261024.waves_done'
date -Is
git rev-parse HEAD
tmux list-panes -t "$session_name" -F '#{pane_pid} #{pane_current_command}'
