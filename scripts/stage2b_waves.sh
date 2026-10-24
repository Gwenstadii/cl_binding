#!/usr/bin/env bash
# Two waves of stage 2B runs; started by run_stage2b_v1.sh inside tmux. Each run is a separate stage2_run.py.
set -uo pipefail
cd "$(dirname "$0")/.."
alias_arg="d09ebc89fc415015efa908a281c06e7b42499227cdcb752b31e55a0e1cf78050=642f54014f818e592cd23a80545850deefe88cf6b787d4f5e9b626edee2607bc"
run() {
  .venv/bin/python -u scripts/stage2_run.py --config "configs/$1.json" --run-dir ".server/$2" --branches $3 --gpus $4 \
    --scene-alias "$alias_arg" > ".server/$2.launcher.log" 2>&1
  echo "$(date -Is) $2 exit=$?" >> .server/stage2b_20261024.waves.log
}
run stage2b_seed2_v1 stage2b_seed2_20261024_01 "ft4 er" "0 1" &
run stage2b_seed3_v1 stage2b_seed3_20261024_01 "ft4 er" "1 0" &
wait
run stage2b_er_mem064_v1 stage2b_mem064_20261024_01 er 0 &
run stage2b_er_mem016_v1 stage2b_mem016_20261024_01 er 1 &
run stage2b_er_mem004_v1 stage2b_mem004_20261024_01 er 0 &
run stage2b_er_mem001_v1 stage2b_mem001_20261024_01 er 1 &
wait
echo "$(date -Is) waves_done" >> .server/stage2b_20261024.waves.log
