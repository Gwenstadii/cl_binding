#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/cl_binding"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export LIBERO_CONFIG_PATH="$PWD/.server/libero"
trap 'rc=$?; echo "$rc" > .server/r0_validation_20261021/exit_code.txt' EXIT
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/stage1_r0_eval.py verify --output .server/r0_validation_20261021/predicates > .server/r0_validation_20261021/predicates.log 2>&1
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/stage1_r0_train.py --run-dir outputs/r0/pilot_20261021 --max-steps 2 > .server/r0_validation_20261021/pilot.log 2>&1
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/stage1_r0_train.py --run-dir outputs/r0/pilot_20261021 --resume outputs/r0/pilot_20261021/checkpoints/000002 --max-steps 3 > .server/r0_validation_20261021/resume.log 2>&1
printf 'R0_PREREQUISITES_PASS\n' > .server/r0_validation_20261021/COMPLETE
