#!/usr/bin/env bash
set -euo pipefail
source /home/lzx/cl_binding/.server/k1_screen_20261101_01/code/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2
cd /home/lzx/cl_binding/.server/k25_back_retrain_20261124_01
exec "$PY" -u train_back_qualified.py --config config.json
