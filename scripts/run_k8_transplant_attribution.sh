#!/usr/bin/env bash
set -e -o pipefail

run_dir="$(cd "$(dirname "$0")" && pwd)"
source /home/lzx/cl_binding/.server/k1_screen_20261101_01/code/env.sh
set -u
worker="$run_dir/k8_transplant_attribution.py"
date -Iseconds > "$run_dir/started"
sha256sum "$worker" "$run_dir/run_k8_transplant_attribution.sh" > "$run_dir/code.sha256"
CUDA_VISIBLE_DEVICES=0 "$PY" "$worker" --output "$run_dir" --gate-only > "$run_dir/gate.log" 2>&1

pids=()
for device in 0 1; do
    CUDA_VISIBLE_DEVICES="$device" "$PY" "$worker" --output "$run_dir" --worker "$device" \
        > "$run_dir/worker$device.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
date -Iseconds > "$run_dir/finished"
echo "$status" > "$run_dir/exit_code"
exit "$status"
