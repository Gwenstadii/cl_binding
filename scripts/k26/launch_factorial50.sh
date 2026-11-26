#!/usr/bin/env bash
set -euo pipefail
source /home/lzx/cl_binding/.server/k1_screen_20261101_01/code/env.sh
run=/home/lzx/cl_binding/.server/k25_factorial50_20261126_01
cd "$run"
exec 9>launch.lock
flock -n 9 || { echo 'Another launch owns this run'; exit 1; }
for ((i=0;i<200;i++)); do
    [[ -f protocol.json ]] && break
    if compgen -G 'FAILED_prepare*' >/dev/null; then
        echo 'Preparation failed; refusing to evaluate'
        exit 1
    fi
    sleep 3
done
[[ -f protocol.json ]]
if [[ ! -f reuse_manifest.json ]]; then
    "$PY" -u factorial50.py reuse
fi
exec "$PY" -u factorial50.py controller
