#!/usr/bin/env bash
set -euo pipefail
cd /home/lzx/cl_binding
output=.server/dual_parent12000_unseen_back_20261023_01
session=dual-parent-back-20261023
test ! -e "$output"
if tmux has-session -t "$session" 2>/dev/null; then
  printf '%s\n' 'Supplement already has a tmux session; refusing duplicate launch.' >&2
  exit 1
fi
.venv/bin/python - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
run=Path('.server/dual12000_then_right_20261023_01')
report=json.loads((run/'run_summary.json').read_text())
assert report['status']=='complete'
assert json.loads((run/'COMPLETE.json').read_text())['status']=='complete'
assert report['source_commit']=='5514c303d5b516a5b565ef777972d69f5bf83480'
parent=Path(report['initial_checkpoint'])
assert hashlib.sha256((parent/'model.safetensors').read_bytes()).hexdigest()==report['initial_model_sha256']=='15df8ed2839c281ba5d55b411b43a15177013b4d12b0e5cb98baefdcd60313f7'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==report['source_commit']
free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
assert int(free.splitlines()[0].strip())>=8192
PY
tmux new-session -d -s "$session" "env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl LIBERO_CONFIG_PATH=/home/lzx/cl_binding/.server/libero TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 .venv/bin/python -u scripts/stage1_r0_eval.py evaluate --config .server/dual12000_then_right_20261023_01/config.json --checkpoint outputs/r0/dual_left_front_20261023_01/checkpoints/012000 --partition paired --instruction-goals back --output $output --trace > $output.log 2>&1"
date -Is
tmux list-panes -t "$session" -F '#{pane_pid} #{pane_current_command}'
