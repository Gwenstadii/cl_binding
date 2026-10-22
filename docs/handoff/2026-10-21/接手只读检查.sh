#!/usr/bin/env bash
# 在已登录的目标服务器执行；不安装、不下载、不训练、不评估、不删除。
# 若本地复制到服务器，可执行：bash /path/to/接手只读检查.sh
set -euo pipefail
hostname
whoami
date -Is
df -h /
cd /home/lzx/cl_binding
pwd
git status --short --branch
git rev-parse HEAD
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
tmux ls || true
.venv/bin/python --version
cat .server/r0_20261021_174358/result_summary.json
cat outputs/r0/r0_20261021_174358/manifest.json
cat outputs/r0/r0_20261021_174358/training_finished_006000.json
cat .server/r0_20261021_174358/eval_003000_screen/evaluation.json
cat .server/r0_20261021_174358/eval_006000_screen/evaluation.json
.venv/bin/python - <<'PY'
from pathlib import Path
import json
root=Path('/home/lzx/cl_binding')
report=json.loads((root/'.server/r0_validation_20261021/predicates_v3/predicate_checks.json').read_text())
print('OPEN_LOOP_REPLAY_FAILURES',report['open_loop_replay_failures'])
for split in ('train','validation'):
    path=root/f'datasets/stage1_r0_{split}_v1/r0_conversion.json'
    item=json.loads(path.read_text())
    print(split,'episodes',len(item['episodes']),'frames',sum(e['frames'] for e in item['episodes']))
markers=list((root/'outputs/r0/r0_20261021_174358/checkpoints').glob('*/CHECKPOINT_COMPLETE.json'))
print('COMPLETE_CHECKPOINTS',len(markers))
PY
