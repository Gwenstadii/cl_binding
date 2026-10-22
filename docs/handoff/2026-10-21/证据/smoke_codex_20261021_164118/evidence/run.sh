#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/cl_binding"
export SMOKE_RUN_ID="$1"
export EVIDENCE="$PWD/.server/smoke/$SMOKE_RUN_ID"
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$PWD/.server/libero"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
TRAIN_REL="outputs/smoke/train_$SMOKE_RUN_ID"
ROLLOUT_REL="outputs/smoke/rollout_$SMOKE_RUN_ID"
GPU_MONITOR_PID=""
finish() {
 local rc=$?
 trap - EXIT
 if [[ -n "$GPU_MONITOR_PID" ]]; then kill "$GPU_MONITOR_PID" 2>/dev/null || true; wait "$GPU_MONITOR_PID" 2>/dev/null || true; fi
 printf '%s\n' "$rc" > "$EVIDENCE/exit_code.txt"
 date -Is > "$EVIDENCE/finished_at.txt"
 if (( rc != 0 )); then printf 'FAILED exit=%s\n' "$rc" > "$EVIDENCE/FAILED"; fi
 exit "$rc"
}
trap finish EXIT
exec 9> .server/smoke/stage1_execution.lock
flock -n 9 || { echo "Another tracked smoke owns the lock"; exit 1; }
[[ ! -e "$TRAIN_REL" && ! -e "$ROLLOUT_REL" ]]
printf '# Commands executed in %s\n' "$PWD" > "$EVIDENCE/commands.sh"
run_stage() {
 local name="$1"; shift
 printf '%q ' "$@" >> "$EVIDENCE/commands.sh"
 printf '\n' >> "$EVIDENCE/commands.sh"
 "$@" 2>&1 | tee "$EVIDENCE/$name.log"
}
mkdir "$EVIDENCE/prior_shared_reports"
for item in resources.json smoke/model_load.json smoke/raw_inspection.json; do
 if [[ -f ".server/$item" ]]; then cp -p ".server/$item" "$EVIDENCE/prior_shared_reports/$(basename "$item")"; fi
done
git status --short > "$EVIDENCE/git_status_before.txt"
nvidia-smi > "$EVIDENCE/gpu_before.txt"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,utilization.gpu --format=csv -l 1 > "$EVIDENCE/gpu_samples.csv" &
GPU_MONITOR_PID=$!
.venv/bin/python - <<'PY'
import hashlib,json,os,platform,subprocess
from datetime import datetime
from pathlib import Path
import torch
r=Path(os.environ['EVIDENCE'])
record={
 'run_id':os.environ['SMOKE_RUN_ID'],'purpose':'independent_engineering_smoke_rerun',
 'started_at':datetime.now().astimezone().isoformat(),'host':platform.node(),
 'user':os.environ.get('USER'),'project':str(Path.cwd()),
 'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'python':platform.python_version(),'torch':torch.__version__,
 'script_sha256':hashlib.sha256(Path('scripts/stage1_smoke.py').read_bytes()).hexdigest(),
 'resources':json.loads(Path('.server/resources.json').read_text()),
 'settings':{'training_physical_gpu':0,'rollout_physical_gpu':1,'steps':5,'rollout_steps_per_task':8,'batch_size':1,'learning_rate':1e-5,'autocast':'bfloat16','num_workers':0},
 'training_dataset':'datasets/stage1_smoke_old','n1_dataset':'datasets/stage1_smoke_n1',
 'train_output':'outputs/smoke/train_'+os.environ['SMOKE_RUN_ID'],
 'rollout_output':'outputs/smoke/rollout_'+os.environ['SMOKE_RUN_ID'],
}
(r/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
PY
run_stage 00_resources .venv/bin/python -u scripts/download_stage1.py --plan
.venv/bin/python - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('.server/resources.json').read_text())
i=r['last_local_inventory']
assert i['models']['lerobot/smolvla_base']['status']=='reuse'
assert len(i['demonstrations'])==4
assert all(v['status']=='reuse' for v in i['demonstrations'].values())
assert Path(r['HuggingFaceTB/SmolVLM2-500M-Video-Instruct']['metadata_path'],'config.json').is_file()
print('RESOURCE_ASSERTIONS_PASS',flush=True)
PY
run_stage 01_inspect .venv/bin/python -u scripts/stage1_smoke.py inspect
cp .server/smoke/raw_inspection.json "$EVIDENCE/raw_inspection.json"
run_stage 02_convert_old .venv/bin/python -u scripts/stage1_smoke.py convert --goals front left right --episodes-per-task 1 --output datasets/stage1_smoke_old
run_stage 03_convert_n1 .venv/bin/python -u scripts/stage1_smoke.py convert --goals back --episodes-per-task 1 --output datasets/stage1_smoke_n1
cp datasets/stage1_smoke_old/smoke_conversion.json "$EVIDENCE/conversion_old.json"
cp datasets/stage1_smoke_n1/smoke_conversion.json "$EVIDENCE/conversion_n1.json"
run_stage 04_load env CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/stage1_smoke.py load
cp .server/smoke/model_load.json "$EVIDENCE/model_load.json"
run_stage 05_train env CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/stage1_smoke.py train --dataset datasets/stage1_smoke_old --run-dir "$TRAIN_REL" --steps 5
run_stage 06_rollout env CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/stage1_smoke.py rollout --checkpoint "$TRAIN_REL/checkpoint" --output "$ROLLOUT_REL" --steps 8
.venv/bin/python - <<'PY'
import json,math,os
from pathlib import Path
root=Path(os.environ['EVIDENCE'])
m=json.loads((root/'manifest.json').read_text())
load=json.loads((root/'model_load.json').read_text())
train=json.loads(Path(m['train_output'],'train_smoke.json').read_text())
roll=json.loads(Path(m['rollout_output'],'rollout_smoke.json').read_text())
assert load['status']=='pass' and load['strict_loading'] is True
assert train['status']=='pass' and len(train['steps'])==5
assert all(math.isfinite(s['loss']) and math.isfinite(s['gradient_norm']) and s['gradient_norm']>0 for s in train['steps'])
assert train['updated_projection_max_delta']>0
assert train['action_shape']==[1,50,7]
assert all(c['std']>0 for c in train['decoded_cameras'].values())
assert roll['status']=='pass' and len(roll['tasks'])==4
assert {t['goal'] for t in roll['tasks']}=={'front','left','right','back'}
assert all(t['status']=='pass' and t['policy_steps']==8 for t in roll['tasks'])
assert all(len(a['executed'])==7 and all(math.isfinite(x) for x in a['executed']) for t in roll['tasks'] for a in t['actions'])
summary={'status':'pass','run_id':m['run_id'],'git_commit':m['git_commit'],'strict_loading':load['strict_loading'],
 'train':train,'rollout_tasks':[{k:v for k,v in t.items() if k!='actions'} for t in roll['tasks']],
 'scope':'Engineering data/model/update/reload/short-rollout smoke only; not R0 qualification or task success evaluation.'}
(root/'completion.json').write_text(json.dumps(summary,indent=2)+'\n')
(root/'COMPLETE').write_text('ALL_SMOKE_ASSERTIONS_PASS\n')
print('ALL_SMOKE_ASSERTIONS_PASS',flush=True)
PY
nvidia-smi > "$EVIDENCE/gpu_after.txt"
git status --short > "$EVIDENCE/git_status_after.txt"
