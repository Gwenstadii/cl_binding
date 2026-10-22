#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/cl_binding"
export R0_RUN_ID="$1"
export CONTROL="$PWD/.server/$R0_RUN_ID"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export LIBERO_CONFIG_PATH="$PWD/.server/libero"
TRAIN="outputs/r0/$R0_RUN_ID"
export TRAIN
trap 'rc=$?; echo "$rc" > "$CONTROL/exit_code.txt"; date -Is > "$CONTROL/finished_at.txt"' EXIT
exec 9>.server/r0_development.lock
flock -n 9 || { echo Another_R0_run_is_active;exit 1; }
.venv/bin/python - <<'PY'
import json
from pathlib import Path
assert json.loads(Path('.server/r0_validation_20261021/predicates_v3/predicate_checks.json').read_text())['status']=='pass'
assert Path('.server/r0_validation_20261021/pilot_exit_code.txt').read_text().strip()=='0'
PY
printf 'CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/stage1_r0_train.py --run-dir %s\n' "$TRAIN" > "$CONTROL/commands.sh"
(
 set +e
 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/stage1_r0_train.py --run-dir "$TRAIN" > "$CONTROL/train.log" 2>&1
 rc=$?
 echo "$rc" > "$CONTROL/train_exit_code.txt"
 exit "$rc"
) &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$CONTROL/training_wrapper_pid.txt"
mapfile -t POINTS < <(.venv/bin/python -c 'import json;c=json.load(open("configs/stage1_r0.json"))["training"];print(*range(c["save_every"],c["max_steps"]+1,c["save_every"]),sep="\n")')
gate() { .venv/bin/python -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))["passed_gate"] else 1)' "$1"; }
run_eval() {
 local point="$1" partition="$2"
 local cp="$TRAIN/checkpoints/$point" out="$CONTROL/eval_${point}_${partition}"
 printf 'CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/stage1_r0_eval.py evaluate --checkpoint %s --output %s --partition %s\n' "$cp" "$out" "$partition" >> "$CONTROL/commands.sh"
 CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/stage1_r0_eval.py evaluate --checkpoint "$cp" --output "$out" --partition "$partition" > "$out.log" 2>&1
}
for number in "${POINTS[@]}"; do
 printf -v point '%06d' "$number"
 checkpoint="$TRAIN/checkpoints/$point"
 while [[ ! -f "$checkpoint/CHECKPOINT_COMPLETE.json" ]]; do
  if [[ -f "$CONTROL/train_exit_code.txt" ]]; then
   [[ "$(cat "$CONTROL/train_exit_code.txt")" = 0 ]] || { tail -n 30 "$CONTROL/train.log";exit 1; }
   break 2
  fi
  sleep 10
 done
 run_eval "$point" screen
 if gate "$CONTROL/eval_${point}_screen/evaluation.json"; then
  run_eval "$point" development
  if gate "$CONTROL/eval_${point}_development/evaluation.json"; then
   run_eval "$point" qualification
   .venv/bin/python - "$checkpoint" "$CONTROL/eval_${point}_qualification/evaluation.json" <<'PY'
import json,os,sys
from pathlib import Path
q=json.loads(Path(sys.argv[2]).read_text())
record={'checkpoint':sys.argv[1],'qualification':q,'reason':'single_independent_qualification_completed'}
Path(os.environ['TRAIN'],'STOP_REQUESTED.json').write_text(json.dumps(record,indent=2)+'\n')
Path(os.environ['CONTROL'],'qualification_decision.json').write_text(json.dumps(record,indent=2)+'\n')
PY
   break
  fi
 fi
done
wait "$TRAIN_PID"
.venv/bin/python - <<'PY'
import json,os
from pathlib import Path
c=Path(os.environ['CONTROL']);train=Path(os.environ['TRAIN'])
decision=c/'qualification_decision.json'
q=json.loads(decision.read_text()) if decision.is_file() else None
report={'pipeline_status':'complete','r0_qualified':bool(q and q['qualification']['qualified_R0']),
 'run_id':os.environ['R0_RUN_ID'],'training_dir':str(train),'qualification':q,
 'checkpoints':[p.parent.name for p in sorted((train/'checkpoints').glob('*/CHECKPOINT_COMPLETE.json'))],
 'development_reports':[str(p) for p in sorted(c.glob('eval_*/evaluation.json'))],
 'scope':'R0 development only; no FT/ER continuation launched'}
(c/'result_summary.json').write_text(json.dumps(report,indent=2)+'\n')
(c/'PIPELINE_COMPLETE').write_text('R0_DEVELOPMENT_PIPELINE_COMPLETE\n')
print(json.dumps(report,indent=2),flush=True)
PY
