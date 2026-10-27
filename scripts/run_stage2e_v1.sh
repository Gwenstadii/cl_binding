#!/usr/bin/env bash
# Stage 2E method pilots (approved 2026-10-26): offline parent labels -> DPR memory -> CID / CID+ER1 / DPR4 runs.
set -euo pipefail
cd "$(dirname "$0")/.."
out=.server/stage2e_20261026
test ! -e "$out"; mkdir -p "$out"
log() { echo "$(date -Is) $*" | tee -a "$out/pipeline.log"; }
py=.venv/bin/python
alias_arg="d09ebc89fc415015efa908a281c06e7b42499227cdcb752b31e55a0e1cf78050=642f54014f818e592cd23a80545850deefe88cf6b787d4f5e9b626edee2607bc"
CUDA_VISIBLE_DEVICES=0 $py scripts/stage2_teacher_labels.py --config configs/stage2e_cid_v1.json --mode pseudo \
  --output outputs/stage2a/teacher/pseudo_right_frames_v1 > "$out/pseudo.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $py scripts/stage2_teacher_labels.py --config configs/stage2e_dpr4_v1.json --mode score \
  --output outputs/stage2a/teacher/decision_scores_v1 > "$out/score.log" 2>&1 &
wait
log "teacher labels done"
$py scripts/stage2_memory_export.py --config configs/stage2e_dpr4_v1.json > "$out/dpr_memory.log" 2>&1
log "dpr memory exported"
run() {
  $py -u scripts/stage2_run.py --config "configs/$1.json" --run-dir ".server/$2" --branches $3 --gpus $4 --scene-alias "$alias_arg" > ".server/$2.launcher.log" 2>&1
  log "$2 exit=$?"
}
set +e
run stage2e_cid_v1 stage2e_cid_20261026_01 cid 0 &
run stage2e_cid_er1_v1 stage2e_cid_er1_20261026_01 cid_er 1 &
run stage2e_dpr4_v1 stage2e_dpr4_20261026_01 er 0 &
wait
log "stage2e_done"
