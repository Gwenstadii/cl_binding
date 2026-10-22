#!/usr/bin/env bash
# Bounded corrected-alignment R0 run. Does not launch qualification or N1.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$PWD/.server/libero"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
RUN=${1:-".server/r0_aligned_v2_$(date +%Y%m%d_%H%M%S)"}
TRAIN="outputs/r0/$(basename "$RUN")"
mkdir "$RUN"
exec > "$RUN/session.log" 2>&1
trap 'rc=$?; echo "$rc" > "$RUN/exit_code.txt"; date -Iseconds > "$RUN/finished_at.txt"' EXIT
date -Iseconds > "$RUN/started_at.txt"
git rev-parse HEAD > "$RUN/source_commit.txt"
cp scripts/run_r0_aligned_v2.sh "$RUN/runner.sh"
cp configs/stage1_r0_aligned_v2.json "$RUN/config.json"
printf '%s\n' "$TRAIN" > "$RUN/training_dir.txt"

echo DATA_PREPARATION_STARTED
timeout 30m .venv/bin/python -u scripts/stage1_r0_data.py \
  --config "$RUN/config.json" --output "$RUN/data" > "$RUN/data.log" 2>&1
.venv/bin/python -u scripts/stage1_alignment_verify.py \
  --audit "$RUN/data/audit.json" --output "$RUN/alignment_verification.json" > "$RUN/alignment_verification.log" 2>&1

echo ALIGNED_TRAINING_STARTED
date -Iseconds > "$RUN/training_started_at.txt"
CUDA_VISIBLE_DEVICES=0 timeout 100m .venv/bin/python -u scripts/stage1_r0_train.py \
  --config "$RUN/config.json" --run-dir "$TRAIN" > "$RUN/train.log" 2>&1
test -f "$TRAIN/training_finished_006000.json"
date -Iseconds > "$RUN/training_finished_at.txt"

eval_one() {
    local checkpoint="$1" device="$2" rc
    set +e
    CUDA_VISIBLE_DEVICES="$device" timeout 20m .venv/bin/python -u scripts/stage1_r0_eval.py evaluate \
      --config "$RUN/config.json" --checkpoint "$TRAIN/checkpoints/$checkpoint" \
      --partition development --trace --output "$RUN/step$checkpoint" > "$RUN/step$checkpoint.log" 2>&1
    rc=$?
    echo "$rc" > "$RUN/step${checkpoint}_exit_code.txt"
    return "$rc"
}
echo DEVELOPMENT_EVALUATION_STARTED
eval_one 003000 0 &
PID3000=$!
eval_one 006000 1 &
PID6000=$!
set +e
wait "$PID3000"; RC3000=$?
wait "$PID6000"; RC6000=$?
if ((RC3000 != 0 || RC6000 != 0)); then exit 1; fi
printf 'COMPLETE\n' > "$RUN/COMPLETE"
