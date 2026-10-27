#!/usr/bin/env bash
# Closed-loop check of the open-loop parameter-swap screen (approved 2026-10-25). Evaluation only: each hybrid is
# tested on the four instructions x paired states 20-39 with the registered 400-step protocol; three per GPU in turn.
set -euo pipefail
cd "$(dirname "$0")/.."
out=.server/stage2c_swap_closedloop_20261025
session=stage2c-swap-20261025
test ! -e "$out"
if tmux has-session -t "$session" 2>/dev/null; then echo "session exists" >&2; exit 1; fi
mkdir -p "$out"
cat > "$out/queue.sh" <<'Q'
set -uo pipefail
out=.server/stage2c_swap_closedloop_20261025
evaluate() {
  CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1 .venv/bin/python -u scripts/stage1_r0_eval.py evaluate --config configs/stage2a_replay_v1.json \
    --checkpoint "outputs/stage2a/hybrids/$2" --partition paired --instruction-goals left front right back \
    --output "$out/$2" --trace > "$out/$2.log" 2>&1
  echo "$(date -Is) $2 exit=$?" >> "$out/queue.log"
}
( for h in parent_damage_late_expert ft4_restore_late_expert parent_damage_read_kv; do evaluate 0 $h; done ) &
( for h in ft4_restore_read_kv parent_damage_early_expert ft4_restore_early_expert; do evaluate 1 $h; done ) &
wait
echo "$(date -Is) all_done" >> "$out/queue.log"
Q
tmux new-session -d -s "$session" "bash $out/queue.sh"
date -Is; git rev-parse HEAD
