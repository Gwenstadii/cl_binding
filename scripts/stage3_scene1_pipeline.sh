#!/usr/bin/env bash
# Second task family (STUDY_SCENE1), approved 2026-10-25. Registered before any SCENE1 training or evaluation:
#   data -> predicate/reset checks -> fixed 12000-step left+front parent -> paired 4x20 parent evaluation ->
#   gate (parent ever-success >= 10/20 on both left and front) -> FT4, ER256, ER16, ER1 from that parent.
# Every step refuses to overwrite; a failed gate stops the pipeline and is reported, not retried.
set -euo pipefail
cd "$(dirname "$0")/.."
run=.server/stage3_scene1_20261025
parent_cfg=configs/stage3_scene1_dual_left_front_v1.json
parent_dir=outputs/stage3/scene1_dual_left_front_20261025_01
py=.venv/bin/python
log() { echo "$(date -Is) $*" | tee -a "$run/pipeline.log"; }
mkdir -p "$run/configs"

until test -s .server/scene1_download_20261025/manifest.json; do sleep 60; done
log "download complete"

$py scripts/stage1_r0_data.py --config "$parent_cfg" --output "$run/data_left_front" > "$run/data_left_front.log" 2>&1
$py scripts/stage1_r0_data.py --config configs/stage3_scene1_right_data_v1.json --output "$run/data_right" > "$run/data_right.log" 2>&1
log "data converted"
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 $py scripts/stage1_r0_eval.py verify --config "$parent_cfg" --output "$run/verify" > "$run/verify.log" 2>&1
log "predicate and common-reset checks passed"

CUDA_VISIBLE_DEVICES=0 $py -u scripts/stage1_r0_train.py --config "$parent_cfg" --run-dir "$parent_dir" > "$run/parent_train.log" 2>&1
test -s "$parent_dir/training_finished_012000.json"
log "parent trained"

eval_part() {
  CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1 $py -u scripts/stage1_r0_eval.py evaluate --config "$parent_cfg" \
    --checkpoint "$parent_dir/checkpoints/012000" --partition paired --instruction-goals $2 --output "$run/parent_eval_part$1" --trace \
    > "$run/parent_eval_part$1.log" 2>&1
}
eval_part 0 "left front" & eval_part 1 "right back" & wait
mkdir -p "$run/parent_eval"
cat "$run/parent_eval_part0/episodes.jsonl" "$run/parent_eval_part1/episodes.jsonl" > "$run/parent_eval/episodes.jsonl"
log "parent evaluated"

$py - "$run" "$parent_dir" <<'PY'
import hashlib, json, sys
from pathlib import Path
run, parent = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(l) for l in (run / "parent_eval/episodes.jsonl").read_text().splitlines() if l.strip()]
assert len(rows) == 80 and len({(r["init_index"], r["requested_goal"]) for r in rows}) == 80
ever = {g: sum(r["correct_ever"] for r in rows if r["requested_goal"] == g) for g in ("left", "front", "right", "back")}
terminal = {g: sum(r["terminal_category"] == "correct" for r in rows if r["requested_goal"] == g) for g in ("left", "front", "right", "back")}
passed = ever["left"] >= 10 and ever["front"] >= 10
gate = {"ever_success": ever, "terminal_correct": terminal, "rule": "left and front ever-success >= 10/20", "passed": passed}
(run / "parent_gate.json").write_text(json.dumps(gate, indent=1) + "\n")
print(json.dumps(gate))
if not passed:
    sys.exit("Parent gate failed: stop before adaptation")
sha = hashlib.sha256((parent / "checkpoints/012000/model.safetensors").read_bytes()).hexdigest()
for name in ("stage3_scene1_replay_v1", "stage3_scene1_er_mem016_v1", "stage3_scene1_er_mem001_v1"):
    config = json.loads(Path(f"configs/{name}.json").read_text())
    config["parent"]["model_sha256"] = sha
    (run / "configs" / f"{name}.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
PY
log "gate passed; configs resolved"

$py scripts/stage2_memory_export.py --config "$run/configs/stage3_scene1_replay_v1.json" > "$run/memory256.log" 2>&1
$py scripts/stage2_memory_export.py --config "$run/configs/stage3_scene1_er_mem016_v1.json" > "$run/memory016.log" 2>&1
$py scripts/stage2_memory_export.py --config "$run/configs/stage3_scene1_er_mem001_v1.json" > "$run/memory001.log" 2>&1
log "memories exported"

adapt() {
  $py -u scripts/stage2_run.py --config "$run/configs/$1.json" --run-dir ".server/$2" --branches $3 --gpus $4 > ".server/$2.launcher.log" 2>&1
  log "$2 exit=$?"
}
set +e
adapt stage3_scene1_replay_v1 stage3_scene1_ft4_er_20261025_01 "ft4 er" "0 1" &
adapt stage3_scene1_er_mem016_v1 stage3_scene1_mem016_20261025_01 er 1 &
adapt stage3_scene1_er_mem001_v1 stage3_scene1_mem001_20261025_01 er 0 &
wait
log "pipeline_done"
