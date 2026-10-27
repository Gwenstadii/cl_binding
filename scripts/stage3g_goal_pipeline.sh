#!/usr/bin/env bash
# LIBERO-Goal family (approved 2026-10-25), registered before any LIBERO-Goal training or evaluation.
# Slots: left = bowl->stove, front = bowl->top of cabinet (old); right = bowl->plate (new);
# back = open top drawer and put bowl inside (never trained).
# Constant-recipe 12000-step parent -> gate (stove and cabinet ever-success >= 10/20 on paired 20-39);
# on failure, one cosine 18000-step parent selected on development 0-19, same gate; then FT4/ER256/ER16/ER1.
set -euo pipefail
cd "$(dirname "$0")/.."
run=.server/stage3g_goal_20261025
py=.venv/bin/python
test ! -e "$run"; mkdir -p "$run/configs"
log() { echo "$(date -Is) $*" | tee -a "$run/pipeline.log"; }

until test -s .server/goal_download_20261025/manifest.json; do sleep 60; done
log "download complete"
$py scripts/stage1_r0_data.py --config configs/stage3g_goal_dual_v1.json --output "$run/data_old" > "$run/data_old.log" 2>&1
$py scripts/stage1_r0_data.py --config configs/stage3g_goal_right_data_v1.json --output "$run/data_new" > "$run/data_new.log" 2>&1
log "data converted"
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 $py scripts/stage1_r0_eval.py verify --config configs/stage3g_goal_dual_v1.json --output "$run/verify" > "$run/verify.log" 2>&1
log "predicate and common-reset checks passed"

evaluate_parent() {  # $1 config, $2 checkpoint, $3 tag
  part() {
    CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1 $py -u scripts/stage1_r0_eval.py evaluate --config "$cfg" --checkpoint "$ckpt" \
      --partition paired --instruction-goals $2 --output "$run/${tag}_eval_part$1" --trace > "$run/${tag}_eval_part$1.log" 2>&1
  }
  cfg=$1; ckpt=$2; tag=$3
  part 0 "left front" & part 1 "right back" & wait
  mkdir -p "$run/${tag}_eval"
  cat "$run/${tag}_eval_part0/episodes.jsonl" "$run/${tag}_eval_part1/episodes.jsonl" > "$run/${tag}_eval/episodes.jsonl"
}
gate() {  # $1 tag; exit 0 when passed
  $py - "$run" "$1" <<'PY'
import json, sys
from pathlib import Path
run, tag = Path(sys.argv[1]), sys.argv[2]
rows = [json.loads(l) for l in (run / f"{tag}_eval/episodes.jsonl").read_text().splitlines() if l.strip()]
assert len(rows) == 80
ever = {g: sum(r["correct_ever"] for r in rows if r["requested_goal"] == g) for g in ("left", "front", "right", "back")}
passed = ever["left"] >= 10 and ever["front"] >= 10
(run / f"{tag}_gate.json").write_text(json.dumps({"ever_success": ever, "rule": "left and front ever-success >= 10/20", "passed": passed}, indent=1) + "\n")
print(json.dumps({"tag": tag, "ever": ever, "passed": passed}))
sys.exit(0 if passed else 1)
PY
}

parent_dir=outputs/stage3/goal_dual_constant_20261025_01
CUDA_VISIBLE_DEVICES=0 $py -u scripts/stage1_r0_train.py --config configs/stage3g_goal_dual_v1.json --run-dir "$parent_dir" > "$run/parent_train.log" 2>&1
evaluate_parent configs/stage3g_goal_dual_v1.json "$parent_dir/checkpoints/012000" constant
if gate constant; then
  checkpoint="$parent_dir/checkpoints/012000"; tag=constant
  log "constant parent passed the gate"
else
  log "constant parent failed the gate; registered cosine fallback"
  cos_dir=outputs/stage3/goal_dual_cosine_20261025_01; cfg=configs/stage3g_goal_dual_cosine_v1.json
  CUDA_VISIBLE_DEVICES=0 $py -u scripts/stage1_r0_train.py --config "$cfg" --run-dir "$cos_dir" > "$run/cosine_train.log" 2>&1
  dev() {
    CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1 $py -u scripts/stage1_r0_eval.py evaluate --config "$cfg" \
      --checkpoint "$cos_dir/checkpoints/$2" --partition development --instruction-goals left front --output "$run/dev_$2" > "$run/dev_$2.log" 2>&1
  }
  dev 0 006000 & dev 1 012000 & dev 0 018000 & wait
  step=$($py - "$run" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1]); report = {}
for step in ("006000", "012000", "018000"):
    rows = [json.loads(l) for l in (run / f"dev_{step}/episodes.jsonl").read_text().splitlines() if l.strip()]
    term = {g: {r["init_index"] for r in rows if r["requested_goal"] == g and r["terminal_category"] == "correct"} for g in ("left", "front")}
    report[step] = {"left_terminal": len(term["left"]), "front_terminal": len(term["front"]), "joint_terminal": len(term["left"] & term["front"]),
                    "front_ever": sum(r["correct_ever"] for r in rows if r["requested_goal"] == "front")}
passing = [s for s, v in report.items() if min(v["left_terminal"], v["front_terminal"], v["joint_terminal"]) >= 16]
choice = passing[0] if passing else max(report, key=lambda s: (report[s]["front_ever"], s))
(run / "selection.json").write_text(json.dumps({"development": report, "passing": passing, "selected": choice}, indent=1) + "\n")
print(choice)
PY
)
  evaluate_parent "$cfg" "$cos_dir/checkpoints/$step" cosine
  gate cosine || { log "cosine fallback failed the gate; stop"; exit 1; }
  checkpoint="$cos_dir/checkpoints/$step"; tag=cosine
  log "cosine parent (step $step) passed the gate"
fi
mkdir -p "$run/parent_eval"; cp "$run/${tag}_eval/episodes.jsonl" "$run/parent_eval/episodes.jsonl"

$py - "$run" "$checkpoint" <<'PY'
import hashlib, json, sys
from pathlib import Path
run, checkpoint = Path(sys.argv[1]), sys.argv[2]
sha = hashlib.sha256((Path(checkpoint) / "model.safetensors").read_bytes()).hexdigest()
for name in ("stage3g_goal_replay_v1", "stage3g_goal_er_mem016_v1", "stage3g_goal_er_mem001_v1"):
    config = json.loads(Path(f"configs/{name}.json").read_text())
    config["parent"].update(checkpoint=checkpoint, model_sha256=sha)
    (run / "configs" / f"{name}.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
PY
log "configs resolved for $checkpoint"
$py scripts/stage2_memory_export.py --config "$run/configs/stage3g_goal_replay_v1.json" > "$run/memory256.log" 2>&1
$py scripts/stage2_memory_export.py --config "$run/configs/stage3g_goal_er_mem016_v1.json" > "$run/memory016.log" 2>&1
$py scripts/stage2_memory_export.py --config "$run/configs/stage3g_goal_er_mem001_v1.json" > "$run/memory001.log" 2>&1
adapt() {
  $py -u scripts/stage2_run.py --config "$run/configs/$1.json" --run-dir ".server/$2" --branches $3 --gpus $4 > ".server/$2.launcher.log" 2>&1
  log "$2 exit=$?"
}
set +e
adapt stage3g_goal_replay_v1 stage3g_goal_ft4_er_20261025_01 "ft4 er" "0 1" &
adapt stage3g_goal_er_mem016_v1 stage3g_goal_mem016_20261025_01 er 1 &
adapt stage3g_goal_er_mem001_v1 stage3g_goal_mem001_20261025_01 er 0 &
wait
log "pipeline_done"
