#!/usr/bin/env bash
# Option B (researcher decision 2026-10-25): retrain the STUDY_SCENE1 left+front parent with the handoff's 2B
# candidate recipe (cosine 1e-4 -> 1e-5, 18000 steps), select a candidate on development states 0-19 only,
# apply the ORIGINAL registered gate on paired 20-39, and only then run FT4/ER256/ER16/ER1.
set -euo pipefail
cd "$(dirname "$0")/.."
run=.server/stage3b_scene1_20261025
cfg=configs/stage3_scene1_dual_left_front_cosine_v1.json
parent_dir=outputs/stage3/scene1b_dual_left_front_cosine_20261025_01
py=.venv/bin/python
test ! -e "$run"; mkdir -p "$run/configs"
log() { echo "$(date -Is) $*" | tee -a "$run/pipeline.log"; }

CUDA_VISIBLE_DEVICES=1 $py -u scripts/stage1_r0_train.py --config "$cfg" --run-dir "$parent_dir" > "$run/parent_train.log" 2>&1
test -s "$parent_dir/training_finished_018000.json"
log "cosine parent trained"

dev() {
  CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1 $py -u scripts/stage1_r0_eval.py evaluate --config "$cfg" \
    --checkpoint "$parent_dir/checkpoints/$2" --partition development --instruction-goals left front --output "$run/dev_$2" > "$run/dev_$2.log" 2>&1
}
dev 0 006000 & dev 1 012000 & dev 0 018000 & wait
selected=$($py - "$run" <<'PY'
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
(run / "selection.json").write_text(json.dumps({"development": report, "passing": passing, "selected": choice,
    "rule": "first passing candidate; otherwise most front ever-successes on development, later on ties"}, indent=1) + "\n")
print(choice)
PY
)
log "selected candidate $selected"

part() {
  CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1 $py -u scripts/stage1_r0_eval.py evaluate --config "$cfg" \
    --checkpoint "$parent_dir/checkpoints/$selected" --partition paired --instruction-goals $2 --output "$run/parent_eval_part$1" --trace \
    > "$run/parent_eval_part$1.log" 2>&1
}
part 0 "left front" & part 1 "right back" & wait
mkdir -p "$run/parent_eval"
cat "$run/parent_eval_part0/episodes.jsonl" "$run/parent_eval_part1/episodes.jsonl" > "$run/parent_eval/episodes.jsonl"
log "selected parent evaluated"

$py - "$run" "$parent_dir" "$selected" <<'PY'
import hashlib, json, sys
from pathlib import Path
run, parent, step = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rows = [json.loads(l) for l in (run / "parent_eval/episodes.jsonl").read_text().splitlines() if l.strip()]
assert len(rows) == 80
ever = {g: sum(r["correct_ever"] for r in rows if r["requested_goal"] == g) for g in ("left", "front", "right", "back")}
passed = ever["left"] >= 10 and ever["front"] >= 10
(run / "parent_gate.json").write_text(json.dumps({"selected_step": step, "ever_success": ever,
    "rule": "registered original gate: left and front ever-success >= 10/20", "passed": passed}, indent=1) + "\n")
print(json.dumps({"ever": ever, "passed": passed}))
if not passed:
    sys.exit("Option B parent failed the original gate: stop before adaptation")
checkpoint = f"{parent}/checkpoints/{step}"
sha = hashlib.sha256((Path(checkpoint) / "model.safetensors").read_bytes()).hexdigest()
for name in ("stage3b_scene1_replay_v1", "stage3b_scene1_er_mem016_v1", "stage3b_scene1_er_mem001_v1"):
    config = json.loads(Path(f"configs/{name}.json").read_text())
    config["parent"].update(checkpoint=checkpoint, model_sha256=sha, status=f"option B cosine parent, development-selected step {step}")
    (run / "configs" / f"{name}.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
PY
log "original gate passed; configs resolved"
$py scripts/stage2_memory_export.py --config "$run/configs/stage3b_scene1_replay_v1.json" > "$run/memory256.log" 2>&1
$py scripts/stage2_memory_export.py --config "$run/configs/stage3b_scene1_er_mem016_v1.json" > "$run/memory016.log" 2>&1
$py scripts/stage2_memory_export.py --config "$run/configs/stage3b_scene1_er_mem001_v1.json" > "$run/memory001.log" 2>&1
adapt() {
  $py -u scripts/stage2_run.py --config "$run/configs/$1.json" --run-dir ".server/$2" --branches $3 --gpus $4 > ".server/$2.launcher.log" 2>&1
  log "$2 exit=$?"
}
set +e
adapt stage3b_scene1_replay_v1 stage3b_scene1_ft4_er_20261025_01 "ft4 er" "0 1" &
adapt stage3b_scene1_er_mem016_v1 stage3b_scene1_mem016_20261025_01 er 1 &
adapt stage3b_scene1_er_mem001_v1 stage3b_scene1_mem001_20261025_01 er 0 &
wait
log "option_b_done"
