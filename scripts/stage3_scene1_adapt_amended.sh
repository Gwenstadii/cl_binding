#!/usr/bin/env bash
# Option A (researcher decision 2026-10-25, AFTER seeing the parent evaluation): the registered gate
# "left and front ever-success >= 10/20" failed (front 3/20). Amended gate, disclosed as post hoc:
# "goal-directed (success + carried within 6 cm of the requested compartment) >= 10/20 for left and front",
# because the substitution test scores where old instructions send the book, not whether it is inserted.
set -euo pipefail
cd "$(dirname "$0")/.."
run=.server/stage3_scene1_20261025
parent_dir=outputs/stage3/scene1_dual_left_front_20261025_01
py=.venv/bin/python
log() { echo "$(date -Is) $*" | tee -a "$run/pipeline.log"; }
test ! -e "$run/gate_amendment.json"
cp /home/lzx/ytz_review_20261024/results/scene1_parent_phase.json "$run/parent_phase_audit.json"
$py - "$run" "$parent_dir" <<'PY'
import hashlib, json, sys
from pathlib import Path
run, parent = Path(sys.argv[1]), Path(sys.argv[2])
audit = json.loads((run / "parent_phase_audit.json").read_text())
eps = {g: audit["models"]["p0"]["episodes"][g] for g in ("left", "front")}
directed = {g: sum(e["correct_ever"] or e["category"] == "near_requested_not_satisfied" for e in v) for g, v in eps.items()}
passed = all(v >= 10 for v in directed.values())
record = {"original_gate": json.loads((run / "parent_gate.json").read_text()), "amended_rule": "goal-directed >= 10/20 for left and front",
          "goal_directed": directed, "passed": passed, "decided_by": "researcher, 2026-10-25, after seeing the parent evaluation",
          "disclosure": "post hoc amendment; report the failed registered gate alongside every SCENE1 adaptation result"}
(run / "gate_amendment.json").write_text(json.dumps(record, indent=1) + "\n")
print(json.dumps(record))
if not passed:
    sys.exit("Amended gate failed as well")
sha = hashlib.sha256((parent / "checkpoints/012000/model.safetensors").read_bytes()).hexdigest()
for name in ("stage3_scene1_replay_v1", "stage3_scene1_er_mem016_v1", "stage3_scene1_er_mem001_v1"):
    config = json.loads(Path(f"configs/{name}.json").read_text())
    config["parent"]["model_sha256"] = sha
    config["parent"]["gate"] = "amended goal-directed gate (post hoc, disclosed); registered ever-success gate failed"
    (run / "configs" / f"{name}.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
PY
log "amended gate passed; configs resolved"
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
log "option_a_done"
