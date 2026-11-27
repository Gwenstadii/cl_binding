#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/cl_binding"
run="$PWD/.server/r1_then_front_v1"
parent="$PWD/.server/r1_left_right_20261104_01/train_r1/checkpoints/024000"
py="$PWD/.venv/bin/python"
mkdir "$run"
exec > >(tee "$run/run.log") 2>&1
trap 'printf "%s\n" "$?" > "$run/FAILED"' ERR
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
test -f "$parent/CHECKPOINT_COMPLETE.json"
nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]' && exit 1
"$py" - "$run" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
config = json.loads(Path('.server/r1_left_right_eval_v1.json').read_text())
config['purpose'] = 'R1-24000 front-only adaptation with development gate and paired final evaluation'
config['old_goals'] = ['front']
config['data'].update(
    train_output='datasets/stage1_single_front_train_aligned_v1',
    validation_output='datasets/stage1_single_front_validation_aligned_v1',
    normalization_source='datasets/stage1_dual_left_right_train_aligned_v1',
)
config['training'].update(samples_per_goal=6, save_every=6000, max_steps=24000)
config['evaluation'].update(development_required_correct=16, paired_indices=list(range(20, 40)))
(run / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
PY
selected=
for step in 12000 18000 24000; do
    printf -v label '%06d' "$step"
    if [[ "$step" == 12000 ]]; then
        init=(--init-checkpoint "$parent")
    else
        init=(--resume "$checkpoint")
    fi
    "$py" scripts/stage1_r0_train.py --config "$run/config.json" --run-dir "$run/train" --max-steps "$step" "${init[@]}"
    checkpoint="$run/train/checkpoints/$label"
    "$py" scripts/stage1_r0_eval.py evaluate --config "$run/config.json" --output "$run/screen_$label" --checkpoint "$checkpoint" --partition development --instruction-goals front
    if "$py" -c 'import json,sys; c=json.load(open(sys.argv[1]))["counts"]["front"]; sys.exit(not(c["total"]==20 and c["correct"]>=16))' "$run/screen_$label/evaluation.json"; then
        selected="$label"
        break
    fi
done
printf '%s\n' "${selected:-unmet}" > "$run/development_gate"
"$py" scripts/stage1_r0_eval.py evaluate --config "$run/config.json" --output "$run/final" --checkpoint "$checkpoint" --partition paired --instruction-goals left right front
"$py" scripts/stage1_r0_eval.py evaluate --config "$run/config.json" --output "$run/parent" --checkpoint "$parent" --partition paired --instruction-goals left right front
touch "$run/COMPLETE"
