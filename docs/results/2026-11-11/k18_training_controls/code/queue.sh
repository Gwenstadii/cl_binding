set -euo pipefail
p=/home/lzx/cl_binding/.server/k18_training_controls_20261111_01
rt=/home/lzx/cl_binding_stage2a
g=$1
cd "$rt"
export CUDA_VISIBLE_DEVICES="$g"
export PYTHONPATH="$rt/scripts"
if test "$g" = 0; then
  jobs=(20261021:no_rot 20261121:full 20261122:no_rot 20261021:random 20261121:norm 20261122:random)
else
  jobs=(20261021:full 20261121:no_rot 20261121:random 20261021:norm 20261122:full 20261122:norm)
fi
for job in "${jobs[@]}"; do
  seed=${job%:*}
  arm=${job#*:}
  if test "$arm" = norm; then
    until test -f "$p/runs/${seed}_no_rot/COMPLETE.json"; do sleep 5; done
  fi
  if test -f "$p/runs/${seed}_${arm}/COMPLETE.json"; then continue; fi
  .venv/bin/python -u "$p/train_controls.py" --seed "$seed" --arm "$arm" > "$p/train_${seed}_${arm}.log" 2>&1
done
