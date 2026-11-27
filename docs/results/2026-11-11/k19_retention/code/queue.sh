set -euo pipefail
p=/home/lzx/cl_binding/.server/k19_retention_20261111_01
rt=/home/lzx/cl_binding_stage2a
g=$1
cd "$rt"
export CUDA_VISIBLE_DEVICES="$g"
export PYTHONPATH="$rt/scripts"
until test -s "$p/teacher_READY.json"; do sleep 5; done
if test "$g" = 0; then
  arms=(both mu both_replay)
else
  arms=(replay d)
fi
for arm in "${arms[@]}"; do
  .venv/bin/python -u "$p/train_k19.py" train --arm "$arm" >"$p/train_20261021_$arm.log" 2>&1
done
