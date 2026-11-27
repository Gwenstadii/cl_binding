set -euo pipefail
p=/home/lzx/cl_binding/.server/k19_retention_20261111_01
rt=/home/lzx/cl_binding_stage2a
until test -s "$p/selection.json"; do
  if ! tmux has-session -t =k19-selection 2>/dev/null; then exit 1; fi
  sleep 5
done
arm=$("$rt/.venv/bin/python" -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s["candidate"] if s["passes_gate"] or "--near-miss" in sys.argv else "")' "$p/selection.json" "$@")
if test -z "$arm"; then
  echo SCREEN_GATE_NOT_MET
  exit 0
fi
case "$arm" in freeze_kv|mu|d|both|replay|both_replay|replay_full|replay_soft) ;; *) exit 2 ;; esac
for g in 0 1; do
  seed=20261121
  if test "$g" = 1; then seed=20261122; fi
  tmux new-session -d -s "k19-confirm-train-$g" "cd $rt && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$rt/scripts .venv/bin/python -u $p/train_k19.py train --arm $arm --seed $seed >$p/train_${seed}_$arm.log 2>&1"
done
for w in 0 1 2 3 4 5; do
  g=0
  if test "$w" -ge 3; then g=1; fi
  tmux new-session -d -s "k19-confirm-$w" "cd $rt && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$rt/scripts .venv/bin/python -u $p/eval_k19.py confirm --lane $w --lanes 6 --arms $arm >$p/confirm_$w.log 2>&1"
done
tmux new-session -d -s k19-confirm-probe "cd $rt && export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$rt/scripts && .venv/bin/python -u $p/eval_k19.py probe --seed 20261121 --arms $arm && .venv/bin/python -u $p/eval_k19.py probe --seed 20261122 --arms $arm" 
echo "CONFIRM_STARTED $arm"
