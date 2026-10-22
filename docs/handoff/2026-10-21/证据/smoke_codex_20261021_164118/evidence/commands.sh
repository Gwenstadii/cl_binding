# Commands executed in /home/lzx/cl_binding
.venv/bin/python -u scripts/download_stage1.py --plan 
.venv/bin/python -u scripts/stage1_smoke.py inspect 
.venv/bin/python -u scripts/stage1_smoke.py convert --goals front left right --episodes-per-task 1 --output datasets/stage1_smoke_old 
.venv/bin/python -u scripts/stage1_smoke.py convert --goals back --episodes-per-task 1 --output datasets/stage1_smoke_n1 
env CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/stage1_smoke.py load 
env CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/stage1_smoke.py train --dataset datasets/stage1_smoke_old --run-dir outputs/smoke/train_codex_20261021_164118 --steps 5 
env CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/stage1_smoke.py rollout --checkpoint outputs/smoke/train_codex_20261021_164118/checkpoint --output outputs/smoke/rollout_codex_20261021_164118 --steps 8 
