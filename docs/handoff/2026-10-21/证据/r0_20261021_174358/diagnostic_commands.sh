# Executed after bounded training completed. No optimizer updates or qualification rollouts.
cd /home/lzx/cl_binding
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u .server/r0_20261021_174358/offline_action_diagnostic.py > .server/r0_20261021_174358/offline_action_diagnostic.log 2>&1
# Observation alignment check restored demo_0 XML/state for all four goals and
# compared raw/rotated camera MAE before/after actions[0]; results are archived in
# control/observation_alignment_check.json. It made no training updates.
