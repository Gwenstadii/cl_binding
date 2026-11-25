import json,hashlib
from pathlib import Path
from datetime import datetime
p=Path('/home/lzx/cl_binding/.server/k25_b0_back_hist_20261125_01')
read=lambda x:json.loads(x.read_text())
stop=read(p/'STOPPED_BY_USER.json')
assert stop['remaining_owned_training_or_evaluation_processes']==[]
primary={}
for f in sorted(p.glob('dev_*_COMPLETE.json')):
    r=read(f);assert r['n']==20
    step=int(f.name.split('_')[1]);primary[str(step)]=r
historical=read(p/'historical_matrix.json')
partial=[]
for folder in p.iterdir():
    if folder.is_dir() and (folder/'episodes.jsonl').exists() and not (folder/'evaluation.json').exists():
        rows=[json.loads(x) for x in (folder/'episodes.jsonl').read_text().splitlines()]
        partial.append(dict(folder=folder.name,completed_episodes=len(rows),expected=20,
                            excluded_from_complete_20_episode_comparisons=True))
failure=read(p/'FAILED_controller.json') if (p/'FAILED_controller.json').exists() else None
if failure:
    assert datetime.fromisoformat(failure['at_utc'])>=datetime.fromisoformat(stop['at_utc'])
    assert failure['error']=='Training ended before all scheduled checkpoints were saved'
summary=dict(status='stopped_by_user',stop=stop,primary_completed=primary,
    historical_completed={k:historical[k] for k in ['new','previous_B0','confirmations']},
    partial_evaluations=partial,not_run_checkpoint_24000=True,
    final_saved_weights_evaluated=False,
    post_stop_scheduler_marker_interpretation='Unfinished fixed budget after user cancellation, not a pre-stop model/training failure' if failure else None)
(p/'STOP_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(primary={s:{k:r[k] for k in ['terminal','ever','stable_released_final']} for s,r in primary.items()},partial=partial)))
