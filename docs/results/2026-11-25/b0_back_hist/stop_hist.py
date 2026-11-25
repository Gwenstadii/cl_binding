import os,signal,json,time
from pathlib import Path
from datetime import datetime,timezone
run=Path('/home/lzx/cl_binding/.server/k25_b0_back_hist_20261125_01')
now=lambda:datetime.now(timezone.utc).isoformat()
request=dict(at_utc=now(),reason='User explicitly requested stopping training and analyzing existing evidence')
(run/'STOP_REQUESTED_BY_USER.json').write_text(json.dumps(request,indent=2)+'\n')
stop=run/'training/STOP_REQUESTED.json';stop.write_text(json.dumps(request,indent=2)+'\n')
trainers=[];stopped=[]
names={'back_first_historical.py','back_first_historical_matrix.py','historical_stage1_r0_eval.py',
       'back_first_historical_baseline.py','watch_hist.py'}
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
    except (OSError,UnicodeError):continue
    scripts=[Path(x) for x in argv[1:] if x.endswith('.py')]
    if any(x==run/'historical_stage1_r0_train.py' for x in scripts):trainers.append(int(p.name))
    elif any(x.parent==run and x.name in names for x in scripts):
        os.kill(int(p.name),signal.SIGTERM);stopped.append(dict(pid=int(p.name),argv=argv))
def alive(pid):
    p=Path(f'/proc/{pid}/status')
    return p.exists() and '\nState:\tZ' not in p.read_text()
deadline=time.monotonic()+55
while any(alive(pid) for pid in trainers) and time.monotonic()<deadline:time.sleep(.5)
forced=[]
for pid in trainers:
    if alive(pid):os.kill(pid,signal.SIGTERM);forced.append(pid)
time.sleep(1)
assert not any(alive(pid) for pid in trainers)
with (run/'training/train_steps.jsonl').open('rb') as f:
    f.seek(0,2);f.seek(max(0,f.tell()-8192));last=json.loads(f.read().splitlines()[-1])
checkpoints=sorted(p.name for p in (run/'training/checkpoints').iterdir() if p.is_dir() and (p/'CHECKPOINT_COMPLETE.json').exists())
record=dict(**request,stopped_at_utc=now(),status='stopped_by_user',last_training_step=last['step'],
    saved_checkpoints=checkpoints,last_saved_checkpoint=checkpoints[-1],trainers=trainers,
    stopped_controllers_evaluators=stopped,forced_trainer_termination=forced,
    all_checkpoints_and_partial_evaluation_files_retained=True,
    completed_evaluation_markers=[p.name for p in sorted(run.glob('*_COMPLETE.json'))])
(run/'STOPPED_BY_USER.json').write_text(json.dumps(record,indent=2)+'\n')
(run/'status.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
