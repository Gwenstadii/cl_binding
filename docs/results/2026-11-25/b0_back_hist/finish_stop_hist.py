import os,signal,json,time
from pathlib import Path
from datetime import datetime,timezone
run=Path('/home/lzx/cl_binding/.server/k25_b0_back_hist_20261125_01')
names={'back_first_historical.py','back_first_historical_matrix.py','historical_stage1_r0_eval.py',
       'back_first_historical_baseline.py','watch_hist.py','historical_stage1_r0_train.py'}
def owned():
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            cwd=(p/'cwd').resolve()
        except (OSError,UnicodeError):continue
        scripts=[(cwd/Path(x)).resolve() for x in argv[1:] if x.endswith('.py')]
        if any(x.parent==run and x.name in names for x in scripts):found.append((int(p.name),argv))
    return found
stopped=owned()
for pid,argv in stopped:
    try:os.kill(pid,signal.SIGTERM)
    except ProcessLookupError:pass
time.sleep(1)
assert not owned(),owned()
record=json.loads((run/'STOPPED_BY_USER.json').read_text())
record['additional_controllers_evaluators_stopped']=[dict(pid=p,argv=a) for p,a in stopped]
record['all_owned_processes_exited_verified_at_utc']=datetime.now(timezone.utc).isoformat()
record['remaining_owned_training_or_evaluation_processes']=[]
record['post_cancellation_scheduler_markers']=[p.name for p in run.glob('FAILED_*.json')]
(run/'STOPPED_BY_USER.json').write_text(json.dumps(record,indent=2)+'\n')
(run/'status.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
for p in sorted(run.glob('dev_*_COMPLETE.json')):print(p.name,p.read_text())
if (run/'historical_matrix.json').exists():
    m=json.loads((run/'historical_matrix.json').read_text())
    print('HISTORICAL_COUNTS',json.dumps({k:{s:r['counts']['back']['correct'] for s,r in m[k].items()} for k in ['new','previous_B0','confirmations']}))
