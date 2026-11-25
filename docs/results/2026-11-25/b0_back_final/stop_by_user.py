import os,signal,json,time
from pathlib import Path
from datetime import datetime,timezone
root=Path('/home/lzx/cl_binding/.server')
targets={'k25_back_stable_20261124_01':'back_stable_train.py','k25_b0_back_20261125_01':'back_only_first.py','k25_back36_branch_20261125_01':'back_stable_branch36.py'}
found=[]
for item in Path('/proc').iterdir():
    if not item.name.isdigit():continue
    try:
        argv=(item/'cmdline').read_bytes().split(b'\0');cwd=(item/'cwd').resolve()
    except (OSError,RuntimeError):continue
    for folder,script in targets.items():
        if cwd==root/folder and any(Path(s.decode()).name==script for s in argv if s):
            assert b'python' in argv[0];found.append((int(item.name),folder,script))
records=[]
for pid,folder,script in found:
    path=root/folder;before=json.loads((path/'status.json').read_text())
    snapshots=sorted(p.name for p in path.iterdir() if p.is_dir() and p.name.isdigit() and (p/'READY.json').exists())
    os.kill(pid,signal.SIGTERM)
    for _ in range(25):
        state=Path(f'/proc/{pid}/status')
        if not state.exists() or '\nState:\tZ' in state.read_text():break
        time.sleep(.2)
    state=Path(f'/proc/{pid}/status');stopped=not state.exists() or '\nState:\tZ' in state.read_text();assert stopped
    record=dict(pid=pid,script=script,stopped_at_utc=datetime.now(timezone.utc).isoformat(),
        reason='user requested terminating further Back training after failure to qualify',previous_status=before,
        last_saved_checkpoint=snapshots[-1] if snapshots else None,all_saved_checkpoints_retained=True,stopped=True)
    (path/'STOPPED_BY_USER.json').write_text(json.dumps(record,indent=2)+'\n')
    (path/'status.json').write_text(json.dumps(dict(status='stopped_by_user',**record),indent=2)+'\n')
    records.append(record)
b0=root/'k25_b0_back_20261125_01';complete=json.loads((b0/'COMPLETE.json').read_text());assert not complete['qualified']
(b0/'NO_FURTHER_TRAINING.json').write_text(json.dumps(dict(requested_at_utc=datetime.now(timezone.utc).isoformat(),stopped_records=records,b0_finished_not_qualified=True),indent=2)+'\n')
print(json.dumps(dict(stopped=records,b0_complete=complete),indent=2))
