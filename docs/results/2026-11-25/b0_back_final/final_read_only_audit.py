import json,gzip,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone
root=Path('/home/lzx/cl_binding/.server')
b=root/'k25_b0_back_20261125_01'
out={'audit_at_utc':datetime.now(timezone.utc).isoformat()}
out['b0_complete']=json.loads((b/'COMPLETE.json').read_text())
out['stop']=json.loads((root/'k25_back_stable_20261124_01/STOPPED_BY_USER.json').read_text())
out['no_further_training']=json.loads((b/'NO_FURTHER_TRAINING.json').read_text())
out['r1_manifest']=json.loads((root/'r1_left_right_20261104_01/train_r1/manifest.json').read_text())
out['b0_manifest']=json.loads((b/'manifest.json').read_text())
s=root/'k25_back_stable_20261124_01'
attempts=json.loads((s/'attempts.json').read_text())
out['old_stable_attempts']={'count':len(attempts),'qualified_marker_exists':(s/'QUALIFIED.json').exists(),
    'best_terminal':max(attempts,key=lambda x:x['development']['terminal']),
    'best_stable':max(attempts,key=lambda x:x['development']['stable_released_final']),
    'last':attempts[-1]}
out['trace_audits']={}
for step in [12000,16000,24000]:
    p=b/f'dev_{step}.jsonl.gz'
    with gzip.open(p,'rt') as f:rows=[json.loads(line) for line in f]
    details=[]
    for r in rows:
        t=r['trace'];ss=t['success'];on=[i for i,m in enumerate(ss) if m&8]
        d={'init':r.get('init'),'terminal':r['terminal']['back'],'ever':r['ever']['back'],
           'stable':r['stable_released_final'],'first_back_step':on[0]+1 if on else None,
           'last_back_step':on[-1]+1 if on else None,'back_frames':len(on),
           'first_back_contacts':t['contacts'][on[0]] if on else None,
           'first_back_gripper':t['actions'][on[0]][-1] if on else None,
           'last10_contacts':t['contacts'][-10:]}
        details.append(d)
    out['trace_audits'][str(step)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
       'trace_keys':list(rows[0]['trace']),'row_keys':list(rows[0]),'n':len(rows),
       'ever_but_final_failed':sum(d['ever'] and not d['terminal'] for d in details),
       'first_back_both_fingers':sum(d['first_back_contacts']==3 for d in details),
       'first_back_gripper_positive':sum(d['first_back_gripper'] is not None and d['first_back_gripper']>0 for d in details),
       'rows':details}
out['owned_training_processes']=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        argv=(p/'cmdline').read_bytes().split(b'\0')
        if any(Path(x.decode()).name in ['back_stable_train.py','back_only_first.py','back_stable_branch36.py'] for x in argv if x):
            out['owned_training_processes'].append({'pid':int(p.name),'cwd':str((p/'cwd').resolve())})
    except (OSError,UnicodeError):pass
out['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True)
dest=b/'final_read_only_audit.json';dest.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['b0_manifest','r1_manifest','no_further_training','trace_audits']},indent=2))
for step,v in out['trace_audits'].items():print(step,json.dumps({k:x for k,x in v.items() if k!='rows'}))
