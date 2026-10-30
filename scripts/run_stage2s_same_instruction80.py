"""Detached sequential controller: engineering -> evaluation -> audit -> integrity."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from audit_stage2s_trajectories import info

def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def command(args,cwd=None):return subprocess.check_output(args,cwd=cwd,text=True).strip()

def main(a):
    cfg=json.loads(a.config.read_text());out=Path(cfg['output_root']);runtime=Path(cfg['runtime_root']);code=a.code_dir
    if (out/'launcher.json').exists():raise FileExistsError('Already dispatched')
    for key in ('teacher','student'):assert info(Path(cfg[key])/'model.safetensors')['sha256']==cfg[key+'_model_sha256']
    for p,h in cfg['expected_model_sources'].items():assert info(Path(p))['sha256']==h
    gpu=command(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
    entry=next([int(x.strip()) for x in line.split(',')] for line in gpu.splitlines() if int(line.split(',')[0])==cfg['gpu'])
    assert entry[1]<1536 and entry[2]<=5,'GPU occupied; no preemption'
    assert shutil.disk_usage(out).free>=cfg['minimum_disk_free_bytes']
    report={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'registered_commit':a.registered_commit,
            'configuration':cfg,'runtime_commit':command(['git','rev-parse','HEAD'],runtime),
            'runtime_dirty_before':command(['git','status','--short'],runtime).splitlines(),'gpu_before':gpu,
            'code_files':{p.name:info(p) for p in code.glob('*.py')},'phases':[]}
    write(out/'registered_config.json',cfg);write(out/'launcher.json',report)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(cfg['gpu']),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'MUJOCO_GL':'egl','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4',
         'LIBERO_CONFIG_PATH':str(runtime/'.server/libero'),'PYTHONPATH':os.pathsep.join((str(code),str(runtime/'scripts')))}
    started=time.monotonic()
    def run(name,cmd,limit):
        remaining=cfg['total_deadline_seconds']-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('Overall budget exhausted')
        row={'name':name,'status':'running','command':cmd,'started_at':datetime.now(timezone.utc).isoformat()}
        report['phases'].append(row);write(out/'launcher.json',report);t=time.monotonic()
        with (out/(name+'.log')).open('w') as f:
            p=subprocess.run(cmd,cwd=runtime,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=min(limit,remaining))
        row.update(status='complete' if p.returncode==0 else 'failed',returncode=p.returncode,seconds=time.monotonic()-t)
        write(out/'launcher.json',report)
        if p.returncode:raise RuntimeError(name+' failed; retained without retry')
    try:
        python=str(runtime/'.venv/bin/python')
        base=[python,'-u',str(code/'stage2s_same_instruction80.py'),'--config',str(a.config),'--runtime-scripts',str(runtime/'scripts')]
        for phase in ('engineering','evaluation'):
            run(phase,base+['--phase',phase,'--output',str(out/phase)],600 if phase=='engineering' else 3600)
            assert (out/phase/'COMPLETE.json').exists()
        run('audit',[python,str(code/'audit_stage2s_trajectories.py'),'--root',str(out)],900)
        after={'models':{k:info(Path(cfg[k])/'model.safetensors')['sha256'] for k in ('teacher','student')},
               'sources':{p:info(Path(p))['sha256'] for p in cfg['expected_model_sources']},
               'runtime_commit':command(['git','rev-parse','HEAD'],runtime),
               'runtime_dirty':command(['git','status','--short'],runtime).splitlines(),
               'gpu_after':command(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'])}
        after['passed']=(all(after['models'][k]==cfg[k+'_model_sha256'] for k in ('teacher','student'))
            and after['sources']==cfg['expected_model_sources'] and after['runtime_commit']==report['runtime_commit']
            and after['runtime_dirty']==report['runtime_dirty_before'])
        write(out/'post_run_check.json',after);assert after['passed'],'Integrity changed'
        report['status']='complete';write(out/'COMPLETE.json',{'status':'complete','episodes':24,'report':'report.md','csv':'results.csv'})
    except Exception as e:
        report.update(status='failed',error=repr(e));write(out/'FAILED.json',report);raise
    finally:
        report.update(finished_at=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started)
        write(out/'launcher.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--code-dir',type=Path,required=True)
    p.add_argument('--registered-commit',required=True);main(p.parse_args())
