"""Resume exact right-task state; select on new-task development, then paired20."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from stage1_protocol import GOALS, new_goal_development_gate
from stage1_sequential_run import compare_sequential, processor_hashes
from stage1_smoke import ROOT, dump


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=ROOT/'configs/stage1_right_continue_v1.json')
    p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args()
    plan=json.loads(a.plan.read_text())
    source_config=ROOT/plan['source_training_config']
    config=json.loads(source_config.read_text())
    train=ROOT/plan['training_run']
    start=plan['resume_step']
    parent=train/'checkpoints'/f'{start:06d}'
    assert plan['candidate_steps']==sorted(set(plan['candidate_steps']))
    assert start < plan['candidate_steps'][0] and plan['candidate_steps'][-1]==plan['maximum_step']
    assert config['old_goals']==['right'] and config['sequential']['replay'] is False
    assert json.loads((parent/'CHECKPOINT_COMPLETE.json').read_text())['status']=='pass'
    assert read_rows(train/'train_steps.jsonl')[-1]['step']==start, 'Training has already advanced; refuse duplicate continuation'
    assert not (train/'STOP_REQUESTED.json').exists(), 'Existing stop request requires inspection'
    run=a.run_dir.resolve()
    if run.exists():raise FileExistsError(run)
    run.mkdir(parents=True)
    dump(run/'plan.json',plan)
    evaluation=json.loads(source_config.read_text())
    e=evaluation['evaluation']
    e.update(development_indices=plan['development_indices'],
             development_required_correct=plan['development_required_correct'],
             qualification_indices=[i for i in range(30) if i not in plan['development_indices']],
             qualification_required_correct=8,paired_indices=plan['paired_indices'],reserve_indices=[])
    evaluation_path=run/'evaluation_config.json'
    dump(evaluation_path,evaluation)
    report={'status':'running','purpose':plan['purpose'],'source_commit':subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'resume_checkpoint':str(parent),'training_run':str(train),'training_config_unchanged':True,
        'optimizer_scheduler_rng_resume_requested':True,'learning_rate_changed':False,'normalization_changed':False,
        'selection_rule':'First fixed candidate passing >=16/20 ever AND terminal right development successes; paired results never select checkpoints. If none passes, evaluate fixed 18000 endpoint as unqualified.',
        'paired_set_previously_used':True,'candidate_reports':[],'phases':[]}
    parent_hashes=processor_hashes(parent)
    parent_model_sha256=hashlib.sha256((parent/'model.safetensors').read_bytes()).hexdigest()
    report['resume_model_sha256']=parent_model_sha256
    started=time.monotonic()
    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','MUJOCO_GL':'egl',
         'LIBERO_CONFIG_PATH':str(ROOT/'.server/libero'),'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4'}
    workers=[];handles=[]

    def save():dump(run/'run_summary.json',report)

    def launch(name,script,arguments,gpu):
        command=[sys.executable,'-u',str(ROOT/'scripts'/script),*map(str,arguments)]
        handle=(run/f'{name}.log').open('w');handles.append(handle)
        worker=subprocess.Popen(command,cwd=ROOT,env={**env,'CUDA_VISIBLE_DEVICES':str(gpu)},
                                stdout=handle,stderr=subprocess.STDOUT)
        workers.append(worker)
        report['phases'].append({'name':name,'command':command,'gpu':gpu,'pid':worker.pid,
                                'started_elapsed_s':time.monotonic()-started})
        save()
        return worker

    def finish(name,worker,timeout):
        rc=worker.wait(timeout=timeout)
        phase=next(p for p in report['phases'] if p['name']==name)
        phase.update(returncode=rc,finished_elapsed_s=time.monotonic()-started)
        save()
        if rc:raise RuntimeError(f'{name} failed with exit {rc}; inspect log')

    def develop(step):
        name=f'dev_{step:06d}'
        worker=launch(name,'stage1_r0_eval.py',['evaluate','--config',evaluation_path,
            '--checkpoint',train/'checkpoints'/f'{step:06d}','--partition','development',
            '--instruction-goals','right','--output',run/name,'--trace'],1)
        finish(name,worker,15*60)
        return new_goal_development_gate(read_rows(run/name/'episodes.jsonl'),'right',20,
                                         plan['development_required_correct'])

    try:
        save()
        trainer=launch('train','stage1_r0_train.py',['--config',source_config,'--run-dir',train,
            '--resume',parent,'--max-steps',plan['maximum_step'],'--stop-file',run/'STOP_REQUESTED.json'],0)
        report['baseline_6000_development']=develop(start)
        restored=json.loads((train/f'resume_{start:06d}.json').read_text())
        assert restored['status']=='pass' and restored['restored_step']==start
        report['optimizer_scheduler_rng_restored']=True
        report['resume_verification']=restored
        save()
        selected=None
        for step in plan['candidate_steps']:
            checkpoint=train/'checkpoints'/f'{step:06d}'
            while not (checkpoint/'CHECKPOINT_COMPLETE.json').exists():
                if trainer.poll() is not None:raise RuntimeError(f'Training stopped before {step}; inspect train log')
                if time.monotonic()-started>100*60:raise TimeoutError('Continuation training exceeded 100 minutes')
                time.sleep(5)
            score={'step':step,**develop(step)}
            report['candidate_reports'].append(score)
            save()
            if score['passed']:
                selected=checkpoint
                if trainer.poll() is None:dump(run/'STOP_REQUESTED.json',{'selected_step':step,'reason':'right_development_gate_passed'})
                break
        finish('train',trainer,5*60)
        if selected is None:selected=train/'checkpoints'/f"{plan['maximum_step']:06d}"
        report['selected_checkpoint']=str(selected)
        report['selected_step']=int(selected.name)
        report['development_gate_passed']=any(row['passed'] for row in report['candidate_reports'])
        assert processor_hashes(selected)==parent_hashes, 'Normalization changed'
        assert hashlib.sha256((parent/'model.safetensors').read_bytes()).hexdigest()==parent_model_sha256
        report['normalization_verified_unchanged']=True
        save()
        paired=[]
        for gpu,goals in enumerate((('front','left'),('right','back'))):
            name=f'paired20_part{gpu}'
            worker=launch(name,'stage1_r0_eval.py',['evaluate','--config',evaluation_path,
                '--checkpoint',selected,'--partition','paired','--instruction-goals',*goals,
                '--output',run/name,'--trace'],gpu)
            paired.append((name,worker))
        after=[]
        for name,worker in paired:
            finish(name,worker,30*60)
            after.extend(read_rows(run/name/'episodes.jsonl'))
        assert len(after)==80
        (run/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in after))
        left=compare_sequential(read_rows(ROOT/plan['original_left_baseline']),after,'left','right')
        prior=compare_sequential(read_rows(ROOT/plan['right_6000_baseline']),after,'left','right')
        dump(run/'comparison_to_original_left.json',left)
        dump(run/'comparison_to_right6000.json',prior)
        report.update(status='complete',right_paired_counts=left['after']['task_success_counts']['right'],
            left_paired_counts=left['after']['task_success_counts']['left'],
            result='development_qualified_continuation' if report['development_gate_passed'] else 'continuation_budget_exhausted_without_development_gate',
            finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-started)
        save()
        dump(run/'COMPLETE.json',{'status':'complete','result':report['result'],'selected_step':report['selected_step']})
    except BaseException as error:
        report.update(status='failed',error=str(error),seconds=time.monotonic()-started)
        save()
        raise
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
                try:worker.wait(timeout=30)
                except subprocess.TimeoutExpired:worker.kill();worker.wait()
        for handle in handles:handle.close()


if __name__=='__main__':main()
