"""Bounded single-task acquisition followed by a fixed four-instruction pilot."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from stage1_protocol import GOALS, instruction_response_summary
from stage1_smoke import ROOT, dump


def read_rows(folder):
    return [json.loads(line) for line in (folder/'episodes.jsonl').read_text().splitlines()]


def compare_responses(base, adapted, goal):
    before={(r['init_index'],r['requested_goal']):r for r in base}
    after={(r['init_index'],r['requested_goal']):r for r in adapted}
    if before.keys()!=after.keys():raise ValueError('Pre/post evaluation conditions differ')
    for key, row in after.items():
        for field in ('initial_state_sha256','scene_xml_sha256','seed'):
            if row[field]!=before[key][field]:raise ValueError(f'Pre/post mismatch: {key} {field}')
    transitions={}
    for requested in GOALS:
        if requested==goal:continue
        prior=[key for key,row in before.items() if key[1]==requested and row['terminal_category']=='correct']
        transitions[requested]={'base_correct':len(prior),
            'base_correct_to_correct':sum(after[k]['terminal_category']=='correct' for k in prior),
            'base_correct_to_trained_goal':sum(after[k]['terminal_goal']==goal for k in prior),
            'base_correct_to_unfinished':sum(after[k]['terminal_category']=='unfinished' for k in prior),
            'base_correct_to_ambiguous':sum(after[k]['terminal_category']=='ambiguous' for k in prior),
            'base_correct_to_other_wrong':sum(after[k]['terminal_category']=='wrong_goal' and after[k]['terminal_goal']!=goal for k in prior)}
    return {'base':instruction_response_summary(base,goal),'adapted':instruction_response_summary(adapted,goal),
            'base_correct_transitions':transitions,'matched_pairs':len(after),
            'interpretation':'Other instructions are unsupervised in this adaptation. Trained-goal intrusion alone is not forgetting of an acquired skill.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/stage1_single_left_v1.json')
    p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args()
    config=json.loads(a.config.read_text())
    goal=config['pilot']['trained_goal']
    candidates=config['pilot']['candidate_steps']
    assert config['old_goals']==[goal] and goal in GOALS
    assert candidates==sorted(set(candidates)) and candidates[-1]==config['training']['max_steps']
    assert all(s%config['training']['save_every']==0 for s in candidates)
    run=a.run_dir.resolve()
    if run.exists():raise FileExistsError(run)
    run.mkdir(parents=True)
    train=ROOT/'outputs/r0'/run.name
    if train.exists():raise FileExistsError(train)
    cfg=run/'config.json'
    dump(cfg,config)
    report={'status':'running','purpose':config['purpose'],'trained_goal':goal,'run_dir':str(run),
            'training_dir':str(train),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate_reports':[],
            'selection_rule':'First of the fixed candidate checkpoints to reach the trained-goal development threshold; no other instruction or paired score is used for selection.',
            'qualification_partition_used':False,'phases':[]}
    started=time.monotonic()
    base_env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','MUJOCO_GL':'egl',
              'LIBERO_CONFIG_PATH':str(ROOT/'.server/libero'),'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4'}
    processes=[];handles=[]

    def save():
        dump(run/'run_summary.json',report)

    def command(script,*args):
        return [sys.executable,'-u',str(ROOT/'scripts'/script),*map(str,args)]

    def launch(name,args,gpu=None):
        env=dict(base_env)
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        output=(run/f'{name}.log').open('w');handles.append(output)
        worker=subprocess.Popen(args,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
        processes.append(worker)
        report['phases'].append({'name':name,'command':args,'gpu':gpu,'pid':worker.pid,'started_elapsed_s':time.monotonic()-started})
        save()
        return worker

    def finish(name,worker,timeout):
        rc=worker.wait(timeout=timeout)
        phase=next(r for r in report['phases'] if r['name']==name)
        phase.update(returncode=rc,finished_elapsed_s=time.monotonic()-started)
        save()
        if rc:raise RuntimeError(f'{name} failed with exit {rc}; inspect its log')

    def evaluate(name,checkpoint=None,partition='development',all_goals=False):
        args=command('stage1_r0_eval.py','evaluate','--config',cfg,'--partition',partition,'--output',run/name,'--trace')
        args+=['--checkpoint',str(checkpoint)] if checkpoint else ['--base-policy']
        if all_goals:args+=['--instruction-goals',*GOALS]
        return launch(name,args,gpu=1)

    try:
        save()
        data=launch('data',command('stage1_r0_data.py','--config',cfg,'--output',run/'data'))
        finish('data',data,20*60)
        verify=launch('alignment_verify',command('stage1_alignment_verify.py','--audit',run/'data/audit.json','--output',run/'alignment_verification.json'))
        finish('alignment_verify',verify,5*60)
        trainer=launch('train',command('stage1_r0_train.py','--config',cfg,'--run-dir',train),gpu=0)
        baseline=evaluate('base_paired',partition='paired',all_goals=True)
        selected=None
        for step in candidates:
            checkpoint=train/'checkpoints'/f'{step:06d}'
            while not (checkpoint/'CHECKPOINT_COMPLETE.json').exists():
                if trainer.poll() is not None:raise RuntimeError(f'Training ended before checkpoint {step}; exit {trainer.returncode}')
                if time.monotonic()-started>90*60:raise TimeoutError('Single-task acquisition exceeded 90-minute run budget')
                time.sleep(5)
            if 'baseline_complete' not in report:
                finish('base_paired',baseline,15*60)
                report['baseline_complete']=True
                save()
            name=f'dev_{step:06d}'
            worker=evaluate(name,checkpoint=checkpoint)
            finish(name,worker,10*60)
            scores=json.loads((run/name/'evaluation.json').read_text())
            report['candidate_reports'].append({'step':step,'counts':scores['counts'],'passed_gate':scores['passed_gate']})
            save()
            print(json.dumps(report['candidate_reports'][-1]),flush=True)
            if scores['passed_gate']:
                selected=checkpoint
                report['selected_checkpoint']=str(selected)
                report['selected_step']=step
                # Stop future updates. The selected, already verified checkpoint
                # stays immutable even if training has progressed past it.
                if trainer.poll() is None:dump(train/'STOP_REQUESTED.json',{'reason':'single_task_development_gate_passed','selected_step':step})
                break
        finish('train',trainer,5*60)
        if selected:
            paired=evaluate('adapted_paired',checkpoint=selected,partition='paired',all_goals=True)
            finish('adapted_paired',paired,15*60)
            comparison=compare_responses(read_rows(run/'base_paired'),read_rows(run/'adapted_paired'),goal)
            dump(run/'instruction_comparison.json',comparison)
            report['paired_trained_goal_correct']=comparison['adapted']['trained_instruction_correct']
            report['paired_trained_goal_total']=comparison['adapted']['initial_states']
            report['high_success_on_paired']=report['paired_trained_goal_correct']>=config['pilot']['paired_required_correct']
            report['result']='high_success_single_task_pilot' if report['high_success_on_paired'] else 'development_passed_but_paired_skill_not_high'
            report['comparison']=comparison
        else:
            report['result']='single_task_development_gate_not_reached'
            report['high_success_on_paired']=False
        report.update(status='complete',seconds=time.monotonic()-started,
                      finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        dump(run/'COMPLETE.json',{'status':'complete','result':report['result'],'seconds':report['seconds']})
        print(json.dumps({'status':'complete','result':report['result'],'run_dir':str(run)}),flush=True)
    except BaseException as error:
        report.update(status='failed',error=str(error),seconds=time.monotonic()-started)
        save()
        raise
    finally:
        for worker in processes:
            if worker.poll() is None:
                worker.terminate()
                try:worker.wait(timeout=30)
                except subprocess.TimeoutExpired:worker.kill();worker.wait()
        for handle in handles:handle.close()


if __name__=='__main__':
    main()
