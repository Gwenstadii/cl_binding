"""Explicit benchmark-success follow-up at the fixed 6000-step endpoint.

The original terminal-state acquisition protocol is retained without edits.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from stage1_protocol import GOALS, instruction_response_summary


def metric_summary(rows, trained_goal):
    terminal=instruction_response_summary(rows,trained_goal)
    first={g:{c:0 for c in (*GOALS,'unreached','ambiguous')} for g in GOALS}
    counts={g:{'total':0,'success_ever':0,'terminal_correct':0} for g in GOALS}
    control={r['init_index'] for r in rows if r['requested_goal']==trained_goal and r['correct_ever']}
    other=[r for r in rows if r['requested_goal']!=trained_goal]
    conditional=[r for r in other if r['init_index'] in control]
    for r in rows:
        g=r['requested_goal'];counts[g]['total']+=1
        counts[g]['success_ever']+=int(r['correct_ever'])
        counts[g]['terminal_correct']+=int(r['terminal_category']=='correct')
        outcome=r['first_goal'] or ('unreached' if all(t is None for t in r['first_goal_times'].values()) else 'ambiguous')
        first[g][outcome]+=1
    def intrusion(selected):
        return {'total':len(selected),'first_trained_goal':sum(r['first_goal']==trained_goal for r in selected),
                'trained_goal_ever_without_requested_success':sum(r['first_goal_times'][trained_goal] is not None and not r['correct_ever'] for r in selected),
                'terminal_trained_goal':sum(r['terminal_goal']==trained_goal for r in selected)}
    return {'task_success_counts':counts,'first_target_matrix':first,'terminal_summary':terminal,
            'other_instruction_intrusion':intrusion(other),
            'conditioned_on_trained_success_ever':{'control_indices':sorted(control),**intrusion(conditional)}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args();run=a.run_dir.resolve()
    output=run/'benchmark_followup'
    output.mkdir()
    script=Path(__file__)
    report={'status':'waiting_for_original_protocol','fixed_endpoint_step':6000,
            'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
            'underlying_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'reason':'LIBERO reference metric accumulates success; 400-step terminal correctness is an additional retention-of-result criterion.',
            'original_protocol_unchanged':True}
    def save():
        (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    save();started=time.monotonic()
    while True:
        try:original=json.loads((run/'run_summary.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(5);continue
        if original['status']=='failed':raise RuntimeError('Original protocol failed; inspect its summary')
        if original['status']=='complete':break
        if time.monotonic()-started>90*60:raise TimeoutError('Original protocol did not finish in follow-up budget')
        time.sleep(5)
    config=json.loads((run/'config.json').read_text());goal=config['pilot']['trained_goal']
    trained=ROOT/'outputs/r0'/run.name
    checkpoint=trained/'checkpoints/006000'
    if not (checkpoint/'CHECKPOINT_COMPLETE.json').exists():
        report.update(status='complete',result='earlier_terminal_qualified_checkpoint_selected_no_fixed_6000_model')
        save();return
    dev=[json.loads(l) for l in (run/'dev_006000/episodes.jsonl').read_text().splitlines()]
    successes=sum(r['correct_ever'] for r in dev)
    report.update(development_success_ever=successes,development_total=len(dev),
                  development_terminal_correct=sum(r['terminal_category']=='correct' for r in dev))
    if successes<config['evaluation']['development_required_correct']:
        report.update(status='complete',result='fixed_6000_endpoint_not_high_even_by_accumulated_success')
        save();return
    report['status']='evaluating_fixed_endpoint';save()
    if original.get('selected_step')==6000:
        adapted=run/'adapted_paired';report['reused_original_paired_evaluation']=True
    else:
        adapted=output/'adapted_paired'
        args=[sys.executable,'-u',str(ROOT/'scripts/stage1_r0_eval.py'),'evaluate','--config',str(run/'config.json'),
              '--checkpoint',str(checkpoint),'--partition','paired','--instruction-goals',*GOALS,'--trace','--output',str(adapted)]
        report['command']=args;save()
        with (output/'evaluation.log').open('w') as log:
            subprocess.run(args,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OMP_NUM_THREADS':'4'},
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=15*60)
    read=lambda folder:[json.loads(l) for l in (folder/'episodes.jsonl').read_text().splitlines()]
    base=read(run/'base_paired');after=read(adapted)
    before={(r['init_index'],r['requested_goal']):r for r in base}
    for row in after:
        prior=before[row['init_index'],row['requested_goal']]
        assert all(row[k]==prior[k] for k in ('initial_state_sha256','scene_xml_sha256','seed'))
    report['base']=metric_summary(base,goal);report['adapted']=metric_summary(after,goal)
    c=report['adapted']['task_success_counts'][goal]
    report.update(status='complete',paired_success_ever=c['success_ever'],paired_total=c['total'],
                  high_success_by_accumulated_metric=c['success_ever']>=config['pilot']['paired_required_correct'],
                  result='fixed_endpoint_four_instruction_probe_complete',seconds=time.monotonic()-started)
    save();print(json.dumps(report),flush=True)


if __name__=='__main__':main()
