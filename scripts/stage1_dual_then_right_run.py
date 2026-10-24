"""Explore right-only adaptation from the explicitly selected joint 12000 model."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from stage1_dual_goal_run import dump, read_rows, sha256

ROOT=Path(__file__).resolve().parents[1]
REQUESTS=('left','front','right')
TARGETS=('front','left','right','back')


def validate_config(config):
    s=config['sequential'];t=config['training'];e=config['evaluation']
    if config['old_goals']!=['right'] or s['retained_goals']!=['left','front'] or s['new_goal']!='right':
        raise ValueError('Expected two retained goals and right-only adaptation')
    if s['evaluation_goals']!=list(REQUESTS) or e['paired_indices']!=list(range(20,40)):
        raise ValueError('Each of the three instructions needs the fixed twenty paired states')
    if t['max_steps']!=s['fixed_endpoint_step'] or t['max_steps']!=12000 or t['samples_per_goal']!=6:
        raise ValueError('Expected exactly12000 right-only updates with batch6')
    if s['replay'] is not False or s['distillation'] is not False:
        raise ValueError('This is the no-replay/no-distillation exploratory branch')
    if config['data']['normalization_source']!='datasets/stage1_dual_left_front_train_aligned_v1':
        raise ValueError('Joint old-training normalization must be inherited')
    if config['data']['action_alignment']!='post_action_obs_to_next_action' or e['horizon']!=400:
        raise ValueError('Data alignment and400-step evaluation must stay fixed')
    if s['evaluation_attempts'] not in (1,2):raise ValueError('Runtime retries must be bounded')
    if s['initial_checkpoint']!='outputs/r0/dual_left_front_20261023_01/checkpoints/012000':
        raise ValueError('Use the user-selected12000 model, not the18000 endpoint')


def summarize_requests(rows, indices, seed):
    expected={(i,g) for i in indices for g in REQUESTS}
    keyed={(r['init_index'],r['requested_goal']):r for r in rows}
    if len(set(indices))!=len(indices) or len(rows)!=len(expected) or set(keyed)!=expected:
        raise ValueError('Missing, repeated or unexpected evaluation conditions')
    for i in indices:
        for field in ('seed','initial_state_sha256','scene_xml_sha256','steps'):
            if len({keyed[i,g][field] for g in REQUESTS})!=1:raise ValueError(f'Unpaired reset: {i} {field}')
        if keyed[i,'left']['seed']!=seed+i or keyed[i,'left']['steps']!=400:
            raise ValueError('Reset seed or horizon differs from the locked protocol')
    if len({keyed[i,'left']['initial_state_sha256'] for i in indices})!=len(indices):
        raise ValueError('Repeated physical initial state under different indices')
    if len({r['scene_xml_sha256'] for r in rows})!=1:raise ValueError('Scene XML changed')
    counts={g:{'total':len(indices),'success_ever':0,'terminal_correct':0,'correct_then_destroyed':0} for g in REQUESTS}
    matrix={g:{t:0 for t in (*TARGETS,'unfinished','ambiguous')} for g in REQUESTS}
    for row in rows:
        g=row['requested_goal'];category=row['terminal_category'];target=row['terminal_goal']
        if category not in ('correct','wrong_goal','unfinished','ambiguous'):raise ValueError('Unknown terminal category')
        if row['correct_ever']!=(row['first_goal_times'][g] is not None):raise ValueError('Inconsistent ever-success')
        if category=='correct' and (target!=g or not row['correct_ever']):raise ValueError('Invalid correct classification')
        if category=='wrong_goal' and (target not in TARGETS or target==g):raise ValueError('Invalid wrong target')
        if category in ('unfinished','ambiguous') and target is not None:raise ValueError('Unexpected unique terminal target')
        counts[g]['success_ever']+=int(row['correct_ever'])
        counts[g]['terminal_correct']+=int(category=='correct')
        counts[g]['correct_then_destroyed']+=int(row['correct_then_destroyed'])
        matrix[g][target if target is not None else category]+=1
    joint=[i for i in indices if all(keyed[i,g]['terminal_category']=='correct' for g in ('left','front'))]
    return {'episodes':len(rows),'initial_states':len(indices),'indices':list(indices),
            'task_success_counts':counts,'terminal_matrix':matrix,
            'joint_old_terminal_correct':len(joint),'joint_old_terminal_correct_indices':joint,
            'scope':'Exploratory checkpoint selected after development review; not a formally qualified R0.'}


def compare_snapshots(before, after, indices, seed):
    pre_summary=summarize_requests(before,indices,seed);post_summary=summarize_requests(after,indices,seed)
    pre={(r['init_index'],r['requested_goal']):r for r in before}
    post={(r['init_index'],r['requested_goal']):r for r in after}
    for key in pre:
        for field in ('initial_state_sha256','scene_xml_sha256','seed','steps'):
            if pre[key][field]!=post[key][field]:raise ValueError(f'Before/after mismatch: {key} {field}')
    old={}
    for goal in ('left','front'):
        old[goal]={}
        for label,predicate in (('previously_successful_ever',lambda r:r['correct_ever']),
                                ('previously_terminal_correct',lambda r:r['terminal_category']=='correct')):
            eligible=[i for i in indices if predicate(pre[i,goal])]
            lost=[post[i,goal] for i in eligible if not predicate(post[i,goal])]
            old[goal][label]={'total':len(eligible),'retained':len(eligible)-len(lost),'lost':len(lost),
                'lost_and_first_new_right':sum(r['first_goal']=='right' for r in lost),
                'lost_and_terminal_new_right':sum(r['terminal_goal']=='right' for r in lost),
                'lost_and_terminal_unfinished':sum(r['terminal_category']=='unfinished' for r in lost),
                'lost_and_terminal_ambiguous':sum(r['terminal_category']=='ambiguous' for r in lost),
                'lost_and_terminal_other_wrong':sum(r['terminal_category']=='wrong_goal' and r['terminal_goal']!='right' for r in lost)}
    joint=pre_summary['joint_old_terminal_correct_indices']
    joint_retained=[i for i in joint if all(post[i,g]['terminal_category']=='correct' for g in ('left','front'))]
    return {'before':pre_summary,'after':post_summary,'matched_pairs':len(pre),
            'old_goal_transitions':old,'previously_joint_correct_states':{
                'total':len(joint),'retained_joint_terminal_correct':len(joint_retained),
                'both_old_instructions_terminal_new_right':sum(all(post[i,g]['terminal_goal']=='right' for g in ('left','front')) for i in joint)},
            'interpretation':'Report each old skill on its actual pre-success denominator. Wrong-new-target, late result destruction and unfinished are distinct. This exploratory branch alone does not prove an internal causal pathway.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/stage1_dual12000_then_right_v1.json')
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text());validate_config(config)
    from stage1_sequential_run import processor_hashes
    s=config['sequential'];parent=ROOT/s['initial_checkpoint'];run=args.run_dir.resolve()
    train=ROOT/'outputs/sequential'/run.name
    if run.exists() or train.exists():raise FileExistsError('Use new run directories; no implicit restart')
    parent_sha=sha256(parent/'model.safetensors')
    if parent_sha!=s['initial_model_sha256']:raise ValueError('Wrong parent model')
    parent_marker=json.loads((parent/'CHECKPOINT_COMPLETE.json').read_text())
    if parent_marker['status']!='pass' or parent_marker['step']!=12000:raise ValueError('Parent is not verified12000')
    inherited=processor_hashes(parent)
    if len(inherited)!=2:raise ValueError('Parent must have both saved normalization processors')
    run.mkdir(parents=True);cfg=run/'config.json';dump(cfg,config)
    report={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'initial_checkpoint':str(parent),'initial_model_sha256':parent_sha,'training_dir':str(train),
        'starting_point_status':s['starting_point_status'],'selection_disclosure':s['selection_disclosure'],
        'normalization_source':config['data']['normalization_source'],'fixed_training_steps':12000,
        'evaluation_goals':list(REQUESTS),'episodes_each_before_and_after':60,'phases':[]}
    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','MUJOCO_GL':'egl',
         'LIBERO_CONFIG_PATH':str(ROOT/'.server/libero'),'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4'}
    started=time.monotonic();workers=[];handles=[]
    def save():dump(run/'run_summary.json',report)
    def launch(name,script,arguments,gpu=None):
        command=[sys.executable,'-u',str(ROOT/'scripts'/script),*map(str,arguments)]
        handle=(run/f'{name}.log').open('w');handles.append(handle)
        child_env=dict(env)
        if gpu is not None:child_env['CUDA_VISIBLE_DEVICES']=str(gpu)
        worker=subprocess.Popen(command,cwd=ROOT,env=child_env,stdout=handle,stderr=subprocess.STDOUT)
        workers.append(worker);report['phases'].append({'name':name,'command':command,'gpu':gpu,'pid':worker.pid,
                                                       'started_elapsed_s':time.monotonic()-started});save();return worker
    def finish(name,worker,timeout):
        code=worker.wait(timeout=timeout);phase=next(p for p in report['phases'] if p['name']==name)
        phase.update(returncode=code,finished_elapsed_s=time.monotonic()-started);save()
        if code:raise RuntimeError(f'{name} failed with exit{code}')
    def stop(worker):
        if worker.poll() is None:
            worker.terminate()
            try:worker.wait(timeout=30)
            except subprocess.TimeoutExpired:worker.kill();worker.wait()
    def evaluate(label,checkpoint,parts):
        for attempt in range(1,s['evaluation_attempts']+1):
            running=[]
            for gpu,goals in parts:
                name=f'{label}_attempt{attempt}_part{gpu}';output=run/name
                worker=launch(name,'stage1_r0_eval.py',['evaluate','--config',cfg,'--checkpoint',checkpoint,
                    '--partition','paired','--instruction-goals',*goals,'--output',output,'--trace'],gpu)
                running.append((name,worker,output))
            try:
                for name,worker,_ in running:finish(name,worker,40*60)
            except (RuntimeError,subprocess.TimeoutExpired) as error:
                for _,worker,_ in running:stop(worker)
                report.setdefault('evaluation_runtime_failures',[]).append({'label':label,'attempt':attempt,'error':str(error)});save()
                if attempt==s['evaluation_attempts']:raise
                continue
            rows=[row for _,_,folder in running for row in read_rows(folder/'episodes.jsonl')]
            summary=summarize_requests(rows,config['evaluation']['paired_indices'],config['seed'])
            folder=run/label;folder.mkdir();(folder/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            dump(folder/'summary.json',summary);return rows,summary
        raise RuntimeError('Evaluation attempts exhausted')
    try:
        save();dump(run/'parent_checkpoint_verification.json',parent_marker)
        worker=launch('data','stage1_r0_data.py',['--config',cfg,'--output',run/'data']);finish('data',worker,30*60)
        worker=launch('alignment','stage1_alignment_verify.py',['--audit',run/'data/audit.json','--output',run/'alignment_verification.json']);finish('alignment',worker,10*60)
        report['current_stage']='training_right_and_evaluating_frozen_parent';save()
        trainer=launch('train','stage1_r0_train.py',['--config',cfg,'--run-dir',train,'--init-checkpoint',parent],0)
        before,pre_summary=evaluate('before',parent,[(1,REQUESTS)])
        report['before']=pre_summary;save();finish('train',trainer,3*3600+10*60)
        final=train/'checkpoints/012000';finished=json.loads((train/'training_finished_012000.json').read_text())
        marker=json.loads((final/'CHECKPOINT_COMPLETE.json').read_text());manifest=json.loads((train/'manifest.json').read_text())
        if finished['step']!=finished['target'] or finished['step']!=12000 or not finished['frozen_parameters_unchanged']:
            raise ValueError('Training did not finish exact12000 with frozen VLM')
        if marker['status']!='pass' or not marker['frozen_hash_unchanged']:raise ValueError('Invalid final checkpoint')
        if manifest['initialization_model_sha256']!=parent_sha or not manifest['optimizer_reset'] or manifest['config']!=config or manifest['git_commit']!=report['source_commit']:
            raise ValueError('Training initialization/config does not match the plan')
        trace=read_rows(train/'train_steps.jsonl')
        if [r['step'] for r in trace]!=list(range(1,12001)) or not all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in trace):
            raise ValueError('Discontinuous or nonfinite training')
        if processor_hashes(final)!=inherited or processor_hashes(parent)!=inherited or sha256(parent/'model.safetensors')!=parent_sha:
            raise ValueError('Normalization or immutable parent changed')
        report.update(current_stage='evaluating_fixed_right12000',normalization_verified_unchanged=True);save()
        after,post_summary=evaluate('after',final,[(0,('left','front')),(1,('right',))])
        comparison=compare_snapshots(before,after,config['evaluation']['paired_indices'],config['seed'])
        dump(run/'comparison.json',comparison)
        new_counts=post_summary['task_success_counts']['right']
        report.update(status='complete',current_stage='finished',finished_at=datetime.now(timezone.utc).isoformat(),
            seconds=time.monotonic()-started,final_checkpoint=str(final),final_model_sha256=sha256(final/'model.safetensors'),
            after=post_summary,comparison=comparison,accepted_evaluation_episodes=120,
            new_task_reached_descriptive_threshold=all(new_counts[k]>=s['new_goal_descriptive_success_threshold'] for k in ('success_ever','terminal_correct')),
            actual_completed_worker_episodes_including_runtime_retries=sum(len(read_rows(run/p['name']/'episodes.jsonl'))
                for p in report['phases'] if p['name'].startswith(('before_','after_')) and (run/p['name']/'episodes.jsonl').exists()))
        save()
        lines=['# 双目标12000步起点 → 右格12000步','',
            '用户指定的探索分支；起点未通过原联合资格，不改写为合格R0。每个阶段左／前／右各20回合。','',
            '| 指令 | 训练前曾成功／最终正确 | 训练后曾成功／最终正确 |','|---|---:|---:|']
        for goal in REQUESTS:
            a=pre_summary['task_success_counts'][goal];b=post_summary['task_success_counts'][goal]
            lines.append(f"| {goal} | {a['success_ever']}/20、{a['terminal_correct']}/20 | {b['success_ever']}/20、{b['terminal_correct']}/20 |")
        lines+=['','详细目标替代、未完成与原先成功分母见`comparison.json`；只做行为层诊断，不宣称内部因果通路。']
        (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        files=[run/'config.json',run/'run_summary.json',run/'report.md',run/'comparison.json',
            run/'parent_checkpoint_verification.json',run/'alignment_verification.json',run/'data/audit.json',run/'data/COMPLETE.json',
            train/'manifest.json',train/'trainable_parameters.json',train/'training_finished_012000.json',final/'CHECKPOINT_COMPLETE.json']
        for folder in run.iterdir():
            if folder.is_dir() and (folder.name.startswith(('before','after'))):
                files+=list(folder.glob('*.json'))+list(folder.glob('episodes.jsonl'))
        files=list(dict.fromkeys(files))
        def relative(path):return path.relative_to(run) if path.is_relative_to(run) else Path('training')/path.relative_to(train)
        dump(run/'evidence_manifest.json',[{'archive_path':str(relative(p)),'server_path':str(p),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in files])
        with tarfile.open(run/'evidence.tar.gz','w:gz') as archive:
            for path in files:archive.add(path,arcname=str(relative(path)))
            archive.add(run/'evidence_manifest.json',arcname='evidence_manifest.json')
        dump(run/'COMPLETE.json',{'status':'complete','fixed_new_task_steps':12000,'accepted_evaluation_episodes':120,
                                'starting_point_formally_qualified':False,'evidence_archive_ready':True})
    except BaseException as error:
        report.update(status='failed',error=str(error),seconds=time.monotonic()-started);save();raise
    finally:
        for worker in workers:stop(worker)
        for handle in handles:handle.close()


if __name__=='__main__':main()
