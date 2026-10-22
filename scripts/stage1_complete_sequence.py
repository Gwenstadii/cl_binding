"""Finish front/back sequential adaptation and both paired20 evaluations unattended."""

import argparse
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from stage1_protocol import GOALS
from stage1_sequential_run import compare_sequential, processor_hashes
from stage1_single_task_benchmark_followup import metric_summary
from stage1_smoke import ROOT, dump


def validate_plan(plan):
    if [s['goal'] for s in plan['stages']] != ['front','back']:
        raise ValueError('This run must complete left -> right -> front -> back')
    if any(s['steps'] != 12000 for s in plan['stages']):
        raise ValueError('Each remaining task must receive exactly 12000 steps')
    if plan['paired_indices'] != list(range(30,50)):
        raise ValueError('Expected the existing twenty paired initial states')
    if len({s['name'] for s in plan['stages']}) != 2:
        raise ValueError('Stage names must be unique')
    for i,stage in enumerate(plan['stages']):
        if stage['exposed_goals'] != ['left','right','front','back'][:i+3]:
            raise ValueError('Incorrect exposure history')
    if plan['evaluation_attempts'] not in (1,2):
        raise ValueError('Evaluation retries must be bounded')


def summarize_snapshot(rows, exposed, current):
    summary=metric_summary(rows,current)
    if len(rows)!=80 or any(c['total']!=20 for c in summary['task_success_counts'].values()):
        raise ValueError('Every stage needs all four instructions, twenty episodes each')
    if len(set(exposed))!=len(exposed) or not set(exposed)<=set(GOALS):
        raise ValueError('Invalid task exposure history')
    counts=summary['task_success_counts']
    summary['training_exposure_goals']=exposed
    summary['training_exposure_is_not_a_mastery_claim']=True
    summary['exposed_task_aggregate']={
        'episodes':20*len(exposed),
        'success_ever':sum(counts[g]['success_ever'] for g in exposed),
        'terminal_correct':sum(counts[g]['terminal_correct'] for g in exposed)}
    return summary


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(8*1024*1024):digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=ROOT/'configs/stage1_four_task_sequence_v1.json')
    p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());validate_plan(plan)
    run=a.run_dir.resolve()
    training_root=ROOT/'outputs/sequential'/run.name
    if run.exists() or training_root.exists():raise FileExistsError('Use new experiment directories')
    parent=ROOT/plan['source_checkpoint']
    if sha256(parent/'model.safetensors')!=plan['source_model_sha256']:
        raise ValueError('The starting point must be the selected right9000 model')
    assert json.loads((parent/'CHECKPOINT_COMPLETE.json').read_text())['status']=='pass'
    base=json.loads((ROOT/plan['base_config']).read_text())
    run.mkdir(parents=True)
    dump(run/'plan.json',plan)
    snapshots={};episode_sets={};acquisition={}
    for label,exposed,current in (('left',['left'],'left'),('left_right',['left','right'],'right')):
        path=ROOT/plan['baselines'][label]
        episode_sets[label]=rows(path)
        snapshots[label]=summarize_snapshot(episode_sets[label],exposed,current)
        acquisition[current]=label
        (run/'baselines').mkdir(exist_ok=True)
        shutil.copyfile(path,run/'baselines'/f'{label}_episodes.jsonl')
    # Verify baseline conditions before starting either new training task.
    compare_sequential(episode_sets['left'],episode_sets['left_right'],'left','right')
    report={'status':'running','purpose':plan['purpose'],
            'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_checkpoint':str(parent),'source_model_sha256':plan['source_model_sha256'],
            'sequence':['left','right','front','back'],'stages':[],'phases':[],
            'selection_rule':'Fixed +12000 endpoint for each new task; continue to back regardless of front scores.',
            'cached_baseline_episodes':160,'planned_new_evaluation_episodes':160,
            'paired_set_previously_used':True}
    started=time.monotonic();workers=[];handles=[]
    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','MUJOCO_GL':'egl',
         'LIBERO_CONFIG_PATH':str(ROOT/'.server/libero'),'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4'}

    def render():
        lines=['# 四任务顺序训练验收报告','',f"运行状态：**{report['status']}**",
               '',f"源码：`{report['source_commit']}`",'',
               '顺序：左格 → 右格9000步 → 前格12000步 → 后格12000步。',
               '每个阶段均用四条指令各20回合；三任务阶段尚未训练的后格只作诊断，不计入三任务汇总。','',
               '| 模型阶段 | 左格 曾成功/最终正确 | 右格 曾成功/最终正确 | 前格 曾成功/最终正确 | 后格 曾成功/最终正确 |',
               '|---|---:|---:|---:|---:|']
        for label,snapshot in snapshots.items():
            counts=snapshot['task_success_counts']
            cells=[f"{counts[g]['success_ever']}/20、{counts[g]['terminal_correct']}/20" for g in ('left','right','front','back')]
            lines.append('| '+label+' | '+' | '.join(cells)+' |')
        for stage in report['stages']:
            if stage.get('status')!='complete':continue
            label=stage['name'];snapshot=snapshots[label]
            lines+=['',f"## {label}",'',f"固定终点：`{stage['checkpoint']}`，本任务12000步。",'',
                    '| 请求指令 | 实际前格 | 实际左格 | 实际右格 | 实际后格 | 未完成 | 歧义 |',
                    '|---|---:|---:|---:|---:|---:|---:|']
            for goal in GOALS:
                matrix=snapshot['terminal_summary']['terminal_matrix'][goal]
                lines.append('| '+goal+' | '+' | '.join(str(matrix[g]) for g in (*GOALS,'unfinished','ambiguous'))+' |')
            aggregate=snapshot['exposed_task_aggregate']
            lines+=['',f"已训练过任务汇总：曾成功 {aggregate['success_ever']}/{aggregate['episodes']}，最终正确 {aggregate['terminal_correct']}/{aggregate['episodes']}。这不代表所有这些任务都曾学好。",'',
                    '此前已为0的任务不会再次算作本阶段新遗忘；详细分母见立即前一阶段及历史获得阶段的比较文件。']
        lines+=['','## 验收文件','',
                '- `run_summary.json`：阶段、命令、退出码及训练完整性。',
                '- `stage3_front/summary.json`、`stage4_back/summary.json`：两阶段的成功率及目标矩阵。',
                '- 各阶段 `episodes.jsonl`：完整80回合；`comparison_to_previous.json`：立即前后配对。',
                '- `sequence_summary.json`：所有阶段的完整对照。',
                '- `COMPLETE.json`：仅全部训练及两次评估成功后生成。',
                '- `evidence.tar.gz`：小文件证据包，不含权重、数据、视频及逐步大日志。','',
                '本轮只完成固定预算的探索性顺序微调，不预先保证各新任务高成功率，也不证明语言信息路径的因果机制。']
        if report.get('error'):lines+=['',f"失败信息：{report['error']}"]
        (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    def save():
        dump(run/'run_summary.json',report)
        dump(run/'sequence_summary.json',snapshots)
        render()

    def launch(name,script,arguments,gpu=None):
        command=[sys.executable,'-u',str(ROOT/'scripts'/script),*map(str,arguments)]
        handle=(run/f'{name}.log').open('w');handles.append(handle)
        child_env=dict(env)
        if gpu is not None:child_env['CUDA_VISIBLE_DEVICES']=str(gpu)
        worker=subprocess.Popen(command,cwd=ROOT,env=child_env,stdout=handle,stderr=subprocess.STDOUT)
        workers.append(worker)
        report['phases'].append({'name':name,'command':command,'gpu':gpu,'pid':worker.pid,
                                'started_elapsed_s':time.monotonic()-started})
        save();return worker

    def finish(name,worker,timeout):
        code=worker.wait(timeout=timeout)
        phase=next(p for p in report['phases'] if p['name']==name)
        phase.update(returncode=code,finished_elapsed_s=time.monotonic()-started)
        save()
        if code:raise RuntimeError(f'{name} failed with exit {code}; inspect log')

    def stop(worker):
        if worker.poll() is None:
            worker.terminate()
            try:worker.wait(timeout=30)
            except subprocess.TimeoutExpired:worker.kill();worker.wait()

    def evaluate(stage,checkpoint,cfg,folder):
        for attempt in range(1,plan['evaluation_attempts']+1):
            parts=[]
            for gpu,goals in enumerate((('front','left'),('right','back'))):
                name=f"{stage['name']}_eval{attempt}_part{gpu}"
                output=folder/f'eval{attempt}_part{gpu}'
                worker=launch(name,'stage1_r0_eval.py',['evaluate','--config',cfg,'--checkpoint',checkpoint,
                    '--partition','paired','--instruction-goals',*goals,'--output',output,'--trace'],gpu)
                parts.append((name,worker,output))
            try:
                for name,worker,_ in parts:finish(name,worker,35*60)
            except (RuntimeError,subprocess.TimeoutExpired) as error:
                for _,worker,_ in parts:stop(worker)
                stage.setdefault('evaluation_runtime_failures',[]).append({'attempt':attempt,'error':str(error)})
                save()
                if attempt==plan['evaluation_attempts']:raise
                continue
            combined=[row for _,_,output in parts for row in rows(output/'episodes.jsonl')]
            stage['evaluation_attempt_used']=attempt
            return combined
        raise RuntimeError('Evaluation attempts exhausted')

    try:
        save()
        # Prepare and audit both future task datasets before spending GPU training time.
        for item in plan['stages']:
            stage={'name':item['name'],'goal':item['goal'],'requested_steps':12000,'status':'preparing_data'}
            report['stages'].append(stage)
            folder=run/item['name'];folder.mkdir()
            cfg=copy.deepcopy(base);cfg.pop('sequential',None)
            cfg.update(purpose=plan['purpose'],old_goals=[item['goal']])
            for split in ('train','validation'):
                cfg['data'][f'{split}_output']=f"datasets/stage1_single_{item['goal']}_{split}_aligned_v1"
            cfg['training'].update(max_steps=12000,save_every=plan['checkpoint_every'],walltime_hours=3)
            cfg['evaluation'].update(paired_indices=plan['paired_indices'],
                                     development_indices=list(range(20)),development_required_correct=16,
                                     qualification_indices=list(range(20,30)),qualification_required_correct=8,
                                     reserve_indices=[])
            cfg['continual_stage']={'goal':item['goal'],'exposed_goals':item['exposed_goals'],
                                    'fixed_endpoint':12000,'optimizer_reset_at_task_boundary':True}
            path=folder/'config.json';dump(path,cfg)
            name=item['name']+'_data'
            worker=launch(name,'stage1_r0_data.py',['--config',path,'--output',folder/'data'])
            finish(name,worker,25*60)
            name=item['name']+'_alignment'
            worker=launch(name,'stage1_alignment_verify.py',['--audit',folder/'data/audit.json',
                           '--output',folder/'alignment_verification.json'])
            finish(name,worker,10*60)
            stage['status']='data_ready';save()
        previous='left_right'
        for item,stage in zip(plan['stages'],report['stages'],strict=True):
            folder=run/item['name'];cfg=folder/'config.json'
            train=training_root/item['name']
            inherited=processor_hashes(parent);parent_sha=sha256(parent/'model.safetensors')
            stage.update(status='training',initial_checkpoint=str(parent),initial_model_sha256=parent_sha,
                         training_dir=str(train))
            name=item['name']+'_train'
            worker=launch(name,'stage1_r0_train.py',['--config',cfg,'--run-dir',train,'--init-checkpoint',parent],0)
            finish(name,worker,3*3600+10*60)
            checkpoint=train/'checkpoints/012000'
            completed=json.loads((train/'training_finished_012000.json').read_text())
            marker=json.loads((checkpoint/'CHECKPOINT_COMPLETE.json').read_text())
            assert completed['step']==completed['target']==12000 and completed['frozen_parameters_unchanged']
            assert marker['status']=='pass' and marker['frozen_hash_unchanged']
            assert processor_hashes(checkpoint)==inherited, 'Normalization numerical statistics changed'
            assert sha256(parent/'model.safetensors')==parent_sha, 'Parent checkpoint changed'
            trace=rows(train/'train_steps.jsonl')
            assert [r['step'] for r in trace]==list(range(1,12001))
            assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in trace)
            stage.update(status='evaluating',checkpoint=str(checkpoint),normalization_verified_unchanged=True,
                         training_integrity={'steps':12000,'continuous':True,'finite_loss_and_gradient':True,
                                             'training_seconds':completed['seconds']})
            save()
            after=evaluate(stage,checkpoint,cfg,folder)
            snapshot=summarize_snapshot(after,item['exposed_goals'],item['goal'])
            comparisons={}
            historical={}
            for old in item['exposed_goals'][:-1]:
                comparisons[old]=compare_sequential(episode_sets[previous],after,old,item['goal'])
                historical[old]=compare_sequential(episode_sets[acquisition[old]],after,old,item['goal'])
            (folder/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in after))
            dump(folder/'summary.json',snapshot)
            dump(folder/'comparison_to_previous.json',comparisons)
            dump(folder/'comparison_to_acquisition.json',historical)
            episode_sets[item['name']]=after;snapshots[item['name']]=snapshot
            acquisition[item['goal']]=item['name']
            stage.update(status='complete',counts=snapshot['task_success_counts'],
                         checkpoint_model_sha256=sha256(checkpoint/'model.safetensors'))
            parent=checkpoint;previous=item['name'];save()
        report.update(status='complete',seconds=time.monotonic()-started,
                      finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_evaluation_episodes=160)
        save()
        # Export only small, reviewable artifacts. Include manifests, not model states.
        manifest=[]
        files=[run/'plan.json',run/'run_summary.json',run/'sequence_summary.json',run/'report.md']
        files+=list((run/'baselines').glob('*.jsonl'))
        for stage in report['stages']:
            folder=run/stage['name'];train=Path(stage['training_dir'])
            files += [p for p in folder.glob('*.json')] + [folder/'episodes.jsonl']
            files += [folder/'data/audit.json',folder/'data/COMPLETE.json']
            files += [train/'manifest.json',train/'training_finished_012000.json',
                      train/'checkpoints/012000/CHECKPOINT_COMPLETE.json']
        def archive_path(path):
            return path.relative_to(run) if path.is_relative_to(run) else Path('training')/path.relative_to(training_root)
        for path in files:manifest.append({'server_path':str(path),'archive_path':str(archive_path(path)),
                                          'bytes':path.stat().st_size,'sha256':sha256(path)})
        dump(run/'evidence_manifest.json',manifest)
        with tarfile.open(run/'evidence.tar.gz','w:gz') as tar:
            for path in files:
                tar.add(path,arcname=str(archive_path(path)))
            tar.add(run/'evidence_manifest.json',arcname='evidence_manifest.json')
        dump(run/'COMPLETE.json',{'status':'complete','new_training_tasks':2,'steps_each':12000,
                                  'evaluation_episodes_each':80,'evidence_archive_ready':True})
    except BaseException as error:
        report.update(status='failed',error=str(error),seconds=time.monotonic()-started)
        save();raise
    finally:
        for worker in workers:stop(worker)
        for handle in handles:handle.close()


if __name__=='__main__':main()
