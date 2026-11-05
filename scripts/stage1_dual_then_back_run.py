"""Complete the joint-left/front -> right -> back fixed-budget sequence."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from stage1_dual_goal_run import dump, read_rows, sha256
from stage1_dual_then_right_run import summarize_requests
from stage1_four_condition_protocol import compare_acquired, compare_snapshots, summarize_snapshot
from stage1_summarize_dual_unseen import GOALS, validate

ROOT = Path(__file__).resolve().parents[1]
PARENT = 'outputs/sequential/dual12000_then_right_20261023_01/checkpoints/012000'
PARENT_SHA = '8c2dc6c371958027239a8c709a346842291bfb631b5a0ac29f4c04c394b85249'


def validate_config(config):
    s, d, t, e = (config[k] for k in ('sequential', 'data', 'training', 'evaluation'))
    if config['old_goals'] != ['back'] or s['new_goal'] != 'back' or s['retained_goals'] != ['left', 'front', 'right']:
        raise ValueError('Expected back-only adaptation after left/front and right')
    if s['initial_checkpoint'] != PARENT or s['initial_model_sha256'] != PARENT_SHA:
        raise ValueError('Wrong starting checkpoint')
    if s['source_run'] != '.server/dual12000_then_right_20261023_01':
        raise ValueError('Wrong cached source run')
    if s['evaluation_goals'] != list(GOALS) or e['paired_indices'] != list(range(20, 40)) or e['horizon'] != 400:
        raise ValueError('Four instructions need twenty matched 400-step rollouts')
    if t['max_steps'] != s['fixed_endpoint_step'] or t['max_steps'] != 12000 or t['samples_per_goal'] != 6:
        raise ValueError('Fixed back12000 updates and batch6 required')
    if s['replay'] is not False or s['distillation'] is not False or s['evaluation_attempts'] not in (1, 2):
        raise ValueError('No replay/distillation; runtime retries must be bounded')
    if d['normalization_source'] != 'datasets/stage1_dual_left_front_train_aligned_v1' or d['action_alignment'] != 'post_action_obs_to_next_action':
        raise ValueError('Keep joint normalization and verified alignment')
    if d['train_per_goal'] != 40 or d['train_output'] != 'datasets/stage1_single_back_train_aligned_v1' or d['validation_output'] != 'datasets/stage1_single_back_validation_aligned_v1':
        raise ValueError('Use the existing back40/10 dataset specification')


def load_cached(config):
    """Reuse only the right-stage endpoint, never the earlier dual-parent back test."""
    source = ROOT/config['sequential']['source_run']
    original = json.loads((source/'run_summary.json').read_text())
    source_config = json.loads((source/'config.json').read_text())
    if original['status'] != 'complete' or json.loads((source/'COMPLETE.json').read_text())['status'] != 'complete':
        raise ValueError('Prior stage incomplete')
    if Path(original['final_checkpoint']).resolve() != (ROOT/PARENT).resolve() or original['final_model_sha256'] != PARENT_SHA:
        raise ValueError('Cached rows belong to another model')
    for field in ('model', 'evaluation', 'seed'):
        if source_config[field] != config[field]:
            raise ValueError(f'Cached inference configuration differs: {field}')
    if source_config['data']['normalization_source'] != config['data']['normalization_source']:
        raise ValueError('Cached normalization differs')
    evidence = json.loads((source/'evidence_manifest.json').read_text())
    required = {'before/episodes.jsonl', 'after/episodes.jsonl', 'config.json', 'run_summary.json',
                'after_attempt1_part0/evaluation.json', 'after_attempt1_part1/evaluation.json'}
    if not required <= {entry['archive_path'] for entry in evidence}:
        raise ValueError('Source evidence manifest omits a cached source')
    for entry in evidence:
        path = Path(entry['server_path'])
        if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError(f'Original source evidence changed: {path}')
    files = [source/'run_summary.json', source/'config.json', source/'COMPLETE.json',
             source/'before/episodes.jsonl', source/'after/episodes.jsonl', source/'evidence_manifest.json']
    for part, goals in ((0, ['left', 'front']), (1, ['right'])):
        path = source/f'after_attempt1_part{part}/evaluation.json'
        # The completed source used its first runtime attempt. Do not silently
        # reuse first-attempt rows if a future source has a different history.
        provenance = json.loads(path.read_text())
        if provenance['status'] != 'complete' or provenance['instruction_goals'] != goals or Path(provenance['checkpoint']).resolve() != (ROOT/PARENT).resolve():
            raise ValueError('Wrong cached evaluation provenance')
        if provenance['git_commit'] != original['source_commit'] or provenance['horizon'] != 400 or provenance['n_action_steps'] != config['model']['n_action_steps'] or provenance['normalization_source'] != config['data']['normalization_source']:
            raise ValueError('Cached evaluation source version differs')
        files.append(path)
    for script in ('stage1_r0_eval.py', 'stage1_protocol.py', 'stage1_smoke.py'):
        old = subprocess.check_output(['git', 'show', original['source_commit']+':scripts/'+script], cwd=ROOT)
        if old != (ROOT/'scripts'/script).read_bytes():
            raise ValueError('Evaluation implementation changed; cannot silently reuse cached rows')
    cached = read_rows(source/'after/episodes.jsonl')
    historical = read_rows(source/'before/episodes.jsonl')
    for items in (cached, historical):
        summarize_requests(items, config['evaluation']['paired_indices'], config['seed'])
    pre = {(r['init_index'], r['requested_goal']): r for r in historical}
    for row in cached:
        if any(pre[row['init_index'], row['requested_goal']][field] != row[field] for field in ('initial_state_sha256', 'scene_xml_sha256', 'seed', 'steps')):
            raise ValueError('Historical and immediate baseline conditions differ')
    return cached, historical, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/stage1_dual_right_then_back_v1.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()); validate_config(config)
    from stage1_sequential_run import processor_hashes
    s = config['sequential']; parent = ROOT/s['initial_checkpoint']; run = args.run_dir.resolve()
    training = ROOT/'outputs/sequential'/run.name
    if run.exists() or training.exists():
        raise FileExistsError('Refusing to overwrite an experiment')
    if sha256(parent/'model.safetensors') != PARENT_SHA:
        raise ValueError('Wrong parent bytes')
    marker = json.loads((parent/'CHECKPOINT_COMPLETE.json').read_text())
    if marker['status'] != 'pass' or marker['step'] != 12000:
        raise ValueError('Parent checkpoint not accepted')
    inherited = processor_hashes(parent)
    if len(inherited) != 2:
        raise ValueError('Expected both saved normalization processors')
    cached, historical, source_files = load_cached(config)
    run.mkdir(parents=True); cfg = run/'config.json'; dump(cfg, config)
    baseline = run/'baselines'; baseline.mkdir()
    for name, items in (('parent_cached60', cached), ('joint_acquisition60', historical)):
        (baseline/f'{name}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in items))
    dump(baseline/'source_files.json', [{'server_path': str(p), 'sha256': sha256(p)} for p in source_files])
    source = ROOT/s['source_run']
    for src, name in ((source/'run_summary.json','previous_run_summary.json'), (source/'config.json','previous_config.json'),
                      (source/'evidence_manifest.json','previous_evidence_manifest.json'),
                      (source/'after_attempt1_part0/evaluation.json','cached_eval_part0.json'),
                      (source/'after_attempt1_part1/evaluation.json','cached_eval_part1.json')):
        shutil.copyfile(src, baseline/name)
    report = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'initial_checkpoint': str(parent), 'initial_model_sha256': PARENT_SHA,
              'training_dir': str(training), 'sequence': ['joint_left_front12000', 'right12000', 'back12000'],
              'fixed_training_steps': 12000, 'new_goal': 'back', 'cached_parent_evaluation_episodes': 60,
              'planned_new_evaluation_episodes': 100, 'starting_point_formally_qualified': False,
              'normalization_source': config['data']['normalization_source'], 'phases': []}
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'MUJOCO_GL': 'egl',
           'LIBERO_CONFIG_PATH': str(ROOT/'.server/libero'), 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4'}
    workers = []; handles = []; started = time.monotonic()
    def save():
        dump(run/'run_summary.json', report)
    def launch(name, script, arguments, gpu=None):
        command = [sys.executable, '-u', str(ROOT/'scripts'/script), *map(str, arguments)]
        handle = (run/f'{name}.log').open('w'); handles.append(handle)
        child_env = dict(env)
        if gpu is not None:
            child_env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        worker = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=handle, stderr=subprocess.STDOUT)
        workers.append(worker)
        report['phases'].append({'name': name, 'command': command, 'gpu': gpu, 'pid': worker.pid,
                                 'started_elapsed_s': time.monotonic()-started})
        save(); return worker
    def finish(name, worker, timeout):
        code = worker.wait(timeout=timeout)
        phase = next(p for p in report['phases'] if p['name'] == name)
        phase.update(returncode=code, finished_elapsed_s=time.monotonic()-started); save()
        if code:
            raise RuntimeError(f'{name} failed: exit{code}')
    def stop(worker):
        if worker.poll() is None:
            worker.terminate()
            try: worker.wait(timeout=30)
            except subprocess.TimeoutExpired: worker.kill(); worker.wait()
    def evaluate(label, checkpoint, parts):
        for attempt in range(1, s['evaluation_attempts']+1):
            running = []
            for gpu, goals in parts:
                name = f'{label}_attempt{attempt}_part{gpu}'; folder = run/name
                worker = launch(name, 'stage1_r0_eval.py', ['evaluate', '--config', cfg, '--checkpoint', checkpoint,
                    '--partition', 'paired', '--instruction-goals', *goals, '--output', folder, '--trace'], gpu)
                running.append((name, worker, folder))
            try:
                for name, worker, _ in running: finish(name, worker,40*60)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                for _, worker, _ in running: stop(worker)
                report.setdefault('evaluation_runtime_failures', []).append({'label': label, 'attempt': attempt, 'error': str(error)})
                save()
                if attempt == s['evaluation_attempts']: raise
                continue
            items = [row for _, _, folder in running for row in read_rows(folder/'episodes.jsonl')]
            goals = tuple(g for _, group in parts for g in group)
            validate(items, goals, config['evaluation']['paired_indices'], config['seed'])
            return items
        raise RuntimeError('Evaluation attempts exhausted')
    try:
        save(); dump(run/'parent_checkpoint_verification.json', marker)
        worker = launch('data', 'stage1_r0_data.py', ['--config', cfg, '--output', run/'data']); finish('data', worker,30*60)
        worker = launch('alignment', 'stage1_alignment_verify.py', ['--audit', run/'data/audit.json', '--output', run/'alignment_verification.json']); finish('alignment', worker,10*60)
        report['current_stage'] = 'training_back_and_evaluating_frozen_parent_back'; save()
        trainer = launch('train', 'stage1_r0_train.py', ['--config', cfg, '--run-dir', training, '--init-checkpoint', parent], 0)
        back = evaluate('before_back', parent, [(1, ('back',))])
        before = cached+back; pre_summary = summarize_snapshot(before, config['evaluation']['paired_indices'], config['seed'])
        (run/'before').mkdir(); (run/'before/episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in before))
        dump(run/'before/summary.json', pre_summary); report['before'] = pre_summary; save()
        finish('train', trainer,3*3600+10*60)
        final = training/'checkpoints/012000'
        completed = json.loads((training/'training_finished_012000.json').read_text())
        accepted = json.loads((final/'CHECKPOINT_COMPLETE.json').read_text())
        manifest = json.loads((training/'manifest.json').read_text())
        if completed['step'] != completed['target'] or completed['step'] != 12000 or not completed['frozen_parameters_unchanged'] or accepted['status'] != 'pass' or not accepted['frozen_hash_unchanged']:
            raise ValueError('Invalid fixed12000 endpoint')
        if manifest['config'] != config or manifest['git_commit'] != report['source_commit'] or manifest['initialization_model_sha256'] != PARENT_SHA or not manifest['optimizer_reset'] or not manifest['scheduler_reset']:
            raise ValueError('Wrong training initialization/config')
        trace = read_rows(training/'train_steps.jsonl')
        if [r['step'] for r in trace] != list(range(1,12001)) or not all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in trace):
            raise ValueError('Discontinuous/nonfinite training')
        for step in (3000,6000,9000,12000):
            saved = json.loads((training/'checkpoints'/f'{step:06d}'/'CHECKPOINT_COMPLETE.json').read_text())
            if saved['step'] != step or saved['status'] != 'pass' or not saved['frozen_hash_unchanged']:
                raise ValueError('A checkpoint verification failed')
        if processor_hashes(final) != inherited or processor_hashes(parent) != inherited or sha256(parent/'model.safetensors') != PARENT_SHA:
            raise ValueError('Parent/normalization changed')
        report.update(current_stage='evaluating_back12000_all_four_instructions', normalization_verified_unchanged=True); save()
        after = evaluate('after', final, [(0, ('front','left')), (1, ('right','back'))])
        post_summary = summarize_snapshot(after, config['evaluation']['paired_indices'], config['seed'])
        (run/'after').mkdir(); (run/'after/episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in after))
        dump(run/'after/summary.json', post_summary)
        immediate = compare_snapshots(before, after, config['evaluation']['paired_indices'], config['seed'])
        acquisition = {g: [r for r in (cached if g == 'right' else historical) if r['requested_goal'] == g] for g in ('left','front','right')}
        original = compare_acquired(acquisition, after, config['evaluation']['paired_indices'], config['seed'])
        dump(run/'comparison_to_previous.json', immediate); dump(run/'comparison_to_acquisition.json', original)
        new = post_summary['task_success_counts']['back']
        report.update(status='complete', current_stage='finished', finished_at=datetime.now(timezone.utc).isoformat(),
            seconds=time.monotonic()-started, after=post_summary, final_checkpoint=str(final), final_model_sha256=sha256(final/'model.safetensors'),
            accepted_evaluation_episodes=160, accepted_new_evaluation_episodes=100,
            actual_completed_new_worker_episodes_including_runtime_retries=sum(len(read_rows(run/p['name']/'episodes.jsonl')) for p in report['phases'] if p['name'].startswith(('before_back','after')) and (run/p['name']/'episodes.jsonl').exists()),
            new_task_reached_descriptive_threshold=all(new[k] >= 16 for k in ('success_ever','terminal_correct')))
        save()
        lines = ['# 左＋前 → 右 → 后：固定四任务终点评估', '', '每指令20回合。前测复用同一右格终点已有60回合，仅补它的后格20次。', '',
                 '| 指令 | 前测曾成功／最终正确 | 后测曾成功／最终正确 |', '|---|---:|---:|---:|']
        for goal in GOALS:
            a,b=pre_summary['task_success_counts'][goal],post_summary['task_success_counts'][goal]
            lines.append(f"| {goal} | {a['success_ever']}/20、{a['terminal_correct']}/20 | {b['success_ever']}/20、{b['terminal_correct']}/20 |")
        lines += ['', '立即前阶段已为0的旧任务不再算新增遗忘；另按历史获得阶段给出旧技能结果。正式R0未合格，单种子历史状态探索，不证明内部因果机制。']
        (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        files=[p for p in run.rglob('*') if p.is_file() and (p.suffix=='.json' or p.name=='episodes.jsonl' or p.parent==baseline or p.name=='report.md')]
        files += [training/'manifest.json',training/'trainable_parameters.json',training/'training_finished_012000.json']
        files += [training/'checkpoints'/f'{step:06d}'/'CHECKPOINT_COMPLETE.json' for step in (3000,6000,9000,12000)]
        def relative(path): return path.relative_to(run) if path.is_relative_to(run) else Path('training')/path.relative_to(training)
        dump(run/'evidence_manifest.json',[{'server_path':str(p),'archive_path':str(relative(p)),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in files])
        with tarfile.open(run/'evidence.tar.gz','w:gz') as archive:
            for path in files: archive.add(path,arcname=str(relative(path)))
            archive.add(run/'evidence_manifest.json',arcname='evidence_manifest.json')
        dump(run/'COMPLETE.json',{'status':'complete','fixed_back_updates':12000,'four_instructions_each20':True,
            'accepted_before_and_after_episodes':160,'new_model_evaluation_episodes':100,'cached_episodes':60,'evidence_archive_ready':True})
    except BaseException as error:
        report.update(status='failed',error=str(error),seconds=time.monotonic()-started); save(); raise
    finally:
        for worker in workers: stop(worker)
        for handle in handles: handle.close()


if __name__ == '__main__':
    main()
