"""Acquire a left/front R0 from the base; select on development, then test once."""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
GOALS = ('left', 'front')
ALL_GOALS = ('front', 'left', 'right', 'back')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config):
    """Fix budgets and data partitions before seeing any model results."""
    d, e, t = config['dual_goal'], config['evaluation'], config['training']
    if config['old_goals'] != list(GOALS):
        raise ValueError('This R0 must jointly train left and front')
    if d['candidate_steps'] != [6000, 12000, 18000] or t['max_steps'] != 18000:
        raise ValueError('Expected fixed 6000/12000/18000 development candidates')
    if t['save_every'] != 6000 or t['samples_per_goal'] != 3:
        raise ValueError('Expected balanced six-sample batches and 6000-step checkpoints')
    expected = {'development_indices': list(range(20)),
                'qualification_indices': list(range(20, 40)),
                'paired_indices': [], 'reserve_indices': list(range(40, 50))}
    if any(e[key] != value for key, value in expected.items()):
        raise ValueError('Development, final evaluation and reserve partitions changed')
    if d['qualification_subset'] != list(range(20, 30)):
        raise ValueError('The ten-state qualification subset must be fixed in advance')
    if d.get('final_indices', e['qualification_indices']) != list(range(20, 40)):
        raise ValueError('Final evaluation must use twenty states per instruction')
    if (d['development_joint_required'], d['final_required'], d['qualification_required']) != (16, 16, 8):
        raise ValueError('Expected fixed development/final/fresh thresholds of 16/16/8')
    if e['development_required_correct'] != 16 or e['qualification_required_correct'] != 16:
        raise ValueError('The evaluation script must use the same individual-goal threshold')
    if e['horizon'] != 400 or config['data'].get('action_alignment') != 'post_action_obs_to_next_action':
        raise ValueError('Expected 400-step evaluations and verified next-action alignment')
    if config['data'].get('normalization_source') not in (None, config['data']['train_output']):
        raise ValueError('R0 must compute normalization from its own joint training data')
    if config.get('sequential') or config.get('continual_stage'):
        raise ValueError('This acquisition run must start from the base, not a previous task')
    if d.get('evaluation_attempts', 2) not in (1, 2):
        raise ValueError('Runtime retries must be bounded to at most two attempts')


def summarize_dual(rows, indices, required):
    """Validate every reset pair, retain failures, and count joint instruction success."""
    if not indices or len(set(indices)) != len(indices) or not 0 < required <= len(indices):
        raise ValueError('Invalid evaluation denominator or threshold')
    expected = {(index, goal) for index in indices for goal in GOALS}
    keyed = {}
    for row in rows:
        key = (row['init_index'], row['requested_goal'])
        if key in keyed or key not in expected:
            raise ValueError('Duplicate or unexpected initial-state/instruction pair')
        if row['steps'] != 400:
            raise ValueError('Evaluation horizon changed')
        category, terminal = row['terminal_category'], row['terminal_goal']
        if category not in ('correct', 'wrong_goal', 'unfinished', 'ambiguous'):
            raise ValueError('Unknown terminal category')
        if category == 'correct' and (terminal != key[1] or not row['correct_ever']):
            raise ValueError('Terminal-correct episode has inconsistent goal or ever-success')
        if category == 'wrong_goal' and (terminal not in ALL_GOALS or terminal == key[1]):
            raise ValueError('Wrong-goal episode has inconsistent terminal target')
        if category in ('unfinished', 'ambiguous') and terminal is not None:
            raise ValueError('Unfinished/ambiguous episode must not have a unique terminal goal')
        keyed[key] = row
    if set(keyed) != expected:
        raise ValueError('Missing evaluation conditions; no denominator may be shortened')
    for index in indices:
        for field in ('seed', 'initial_state_sha256', 'scene_xml_sha256', 'steps'):
            values = [keyed[index, goal][field] for goal in GOALS]
            if not all(value == values[0] for value in values):
                raise ValueError(f'Mismatched paired condition: init={index}, field={field}')
    if len({keyed[index, GOALS[0]]['initial_state_sha256'] for index in indices}) != len(indices):
        raise ValueError('Different initial-state indices resolved to repeated actual simulator states')
    if len({keyed[index, GOALS[0]]['scene_xml_sha256'] for index in indices}) != 1:
        raise ValueError('The common scene must stay fixed across initial states')
    counts = {}
    matrix = {goal: {target: 0 for target in (*ALL_GOALS, 'unfinished', 'ambiguous')} for goal in GOALS}
    for goal in GOALS:
        selected = [keyed[index, goal] for index in indices]
        counts[goal] = {'total': len(indices),
                        'success_ever': sum(bool(row['correct_ever']) for row in selected),
                        'terminal_correct': sum(row['terminal_category'] == 'correct' for row in selected),
                        'correct_then_destroyed': sum(bool(row.get('correct_then_destroyed', False)) for row in selected)}
        for row in selected:
            target = row['terminal_category'] if row['terminal_category'] in ('unfinished', 'ambiguous') else row['terminal_goal']
            matrix[goal][target] += 1
    joint_terminal = [index for index in indices if all(keyed[index, goal]['terminal_category'] == 'correct' for goal in GOALS)]
    joint_ever = [index for index in indices if all(keyed[index, goal]['correct_ever'] for goal in GOALS)]
    passed = len(joint_terminal) >= required and all(count['terminal_correct'] >= required for count in counts.values())
    return {'initial_states': len(indices), 'episodes': len(rows), 'indices': list(indices),
            'task_success_counts': counts, 'terminal_matrix': matrix,
            'joint_success_ever': len(joint_ever), 'joint_terminal_correct': len(joint_terminal),
            'joint_terminal_correct_indices': joint_terminal, 'required_correct': required, 'passed': passed,
            'paired_reset_and_inference_seed_verified': True,
            'joint_definition': 'Both left and front instructions finish correctly from the same initial state; denominator is initial states, not episodes.'}


def final_qualification(development_passed, final_rows, config):
    """The final holdout reports performance; it never selects a different checkpoint."""
    d = config['dual_goal']
    final = summarize_dual(final_rows, config['evaluation']['qualification_indices'], d['final_required'])
    fresh = summarize_dual([row for row in final_rows if row['init_index'] in d['qualification_subset']],
                           d['qualification_subset'], d['qualification_required'])
    historical = summarize_dual([row for row in final_rows if row['init_index'] in range(30, 40)],
                                list(range(30, 40)), 8)
    return {'qualified_R0': bool(development_passed and final['passed'] and fresh['passed']),
            'development_gate_passed': bool(development_passed), 'final_twenty': final,
            'qualification_subset_twenty_to_twenty_nine': fresh,
            'historically_exposed_subset_thirty_to_thirty_nine': historical,
            'scope': 'Final twenty mixes a ten-state qualification subset with ten historically evaluated states. Report both subsets and do not claim all twenty states are previously unused.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/stage1_dual_left_front_v1.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    validate_config(config)
    run = args.run_dir.resolve()
    train = ROOT / 'outputs/r0' / run.name
    if run.exists() or train.exists():
        raise FileExistsError('Use new run and training directories; no implicit restart')
    # GPU libraries remain in the existing child scripts; importing this module is stdlib-only.
    from download_stage1 import BASE_ID, find_model
    from stage1_sequential_run import processor_hashes
    base = find_model(BASE_ID)
    if base is None:
        raise FileNotFoundError('The complete local smolvla_base is required; no automatic download')
    base_sha = sha256(base / 'model.safetensors')
    run.mkdir(parents=True)
    cfg = run / 'config.json'
    dump(cfg, config)
    report = {'status': 'running', 'purpose': config['purpose'],
              'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'base_checkpoint': str(base), 'base_model_sha256': base_sha,
              'initialization': 'Original smolvla_base; no adapted checkpoint; fresh optimizer/scheduler/RNG.',
              'training_run': str(train), 'normalization_source': str(ROOT / config['data']['train_output']),
              'maximum_training_steps': 18000, 'batch_size': 6, 'samples_per_goal': 3,
              'selection_rule': 'First of 6000/12000/18000 with >=16/20 terminal-correct left, front and same-state joint successes on development. Pause training for each evaluation; if none passes, evaluate fixed 18000 as unqualified. Final scores never change selection or training budget.',
              'optimizer_scheduler_rng_restored_between_candidates': [],
              'candidate_reports': [], 'phases': [], 'qualified_R0': False,
              'final_evaluation_episodes_planned': 40,
              'final_mixed_set_disclosure': 'States 20..29 are the qualification subset; 30..39 were previously evaluated in older experiments. Development 0..19 is disjoint from all final states.'}
    started = time.monotonic()
    workers, handles = [], []
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'MUJOCO_GL': 'egl',
           'LIBERO_CONFIG_PATH': str(ROOT / '.server/libero'), 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4'}

    def save():
        dump(run / 'run_summary.json', report)
        lines = ['# 左＋前双目标起点 R0', '', f"运行状态：**{report['status']}**", '',
                 '从原始 SmolVLA base 联合学习左、前；每批各3条，共6条。冻结VLM，训练原生动作专家及相关投影。',
                 '最多18000步，在6000、12000、18000步暂停训练，用两条指令各20回合开发评估。',
                 '选第一个达到每条指令及同初态双指令最终均正确至少16/20的候选；最终评估不会改变选点。', '',
                 '| 候选步数 | 左 曾成功/最终正确 | 前 曾成功/最终正确 | 同初态两条最终均正确 | 开发通过 |',
                 '|---|---:|---:|---:|---|']
        for candidate in report['candidate_reports']:
            score = candidate['summary']; counts = score['task_success_counts']
            cells = [f"{counts[goal]['success_ever']}/20、{counts[goal]['terminal_correct']}/20" for goal in GOALS]
            lines.append('| ' + str(candidate['step']) + ' | ' + ' | '.join(cells) + f" | {score['joint_terminal_correct']}/20 | {score['passed']} |")
        if 'final_qualification' in report:
            result = report['final_qualification']
            lines += ['', f"选定步数：**{report['selected_step']}**；正式起点合格：**{result['qualified_R0']}**。", '',
                      '最终40回合：每条指令在20..39初态各20次；20..29单独作为10个资格初态，30..39是历史已评估初态，不能把全部20个说成全新。', '',
                      '| 最终子集 | 左 曾成功/最终正确 | 前 曾成功/最终正确 | 同初态两条最终均正确 |',
                      '|---|---:|---:|---:|']
            for name, key in (('全部20初态', 'final_twenty'), ('资格10初态20..29', 'qualification_subset_twenty_to_twenty_nine'),
                              ('历史10初态30..39', 'historically_exposed_subset_thirty_to_thirty_nine')):
                score = result[key]; n = score['initial_states']; counts = score['task_success_counts']
                cells = [f"{counts[goal]['success_ever']}/{n}、{counts[goal]['terminal_correct']}/{n}" for goal in GOALS]
                lines.append('| ' + name + ' | ' + ' | '.join(cells) + f" | {score['joint_terminal_correct']}/{n} |")
        if report.get('error'):
            lines += ['', f"运行失败：{report['error']}"]
        lines += ['', '每回合固定执行400步；曾成功与最终正确分开，失败保留在分母中。只有训练、完整最终评估和证据归档均完成，才生成COMPLETE.json；complete不等于起点合格。']
        (run / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def launch(name, script, arguments, gpu=None):
        command = [sys.executable, '-u', str(ROOT / 'scripts' / script), *map(str, arguments)]
        handle = (run / f'{name}.log').open('w', encoding='utf-8'); handles.append(handle)
        child_env = dict(env)
        if gpu is not None:
            child_env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        worker = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=handle, stderr=subprocess.STDOUT)
        workers.append(worker)
        report['phases'].append({'name': name, 'command': command, 'gpu': gpu, 'pid': worker.pid,
                                 'started_elapsed_s': time.monotonic() - started})
        save()
        return worker

    def finish(name, worker, timeout):
        code = worker.wait(timeout=timeout)
        phase = next(item for item in report['phases'] if item['name'] == name)
        phase.update(returncode=code, finished_elapsed_s=time.monotonic() - started)
        save()
        if code:
            raise RuntimeError(f'{name} failed with exit {code}; inspect its log')

    def stop(worker):
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker.kill(); worker.wait()

    def evaluate(label, checkpoint, partition, indices):
        for attempt in range(1, config['dual_goal'].get('evaluation_attempts', 2) + 1):
            parts = []
            for gpu, goal in enumerate(GOALS):
                name = f'{label}_attempt{attempt}_{goal}'
                output = run / name
                worker = launch(name, 'stage1_r0_eval.py', ['evaluate', '--config', cfg, '--checkpoint', checkpoint,
                    '--partition', partition, '--instruction-goals', goal, '--output', output, '--trace'], gpu)
                parts.append((name, worker, output))
            try:
                for name, worker, _ in parts:
                    finish(name, worker, 25 * 60)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                for _, worker, _ in parts:
                    stop(worker)
                report.setdefault('evaluation_runtime_failures', []).append({'label': label, 'attempt': attempt, 'error': str(error)})
                save()
                if attempt == config['dual_goal'].get('evaluation_attempts', 2):
                    raise
                continue
            combined = [row for _, _, output in parts for row in read_rows(output / 'episodes.jsonl')]
            if any(row['seed'] != config['seed'] + row['init_index'] for row in combined):
                raise ValueError('Inference/reset seed does not match the predeclared configuration')
            # Invalid/missing conditions are a protocol failure, not a low score or a retry trigger.
            summary = summarize_dual(combined, indices, 16)
            folder = run / label; folder.mkdir()
            (folder / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in combined), encoding='utf-8')
            dump(folder / 'summary.json', summary)
            dump(folder / 'evaluation_provenance.json', {'attempt': attempt, 'worker_outputs': [str(output) for _, _, output in parts],
                                                       'checkpoint': str(checkpoint), 'partition': partition})
            return combined, summary
        raise RuntimeError('Evaluation attempts exhausted')

    try:
        save()
        worker = launch('data', 'stage1_r0_data.py', ['--config', cfg, '--output', run / 'data'])
        finish('data', worker, 30 * 60)
        worker = launch('alignment', 'stage1_alignment_verify.py', ['--audit', run / 'data/audit.json', '--output', run / 'alignment_verification.json'])
        finish('alignment', worker, 10 * 60)
        worker = launch('predicate_reset_verify', 'stage1_r0_eval.py', ['verify', '--config', cfg,
            '--audit', run / 'data/audit.json', '--output', run / 'predicate_verification'], 0)
        finish('predicate_reset_verify', worker, 15 * 60)
        report['data_alignment_predicates_reset_and_training_start_separation_verified'] = True
        previous = None; normalization = None; selected = None
        for step in config['dual_goal']['candidate_steps']:
            report['current_stage'] = f'training_to_{step}'
            arguments = ['--config', cfg, '--run-dir', train, '--max-steps', step]
            if previous is not None:
                arguments += ['--resume', previous]
            name = f'train_to_{step:06d}'
            worker = launch(name, 'stage1_r0_train.py', arguments, 0)
            finish(name, worker, max(3 * 3600, (config['training']['walltime_hours'] + .2) * 3600))
            checkpoint = train / 'checkpoints' / f'{step:06d}'
            completed = json.loads((train / f'training_finished_{step:06d}.json').read_text())
            marker = json.loads((checkpoint / 'CHECKPOINT_COMPLETE.json').read_text())
            manifest = json.loads((train / 'manifest.json').read_text())
            if completed['step'] != step or completed['target'] != step or not completed['frozen_parameters_unchanged']:
                raise ValueError('Training stopped early or changed frozen parameters')
            if marker['status'] != 'pass' or marker['step'] != step or not marker['frozen_hash_unchanged'] or not marker['optimizer_and_rng_saved']:
                raise ValueError('Candidate checkpoint did not pass reload/frozen/state verification')
            if Path(manifest['normalization_source']).resolve() != (ROOT / config['data']['train_output']).resolve() or manifest.get('initialization_checkpoint'):
                raise ValueError('R0 did not start from base with joint-training normalization')
            if manifest['config'] != config or manifest['git_commit'] != report['source_commit']:
                raise ValueError('Training manifest source/config changed')
            trace = read_rows(train / 'train_steps.jsonl')
            if [row['step'] for row in trace] != list(range(1, step + 1)) or not all(math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']) for row in trace):
                raise ValueError('Training trace is discontinuous or nonfinite')
            current_normalization = processor_hashes(checkpoint)
            if not current_normalization or normalization is not None and normalization != current_normalization:
                raise ValueError('Joint normalization disappeared or changed between candidates')
            normalization = current_normalization
            if previous is not None:
                resumed = json.loads((train / f'resume_{int(previous.name):06d}.json').read_text())
                if resumed['status'] != 'pass' or resumed['restored_step'] != int(previous.name) or not resumed['training_config_unchanged'] or resumed['optimizer_state_entries'] <= 0:
                    raise ValueError('Full training state was not restored')
                report['optimizer_scheduler_rng_restored_between_candidates'].append(resumed)
            report['normalization_processor_fingerprints'] = normalization
            report['frozen_hash'] = manifest['frozen_hash']
            report['current_stage'] = f'development_{step}'
            save()
            _, summary = evaluate(f'development_{step:06d}', checkpoint, 'development', config['evaluation']['development_indices'])
            report['candidate_reports'].append({'step': step, 'summary': summary, 'training_completed': completed,
                                               'checkpoint_verification': marker, 'checkpoint_model_sha256': sha256(checkpoint / 'model.safetensors')})
            save()
            previous = checkpoint
            if summary['passed']:
                selected = checkpoint
                break
        if selected is None:
            selected = previous
        report['selected_checkpoint'] = str(selected)
        report['selected_step'] = int(selected.name)
        report['selected_model_sha256'] = sha256(selected / 'model.safetensors')
        report['development_gate_passed'] = any(item['summary']['passed'] for item in report['candidate_reports'])
        report['actual_training_steps'] = report['selected_step']
        report['no_training_beyond_selected_candidate'] = read_rows(train / 'train_steps.jsonl')[-1]['step'] == report['selected_step']
        if not report['no_training_beyond_selected_candidate'] or sha256(base / 'model.safetensors') != base_sha:
            raise ValueError('Training exceeded the selected point or base weights changed')
        report['base_weights_unchanged'] = True
        report['current_stage'] = 'final_evaluation'
        save()
        final_rows, _ = evaluate('final', selected, 'qualification', config['evaluation']['qualification_indices'])
        result = final_qualification(report['development_gate_passed'], final_rows, config)
        dump(run / 'final_qualification.json', result)
        report.update(status='complete', current_stage='finished', final_qualification=result,
                      qualified_R0=result['qualified_R0'], seconds=time.monotonic() - started,
                      result='qualified_dual_goal_R0' if result['qualified_R0'] else 'completed_but_R0_not_qualified',
                      finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      accepted_evaluation_episodes=40 * (len(report['candidate_reports']) + 1))
        report['actual_completed_worker_episodes_including_runtime_retries'] = sum(
            len(read_rows(run / phase['name'] / 'episodes.jsonl')) for phase in report['phases']
            if phase['name'].startswith(('development_', 'final_'))
            and (run / phase['name'] / 'episodes.jsonl').is_file())
        save()
        # Include only small evidence. Weights, optimizer states, videos and step traces stay on the server.
        files = [run / 'config.json', run / 'run_summary.json', run / 'report.md', run / 'alignment_verification.json',
                 run / 'data/audit.json', run / 'data/COMPLETE.json', run / 'predicate_verification/predicate_checks.json',
                 run / 'final_qualification.json', train / 'manifest.json', train / 'trainable_parameters.json']
        for candidate in report['candidate_reports']:
            step = candidate['step']; checkpoint = train / 'checkpoints' / f'{step:06d}'
            files += [train / f'training_finished_{step:06d}.json', checkpoint / 'CHECKPOINT_COMPLETE.json']
        files += list(train.glob('resume_*.json'))
        for folder in run.iterdir():
            if folder.is_dir() and (folder.name.startswith('development_') or folder.name.startswith('final')):
                files += [path for path in folder.glob('*.json')]
                if (folder / 'episodes.jsonl').is_file():
                    files.append(folder / 'episodes.jsonl')
        files = list(dict.fromkeys(files))
        def archive_path(path):
            return path.relative_to(run) if path.is_relative_to(run) else Path('training') / path.relative_to(train)
        evidence = [{'server_path': str(path), 'archive_path': str(archive_path(path)), 'bytes': path.stat().st_size,
                     'sha256': sha256(path)} for path in files]
        dump(run / 'evidence_manifest.json', evidence)
        with tarfile.open(run / 'evidence.tar.gz', 'w:gz') as archive:
            for path in files:
                archive.add(path, arcname=str(archive_path(path)))
            archive.add(run / 'evidence_manifest.json', arcname='evidence_manifest.json')
        dump(run / 'COMPLETE.json', {'status': 'complete', 'qualified_R0': result['qualified_R0'],
             'selected_step': report['selected_step'], 'final_evaluation_episodes': 40,
             'evidence_archive_ready': True, 'finished_at': report['finished_at']})
    except BaseException as error:
        report.update(status='failed', error=str(error), seconds=time.monotonic() - started)
        save()
        raise
    finally:
        for worker in workers:
            stop(worker)
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
