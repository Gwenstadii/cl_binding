"""Fixed left-to-right adaptation with paired 20-episode instruction evaluations."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from stage1_protocol import GOALS, normalization_stat_shape
from stage1_single_task_benchmark_followup import metric_summary
from stage1_smoke import ROOT, dump


def compare_sequential(before, after, old_goal, new_goal):
    """Keep pre-existing old successes and different failure types explicit."""
    if old_goal == new_goal or {old_goal, new_goal} - set(GOALS):
        raise ValueError('Sequential experiment needs two distinct known goals')
    pre_summary = metric_summary(before, old_goal)
    post_summary = metric_summary(after, new_goal)
    pre = {(r['init_index'], r['requested_goal']): r for r in before}
    post = {(r['init_index'], r['requested_goal']): r for r in after}
    if pre.keys() != post.keys():
        raise ValueError('Pre/post conditions differ')
    for key in pre:
        for field in ('initial_state_sha256', 'scene_xml_sha256', 'seed', 'steps'):
            if pre[key][field] != post[key][field]:
                raise ValueError(f'Pre/post mismatch: {key} {field}')
    eligible = [key for key, row in pre.items() if key[1] == old_goal and row['correct_ever']]
    lost = [post[key] for key in eligible if not post[key]['correct_ever']]
    terminal_eligible = [key for key, row in pre.items()
                         if key[1] == old_goal and row['terminal_category'] == 'correct']
    terminal_lost = [post[key] for key in terminal_eligible if post[key]['terminal_category'] != 'correct']
    return {
        'old_goal': old_goal, 'new_goal': new_goal, 'matched_pairs': len(pre),
        'before': pre_summary, 'after': post_summary,
        'old_goal_previously_successful_ever': {
            'total': len(eligible),
            'retained_success_ever': len(eligible) - len(lost),
            'lost_success_ever': len(lost),
            'lost_and_first_new_goal': sum(r['first_goal'] == new_goal for r in lost),
            'lost_and_terminal_new_goal': sum(r['terminal_goal'] == new_goal for r in lost),
            'lost_and_terminal_unfinished': sum(r['terminal_category'] == 'unfinished' for r in lost),
            'lost_and_terminal_ambiguous': sum(r['terminal_category'] == 'ambiguous' for r in lost),
            'lost_and_terminal_other_wrong': sum(r['terminal_category'] == 'wrong_goal'
                                                and r['terminal_goal'] != new_goal for r in lost),
            'lost_initial_indices': sorted(r['init_index'] for r in lost),
        },
        'old_goal_previously_terminal_correct': {
            'total': len(terminal_eligible),
            'retained_terminal_correct': len(terminal_eligible) - len(terminal_lost),
            'lost_terminal_correct': len(terminal_lost),
            'lost_and_terminal_new_goal': sum(r['terminal_goal'] == new_goal for r in terminal_lost),
            'lost_and_terminal_unfinished': sum(r['terminal_category'] == 'unfinished' for r in terminal_lost),
        },
        'interpretation': 'Single-old-skill retention pilot; it does not establish erosion of previously acquired multi-goal language discrimination or a causal information pathway.',
    }


def processor_hashes(checkpoint):
    """Hash tensor contents so serialization order cannot masquerade as a change."""
    import torch
    from safetensors.torch import load_file
    result = {}
    for path in checkpoint.glob('*processor*.safetensors'):
        digest = hashlib.sha256()
        for name, tensor in sorted(load_file(path).items()):
            digest.update(name.encode())
            digest.update(str((tensor.dtype, normalization_stat_shape(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes())
        result[path.name] = digest.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/stage1_left_then_right_v1.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    s = config['sequential']
    assert config['old_goals'] == [s['new_goal']]
    assert s['old_goal'] != s['new_goal'] and s['replay'] is False
    assert config['training']['max_steps'] == s['fixed_endpoint_step']
    assert len(config['evaluation']['paired_indices']) == 20
    assert len(set(config['evaluation']['paired_indices'])) == 20
    parent = ROOT/s['initial_checkpoint']
    assert json.loads((parent/'CHECKPOINT_COMPLETE.json').read_text())['status'] == 'pass'
    parent_hashes = processor_hashes(parent)
    assert len(parent_hashes) == 2
    parent_model_hash = hashlib.sha256((parent/'model.safetensors').read_bytes()).hexdigest()
    run = args.run_dir.resolve()
    train = ROOT/'outputs/sequential'/run.name
    if run.exists() or train.exists():
        raise FileExistsError('Run paths must be new')
    run.mkdir(parents=True)
    cfg = run/'config.json'
    dump(cfg, config)
    started = time.monotonic()
    report = {
        'status': 'running', 'purpose': config['purpose'], 'run_dir': str(run),
        'training_dir': str(train), 'initial_checkpoint': str(parent),
        'initial_model_sha256': parent_model_hash, 'parent_processor_sha256': parent_hashes,
        'processor_hash_kind': 'sorted_tensor_names_dtypes_shapes_and_values_with_singleton_shape_canonicalization',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'selection_rule': 'Always evaluate the fixed +6000 endpoint, including low-new-task-success outcomes; no checkpoint selection using old or paired scores.',
        'phases': [],
    }
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'MUJOCO_GL': 'egl', 'LIBERO_CONFIG_PATH': str(ROOT/'.server/libero'),
           'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4'}
    workers = []
    handles = []

    def save():
        dump(run/'run_summary.json', report)

    def launch(name, script, arguments, gpu=None):
        command = [sys.executable, '-u', str(ROOT/'scripts'/script), *map(str, arguments)]
        child_env = dict(env)
        if gpu is not None:
            child_env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        handle = (run/f'{name}.log').open('w')
        handles.append(handle)
        worker = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=handle, stderr=subprocess.STDOUT)
        workers.append(worker)
        report['phases'].append({'name': name, 'command': command, 'gpu': gpu, 'pid': worker.pid,
                                  'started_elapsed_s': time.monotonic()-started})
        save()
        return worker

    def finish(name, worker, timeout):
        code = worker.wait(timeout=timeout)
        phase = next(p for p in report['phases'] if p['name'] == name)
        phase.update(returncode=code, finished_elapsed_s=time.monotonic()-started)
        save()
        if code:
            raise RuntimeError(f'{name} failed with exit {code}; inspect log')

    def evaluate(name, checkpoint):
        return launch(name, 'stage1_r0_eval.py', [
            'evaluate', '--config', cfg, '--checkpoint', checkpoint,
            '--partition', 'paired', '--instruction-goals', *GOALS,
            '--output', run/name, '--trace'], gpu=1)

    try:
        save()
        data = launch('data', 'stage1_r0_data.py', ['--config', cfg, '--output', run/'data'])
        finish('data', data, 20*60)
        check = launch('alignment_verify', 'stage1_alignment_verify.py', [
            '--audit', run/'data/audit.json', '--output', run/'alignment_verification.json'])
        finish('alignment_verify', check, 5*60)
        trainer = launch('train', 'stage1_r0_train.py', [
            '--config', cfg, '--run-dir', train, '--init-checkpoint', parent], gpu=0)
        before = evaluate('before_paired20', parent)
        finish('before_paired20', before, 30*60)
        finish('train', trainer, 90*60)
        final = train/'checkpoints'/f"{s['fixed_endpoint_step']:06d}"
        assert json.loads((final/'CHECKPOINT_COMPLETE.json').read_text())['status'] == 'pass'
        finished = json.loads((train/f"training_finished_{s['fixed_endpoint_step']:06d}.json").read_text())
        assert finished['step'] == finished['target'] == s['fixed_endpoint_step']
        assert processor_hashes(final) == parent_hashes, 'Normalization changed across tasks'
        assert processor_hashes(parent) == parent_hashes, 'Parent processor changed'
        assert hashlib.sha256((parent/'model.safetensors').read_bytes()).hexdigest() == parent_model_hash
        report['normalization_tensors_unchanged'] = True
        after = evaluate('after_paired20', final)
        finish('after_paired20', after, 30*60)
        rows = lambda folder: [json.loads(line) for line in (run/folder/'episodes.jsonl').read_text().splitlines()]
        comparison = compare_sequential(rows('before_paired20'), rows('after_paired20'), s['old_goal'], s['new_goal'])
        dump(run/'sequential_comparison.json', comparison)
        counts = comparison['after']['task_success_counts'][s['new_goal']]
        report.update(status='complete', result='fixed_endpoint_sequential_pilot_complete',
                      final_checkpoint=str(final), new_goal_counts=counts,
                      new_goal_reached_descriptive_threshold=counts['success_ever'] >= s['new_goal_descriptive_success_threshold'],
                      old_retention=comparison['old_goal_previously_successful_ever'],
                      seconds=time.monotonic()-started,
                      finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        dump(run/'COMPLETE.json', {'status': 'complete', 'result': report['result'], 'seconds': report['seconds']})
        print(json.dumps(report), flush=True)
    except BaseException as error:
        report.update(status='failed', error=str(error), seconds=time.monotonic()-started)
        save()
        raise
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
