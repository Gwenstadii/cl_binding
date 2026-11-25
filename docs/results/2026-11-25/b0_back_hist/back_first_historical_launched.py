"""Back-first using the unchanged historical trainer, with isolated native evaluation.

The trainer runs in its own process so validation cannot consume its RNG state.
No old task checkpoint or old task processor is supplied to that trainer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

RUN = Path(__file__).resolve().parent
ROOT = Path('/home/lzx/cl_binding_stage2a')
ARCHIVE = Path('/home/lzx/cl_binding/.server')
STEPS = (6000, 12000, 18000, 24000)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def env(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4')


def prepare():
    assert not (RUN / 'manifest.json').exists(), 'Never overwrite an existing run'
    parent = ARCHIVE / 'r1_left_right_20261104_01/train_r1'
    historical = read(parent / 'manifest.json')
    config = json.loads(json.dumps(historical['config']))
    assert config['seed'] == 20261021
    assert config['training']['learning_rate'] == 1e-4
    assert config['training']['warmup_steps'] == 100
    assert read(parent / 'training_finished_024000.json')['step'] == 24000
    config['purpose'] = 'B0-Back-Hist_public_init_Back_only_historical_first_stage_recipe'
    config['old_goals'] = ['back']
    config['data']['train_output'] = 'datasets/stage1_single_back_train_aligned_v1'
    config['data']['validation_output'] = 'datasets/stage1_single_back_validation_aligned_v1'
    config['training']['samples_per_goal'] = 6  # Preserve total batch size: 2*3 -> 1*6.
    config['training']['max_steps'] = 24000
    config['evaluation'].update(common_state_source_goal='front', development_indices=list(range(20)),
                                development_required_correct=16)
    converted = read(ROOT / config['data']['train_output'] / 'r0_conversion.json')
    assert converted['spec']['goals'] == ['back'] and converted['spec']['split'] == 'train'
    assert converted['spec']['action_alignment'] == config['data']['action_alignment']
    assert len(converted['episodes']) == 40
    assert sum(ep['frames'] for ep in converted['episodes']) == 5849
    write(RUN / 'config.json', config)
    for name, path in {
        'historical_stage1_r0_train.py': ROOT / 'scripts/stage1_r0_train.py',
        'historical_stage1_r0_eval.py': ROOT / 'scripts/stage1_r0_eval.py',
        'historical_stage1_protocol.py': ROOT / 'scripts/stage1_protocol.py',
    }.items():
        shutil.copyfile(path, RUN / name)
    write(RUN / 'historical_r1_manifest.json', historical)
    write(RUN / 'dev_cases.json', read(ARCHIVE / 'k25_whole4_new_20261124_01/dev_cases.json'))
    write(RUN / 'confirm_cases.json', read(ARCHIVE / 'k25_back_stable_20261124_01/confirmation_cases.json'))
    assert [r['init'] for r in read(RUN / 'dev_cases.json')] == list(range(640, 660))
    assert [r['init'] for r in read(RUN / 'confirm_cases.json')] == list(range(700, 720))
    write(RUN / 'manifest.json', dict(
        code='B0-Back-Hist', started_at_utc=now(), historical_run=str(parent),
        budget=24000, checkpoints=list(STEPS), training_goals=['back'],
        initialization='public lerobot/smolvla_base; no init-checkpoint and no resume',
        normalization='fitted only from this Back training dataset by unchanged trainer',
        historical_recipe=dict(seed=20261021, batch=6, frame_uniform=True, warmup=100,
                               learning_rate=1e-4, after_warmup='constant', amp='bfloat16'),
        task_adaptations=['Left/Right data replaced with Back data and own training statistics',
                          'samples_per_goal 3->6 to preserve total batch of six',
                          'budget24000 from actually successful historical final checkpoint'],
        sources={p.name: digest(p) for p in RUN.glob('*.py')}, config_sha256=digest(RUN / 'config.json'),
        data_split_sha256=converted['spec']['split_sha256'],
        gate='native reset400: terminal Back >=16/20 on dev640-659 and confirm700-719',
        stable_metric='reported separately; terminal plus >=45/last50 Back and no both-finger contact in last10',
        historical_evaluation='unchanged stage1_r0_eval, front common states0-19, native select_action; separately reported',
        training_rng='evaluation is a separate process on GPU1; no intermediate validation RNG consumption in trainer',
        no_automatic_extension=True))


def evaluate(step):
    import gzip
    import torch
    sys.path.insert(0, str(ARCHIVE / 'k25_whole4_new_20261124_01'))
    import whole4_new as audit
    torch.set_num_threads(4)
    checkpoint = RUN / 'training/checkpoints' / f'{step:06d}'
    marker = read(checkpoint / 'CHECKPOINT_COMPLETE.json')
    assert marker['status'] == 'pass' and marker['step'] == step
    training = read(RUN / 'training/manifest.json')
    assert training['normalization_source'] == str(ROOT / 'datasets/stage1_single_back_train_aligned_v1')
    assert 'initialization_checkpoint' not in training
    weights_sha = digest(checkpoint / 'model.safetensors')
    lab = audit.ref.old.original.k21.Lab(family='scene2', models=False)
    # Only this checkpoint is loaded. Its own saved Back pre/post-processors are used.
    rt = audit.ref.old.original.k21.km.make_runtime(lab.ctx, {'FT4': str(checkpoint)}, ['F'])
    lab.rt = audit.ref.old.DualRuntime(rt, rt)
    lab.text = {g: lab.ctx.all_texts[g] for g in audit.ref.ALL}
    lab.alpha = 1.
    assert rt.paths == {'F': str(checkpoint)}
    groups = {}
    try:
        for group in ('dev', 'confirm'):
            cases = read(RUN / f'{group}_cases.json')
            rows = []
            path = RUN / f'{group}_{step}.jsonl.gz'
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), gzip.open(path, 'xt') as log:
                for case in cases:
                    row = audit.ref.episode(lab, case, 'back', 'native')
                    row.update(experiment='B0-Back-Hist', profile='Back_only_public_init', checkpoint_step=step)
                    success = row['trace']['success']
                    row['stable_released_final'] = bool(row['terminal']['back'] and
                        sum(bool(v & 8) for v in success[-50:]) >= 45 and
                        all(v != 3 for v in row['trace']['contacts'][-10:]))
                    log.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
                    log.flush()
                    rows.append(row)
                    print('HIST_NATIVE', step, group, case['init'], row['ever']['back'],
                          row['terminal']['back'], row['stable_released_final'], flush=True)
            result = dict(n=len(rows), terminal=sum(r['terminal']['back'] for r in rows),
                          ever=sum(r['ever']['back'] for r in rows),
                          first=sum(r['first'] == 'back' for r in rows),
                          stable_released_final=sum(r['stable_released_final'] for r in rows),
                          rows_sha256=digest(path), weights_sha256=weights_sha)
            assert result['n'] == 20
            write(RUN / f'{group}_{step}_COMPLETE.json', result)
            groups[group] = result
            if group == 'dev' and result['terminal'] < 16:
                break
    finally:
        lab.ctx.env.close()
    assert weights_sha == digest(checkpoint / 'model.safetensors')
    passed = len(groups) == 2 and all(v['terminal'] >= 16 for v in groups.values())
    write(RUN / f'eval_{step}_COMPLETE.json', dict(step=step, qualified=passed, groups=groups,
          checkpoint=str(checkpoint), completed_at_utc=now(), weights_sha256=weights_sha))


def controller():
    prepare()
    trainer_log = (RUN / 'train_console.log').open('x')
    command = [sys.executable, '-u', str(RUN / 'historical_stage1_r0_train.py'),
               '--config', str(RUN / 'config.json'), '--run-dir', str(RUN / 'training')]
    trainer = subprocess.Popen(command, cwd=ROOT, env=env(0), stdout=trainer_log, stderr=subprocess.STDOUT)
    write(RUN / 'LAUNCHED.json', dict(pid=trainer.pid, command=command, started_at_utc=now()))
    evaluator = None
    evaluator_log = None
    current = None
    attempts = []
    qualified = None
    processed = set()
    try:
        while True:
            if evaluator is not None and evaluator.poll() is not None:
                evaluator_log.close()
                assert evaluator.returncode == 0, f'Evaluation {current} failed; see eval_{current}.log'
                result = read(RUN / f'eval_{current}_COMPLETE.json')
                attempts.append(result)
                write(RUN / 'attempts.json', attempts)
                processed.add(current)
                evaluator = None
                if result['qualified']:
                    qualified = dict(code=f'B0-Back-Hist-{current:06d}', **result)
                    write(RUN / 'QUALIFIED.json', qualified)
                    # Use the original trainer's graceful save-and-stop mechanism.
                    if trainer.poll() is None:
                        write(RUN / 'training/STOP_REQUESTED.json', dict(reason='terminal gate passed', at_utc=now()))
            train_exit = trainer.poll()
            if train_exit is not None:
                assert train_exit == 0, 'Historical trainer failed; see train_console.log'
            if evaluator is None and qualified is None:
                pending = [step for step in STEPS if step not in processed and
                           (RUN / 'training/checkpoints' / f'{step:06d}' / 'CHECKPOINT_COMPLETE.json').exists()]
                if pending:
                    current = pending[0]
                    evaluator_log = (RUN / f'eval_{current}.log').open('x')
                    evaluator = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), 'evaluate', '--step', str(current)],
                          cwd=RUN, env=env(1), stdout=evaluator_log, stderr=subprocess.STDOUT)
            write(RUN / 'status.json', dict(status='training_and_validation' if train_exit is None else 'validation',
                updated_at_utc=now(), trainer_pid=trainer.pid, training_exit_code=train_exit,
                evaluation_step=current if evaluator is not None else None,
                completed_evaluations=sorted(processed), qualified=qualified is not None))
            if train_exit is not None and evaluator is None:
                if qualified or all(step in processed for step in STEPS):
                    break
                raise RuntimeError('Training ended before all scheduled checkpoints were saved')
            time.sleep(5)
        # Reproduce the exact historical evaluation API/reset/noise protocol on the selected
        # checkpoint, or on the last checkpoint when none passes the cross-state gate.
        selected = qualified['step'] if qualified else 24000
        checkpoint = RUN / 'training/checkpoints' / f'{selected:06d}'
        with (RUN / 'historical_evaluation.log').open('x') as log:
            cmd = [sys.executable, '-u', str(RUN / 'historical_stage1_r0_eval.py'), 'evaluate',
                   '--config', str(RUN / 'config.json'), '--checkpoint', str(checkpoint),
                   '--output', str(RUN / 'historical_dev20'), '--partition', 'development',
                   '--instruction-goals', 'back', '--trace']
            write(RUN / 'status.json', dict(status='historical_evaluation', step=selected, updated_at_utc=now()))
            subprocess.run(cmd, cwd=ROOT, env=env(1), stdout=log, stderr=subprocess.STDOUT, check=True)
        write(RUN / 'COMPLETE.json', dict(code='B0-Back-Hist', qualified=qualified is not None,
              selected_step=selected, attempts=attempts,
              historical_evaluation=read(RUN / 'historical_dev20/evaluation.json'),
              finished_at_utc=now()))
        write(RUN / 'status.json', dict(status='qualified' if qualified else 'budget_complete_not_qualified',
                                       step=selected, qualified=qualified is not None, updated_at_utc=now()))
    except BaseException:
        if trainer.poll() is None:
            # Preserve a complete checkpoint rather than leave a failed orchestration training indefinitely.
            stop = RUN / 'training/STOP_REQUESTED.json'
            if stop.parent.exists():
                write(stop, dict(reason='controller failure', at_utc=now()))
        raise
    finally:
        trainer_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('controller', 'evaluate'))
    parser.add_argument('--step', type=int)
    args = parser.parse_args()
    try:
        if args.mode == 'controller':
            controller()
        else:
            evaluate(args.step)
    except BaseException as error:
        import traceback
        write(RUN / f'FAILED_{args.mode}.json', dict(error=str(error), traceback=traceback.format_exc(), at_utc=now()))
        raise
