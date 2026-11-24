"""Complete fixed-endpoint evaluation after a verified singleton-stat guard false alarm."""

import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from stage1_protocol import GOALS
from stage1_sequential_run import compare_sequential, processor_hashes
from stage1_smoke import ROOT, CAMERAS, dump


def validate_normalization(parent, final, config):
    import torch
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    if processor_hashes(parent) != processor_hashes(final):
        raise ValueError('Normalization statistic names, dtypes, non-singleton shapes or values changed')
    for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        if json.loads((parent/name).read_text()) != json.loads((final/name).read_text()):
            raise ValueError(f'Processor configuration changed: {name}')
    cfg = SmolVLAConfig.from_pretrained(parent)
    pre1, post1 = make_pre_post_processors(cfg, pretrained_path=parent)
    pre2, post2 = make_pre_post_processors(cfg, pretrained_path=final)
    generator = torch.Generator().manual_seed(config['seed'])
    batch = {'observation.state': torch.randn(2, 8, generator=generator),
             'action': torch.randn(2, cfg.chunk_size, 7, generator=generator),
             'task': ['pick up the book and place it in the left compartment of the caddy',
                      'pick up the book and place it in the right compartment of the caddy']}
    for camera in CAMERAS:
        batch[camera] = torch.rand(2, 3, 128, 128, generator=generator)
    left, right = pre1(copy.deepcopy(batch)), pre2(copy.deepcopy(batch))
    checked = []
    for key in left:
        if isinstance(left[key], torch.Tensor):
            if key not in right or not torch.equal(left[key], right[key]):
                raise ValueError(f'Preprocessor behavior changed: {key}')
            checked.append(key)
    action = torch.randn(2, cfg.chunk_size, 7, generator=generator, device='cpu').to('cuda')
    if not torch.equal(post1(action.clone()), post2(action.clone())):
        raise ValueError('Postprocessor behavior changed')
    return {'status': 'pass', 'statistic_fingerprints_equal_after_singleton_canonicalization': True,
            'processor_configs_equal': True, 'preprocessor_tensor_outputs_equal': checked,
            'postprocessor_outputs_equal': True,
            'interpretation': 'The original guard treated serialization changes of scalar/count metadata as changed normalization. Non-singleton shapes, dtypes and numerical values remain checked.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    report_path = run/'evaluation_followup_summary.json'
    if report_path.exists():
        raise FileExistsError(report_path)
    original = json.loads((run/'run_summary.json').read_text())
    config = json.loads((run/'config.json').read_text())
    if original.get('error') != 'Normalization changed across tasks':
        raise ValueError('This continuation handles only the diagnosed normalization guard failure')
    parent = Path(original['initial_checkpoint'])
    final = Path(original['training_dir'])/'checkpoints'/f"{config['sequential']['fixed_endpoint_step']:06d}"
    finished = json.loads((final.parent.parent/'training_finished_006000.json').read_text())
    assert finished['step'] == finished['target'] == 6000 and finished['frozen_parameters_unchanged']
    assert json.loads((final/'CHECKPOINT_COMPLETE.json').read_text())['status'] == 'pass'
    assert hashlib.sha256((parent/'model.safetensors').read_bytes()).hexdigest() == original['initial_model_sha256']
    started = time.monotonic()
    report = {'status': 'running', 'training_source_commit': original['source_commit'],
              'evaluation_source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'fixed_checkpoint': str(final), 'original_run_summary_preserved': True,
              'retrained': False, 'phases': []}
    dump(report_path, report)
    workers = []
    handles = []
    env = {**os.environ, 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'MUJOCO_GL':'egl',
           'LIBERO_CONFIG_PATH':str(ROOT/'.server/libero'), 'TOKENIZERS_PARALLELISM':'false', 'OMP_NUM_THREADS':'4'}
    try:
        report['normalization_verification'] = validate_normalization(parent, final, config)
        dump(run/'normalization_verification.json', report['normalization_verification'])
        for gpu, goals in enumerate((('front','left'), ('right','back'))):
            name = f'after_paired20_part{gpu}'
            command = [sys.executable,'-u',str(ROOT/'scripts/stage1_r0_eval.py'),'evaluate',
                       '--config',str(run/'config.json'),'--checkpoint',str(final),'--partition','paired',
                       '--instruction-goals',*goals,'--output',str(run/name),'--trace']
            handle = (run/f'{name}.log').open('w')
            handles.append(handle)
            worker = subprocess.Popen(command,cwd=ROOT,env={**env,'CUDA_VISIBLE_DEVICES':str(gpu)},
                                      stdout=handle,stderr=subprocess.STDOUT)
            workers.append(worker)
            report['phases'].append({'name':name,'command':command,'gpu':gpu,'pid':worker.pid})
        dump(report_path,report)
        after = []
        for phase, worker in zip(report['phases'],workers,strict=True):
            phase['returncode'] = worker.wait(timeout=30*60)
            if phase['returncode']:
                raise RuntimeError(f"{phase['name']} failed; inspect log")
            after.extend(json.loads(line) for line in (run/phase['name']/'episodes.jsonl').read_text().splitlines())
        before = [json.loads(line) for line in (run/'before_paired20/episodes.jsonl').read_text().splitlines()]
        assert len(before) == len(after) == 80
        comparison = compare_sequential(before,after,config['sequential']['old_goal'],config['sequential']['new_goal'])
        dump(run/'sequential_comparison.json',comparison)
        merged = run/'after_paired20'
        merged.mkdir()
        (merged/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in after))
        dump(merged/'evaluation.json',{'status':'complete','merged_parallel_workers':True,
             'instruction_goals':list(GOALS),'episodes':len(after),'horizon':config['evaluation']['horizon'],
             'checkpoint':str(final),'normalization_source':config['data']['normalization_source'],
             'git_commit':report['evaluation_source_commit'],
             'task_success_counts':comparison['after']['task_success_counts']})
        report.update(status='complete',seconds=time.monotonic()-started,
                      finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      old_retention=comparison['old_goal_previously_successful_ever'],
                      new_goal_counts=comparison['after']['task_success_counts'][config['sequential']['new_goal']])
        dump(report_path,report)
        dump(run/'EVALUATION_COMPLETE.json',{'status':'complete','retrained':False,'episodes_before':80,'episodes_after':80})
    except BaseException as error:
        report.update(status='failed',error=str(error),seconds=time.monotonic()-started)
        dump(report_path,report)
        raise
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
                try: worker.wait(timeout=30)
                except subprocess.TimeoutExpired: worker.kill();worker.wait()
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
