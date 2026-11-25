"""Evaluate an archived B0 partial checkpoint with the unchanged historical API.

Provides the matched historical-protocol archived control. It performs no
training or action replacement.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main(run, step):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    archive = Path('/home/lzx/cl_binding/.server')
    baseline = archive / 'k25_b0_back_20261125_01' / f'{step:06d}' / 'trainable.safetensors'
    container = archive / 'r1_left_right_20261104_01/train_r1/checkpoints/024000'
    source = run / 'historical_stage1_r0_eval.py'
    spec = importlib.util.spec_from_file_location('frozen_historical_eval', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_load = module.load_policy
    proof = {}

    def load_baseline(_checkpoint):
        from stage1_r0_train import frozen_hash
        assert _checkpoint == container
        policy = original_load(container)
        params = dict(policy.named_parameters())
        names = {name for name, parameter in params.items() if parameter.requires_grad}
        weights = load_file(str(baseline))
        assert len(names) == 155 and names == set(weights)
        with torch.no_grad():
            for name in names:
                params[name].copy_(weights[name])
            assert all(torch.equal(params[name].detach().cpu(), weights[name].to(params[name].dtype)) for name in names)
        proof.update(overwritten_task_tensors=155, all_archived_B0_tensors_equal=True,
                     container_trainable_weights_remaining=0,
                     processors='R1 statistics, exactly as used by previous B0',
                     no_parent_action_generation=True)
        frozen = frozen_hash(policy)
        assert frozen == '0c0f563fdbc566d8f9d7dd542776678eabd0ed9eb112ee4a8e3e9f7437dc4f5e'
        proof['frozen_hash'] = frozen
        return policy

    module.load_policy = load_baseline
    output = run / f'previous_B0_{step}_historical_dev20'
    args = SimpleNamespace(output=output, checkpoint=container, base_policy=False,
                           partition='development', limit=None, instruction_goals=['back'], trace=True)
    config = json.loads((run / 'config.json').read_text())
    module.evaluate(args, config)
    report = json.loads((output / 'evaluation.json').read_text())
    with baseline.open('rb') as stream:
        weights_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    report.update(checkpoint=str(baseline), frozen_container=str(container),
                  normalization_source=json.loads((container.parent.parent/'manifest.json').read_text())['normalization_source'],
                  runtime_processor_checkpoint=str(container),
                  evaluation_config_used_for_reset_and_partition_only=True,
                  policy_source='archived_B0_all155_task_tensors', reconstruction=proof,
                  weights_sha256=weights_sha,
                  historical_eval_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--step', type=int, choices=(12000,24000), required=True)
    args = parser.parse_args()
    main(args.run, args.step)
