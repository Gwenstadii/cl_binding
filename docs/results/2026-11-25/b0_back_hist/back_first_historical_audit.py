"""Read-only analysis of the Back-first historical-recipe run (no policy execution)."""
import argparse
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import inspect
from datetime import datetime, timezone


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def terminal_demo_statistics(root):
    """Compare only training-demo labels; do not simulate or judge expert success."""
    import h5py
    import numpy as np
    from stage1_smoke import sources
    raw = sources(['left','front','right','back'])
    conversions = {
        'left': root / 'datasets/stage1_dual_left_right_train_aligned_v1/r0_conversion.json',
        'right': root / 'datasets/stage1_dual_left_right_train_aligned_v1/r0_conversion.json',
        'front': root / 'datasets/stage1_single_front_train_aligned_v1/r0_conversion.json',
        'back': root / 'datasets/stage1_single_back_train_aligned_v1/r0_conversion.json'}
    report = {}
    for goal, source in raw.items():
        conversion = read(conversions[goal])
        episodes = [e for e in conversion['episodes'] if e['goal']==goal]
        assert len(episodes)==40
        rows = []
        aligned_actions, aligned_states = [], []
        with h5py.File(source) as stream:
            for episode in episodes:
                demo = stream['data'][episode['demo']]
                actions = demo['actions'][:].astype(np.float64)
                assert len(actions)==episode['frames']+1
                if goal=='back':
                    aligned_actions.append(actions[1:].astype(np.float32))
                    aligned_states.append(np.concatenate((demo['obs']['ee_states'][:-1],
                                                          demo['obs']['gripper_states'][:-1]),axis=1).astype(np.float32))
                rows.append(dict(demo=episode['demo'], aligned_frames=episode['frames'], raw_actions=len(actions),
                    last_action=actions[-1].tolist(),
                    last_translation_l2=float(np.linalg.norm(actions[-1,:3])),
                    last_rotation_l2=float(np.linalg.norm(actions[-1,3:6])),
                    last10_translation_l2_mean=float(np.linalg.norm(actions[-10:,:3],axis=1).mean()),
                    last10_rotation_l2_mean=float(np.linalg.norm(actions[-10:,3:6],axis=1).mean())))
        report[goal] = dict(source=str(source), conversion_sha256=sha(conversions[goal]),
            n=40, positive_final_gripper=sum(r['last_action'][-1]>0 for r in rows),
            aligned_frames_min=min(r['aligned_frames'] for r in rows),
            aligned_frames_max=max(r['aligned_frames'] for r in rows),
            median_final_translation_l2=float(np.median([r['last_translation_l2'] for r in rows])),
            median_final_rotation_l2=float(np.median([r['last_rotation_l2'] for r in rows])),
            median_last10_translation_l2_mean=float(np.median([r['last10_translation_l2_mean'] for r in rows])),
            median_last10_rotation_l2_mean=float(np.median([r['last10_rotation_l2_mean'] for r in rows])), rows=rows)
        if goal=='back':
            metadata = read(conversions[goal].parent/'meta/stats.json')
            checks = {}
            for feature, values in [('action',aligned_actions),('observation.state',aligned_states)]:
                array=np.concatenate(values).astype(np.float64)
                for key,actual in [('mean',array.mean(0)),('std',array.std(0))]:
                    expected=np.asarray(metadata[feature][key],dtype=np.float64)
                    equal=bool(np.allclose(actual,expected,atol=1e-5,rtol=1e-5))
                    assert equal,(feature,key,actual,expected)
                    checks[feature+'.'+key]=dict(matches_raw_aligned_training_data=True,
                        max_absolute_difference=float(np.max(np.abs(actual-expected))))
            report[goal]['own_statistics_raw_training_data_verification']=checks
    return report


def historical_helpers(root, commit):
    import stage1_smoke
    import stage1_protocol
    result = {}
    for module, name in [(stage1_smoke,'load_policy'), (stage1_smoke,'metadata_path'),
                         (stage1_protocol,'balanced_indices')]:
        relative = 'scripts/' + module.__name__ + '.py'
        previous = subprocess.check_output(['git','-C',str(root),'show',commit+':'+relative],text=True)
        old = next(n for n in ast.parse(previous).body if isinstance(n,ast.FunctionDef) and n.name==name)
        current = ast.parse(inspect.getsource(getattr(module,name))).body[0]
        equal = ast.dump(old,include_attributes=False)==ast.dump(current,include_attributes=False)
        result[module.__name__+'.'+name] = dict(current_module=str(Path(module.__file__).resolve()),
                current_module_sha256=sha(module.__file__), historical_function_ast_equal=equal)
        assert equal, (module.__name__,name,'historical helper changed')
    return result


def audit(run):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    root = Path('/home/lzx/cl_binding_stage2a')
    history = Path('/home/lzx/cl_binding/.server/r1_left_right_20261104_01/train_r1')
    old = Path('/home/lzx/cl_binding/.server/k25_b0_back_20261125_01')
    config = read(run / 'config.json')
    historical_manifest = read(run / 'historical_r1_manifest.json')
    source = subprocess.check_output(['git', '-C', str(root), 'show',
                    historical_manifest['git_commit'] + ':scripts/stage1_r0_train.py'])
    source_sha = hashlib.sha256(source).hexdigest()
    assert source_sha == sha(run / 'historical_stage1_r0_train.py')
    eval_blob = subprocess.check_output(['git','-C',str(root),'show',
                    historical_manifest['git_commit']+':scripts/stage1_r0_eval.py'])
    eval_source_sha = hashlib.sha256(eval_blob).hexdigest()
    assert eval_source_sha == sha(run/'historical_stage1_r0_eval.py')
    helpers = historical_helpers(root, historical_manifest['git_commit'])
    from stage1_smoke import find_model
    public = Path(find_model('lerobot/smolvla_base'))
    public_sha = sha(public/'model.safetensors')
    assert public_sha == read(old/'manifest.json')['base_sha256']
    own_stats = read(root / config['data']['train_output'] / 'meta/stats.json')
    lr_stats = read(root / 'datasets/stage1_dual_left_right_train_aligned_v1/meta/stats.json')
    conversion = read(root / config['data']['train_output'] / 'r0_conversion.json')
    lengths = [e['frames'] for e in conversion['episodes']]
    tails = [math.ceil(n * .3) for n in lengths]
    old_tail_fraction = .5 + .5 * sum(t/n for t,n in zip(tails,lengths)) / len(lengths)
    new_tail_fraction = sum(tails) / sum(lengths)
    old_lr = [5e-5*k/300 if k<=300 else 5e-6 + .5*(5e-5-5e-6)*
              (1+math.cos(math.pi*(k-300)/(24000-300))) for k in range(1,24001)]
    new_lr = [1e-4*min(1.,k/100) for k in range(1,24001)]
    recipe_changes = dict(expected_training_tail30_fraction_previous_B0=old_tail_fraction,
                         expected_training_tail30_fraction_historical_recipe=new_tail_fraction,
                         nominal_24000_lr_sum_previous_B0=sum(old_lr),
                         nominal_24000_lr_sum_historical_recipe=sum(new_lr),
                         nominal_24000_lr_sum_ratio=sum(new_lr)/sum(old_lr),
                         previous_seed=20261124, historical_seed=config['seed'],
                         caution='LR sums are descriptive, not effective parameter-update distances; AdamW gradients and moments differ.')
    stats = {}
    for feature in ('action', 'observation.state'):
        bm, bs, lm, ls = [np.asarray(d[feature][k], dtype=np.float64) for d, k in
                          [(own_stats, 'mean'), (own_stats, 'std'), (lr_stats, 'mean'), (lr_stats, 'std')]]
        stats[feature] = dict(back_mean=bm.tolist(), back_std=bs.tolist(), lr_mean=lm.tolist(), lr_std=ls.tolist(),
                             back_mean_in_lr_units=((bm-lm)/ls).tolist(),
                             back_std_over_lr_std=(bs/ls).tolist())
    checkpoints = {}
    reference_config = read(history / 'checkpoints/024000/config.json')
    for cp in sorted((run / 'training/checkpoints').glob('*')):
        if not (cp / 'CHECKPOINT_COMPLETE.json').exists():
            continue
        normal = load_file(str(cp / 'policy_preprocessor_step_5_normalizer_processor.safetensors'))
        checks = {}
        for feature in ('action', 'observation.state'):
            for key in ('mean', 'std'):
                value = normal[f'{feature}.{key}']
                expected = torch.tensor(own_stats[feature][key], dtype=value.dtype)
                checks[f'{feature}.{key}'] = torch.equal(value.flatten(), expected.flatten())
        assert all(checks.values()), (cp, checks)
        saved_config = read(cp / 'config.json')
        config_keys = ['optimizer_betas','optimizer_eps','optimizer_weight_decay','optimizer_grad_clip_norm',
                       'chunk_size','n_action_steps','num_steps','n_obs_steps','normalization_mapping',
                       'max_action_dim','max_state_dim','adapt_to_pi_aloha','input_features','output_features']
        config_checks = {k:saved_config.get(k)==reference_config.get(k) for k in config_keys}
        assert all(config_checks.values()), (cp,config_checks)
        checkpoints[cp.name] = dict(marker=read(cp / 'CHECKPOINT_COMPLETE.json'),
                                   own_Back_training_statistics_bitwise=checks,
                                   historical_model_optimizer_config_equal=config_checks,
                                   model_sha256=sha(cp / 'model.safetensors'),
                                   normalizer_sha256=sha(cp / 'policy_preprocessor_step_5_normalizer_processor.safetensors'))
    trace_summary = {}
    paths = [(p, p.name) for p in sorted(run.glob('*.jsonl.gz'))
             if (run / (p.name.removesuffix('.jsonl.gz') + '_COMPLETE.json')).exists()]
    paths += [(old / f'dev_{step}.jsonl.gz', f'previous_B0_dev_{step}.jsonl.gz') for step in (12000,24000)]
    for path, label in paths:
        with gzip.open(path, 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        items = []
        for row in rows:
            trace = row['trace']; times = [i for i, m in enumerate(trace['success']) if m & 8]
            native_first = next((m for m in trace['success'] if m),0)
            items.append(dict(init=row['init'], first=row['first'], terminal=row['terminal']['back'],
                first_native_predicate_mask=native_first, first_native_unique_Back=native_first==8,
                ever=row['ever']['back'], stable=row['stable_released_final'],
                first_grasp=row['first_grasp'], grasp_confirmed_step=row['grasp_confirmed_step'],
                first_success_step=times[0]+1 if times else None,
                last_success_step=times[-1]+1 if times else None,
                first_success_contacts=trace['contacts'][times[0]] if times else None,
                first_success_gripper=trace['actions'][times[0]][-1] if times else None,
                last10_contacts=trace['contacts'][-10:],
                last10_gripper_commands=[a[-1] for a in trace['actions'][-10:]]))
        trace_summary[label] = dict(rows=items, n=len(items), sha256=sha(path),
            terminal=sum(x['terminal'] for x in items), ever=sum(x['ever'] for x in items),
            stable=sum(x['stable'] for x in items),
            first_native_unique_Back=sum(x['first_native_unique_Back'] for x in items),
            ever_but_failed_final=sum(x['ever'] and not x['terminal'] for x in items),
            first_success_both_contacts=sum(x['first_success_contacts']==3 for x in items))
    paired = {}
    for name, after in trace_summary.items():
        if not name.startswith('dev_'):
            continue
        right = {r['init']:r for r in after['rows']}
        for step in (12000,24000):
            before = trace_summary[f'previous_B0_dev_{step}.jsonl.gz']
            left = {r['init']:r for r in before['rows']}
            assert left.keys() == right.keys()
            paired[f'{name}_vs_previous_B0_{step}'] = {
                metric: dict(new_only=[i for i in left if right[i][metric] and not left[i][metric]],
                             old_only=[i for i in left if left[i][metric] and not right[i][metric]],
                             both=[i for i in left if left[i][metric] and right[i][metric]],
                             neither=[i for i in left if not left[i][metric] and not right[i][metric]])
                for metric in ('terminal','ever','stable')}
    prior = read(old / 'COMPLETE.json')
    train_manifest = read(run / 'training/manifest.json')
    assert 'initialization_checkpoint' not in train_manifest
    assert train_manifest['normalization_source'] == str(root / config['data']['train_output'])
    trainable = read(run / 'training/trainable_parameters.json')
    assert len(trainable['names']) == 155 and trainable['count'] == 99880992
    assert train_manifest['frozen_hash'] == '0c0f563fdbc566d8f9d7dd542776678eabd0ed9eb112ee4a8e3e9f7437dc4f5e'
    out = dict(audited_at_utc=datetime.now(timezone.utc).isoformat(),
        historical_trainer_git_blob_sha256=source_sha, historical_source_exact=True,
        historical_evaluator_git_blob_sha256=eval_source_sha,historical_evaluator_source_exact=True,
        historical_helpers=helpers,
        public_base=dict(path=str(public), model_sha256=public_sha,
                         config_sha256=sha(public/'config.json'),
                         same_initial_weight_file_as_previous_B0=True),
        historical_commit=historical_manifest['git_commit'], previous_B0=prior,
        config=config, manifest=read(run / 'manifest.json'), training_manifest=train_manifest,
        trainable_parameters=trainable,
        statistics_comparison=stats, recipe_changes=recipe_changes,
        verified_checkpoints=checkpoints, trace_summary=trace_summary, paired_comparisons=paired,
        training_demo_terminal_statistics=terminal_demo_statistics(root))
    if (run / 'COMPLETE.json').exists():
        out['complete'] = read(run / 'COMPLETE.json')
    if (run / 'QUALIFIED.json').exists():
        out['qualified'] = read(run / 'QUALIFIED.json')
    path = run / 'read_only_audit.json'
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(path=str(path), historical_source_exact=True,
                         checkpoint_count=len(checkpoints), statistics=stats,
                         trace_counts={k: {n:v[n] for n in ('n','terminal','ever','stable')} for k,v in trace_summary.items()}), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    audit(parser.parse_args().run)
