"""Independent local audit: no imports from project reporting/evaluation code."""
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/results/2026-10-23'
RUN = BASE / 'dual_right_then_back_v1'
PREV = BASE / 'dual12000_then_right_v1'
GOALS = ['front', 'left', 'right', 'back']
INDICES = list(range(20, 40))
ERRORS = []
CHECKS = []

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check(name, condition, detail=None):
    CHECKS.append({'check': name, 'passed': bool(condition)})
    if not condition:
        ERRORS.append({'check': name, 'detail': detail})

def exact(name, actual, expected):
    check(name, actual == expected, {'actual': actual, 'expected': expected})

def indexed(items):
    return {(r['requested_goal'], r['init_index']): r for r in items}

def ever(r):
    return r['first_goal_times'][r['requested_goal']] is not None

def final(r):
    return r['terminal_goal'] == r['requested_goal']

def first_from_times(r):
    times = r['first_goal_times']
    reached = {g: t for g, t in times.items() if t is not None}
    if not reached:
        return None
    first_time = min(reached.values())
    earliest = [g for g, t in reached.items() if t == first_time]
    return earliest[0] if len(earliest) == 1 else 'ambiguous'

def validate_rows(name, items, expected_goals):
    exact(name + '.unique_pairs', len(indexed(items)), len(items))
    exact(name + '.pairs', sorted(indexed(items)), sorted((g, i) for g in expected_goals for i in INDICES))
    for r in items:
        k = f"{name}.{r['requested_goal']}.{r['init_index']}"
        exact(k + '.seed', r['seed'], 20261021 + r['init_index'])
        exact(k + '.steps', r['steps'], 400)
        exact(k + '.time_goal_keys', sorted(r['first_goal_times']), sorted(GOALS))
        exact(k + '.first_goal', r['first_goal'], first_from_times(r))
        exact(k + '.correct_ever', r['correct_ever'], ever(r))
        exact(k + '.correct_then_destroyed', r['correct_then_destroyed'], ever(r) and not final(r))
        category = 'correct' if final(r) else ('unfinished' if r['terminal_goal'] is None else ('ambiguous' if r['terminal_goal'] == 'ambiguous' else 'wrong_goal'))
        exact(k + '.terminal_category', r['terminal_category'], category)
        for g, t in r['first_goal_times'].items():
            check(k + '.time_range.' + g, t is None or isinstance(t, int) and 0 <= t <= 400)
        check(k + '.hash_lengths', all(len(r[field]) == 64 for field in ('initial_state_sha256', 'scene_xml_sha256')))

def condition(r):
    return {k: r[k] for k in ('init_index', 'seed', 'initial_state_sha256', 'scene_xml_sha256', 'steps')}

def summary(items):
    by = indexed(items)
    out = {
        'episodes': len(items),
        'initial_states': len({r['init_index'] for r in items}),
        'indices': sorted({r['init_index'] for r in items}),
        'task_success_counts': {},
        'first_matrix': {},
        'terminal_matrix': {},
    }
    for g in GOALS:
        selected = [r for r in items if r['requested_goal'] == g]
        fc = collections.Counter(first_from_times(r) or 'unreached' for r in selected)
        tc = collections.Counter(r['terminal_goal'] or 'unfinished' for r in selected)
        out['task_success_counts'][g] = {
            'total': len(selected),
            'success_ever': sum(ever(r) for r in selected),
            'terminal_correct': sum(final(r) for r in selected),
            'correct_then_destroyed': sum(ever(r) and not final(r) for r in selected),
            'first_targets': dict(fc),
            'terminal_targets': dict(tc),
            'ever_reached_targets': {t: sum(r['first_goal_times'][t] is not None for r in selected) for t in GOALS},
        }
        out['first_matrix'][g] = {t: fc[t] for t in GOALS + ['unreached', 'ambiguous']}
        out['terminal_matrix'][g] = {t: tc[t] for t in GOALS + ['unfinished', 'ambiguous']}
    joint = [i for i in INDICES if ('left', i) in by and ('front', i) in by and final(by['left', i]) and final(by['front', i])]
    out['joint_left_front_terminal_correct'] = len(joint)
    out['joint_left_front_terminal_correct_indices'] = joint
    return out

def retention(before, after, g):
    b, a = indexed(before), indexed(after)
    out = {}
    for label, predicate in [('previously_successful_ever', ever), ('previously_terminal_correct', final)]:
        eligible = [i for i in INDICES if predicate(b[g, i])]
        kept = [i for i in eligible if predicate(a[g, i])]
        lost = [i for i in eligible if i not in kept]
        selected = [a[g, i] for i in lost]
        out[label] = {
            'total': len(eligible), 'retained': len(kept), 'lost': len(lost),
            'retention_rate': len(kept) / len(eligible) if eligible else None,
            'eligible_indices': eligible, 'retained_indices': kept, 'lost_indices': lost,
            'lost_first_new_target': sum(first_from_times(r) == 'back' for r in selected),
            'lost_terminal_new_target': sum(r['terminal_goal'] == 'back' for r in selected),
            'lost_terminal_unfinished': sum(r['terminal_goal'] is None for r in selected),
            'lost_terminal_ambiguous': sum(r['terminal_goal'] == 'ambiguous' for r in selected),
            'lost_terminal_other_wrong': sum(r['terminal_goal'] not in (None, 'ambiguous', 'back', g) for r in selected),
        }
        exact(f'retention_partition.{g}.{label}', sum(out[label][k] for k in ('lost_terminal_new_target', 'lost_terminal_unfinished', 'lost_terminal_ambiguous', 'lost_terminal_other_wrong')), len(lost))
    return out

def group_indices(items, predicate):
    return sorted(r['init_index'] for r in items if predicate(r))

manifest = read_json(RUN / 'evidence_manifest.json')
file_checks = []
for ent in manifest:
    p = RUN / ent['archive_path']
    result = {'archive_path': ent['archive_path'], 'expected_bytes': ent['bytes'], 'expected_sha256': ent['sha256'], 'actual_bytes': p.stat().st_size, 'actual_sha256': sha(p)}
    result['passed'] = result['actual_bytes'] == result['expected_bytes'] and result['actual_sha256'] == result['expected_sha256']
    check('manifest.' + ent['archive_path'], result['passed'])
    file_checks.append(result)
exact('manifest.no_duplicate_entries', len({x['archive_path'] for x in manifest}), len(manifest))

before = rows(RUN / 'before/episodes.jsonl')
after = rows(RUN / 'after/episodes.jsonl')
cached = rows(RUN / 'baselines/parent_cached60.jsonl')
acquisition = rows(RUN / 'baselines/joint_acquisition60.jsonl')
before_back = rows(RUN / 'before_back_attempt1_part1/episodes.jsonl')
old_after = rows(PREV / 'after/episodes.jsonl')
old_before = rows(PREV / 'before/episodes.jsonl')
original_back = rows(PREV / 'parent_unseen_back/episodes.jsonl')
original80 = old_before + original_back

validate_rows('before', before, GOALS)
validate_rows('after', after, GOALS)
validate_rows('original80', original80, GOALS)
validate_rows('cached60', cached, ['front', 'left', 'right'])
validate_rows('acquisition60', acquisition, ['front', 'left', 'right'])
validate_rows('before_back20', before_back, ['back'])
exact('cached60.byte_identical_previous_after', (RUN / 'baselines/parent_cached60.jsonl').read_bytes(), (PREV / 'after/episodes.jsonl').read_bytes())
exact('acquisition60.byte_identical_previous_before', (RUN / 'baselines/joint_acquisition60.jsonl').read_bytes(), (PREV / 'before/episodes.jsonl').read_bytes())
exact('merged_before.cached60_records_identical', indexed([r for r in before if r['requested_goal'] != 'back']), indexed(old_after))
exact('merged_before.back20_records_identical', indexed([r for r in before if r['requested_goal'] == 'back']), indexed(before_back))
exact('merged_after.worker_records_identical', indexed(after), indexed(rows(RUN / 'after_attempt1_part0/episodes.jsonl') + rows(RUN / 'after_attempt1_part1/episodes.jsonl')))
bi, ai, oi = indexed(before), indexed(after), indexed(original80)
for key in bi:
    exact('paired.before_after.' + str(key), condition(bi[key]), condition(ai[key]))
    exact('paired.original_before.' + str(key), condition(oi[key]), condition(bi[key]))
for i in INDICES:
    for snapshot_name, by in [('original', oi), ('before', bi), ('after', ai)]:
        for g in GOALS:
            exact(f'within_snapshot.{snapshot_name}.{g}.{i}', condition(by[g, i]), condition(by['front', i]))

bs, ass, os = summary(before), summary(after), summary(original80)
run = read_json(RUN / 'run_summary.json')
for stage, computed in [('before', bs), ('after', ass)]:
    exact('summary.file.' + stage, computed, read_json(RUN / stage / 'summary.json'))
    exact('summary.run.' + stage, computed, run[stage])

previous_retention = {g: retention(before, after, g) for g in ['left', 'front', 'right']}
comparison_previous = {'before': bs, 'after': ass, 'matched_pairs': 80, 'retained_goals': previous_retention}
exact('comparison_to_previous.full', comparison_previous, read_json(RUN / 'comparison_to_previous.json'))
history = [r for r in acquisition if r['requested_goal'] in ('left', 'front')] + [r for r in cached if r['requested_goal'] == 'right']
acquisition_retention = {g: retention(history, after, g) for g in ['left', 'front', 'right']}
comparison_acquisition = {'new_goal': 'back', 'matched_pairs': 60, 'retained_goals': acquisition_retention}
exact('comparison_to_acquisition.full', comparison_acquisition, read_json(RUN / 'comparison_to_acquisition.json'))

workers = {}
for name in ['before_back_attempt1_part1', 'after_attempt1_part0', 'after_attempt1_part1']:
    evaluation = read_json(RUN / name / 'evaluation.json')
    episodes = rows(RUN / name / 'episodes.jsonl')
    phase = next(x for x in run['phases'] if x['name'] == name)
    exact('worker.complete.' + name, evaluation['status'], 'complete')
    exact('worker.exit0.' + name, phase['returncode'], 0)
    exact('worker.horizon.' + name, evaluation['horizon'], 400)
    exact('worker.n_action_steps.' + name, evaluation['n_action_steps'], 10)
    exact('worker.source.' + name, evaluation['git_commit'], run['source_commit'])
    exact('worker.counts_terminal_correct.' + name, evaluation['counts'], {g: {'total': sum(r['requested_goal'] == g for r in episodes), 'correct': sum(r['requested_goal'] == g and final(r) for r in episodes)} for g in evaluation['instruction_goals']})
    expected_checkpoint = run['initial_checkpoint'] if name.startswith('before') else run['final_checkpoint']
    exact('worker.checkpoint.' + name, evaluation['checkpoint'], expected_checkpoint)
    workers[name] = {'status': evaluation['status'], 'returncode': phase['returncode'], 'episodes': len(episodes), 'checkpoint': evaluation['checkpoint'], 'attempt': 1}
exact('new_worker_episode_count', sum(w['episodes'] for w in workers.values()), 100)
for phase in run['phases']:
    exact('phase.exit0.' + phase['name'], phase['returncode'], 0)
exact('run.status', run['status'], 'complete')
exact('run.current_stage', run['current_stage'], 'finished')
exact('run.fixed_training_steps', run['fixed_training_steps'], 12000)
exact('run.new_goal', run['new_goal'], 'back')
for key, val in [('cached_parent_evaluation_episodes', 60), ('planned_new_evaluation_episodes', 100), ('accepted_evaluation_episodes', 160), ('accepted_new_evaluation_episodes', 100), ('actual_completed_new_worker_episodes_including_runtime_retries', 100)]:
    exact('run.' + key, run[key], val)
complete = read_json(RUN / 'COMPLETE.json')
exact('COMPLETE.status', complete['status'], 'complete')
exact('COMPLETE.before_after160', complete['accepted_before_and_after_episodes'], 160)
exact('COMPLETE.new100', complete['new_model_evaluation_episodes'], 100)
exact('COMPLETE.cached60', complete['cached_episodes'], 60)

markers = {}
for step in [3000, 6000, 9000, 12000]:
    marker = read_json(RUN / 'training/checkpoints' / f'{step:06d}' / 'CHECKPOINT_COMPLETE.json')
    exact('checkpoint.step.' + str(step), marker['step'], step)
    exact('checkpoint.status.' + str(step), marker['status'], 'pass')
    exact('checkpoint.reload.' + str(step), marker['reload_max_difference'], 0.0)
    exact('checkpoint.frozen.' + str(step), marker['frozen_hash_unchanged'], True)
    exact('checkpoint.optimizer_rng.' + str(step), marker['optimizer_and_rng_saved'], True)
    markers[str(step)] = marker
finished = read_json(RUN / 'training/training_finished_012000.json')
exact('training_finished.status', finished['status'], 'complete')
exact('training_finished.step', finished['step'], 12000)
exact('training_finished.frozen', finished['frozen_parameters_unchanged'], True)
training_manifest = read_json(RUN / 'training/manifest.json')
exact('training_manifest.source', training_manifest['git_commit'], run['source_commit'])
exact('training_manifest.initial_sha', training_manifest['initialization_model_sha256'], run['initial_model_sha256'])
exact('training_manifest.goal', training_manifest['conversion_spec']['goals'], ['back'])
exact('training_manifest.samples_per_goal', training_manifest['config']['training']['samples_per_goal'], 6)
exact('training_manifest.optimizer_reset', training_manifest['optimizer_reset'], True)
exact('training_manifest.scheduler_reset', training_manifest['scheduler_reset'], True)

server_independent = read_json(RUN / 'independent_training_check.json')
exact('independent_server.source_script_sha256', sha(RUN / 'independent_training_check_source.py'), server_independent['audit_script_sha256'])
exact('independent_server.pass', server_independent['status'], 'pass')
exact('independent_server.source', server_independent['source_commit'], run['source_commit'])
exact('independent_server.parent_sha', server_independent['parent_model_sha256'], run['initial_model_sha256'])
exact('independent_server.final_sha', server_independent['final_model_sha256'], run['final_model_sha256'])
exact('independent_server.markers', server_independent['four_checkpoint_acceptance_markers'], markers)
exact('independent_server.parent_normalization', server_independent['normalization_processor_hashes_unchanged'], training_manifest['parent_processor_sha256'])
exact('independent_server.finite_steps', server_independent['continuous_finite_training_updates'], 12000)
exact('independent_server.back_batch6', server_independent['all_batches_back_only_size6'], True)
exact('independent_server.optimizer_scheduler', server_independent['new_optimizer_and_scheduler'], True)

right_previously_final = [ai['right', i] for i in INDICES if final(bi['right', i])]
back_after = [r for r in after if r['requested_goal'] == 'back']
extras = {
    'previously_right_terminal_correct12': {
        'total': len(right_previously_final),
        'first_new_back_indices': group_indices(right_previously_final, lambda r: first_from_times(r) == 'back'),
        'terminal_new_back_indices': group_indices(right_previously_final, lambda r: r['terminal_goal'] == 'back'),
        'terminal_unfinished_indices': group_indices(right_previously_final, lambda r: r['terminal_goal'] is None),
    },
    'back_request_after': {
        'success_ever_indices': group_indices(back_after, ever),
        'terminal_correct_indices': group_indices(back_after, final),
        'correct_then_destroyed_indices': group_indices(back_after, lambda r: ever(r) and not final(r)),
        'never_reached_requested_target_indices': group_indices(back_after, lambda r: not ever(r)),
    },
    'all80_after': {
        'ever_reached_back': sum(r['first_goal_times']['back'] is not None for r in after),
        'first_back': sum(first_from_times(r) == 'back' for r in after),
        'terminal_back': sum(r['terminal_goal'] == 'back' for r in after),
        'terminal_unfinished': sum(r['terminal_goal'] is None for r in after),
        'ever_reached_any_front_left_right': sum(any(r['first_goal_times'][g] is not None for g in ('front', 'left', 'right')) for r in after),
    },
}
result = {
    'status': 'pass' if not ERRORS else 'fail',
    'verified_at': dt.datetime.now(dt.timezone.utc).isoformat(),
    'method': 'Independent Python standard-library recomputation, no imports or calls to project summary/comparison functions. First arrival and ever-success independently derived from first_goal_times. Terminal outcomes use recorded episode terminal_goal labels; raw frame/step traces were not retrieved or re-simulated locally.',
    'audit_script_sha256': sha(Path(__file__)),
    'source_commit': run['source_commit'],
    'checks_count': len(CHECKS),
    'failed_checks': ERRORS,
    'manifest_verification': {'entries': len(file_checks), 'all_sha256_and_byte_lengths_match': all(x['passed'] for x in file_checks), 'files': file_checks},
    'episode_accounting': {'new_completed_worker_rollouts': 100, 'cached_parent_rollouts': 60, 'current_before_after_rows': 160, 'reused_historical_acquisition60_not_new': 60, 'original_joint_snapshot_back_supplement20_not_new': 20, 'new_worker_retries': 0, 'note': 'Historical acquisition60 and original-back supplement20 were evaluated in earlier experiments. They add no new rollouts to this stage. Acquisition reference mixes left/front at original joint snapshot and right at previous snapshot; it is not a simultaneous three-goal-success snapshot.'},
    'pairing': {'before_after_matched_pairs': 80, 'three_real_four_instruction_snapshots': 3, 'each_snapshot_episodes': 80, 'all_240_rows_same_init_state_xml_seed_horizon_across_snapshots_and_instructions': True, 'indices': INDICES, 'horizon': 400, 'seed_rule': '20261021 + init_index', 'cached_parent60_byte_identical_E007_after': True, 'cached_acquisition60_byte_identical_E007_before': True, 'before_merge_and_after_worker_merge_identical': True},
    'actual_run': {'status': run['status'], 'current_stage': run['current_stage'], 'started_at': run['started_at'], 'finished_at': run['finished_at'], 'elapsed_seconds': run['seconds'], 'fixed_training_steps': 12000, 'starting_point_formally_qualified': run['starting_point_formally_qualified'], 'all_six_recorded_phases_exit0': all(p['returncode'] == 0 for p in run['phases']), 'workers': workers, 'checkpoint_markers': markers, 'parent_sha256': run['initial_model_sha256'], 'final_sha256': run['final_model_sha256'], 'new_task_descriptive_threshold_ever_and_final_at_least16': False},
    'independent_server_training_check': {'status': server_independent['status'], 'json_sha256': sha(RUN / 'independent_training_check.json'), 'source_sha256_verified': server_independent['audit_script_sha256'], 'scope': 'Separate server-side audit reads full training trace and model/processor files. This local audit verifies script SHA256 and consistency of its result with manifests/checkpoint markers; it did not independently reread the full trace/model files.', 'reported_continuous_finite_updates': server_independent['continuous_finite_training_updates'], 'reported_back_only_batches_size6': server_independent['all_batches_back_only_size6'], 'reported_new_optimizer_scheduler': server_independent['new_optimizer_and_scheduler'], 'reported_training_trace_sha256': server_independent['training_trace_sha256']},
    'summaries_recomputed': {'original_joint_snapshot80': os, 'previous_right_snapshot80': bs, 'new_back_snapshot80': ass},
    'comparison_to_previous_recomputed': comparison_previous,
    'comparison_to_acquisition_recomputed': comparison_acquisition,
    'additional_findings': extras,
}
(RUN / 'result_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': result['status'], 'checks_count': len(CHECKS), 'failed_checks': ERRORS, 'additional_findings': extras}, ensure_ascii=False, indent=2, default=str))
if ERRORS:
    raise SystemExit(1)
