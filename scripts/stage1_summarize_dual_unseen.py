"""Audit paired dual-goal adaptation and the frozen parent's four instructions."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

GOALS = ('front', 'left', 'right', 'back')
OLD = ('left', 'front')
NEW = ('right', 'back')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(items, goals, indices, seed):
    expected = {(i, g) for i in indices for g in goals}
    keyed = {(r['init_index'], r['requested_goal']): r for r in items}
    if len(items) != len(expected) or set(keyed) != expected:
        raise ValueError('Missing, duplicated or unexpected state/instruction pair')
    for i in indices:
        for field in ('initial_state_sha256', 'scene_xml_sha256', 'seed', 'steps'):
            if len({keyed[i, g][field] for g in goals}) != 1:
                raise ValueError(f'Unmatched {field} at state {i}')
        if keyed[i, goals[0]]['seed'] != seed+i or keyed[i, goals[0]]['steps'] != 400:
            raise ValueError('Wrong seed or horizon')
    if len({keyed[i, goals[0]]['initial_state_sha256'] for i in indices}) != len(indices):
        raise ValueError('Repeated physical state under different indices')
    if len({r['scene_xml_sha256'] for r in items}) != 1:
        raise ValueError('Scene XML changed')
    for r in items:
        times = r['first_goal_times']
        if set(times) != set(GOALS) or r['correct_ever'] != (times[r['requested_goal']] is not None):
            raise ValueError('Inconsistent requested success')
        earliest = min((t for t in times.values() if t is not None), default=None)
        first = [g for g in GOALS if times[g] is not None and times[g] == earliest]
        if r['first_goal'] != (first[0] if len(first) == 1 else None):
            raise ValueError('Inconsistent first target')
        category, target = r['terminal_category'], r['terminal_goal']
        if category == 'correct' and (target != r['requested_goal'] or not r['correct_ever']):
            raise ValueError('Invalid correct terminal target')
        if category == 'wrong_goal' and (target not in GOALS or target == r['requested_goal']):
            raise ValueError('Invalid wrong terminal target')
        if category not in ('correct', 'wrong_goal', 'unfinished', 'ambiguous'):
            raise ValueError('Unknown terminal category')
        if category in ('unfinished', 'ambiguous') and target is not None:
            raise ValueError('Unexpected unique terminal target')
        if category != 'ambiguous' and r['correct_then_destroyed'] != (r['correct_ever'] and category != 'correct'):
            raise ValueError('Inconsistent loss of success state')
    return keyed


def counts(items):
    return {'total': len(items), 'success_ever': sum(r['correct_ever'] for r in items),
            'terminal_correct': sum(r['terminal_category'] == 'correct' for r in items),
            'correct_then_destroyed': sum(r['correct_then_destroyed'] for r in items),
            'first_targets': dict(Counter(r['first_goal'] or ('unreached' if all(t is None for t in r['first_goal_times'].values()) else 'ambiguous') for r in items)),
            'terminal_targets': dict(Counter(r['terminal_goal'] or r['terminal_category'] for r in items)),
            'ever_reached_targets': {g: sum(r['first_goal_times'][g] is not None for r in items) for g in GOALS}}


def parent_summary(items, indices, seed):
    keyed = validate(items, GOALS, indices, seed)
    joint = [i for i in indices if all(keyed[i, g]['terminal_category'] == 'correct' for g in OLD)]
    def intrusion(selected):
        return {'total': len(selected), 'requested_success_ever': sum(r['correct_ever'] for r in selected),
                'requested_terminal_correct': sum(r['terminal_category'] == 'correct' for r in selected),
                'first_old_target': sum(r['first_goal'] in OLD for r in selected),
                'old_target_ever_without_requested_success': sum(not r['correct_ever'] and any(r['first_goal_times'][g] is not None for g in OLD) for r in selected),
                'terminal_old_target': sum(r['terminal_goal'] in OLD for r in selected)}
    return {'episodes': len(items), 'initial_states': len(indices),
            'task_success_counts': {g: counts([r for r in items if r['requested_goal'] == g]) for g in GOALS},
            'joint_old_terminal_correct': len(joint), 'joint_old_terminal_correct_indices': joint,
            'unseen_instructions': intrusion([r for r in items if r['requested_goal'] in NEW]),
            'unseen_conditioned_on_both_old_instructions_correct': {'control_indices': joint,
                **intrusion([r for r in items if r['requested_goal'] in NEW and r['init_index'] in joint])},
            'interpretation': 'Untrained right/back failures measure generalization and old-target bias, not forgetting. First arrival and terminal target are separate.'}


def audit_sequence(folder):
    manifest = read(folder/'evidence_manifest.json')
    for entry in manifest:
        path = folder/entry['archive_path']
        if path.stat().st_size != entry['bytes'] or fingerprint(path) != entry['sha256']:
            raise ValueError(f'Evidence mismatch: {path}')
    run, config, comparison = (read(folder/name) for name in ('run_summary.json', 'config.json', 'comparison.json'))
    if run['status'] != 'complete' or read(folder/'COMPLETE.json')['status'] != 'complete':
        raise ValueError('Sequence incomplete')
    indices, seed = config['evaluation']['paired_indices'], config['seed']
    before, after = rows(folder/'before/episodes.jsonl'), rows(folder/'after/episodes.jsonl')
    pre = validate(before, (*OLD, 'right'), indices, seed)
    post = validate(after, (*OLD, 'right'), indices, seed)
    for key in pre:
        if any(pre[key][field] != post[key][field] for field in ('initial_state_sha256', 'scene_xml_sha256', 'seed', 'steps')):
            raise ValueError(f'Before/after mismatch: {key}')
    for label, items in (('before', before), ('after', after)):
        for goal in (*OLD, 'right'):
            actual = counts([r for r in items if r['requested_goal'] == goal])
            expected = comparison[label]['task_success_counts'][goal]
            if any(actual[k] != expected[k] for k in expected):
                raise ValueError(f'Count mismatch: {label}/{goal}')
    transitions = {}
    for goal in OLD:
        transitions[goal] = {}
        for label, predicate in (('previously_successful_ever', lambda r: r['correct_ever']),
                                 ('previously_terminal_correct', lambda r: r['terminal_category'] == 'correct')):
            eligible = [i for i in indices if predicate(pre[i, goal])]
            lost = [post[i, goal] for i in eligible if not predicate(post[i, goal])]
            value = {'total': len(eligible), 'retained': len(eligible)-len(lost), 'lost': len(lost),
                     'lost_and_first_new_right': sum(r['first_goal'] == 'right' for r in lost),
                     'lost_and_terminal_new_right': sum(r['terminal_goal'] == 'right' for r in lost),
                     'lost_and_terminal_unfinished': sum(r['terminal_category'] == 'unfinished' for r in lost),
                     'lost_and_terminal_ambiguous': sum(r['terminal_category'] == 'ambiguous' for r in lost),
                     'lost_and_terminal_other_wrong': sum(r['terminal_category'] == 'wrong_goal' and r['terminal_goal'] != 'right' for r in lost)}
            if value != comparison['old_goal_transitions'][goal][label]:
                raise ValueError(f'Transition mismatch: {goal}/{label}')
            transitions[goal][label] = value
    joint = [i for i in indices if all(pre[i, g]['terminal_category'] == 'correct' for g in OLD)]
    both = [i for i in joint if all(post[i, g]['terminal_goal'] == 'right' for g in OLD)]
    result = {'total': len(joint), 'retained_joint_terminal_correct': sum(all(post[i, g]['terminal_category'] == 'correct' for g in OLD) for i in joint),
              'both_old_instructions_terminal_new_right': len(both)}
    if result != comparison['previously_joint_correct_states']:
        raise ValueError('Joint-state transition mismatch')
    return {'status': 'pass', 'source_file_hashes_verified': len(manifest), 'matched_pairs': len(pre),
            'old_goal_transitions': transitions, 'previously_joint_correct_states': result,
            'both_old_to_right_indices': both,
            'joint_and_new_right_instruction_correct_after_indices': [i for i in joint if post[i, 'right']['terminal_category'] == 'correct'],
            'source_commit': run['source_commit'], 'verified_at': datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-dir', type=Path, required=True)
    parser.add_argument('--back-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = args.sequence_dir
    result = {'sequence_audit': audit_sequence(folder)}
    if args.back_dir:
        config, run = read(folder/'config.json'), read(folder/'run_summary.json')
        provenance = read(args.back_dir/'evaluation.json')
        if provenance['status'] != 'complete' or provenance['instruction_goals'] != ['back']:
            raise ValueError('Back evaluation incomplete or different instructions')
        checkpoint = provenance['checkpoint']
        if not checkpoint.startswith('/'):
            checkpoint = '/home/lzx/cl_binding/'+checkpoint
        if checkpoint != run['initial_checkpoint'] or provenance['git_commit'] != run['source_commit']:
            raise ValueError('Supplement used different checkpoint or source code')
        if provenance['horizon'] != 400 or provenance['n_action_steps'] != config['model']['n_action_steps']:
            raise ValueError('Supplement inference protocol changed')
        if provenance['normalization_source'] != config['data']['normalization_source']:
            raise ValueError('Supplement normalization source changed')
        combined = rows(folder/'before/episodes.jsonl')+rows(args.back_dir/'episodes.jsonl')
        result['parent_four_instructions'] = parent_summary(combined, config['evaluation']['paired_indices'], config['seed'])
        result['supplement_source_sha256'] = {p.name: fingerprint(p) for p in args.back_dir.iterdir() if p.is_file()}
    result['analysis_script_sha256'] = fingerprint(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
