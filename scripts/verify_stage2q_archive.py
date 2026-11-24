"""Verify downloaded night evidence and independently recount movement (no model).

The 1.35 GB random-norm log remains on the server. Its audit is preserved,
but is deliberately not represented as a local recomputation here.
"""
import argparse
from collections import Counter, defaultdict, deque
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path


def info(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        yield from (json.loads(line) for line in f)


def verify(root):
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    data = root / 'evaluation'
    manifest = load(data / 'manifest.json')
    cfg = manifest['configuration']
    server = load(root / 'audit.json')
    launcher = load(root / 'launcher.json')
    assert launcher['status'] == manifest['status'] == 'complete'
    assert load(root / 'COMPLETE.json')['episodes'] == server['episodes'] == 420
    assert load(root / 'post_run_check.json')['passed'] and server['status'] == 'passed'
    assert manifest == load(data / 'COMPLETE.json')
    assert cfg == load(root / 'registered_config.json') == launcher['configuration']
    local_hashes, remote_only = {}, {}
    for name, expected in manifest['files'].items():
        p = data / name
        if p.exists():
            local_hashes[name] = info(p)
            assert local_hashes[name] == expected, name
        else:
            assert name == 'norms.jsonl.gz', name
            remote_only[name] = expected
    assert remote_only, 'This verifier explicitly documents the partial raw archive'
    for name, expected in launcher['code_files'].items():
        directory = 'tests' if name.startswith('test_') else 'scripts'
        assert info(Path(__file__).resolve().parents[1] / directory / name) == expected, name
    engineering = load(data / 'engineering.json')
    assert engineering['passed'] and len(engineering['checks']) == 32
    assert max(r['max_abs'] for r in engineering['checks']) == 0
    eps = list(rows(data / 'episodes.jsonl'))
    byid = {e['episode_id']: e for e in eps}
    assert len(byid) == len(eps) == 420
    fields = ('model', 'condition', 'site', 'window', 'requested_goal', 'init_index', 'noise_seed')
    common = list(itertools.product(('left', 'front'), (0, 1), (2026102801, 2026102802)))
    expected = {(r, 'native', None, 0, g, i, s) for r in ('R0', 'FT4') for g, i, s in common}
    expected |= {('FT4', 'native', None, 0, 'right', i, s) for i in (0, 1) for s in (2026102801, 2026102802)}
    expected |= {('FT4', 'full_old_response', None, w, g, i, s) for w in (120, 400) for g, i, s in common}
    expected |= {('FT4', c, j, w, g, i, s) for c in ('cross_same', 'cross_wrong', 'random_delta')
                 for j in (1, 3, 5, 7, 9, 11, 13, 15) for w in (120, 400) for g, i, s in common}
    assert {tuple(e[k] for k in fields) for e in eps} == expected
    archived = {r['episode_id']: r for r in server['episode_results']}
    counts, identities, starts = Counter(), {}, defaultdict(list)
    for eid, iterator in itertools.groupby(rows(data / 'events.jsonl.gz'), lambda r: r['episode_id']):
        events = list(iterator)
        assert [r['step'] for r in events] == list(range(401))
        ep, original = byid[eid], archived[eid]
        z0 = events[1]['object_positions']['black_book_1_pos'][2]
        starts[ep['init_index']].append(events[1]['object_positions']['black_book_1_pos'])
        first = {'box': {}, 'near8': {}}
        history = {(metric, goal): deque(maxlen=5) for metric in first for goal in cfg['movement_geometry']['sites']}
        for row in events[1:]:
            counts['physical_steps'] += 1
            p = row['object_positions']['black_book_1_pos']
            distances = {g: sum((p[d] - s['centre'][d]) ** 2 for d in (0, 1))
                         for g, s in cfg['movement_geometry']['sites'].items()}
            for g, s in cfg['movement_geometry']['sites'].items():
                extent = [sum(abs(s['rotation'][d][j]) * s['half_size'][j] for j in range(3)) for d in (0, 1)]
                box = all(abs(p[d] - s['centre'][d]) <= extent[d] for d in (0, 1))
                near = distances[g] < .08 ** 2 and all(distances[g] ** .5 + 1e-9 < v ** .5 for other, v in distances.items() if other != g)
                for metric, hit in (('box', box), ('near8', near)):
                    q = history[metric, g]
                    q.append(hit and p[2] - z0 > .03)
                    if len(q) == 5 and all(q) and g not in first[metric]:
                        first[metric][g] = row['step'] - 4
        for metric, times in first.items():
            assert {g: times.get(g) for g in cfg['movement_geometry']['sites']} == original[metric + '_times'], eid
            minimum = min(times.values(), default=None)
            first_targets = [g for g, t in times.items() if t == minimum]
            assert first_targets == original[metric + '_first_targets'], eid
            assert (ep['requested_goal'] in times) == original[metric + '_correct'], eid
            assert (first_targets == [ep['requested_goal']]) == original[metric + '_first_correct'], eid
            assert (first_targets == ['right']) == original[metric + '_first_right'], eid
        if ep['condition'] == 'native' and ep['model'] == 'R0' or ep['condition'] == 'full_old_response' and ep['window'] == 400:
            identities[ep['condition'], ep['requested_goal'], ep['init_index'], ep['noise_seed']] = events
        counts['episodes'] += 1
    assert counts['episodes'] == 420 and counts['physical_steps'] == 168000
    assert all(max(abs(a - b) for p in ps for a, b in zip(ps[0], p)) <= 1e-6 for ps in starts.values())
    max_identity = 0.
    for g, i, s in common:
        a, b = identities['native', g, i, s], identities['full_old_response', g, i, s]
        assert all(x['predicates'] == y['predicates'] for x, y in zip(a, b))
        max_identity = max(max_identity, max(abs(x - y) for aa, bb in zip(a[1:], b[1:])
                                            for x, y in zip(aa['action_executed'], bb['action_executed'])))
    assert max_identity <= 1e-5
    groups = defaultdict(list)
    for ep in archived.values():
        groups[tuple(ep[k] for k in fields[:5])].append(ep)
    for r in server['per_condition']:
        group = groups[tuple(r[k] for k in fields[:5])]
        assert len(group) == r['episodes'] == 4
        for k in ('box_correct', 'box_first_correct', 'box_first_right', 'near8_correct', 'near8_first_correct', 'terminal_correct', 'native_ever'):
            assert sum(x[k] for x in group) == r[k]
    with (root / 'results.csv').open(newline='', encoding='utf-8') as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 105
    assert csv_rows == [{k: '' if v is None else str(v) for k, v in r.items()} for r in server['per_condition']]
    return {'status': 'passed', 'scope': 'local raw-file hashes, full matrix, independent movement recount and positive-policy identity',
            'counts': dict(counts), 'per_condition_rows': 105, 'positive_identity_action_max_abs': max_identity,
            'verified_raw_files': local_hashes, 'server_only_raw_files': remote_only,
            'random_norm_audit': 'server audit passed; not recomputed locally',
            'server_random_norm_max_relative_difference': server['max_norm_relative_difference']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    output = args.root / 'audit_local_movement_v1.json'
    assert not output.exists(), 'Preserve existing audit'
    result = verify(args.root)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
