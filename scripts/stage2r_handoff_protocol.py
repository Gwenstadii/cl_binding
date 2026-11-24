"""Frozen handoff matrix and movement endpoints, usable without model libraries."""
import itertools
import math
from analyze_stage2p_movement import inside_xy, first_run

GOALS = ('left', 'front', 'right', 'back')
FIELDS = ('source_model', 'prefix_goal', 'boundary', 'continuation_model', 'requested_goal', 'init_index', 'noise_seed')


def jobs():
    return [dict(zip(FIELDS, x)) for x in itertools.product(
        ('R0', 'FT4'), ('left', 'front'), (40, 80, 120),
        ('R0', 'FT4'), ('left', 'front'), (0, 1), (2026102801, 2026102802))]


def movement(events, geometry, initial_z, start=1):
    eligible = {g: [] for g in GOALS}
    near = {g: [] for g in GOALS}
    for row in events:
        if row['step'] < start:
            continue
        p = row['object_positions']['black_book_1_pos']
        assert all(math.isfinite(v) for v in p)
        if p[2] - initial_z <= .03:
            continue
        distances = {g: math.dist(p[:2], s['centre'][:2]) for g, s in geometry.items()}
        for g, site in geometry.items():
            if inside_xy(p, site):
                eligible[g].append(row['step'])
            if distances[g] < .08 and all(distances[g] + 1e-9 < v for other, v in distances.items() if other != g):
                near[g].append(row['step'])
    times = {g: first_run(v, 5) for g, v in eligible.items()}
    earliest = min((t for t in times.values() if t is not None), default=None)
    return {'times': times, 'first_targets': [g for g, t in times.items() if earliest is not None and t == earliest],
            'near8_times': {g: first_run(v, 5) for g, v in near.items()}}


def summarize_episode(job, events, geometry):
    h = job['boundary']
    assert [r['step'] for r in events] == list(range(1, 401))
    z0 = events[0]['object_positions']['black_book_1_pos'][2]
    prefix = movement(events[:h], geometry, z0)
    suffix = movement(events[h:], geometry, z0, h + 1)
    full = movement(events, geometry, z0)
    p = events[h-1]['object_positions']['black_book_1_pos']
    at_boundary = [g for g, site in geometry.items() if p[2] - z0 > .03 and inside_xy(p, site)]
    native_before = any(any(e['predicates'].values()) for e in events[:h])
    goal = job['requested_goal']
    return {'prefix_movement': prefix, 'suffix_movement': suffix, 'full_movement': full,
            'boundary_inside_regions': at_boundary, 'prefix_native_any': native_before,
            'pretarget_eligible': not prefix['first_targets'] and not native_before and not at_boundary,
            'suffix_first_correct': suffix['first_targets'] == [goal],
            'suffix_first_right': suffix['first_targets'] == ['right'],
            'suffix_correct_ever': suffix['times'][goal] is not None,
            'full_correct_ever': full['times'][goal] is not None,
            'terminal_correct': [g for g, value in events[-1]['predicates'].items() if value] == [goal]}
