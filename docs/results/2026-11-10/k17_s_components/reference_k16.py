"""K16: factor the source of the common response over H, S and padding.

All conditions retain exactly the same fixed-gain, actual-query response rule:
    hybrid_mu + 2.959 * (v_F(actual_q) - mu_F).
H selects old mean at 0:2; S at 2:7; P at 7:32. No target-sign selection.
The intervention is on normalized denoising outputs, not physical modules.
N and HSP directly reuse K15 implementations to provide exact references.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
K15_SOURCE = Path('/home/lzx/cl_binding/.server/k15_background_gain_20261109_01')
K15_SHA256 = '8e928d79232cf9346be64965fa77dfcf4c32b2906b3715e13bd278cf700c77a6'
assert hashlib.sha256((K15_SOURCE / 'k15.py').read_bytes()).hexdigest() == K15_SHA256
sys.path.insert(0, str(K15_SOURCE))
import k15

# k15 checks all frozen K14 hashes before importing the original implementation.
from k15 import Lab5, ALPHA, INSTR, LABEL, HORIZON, WINDOW
SOURCE = k15.SOURCE
LOCKED = k15.LOCKED
CONDITIONS = ('N', 'H', 'S', 'HS', 'P', 'HP', 'SP', 'HSP')
BLOCKS = {'H': tuple(range(0, 2)), 'S': tuple(range(2, 7)), 'P': tuple(range(7, 32))}
OLD_BLOCKS = {c: tuple(b for b in BLOCKS if b in c) for c in CONDITIONS}
OLD_INDICES = {c: tuple(i for b in OLD_BLOCKS[c] for i in BLOCKS[b]) for c in CONDITIONS}
assert ALPHA == 2.959 and HORIZON == 400 and WINDOW == 60
assert set(sum(BLOCKS.values(), ())) == set(range(32))
assert sum(map(len, BLOCKS.values())) == 32
assert len(set(OLD_INDICES.values())) == 8


def get_fields(rt, prefixes, qpre, x, t):
    """Evaluate every model/query on one identical observation/draft/time."""
    v = {k: rt.query(k[0], p, x, t).float() for k, p in prefixes.items()}
    actual = rt.query('F', qpre, x, t).float()
    mu_old = (v['R', 'A'] + v['R', 'B']) / 2
    mu_new = (v['F', 'A'] + v['F', 'B']) / 2
    assert mu_old.shape == mu_new.shape == actual.shape == (1, 50, 32)
    return mu_old, mu_new, actual


def combine(rt, condition, mu_old, mu_new, actual, alpha=ALPHA):
    """Choose mean dimensions before FP32 addition, then native-dtype cast."""
    assert condition in CONDITIONS
    mu = mu_new.clone()
    selected = OLD_INDICES[condition]
    if selected:
        mu[..., list(selected)] = mu_old[..., list(selected)]
    raw = mu + alpha * (actual - mu_new)
    assert rt.torch.isfinite(raw).all()
    out = raw.to(rt.vdtype)
    assert rt.torch.isfinite(out).all()
    return out, mu, raw


def hybrid_velocity(rt, condition, prefixes, qpre, x, t, diag):
    mu_old, mu_new, actual = get_fields(rt, prefixes, qpre, x, t)
    out, mu, raw = combine(rt, condition, mu_old, mu_new, actual)
    e = ALPHA * (actual - mu_new)
    for block, dims in BLOCKS.items():
        dims = list(dims)
        delta = (mu_old - mu_new)[..., dims]
        diag['mean_shift_rms_' + block].append(float(delta.square().mean().sqrt()))
        diag['response_rms_' + block].append(float(e[..., dims].square().mean().sqrt()))
    diag['cast_roundoff_rms_real'].append(float((out.float() - raw)[..., :7].square().mean().sqrt()))
    return out


def make_prefixes(lab, obs):
    return {(role, g): lab.rt.prefix(role, obs, INSTR[g]) for role in 'RF' for g in 'AB'}


def dispatch(lab, condition, prefixes, qpre, x, t, diag):
    if condition == 'N':
        return k15.new_velocity(lab.rt, {g: prefixes['F', g] for g in 'AB'}, qpre, x, t, diag)
    if condition == 'HSP':
        return lab.corrected(prefixes, qpre, x, t, diag)
    return hybrid_velocity(lab.rt, condition, prefixes, qpre, x, t, diag)


def gates(lab, snapshots):
    """The same input is tested against all eight masks and both actual q's."""
    torch = lab.torch
    checks = []
    for init in (20, 21, 22):
        snap = snapshots[init]
        controls = [lab.np.asarray(a, dtype=lab.np.float32) for a in snap['controls']]
        obs, _ = lab.rebuild(init, controls, snap['z0'])
        assert lab.ctx.signature(obs) == snap['signature']
        prefixes = make_prefixes(lab, obs)
        x = lab.rt.noise(snap['seed'], snap['boundary'] // 10)
        t = torch.ones(1, dtype=torch.float32, device=x.device)
        goal_checks = []
        zero_outputs = {}
        actual_means = {}
        chosen_means = {}
        for goal in 'AB':
            qpre = lab.rt.prefix('F', obs, INSTR[goal])
            mu_old, mu_new, actual = get_fields(lab.rt, prefixes, qpre, x, t)
            assert torch.equal(actual.to(lab.rt.vdtype),
                               lab.rt.query('F', qpre, x, t))
            native_one, _, _ = combine(lab.rt, 'N', mu_old, mu_new, actual, alpha=1.)
            assert torch.equal(native_one, actual.to(lab.rt.vdtype))
            new_ref = k15.new_velocity(lab.rt, {g: prefixes['F', g] for g in 'AB'},
                                       qpre, x, t, defaultdict(list))
            old_ref = lab.corrected(prefixes, qpre, x, t, defaultdict(list))
            checked = []
            zero_outputs[goal] = {}
            actual_means[goal] = {}
            chosen_means[goal] = {}
            for condition in CONDITIONS:
                output, mu, raw = combine(lab.rt, condition, mu_old, mu_new, actual)
                selected = list(OLD_INDICES[condition])
                unselected = [i for i in range(32) if i not in selected]
                expected = new_ref.clone()
                if selected:
                    expected[..., selected] = old_ref[..., selected]
                    assert torch.equal(output[..., selected], old_ref[..., selected])
                if unselected:
                    assert torch.equal(output[..., unselected], new_ref[..., unselected])
                assert torch.equal(output, expected)
                real_out = dispatch(lab, condition, prefixes, qpre, x, t, defaultdict(list))
                assert torch.equal(real_out, output), ('dispatch differs from formula', condition)
                zero, _, _ = combine(lab.rt, condition, mu_old, mu_new, actual, alpha=0.)
                zero_outputs[goal][condition] = zero
                actual_means[goal][condition] = output.float()
                chosen_means[goal][condition] = mu
                checked.append({
                    'condition': condition, 'selected_indices': selected,
                    'selected_equals_old_bitwise': True,
                    'unselected_equals_new_bitwise': True,
                    'dispatch_matches_formula_bitwise': True,
                    'all32_finite': True,
                    'cast_roundoff_rms_real': float((output.float() - raw)[..., :7].square().mean().sqrt()),
                    'cast_roundoff_max_all32': float((output.float() - raw).abs().max()),
                })
            goal_checks.append({'goal': goal, 'alpha1_N_native_bitwise': True,
                                'conditions': checked})
        mean_roundoff = {}
        for condition in CONDITIONS:
            assert torch.equal(zero_outputs['A'][condition], zero_outputs['B'][condition])
            assert torch.equal(chosen_means['A'][condition], chosen_means['B'][condition])
            mean = (actual_means['A'][condition] + actual_means['B'][condition]) / 2
            mu = chosen_means['A'][condition]
            mean_roundoff[condition] = float((mean - mu)[..., :7].square().mean().sqrt())
            assert mean_roundoff[condition] < .03
        checks.append({'init': init, 'same_snapshot_signature': True,
                       'alpha0_instruction_identity_all_conditions': True,
                       'amplified_mean_roundoff_rms_real': mean_roundoff,
                       'goal_checks': goal_checks})
    result = {'pass': True, 'checks': checks,
              'all32_dimensions_partitioned': True,
              'blocks': {k: list(v) for k, v in BLOCKS.items()},
              'condition_old_indices': {k: list(v) for k, v in OLD_INDICES.items()},
              'floating_point_note': 'FP32 expression; final native dtype rounding is explicit and audited'}
    (HERE / 'gates.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (HERE / 'geometry.json').write_text(json.dumps({
        'sites': lab.ctx.geometry, 'canonical_scene_xml_sha256': lab.ctx.xml_sha}, indent=2))


def run(init, run_gates):
    snapshots = {s['init']: s for s in json.loads((SOURCE / 'snapshots/snapshots.json').read_text())['records']}
    assert set(snapshots) == set(range(20, 40))
    lab = Lab5()
    torch = lab.torch
    snap = snapshots[init]

    def episode(condition, goal):
        started = time.monotonic()
        boundary, z0, seed = snap['boundary'], snap['z0'], snap['seed']
        controls = [lab.np.asarray(a, dtype=lab.np.float32) for a in snap['controls']]
        obs, rows = lab.rebuild(init, controls, z0)
        signature = lab.ctx.signature(obs)
        assert signature == snap['signature']
        diag = defaultdict(list)
        q = INSTR[goal]
        calls = 0
        for r in range(boundary // 10, HORIZON // 10):
            if r * 10 < boundary + WINDOW:
                qpre = lab.rt.prefix('F', obs, q)
                # N needs no old prefix/output for its velocity; use K15 query order.
                if condition == 'N':
                    prefixes = {('F', g): lab.rt.prefix('F', obs, INSTR[g]) for g in 'AB'}
                else:
                    prefixes = make_prefixes(lab, obs)

                def velocity(x, t):
                    nonlocal calls
                    calls += 1
                    return dispatch(lab, condition, prefixes, qpre, x, t, diag)

                chunk = lab.rt.integrate(velocity, lab.rt.noise(seed, r), 10)[..., :7]
            else:
                chunk = lab.native('F', q, obs, seed, r)
            obs, events = lab.execute(obs, lab.rt.physical(chunk), r * 10, z0)
            rows.extend(events)
        assert len(rows) == 400 and calls == 60
        movement = lab.ctx.geo.movement('scene2', lab.ctx.geometry, rows, z0, None)
        window_rows = rows[boundary:boundary + WINDOW]
        record = {
            'init': init, 'condition': condition, 'goal': goal, 'requested': LABEL[goal],
            'seed': seed, 'boundary': boundary, 'window': WINDOW, 'alpha': ALPHA,
            'prefix_model': 'Front-12000', 'prefix_goal': 'Front', 'z0': z0,
            'boundary_signature': signature,
            'mu_old_blocks': list(OLD_BLOCKS[condition]),
            'mu_old_indices': list(OLD_INDICES[condition]),
            'first': movement['first_targets'], 'first_step': movement['first_step'],
            'first_times': movement['times'],
            'ever': {g: any(a['pred'][g] for a in rows) for g in lab.ctx.geo.GOALS},
            'terminal': rows[-1]['pred'],
            'clip_rate_window': sum(a['clip'] for a in window_rows) / len(window_rows),
            'clip_rate_all': sum(a['clip'] for a in rows[boundary:]) / (400 - boundary),
            'diagnostics': {k: [sum(v) / len(v), min(v), max(v)] for k, v in diag.items()},
            'n_window_queries': calls, 'seconds': time.monotonic() - started,
        }
        return record, rows

    try:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            if run_gates:
                gates(lab, snapshots)
            with (HERE / f'i{init}.jsonl').open('x') as records, (HERE / f'i{init}_traces.jsonl').open('x') as traces:
                for condition in CONDITIONS:
                    for goal in 'AB':
                        record, rows = episode(condition, goal)
                        traces.write(json.dumps({'init': init, 'condition': condition, 'goal': goal,
                                                 'rows': rows}, allow_nan=False) + '\n')
                        traces.flush()
                        records.write(json.dumps(record, allow_nan=False) + '\n')
                        records.flush()
                        print(init, condition, goal, record['first'], record['terminal'][LABEL[goal]],
                              f"{record['seconds']:.1f}s", flush=True)
        (HERE / f'i{init}_COMPLETE.json').write_text(json.dumps({
            'init': init, 'n_episodes': 16, 'model_sha256': lab.rt.sha,
            'model_paths': lab.rt.paths, 'processor_check': lab.rt.processor_check,
            'alpha': ALPHA, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_hashes': LOCKED, 'k15_script_sha256': K15_SHA256,
            'condition_old_indices': {k: list(v) for k, v in OLD_INDICES.items()},
        }, indent=2, allow_nan=False))
    finally:
        lab.ctx.env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init', type=int, choices=range(20, 40), required=True)
    parser.add_argument('--gates', action='store_true')
    args = parser.parse_args()
    run(args.init, args.gates)
