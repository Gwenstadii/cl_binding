"""Predeclared same-input mean probes along the first N chunk at each snapshot.

No hybrid endpoint selection; all20 snapshots and both canonical queries included.
The velocity-field units are normalized denoising units, NOT physical velocities.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
K15 = Path('/home/lzx/cl_binding/.server/k15_background_gain_20261109_01')
K15_SHA = '8e928d79232cf9346be64965fa77dfcf4c32b2906b3715e13bd278cf700c77a6'
assert hashlib.sha256((K15 / 'k15.py').read_bytes()).hexdigest() == K15_SHA
sys.path.insert(0, str(K15))
from k15 import ALPHA, INSTR, Lab5, LOCKED, SOURCE

BLOCKS = {'H': [0, 1], 'S': list(range(2, 7)), 'P': list(range(7, 32))}


def summarize_tensor(delta, old, new, contrast):
    result = {}
    for name, ix in BLOCKS.items():
        result[name] = {}
        for horizon, end in (('executed_first10', 10), ('full50', 50)):
            values = {}
            for label, tensor in (('old_minus_new_mean', delta), ('old_mean', old), ('new_mean', new), ('new_contrast', contrast)):
                part = tensor[0, :end, ix]
                values[label] = {'rms': float(part.square().mean().sqrt()), 'norm': float(part.norm())}
            result[name][horizon] = values
    for horizon, end in (('executed_first10', 10), ('full50', 50)):
        result[horizon + '_coordinate_mean_delta'] = delta[0, :end].mean(dim=0).cpu().tolist()
        result[horizon + '_coordinate_rms_delta'] = delta[0, :end].square().mean(dim=0).sqrt().cpu().tolist()
    return result


def main():
    start = time.monotonic()
    snapshots = {s['init']: s for s in json.loads((SOURCE / 'snapshots/snapshots.json').read_text())['records']}
    reference = {(r['init'], r['goal']): r['rows'] for r in map(json.loads, (HERE / 'reference_traces.jsonl').read_text().splitlines()) if r['condition'] == 'N'}
    lab = Lab5()
    torch = lab.torch
    count = 0
    reproduced = 0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16), (HERE / 'same_input_probes.jsonl').open('x') as handle:
        for init in range(20, 40):
            snap = snapshots[init]
            for goal in 'AB':
                obs, prefix_rows = lab.rebuild(init, [lab.np.asarray(c, dtype=lab.np.float32) for c in snap['controls']], snap['z0'])
                assert lab.ctx.signature(obs) == snap['signature']
                prefixes = {(r, g): lab.rt.prefix(r, obs, INSTR[g]) for r in 'RF' for g in 'AB'}
                actual_prefix = lab.rt.prefix('F', obs, INSTR[goal])
                steps = []
                def velocity(x, t):
                    nonlocal count
                    values = {k: lab.rt.query(k[0], p, x, t).float() for k, p in prefixes.items()}
                    actual = lab.rt.query('F', actual_prefix, x, t).float()
                    old = (values['R', 'A'] + values['R', 'B']) / 2
                    new = (values['F', 'A'] + values['F', 'B']) / 2
                    contrast = (values['F', 'A'] - values['F', 'B']) / 2
                    record = {'init': init, 'goal': goal, 'boundary': snap['boundary'], 'denoise_index': len(steps),
                              't': float(t.flatten()[0]), 'x_sha256': hashlib.sha256(x.float().cpu().numpy().tobytes()).hexdigest(),
                              'units': 'normalized action-draft denoising velocity',
                              'blocks': BLOCKS, 'moments': summarize_tensor(old - new, old, new, contrast)}
                    steps.append(record)
                    handle.write(json.dumps(record, allow_nan=False) + '\n')
                    handle.flush()
                    count += 1
                    return (new + ALPHA * (actual - new)).to(lab.rt.vdtype)
                r = snap['boundary'] // 10
                chunk = lab.rt.integrate(velocity, lab.rt.noise(snap['seed'], r), 10)[..., :7]
                _, events = lab.execute(obs, lab.rt.physical(chunk), snap['boundary'], snap['z0'])
                assert len(steps) == 10
                assert prefix_rows + events == reference[init, goal][:snap['boundary'] + 10], (init, goal)
                reproduced += 1
            print(json.dumps({'probe_init': init, 'records': count, 'matched_N_first_chunks': reproduced}), flush=True)
    assert count == 400 and reproduced == 40
    marker = {'pass': True, 'probe_records': count, 'matched_N_first_chunks': reproduced,
              'model_sha256': lab.rt.sha, 'source_hashes': LOCKED, 'k15_script_sha256': K15_SHA,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol_sha256': hashlib.sha256((HERE / 'protocol.json').read_bytes()).hexdigest(),
              'elapsed_seconds': time.monotonic() - start}
    (HERE / 'same_input_probe_COMPLETE.json').write_text(json.dumps(marker, indent=2))
    lab.ctx.env.close()


if __name__ == '__main__':
    main()
