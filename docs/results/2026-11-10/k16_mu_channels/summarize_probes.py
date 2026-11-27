"""Audit K16 same-input probes and describe variation over 20 snapshots.

The 400 denoising queries are repeated measurements, not 400 independent units.
For each snapshot, pool the selected q/denoising rows before describing the
distribution over snapshots. Output units remain normalized denoising velocity.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


INITS = tuple(range(20, 40))
GOALS = ('A', 'B')
BLOCKS = {'H': [0, 1], 'S': list(range(2, 7)), 'P': list(range(7, 32))}
HORIZONS = ('executed_first10', 'full50')
HORIZON_LENGTH = {'executed_first10': 10, 'full50': 50}
PHASES = {'initial_denoise0': (0,), 'all10_denoise': tuple(range(10))}
MOMENT_NAMES = ('old_minus_new_mean', 'old_mean', 'new_mean', 'new_contrast')
MODEL_SHA = {'R': '225322ba28d0ad152fcced19950c43e3e40a435f1835cf64a664226da7830003',
             'F': '8ff1bf8f372d728a3218101a3cc93fb6fb9d484e8fefb39c48d81a0c8ab2c515'}
K15_SHA = '8e928d79232cf9346be64965fa77dfcf4c32b2906b3715e13bd278cf700c77a6'
COORDINATES = ('translation_x', 'translation_y', 'translation_z',
               'rotation_axisangle_0', 'rotation_axisangle_1', 'rotation_axisangle_2',
               'gripper')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def close(a, b, rel_tol=2e-5, abs_tol=2e-6):
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def rms(values):
    return math.sqrt(mean(x * x for x in values))


def ratio(a, b):
    return a / b if b > 1e-12 else None


def distribution(values):
    valid = [x for x in values if x is not None]
    assert all(finite_number(x) for x in valid)
    return {'n_snapshot_clusters': len(values), 'n_valid': len(valid),
            'mean': mean(valid) if valid else None,
            'median': median(valid) if valid else None,
            'min': min(valid) if valid else None,
            'max': max(valid) if valid else None}


def audit_rows(archive, protocol, marker):
    assert marker['pass'] is True
    assert marker['probe_records'] == 400 and marker['matched_N_first_chunks'] == 40
    assert marker['model_sha256'] == MODEL_SHA
    assert marker['k15_script_sha256'] == protocol['k15_script_sha256'] == K15_SHA
    assert marker['source_hashes'] == protocol['source_sha256']
    assert marker['script_sha256'] == sha(archive / 'probe_same_input.py')
    assert marker['protocol_sha256'] == sha(archive / 'protocol.json')
    assert protocol['inits'] == list(INITS) and protocol['alpha'] == 2.959
    assert protocol['blocks'] == BLOCKS
    expected = {(i, q, k) for i in INITS for q in GOALS for k in range(10)}
    rows = {}
    for line in (archive / 'same_input_probes.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row['init'], row['goal'], row['denoise_index']
        assert key in expected and key not in rows, ('unexpected/duplicate probe', key)
        assert row['units'] == 'normalized action-draft denoising velocity'
        assert row['blocks'] == BLOCKS
        assert row['boundary'] in (70, 80, 90, 100, 110)
        assert isinstance(row['x_sha256'], str) and len(row['x_sha256']) == 64
        int(row['x_sha256'], 16)
        assert finite_number(row['t']) and close(row['t'], 1 - key[2] / 10, abs_tol=2e-6)
        for horizon in HORIZONS:
            coordinate_mean = row['moments'][horizon + '_coordinate_mean_delta']
            coordinate_rms = row['moments'][horizon + '_coordinate_rms_delta']
            assert len(coordinate_mean) == len(coordinate_rms) == 32
            assert all(finite_number(x) for x in coordinate_mean + coordinate_rms)
            assert all(r >= 0 and r + 2e-6 >= abs(m) for m, r in zip(coordinate_mean, coordinate_rms))
            for block, dims in BLOCKS.items():
                entries = row['moments'][block][horizon]
                assert set(entries) == set(MOMENT_NAMES)
                for name, value in entries.items():
                    assert set(value) == {'rms', 'norm'}
                    assert all(finite_number(x) and x >= 0 for x in value.values())
                    assert close(value['norm'], value['rms'] * math.sqrt(HORIZON_LENGTH[horizon] * len(dims)))
                expected_rms = math.sqrt(mean(coordinate_rms[d] ** 2 for d in dims))
                assert close(entries['old_minus_new_mean']['rms'], expected_rms)
        rows[key] = row
    assert set(rows) == expected and len(rows) == 400
    for init in INITS:
        boundaries = {rows[init, q, k]['boundary'] for q in GOALS for k in range(10)}
        assert len(boundaries) == 1
        # Both actual q's start from the same action noise. The mean probe itself
        # depends on the canonical pair and identical obs/x/t, not on actual q.
        a, b = rows[init, 'A', 0], rows[init, 'B', 0]
        assert a['x_sha256'] == b['x_sha256'] and a['t'] == b['t']
        assert a['moments'] == b['moments']
        for q in GOALS:
            times = [rows[init, q, k]['t'] for k in range(10)]
            assert all(times[k + 1] < times[k] for k in range(9))
    return rows


def action_scale(metadata):
    assert metadata['action_stats_identical_R_F'] is True
    output = {}
    for role in ('R', 'F'):
        checkpoint = metadata['checkpoints'][role]
        assert checkpoint['config']['normalization_mapping']['ACTION'] == 'MEAN_STD'
        assert checkpoint['config']['max_action_dim'] == 32
        assert checkpoint['config']['output_features']['action']['shape'] == [7]
        assert checkpoint['postprocessor']['steps'][0]['registry_name'] == 'unnormalizer_processor'
        assert checkpoint['postprocessor']['steps'][0]['config']['norm_map']['ACTION'] == 'MEAN_STD'
        values = checkpoint['action_stats']['action.std']['values']
        assert len(values) == 7 and all(finite_number(v) and v > 0 for v in values)
        output[role] = values
    assert output['R'] == output['F']
    return output['F']


def cluster_summary(rows, phase_indices, std):
    per_snapshot = {}
    for init in INITS:
        selected = [rows[init, q, k] for q in GOALS for k in phase_indices]
        horizons = {}
        for horizon in HORIZONS:
            block_values = {}
            for block in BLOCKS:
                points = [row['moments'][block][horizon] for row in selected]
                pooled = {name: rms([p[name]['rms'] for p in points]) for name in MOMENT_NAMES}
                delta, contrast = pooled['old_minus_new_mean'], pooled['new_contrast']
                block_values[block] = {
                    'pooled_rms': pooled,
                    'delta_rms_over_unamplified_new_contrast_rms': ratio(delta, contrast),
                    'delta_rms_over_alpha_amplified_new_contrast_rms': ratio(delta, 2.959 * contrast),
                    'mean_of_record_rms': {name: mean(p[name]['rms'] for p in points) for name in MOMENT_NAMES},
                }
            coordinates = {}
            for d, label in enumerate(COORDINATES):
                signed = mean(row['moments'][horizon + '_coordinate_mean_delta'][d] for row in selected)
                pooled_rms = rms([row['moments'][horizon + '_coordinate_rms_delta'][d] for row in selected])
                coordinates[str(d)] = {
                    'name': label,
                    'delta_signed_mean_normalized': signed,
                    'delta_pooled_rms_normalized': pooled_rms,
                    'action_std': std[d],
                    'delta_signed_mean_times_action_std_scale_reference': signed * std[d],
                    'delta_pooled_rms_times_action_std_scale_reference': pooled_rms * std[d],
                }
            horizons[horizon] = {'blocks': block_values, 'coordinates': coordinates}
        per_snapshot[str(init)] = {'n_repeated_probe_rows': len(selected), 'horizons': horizons}

    aggregate = {}
    for horizon in HORIZONS:
        blocks = {}
        for block in BLOCKS:
            part = [per_snapshot[str(i)]['horizons'][horizon]['blocks'][block] for i in INITS]
            blocks[block] = {
                'pooled_rms_distributions': {name: distribution([p['pooled_rms'][name] for p in part]) for name in MOMENT_NAMES},
                'delta_over_unamplified_contrast_distribution': distribution([p['delta_rms_over_unamplified_new_contrast_rms'] for p in part]),
                'delta_over_alpha_amplified_contrast_distribution': distribution([p['delta_rms_over_alpha_amplified_new_contrast_rms'] for p in part]),
            }
        coordinates = {}
        for d, label in enumerate(COORDINATES):
            part = [per_snapshot[str(i)]['horizons'][horizon]['coordinates'][str(d)] for i in INITS]
            coordinates[str(d)] = {'name': label, 'action_std': std[d],
                'distributions': {metric: distribution([p[metric] for p in part]) for metric in (
                    'delta_signed_mean_normalized', 'delta_pooled_rms_normalized',
                    'delta_signed_mean_times_action_std_scale_reference',
                    'delta_pooled_rms_times_action_std_scale_reference')}}
        aggregate[horizon] = {'blocks': blocks, 'coordinates': coordinates}
    return {'denoise_indices': list(phase_indices),
            'record_count': len(INITS) * len(GOALS) * len(phase_indices),
            'n_snapshot_clusters': len(INITS), 'aggregate': aggregate,
            'per_snapshot': per_snapshot}


def fmt(value):
    return 'NA' if value is None else f'{value:.5g}'


def interval_description(value):
    return f"{fmt(value['median'])} [{fmt(value['min'])}, {fmt(value['max'])}]"


def main(archive):
    protocol = read(archive / 'protocol.json')
    marker = read(archive / 'same_input_probe_COMPLETE.json')
    metadata = read(archive / 'live_action_metadata.json')
    rows = audit_rows(archive, protocol, marker)
    std = action_scale(metadata)
    phases = {name: cluster_summary(rows, indices, std) for name, indices in PHASES.items()}
    sources = ('same_input_probes.jsonl', 'same_input_probe_COMPLETE.json',
               'probe_same_input.py', 'protocol.json', 'live_action_metadata.json')
    report = {
        'experiment': 'K16 same-input mean displacement descriptive probes',
        'audits': {'pass': True, 'unique_probe_records': 400, 'matched_N_first_chunks': 40,
                   'snapshot_clusters': 20, 'initial_A_B_input_and_moments_identical': True,
                   'ten_monotonic_times_1_to_0_1': True,
                   'source_script_and_protocol_hashes_verified': True,
                   'model_hashes_verified': True, 'block_norm_coordinate_moment_consistency': True,
                   'metadata_mean_std_equal_across_models': True},
        'units': 'normalized action-draft denoising velocity, NOT physical velocity',
        'definitions': {
            'delta': 'mu_old - mu_new at one identical observation/action draft/time',
            'new_contrast': '(v_F(canonical Left) - v_F(canonical Right))/2, before alpha amplification',
            'pooling': 'within a snapshot, equally pool both q and selected denoising records: pooled RMS = sqrt(mean(record RMS squared)); then describe the 20 snapshot values',
            'ratio': 'ratio of snapshot-pooled RMS(delta) to snapshot-pooled RMS(new contrast); denominator <=1e-12 yields null and a reported valid-count',
            'first10': 'first ten action-horizon positions of the predicted chunk, later executed; not the first ten denoising calls',
            'initial_denoise0': 't=1 query; A/B action drafts and canonical mean moments are identical',
            'all10_denoise': 't=1,...,0.1 along each requested query N draft path; later A/B drafts can differ',
            'scale_reference': 'multiplication by checkpoint action.std reverses only the per-coordinate scale for a difference; no action.mean intercept applies to differences',
        },
        'interpretation_limits': [
            '400 queries are repeated measurements on 20 physical snapshots, not 400 independent samples',
            'snapshot min/max are descriptive ranges, not confidence intervals; no significance tests performed',
            'RMS magnitude does not establish functional necessity, semantic storage or causal mediation',
            'action.std-scaled values are velocity-field scale references, NOT completed actions, robot command differences, or end-effector displacement',
            'Euler integration proceeds from t=1 toward t=0 with negative dt; a single draft update is x_next=x+dt*v, so delta-v signs must not be read as physical direction signs',
            'later denoising inputs change with the N reference integration; probes do not give a full hybrid policy trajectory',
            'S groups height, rotation and gripper together; padding is not physically executed but can affect future denoising',
        ],
        'action_std': std, 'blocks': BLOCKS, 'phases': phases,
        'source_sha256': {name: sha(archive / name) for name in sources},
        'summarizer_sha256': sha(Path(__file__)),
    }
    (archive / 'probe_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    lines = [
        '# K16 同输入共同响应探针', '',
        '400 条探针、20 个物理快照；40 个首动作块均精确复现 N 的前10个执行步。', '',
        '以下每个数先在同一快照内汇总两句及指定去噪步，再对20个快照取中位数。方括号是快照最小值、最大值，不是置信区间。', '',
        'Δμ＝μ旧−μ新；d新＝续训模型左右输出半差，尚未乘2.959。RMS均为归一化动作草稿去噪速度单位。', '',
        '| 去噪范围 | 动作位置 | 通道 | Δμ RMS：中位数 [最小,最大] | d新 RMS | 两者RMS比值 |',
        '|---|---|---|---|---|---|',
    ]
    for phase_name, phase in phases.items():
        for horizon in HORIZONS:
            for block in BLOCKS:
                value = phase['aggregate'][horizon]['blocks'][block]
                delta = interval_description(value['pooled_rms_distributions']['old_minus_new_mean'])
                contrast = interval_description(value['pooled_rms_distributions']['new_contrast'])
                rel = interval_description(value['delta_over_unamplified_contrast_distribution'])
                lines.append(f'| {phase_name} | {horizon} | {block} | {delta} | {contrast} | {rel} |')
    lines += ['', 'H＝前两执行通道；S＝高度、旋转、夹爪五通道组合；P＝25个填充通道。', '',
              '下表为每个坐标在20快照上的描述；初始去噪与全部去噪、前10与全50动作位置分别列出。', '',
              '| 去噪范围 | 动作位置 | 维度 | Δμ有符号均值 | Δμ RMS | 均值×action.std（仅尺度参考） |',
              '|---|---|---|---|---|---|']
    for phase_name, phase in phases.items():
        for horizon in HORIZONS:
            for d in range(7):
                value = phase['aggregate'][horizon]['coordinates'][str(d)]
                stats = value['distributions']
                signed = interval_description(stats['delta_signed_mean_normalized'])
                energy = interval_description(stats['delta_pooled_rms_normalized'])
                scaled = interval_description(stats['delta_signed_mean_times_action_std_scale_reference'])
                lines.append(f"| {phase_name} | {horizon} | {d} {value['name']} | {signed} | {energy} | {scaled} |")
    lines += ['',
              '**解释边界：** 这些是同输入上的输出差异，不能仅凭大小认定哪个通道造成行为恢复。400条记录不是400个独立样本。', '',
              '乘 action.std 只提供反归一化尺度参考，不是最终动作、控制器实际命令差或末端位移。Euler积分从t=1走向0，步长为负；不能把Δμ的正负直接解释成机器人向哪侧运动。', '',
              '实际通道恢复效果须结合八格闭环评估。']
    (archive / 'probe_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('AUDITED 400 probe rows / 20 physical snapshots; 40 matched N first chunks')
    print('WROTE probe_summary.json and probe_summary.md; no independent-sample or significance claim')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    args = parser.parse_args()
    main(args.archive.resolve())
