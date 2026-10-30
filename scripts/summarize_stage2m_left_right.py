"""Extract existing Left/Right CA measurements. No model calls or new trials."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics

from audit_stage2m_cross_attention import audit


def extract(run, output):
    verified = audit(run)
    assert verified['status'] == 'passed'
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    paired = [json.loads(line) for line in (run / 'paired.jsonl').read_text(encoding='utf-8').splitlines()]
    records = [r for r in paired if r['goal_pair'] == ['left', 'right']]
    layers = manifest['ca_layers']['R0']
    expected = set(itertools.product(manifest['indices'], manifest['seeds'], manifest['flow_indices']))
    rows = []
    for layer in layers:
        row = {'layer_zero_based': layer, 'ca_ordinal': layers.index(layer) + 1}
        for role in ('R0', 'FT4'):
            selected = [r for r in records if r['model'] == role and r['layer'] == layer]
            assert len(selected) == len(expected)
            assert {(r['state_index'], r['noise_seed'], r['flow_index']) for r in selected} == expected
            values = [r['features_primary']['M']['rms_diff'] for r in selected]
            row.update({role + '_queries': len(values), role + '_M_rms_mean': statistics.mean(values),
                        role + '_M_rms_min': min(values), role + '_M_rms_max': max(values),
                        role + '_M_nonzero': sum(v > 0 for v in values)})
        row['FT4_to_R0_mean_ratio'] = row['FT4_M_rms_mean'] / row['R0_M_rms_mean']
        rows.append(row)
    velocity = {role: statistics.mean(r['velocity_primary']['rms_diff'] for r in records
                                      if r['model'] == role and r['layer'] == layers[0])
                for role in ('R0', 'FT4')}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'source_audit.json').write_text(json.dumps(verified, indent=2) + '\n', encoding='utf-8')
    with (output / 'per_layer_left_right.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sources = {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
               for name in ('manifest.json', 'baseline.jsonl', 'paired.jsonl')}
    result = {'status': 'passed', 'new_model_calls': 0, 'new_training_updates': 0,
              'pair': ['left', 'right'], 'physical_states': len(manifest['indices']),
              'inputs_per_model_layer': len(expected), 'paired_records': len(records),
              'message': 'post-o_proj, before residual, first 10 action positions, all 720 channels',
              'velocity_primary_mean': velocity, 'source_sha256': sources,
              'R0_Right_was_not_supervised_in_this_adaptation': True,
              'limits': ['M statistics were recorded during the original run; full M tensors were not archived.',
                         'Source audit validates input pairing and metadata; it does not recompute hidden tensors.',
                         'Numerical instruction sensitivity is not correct goal coding or causal behavior control.'],
              'per_layer': rows}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for role, color in [('R0', '#286091'), ('FT4', '#c75c37')]:
        ax.plot(layers, [r[role + '_M_rms_mean'] for r in rows], marker='o', color=color, label=role)
    ax.set(xticks=layers, xlabel='Transformer block index (zero-based)', ylabel='Mean RMS of M(Left) - M(Right)',
           title='Left / Right CA message difference: same state, action draft and flow time')
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.text(.5, .01, '36 fixed inputs per model/layer; four development starts. R0 was not trained on Right.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output / 'left_right_ca_messages.png', dpi=160)
    plt.close(fig)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.run, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'new_model_calls', 'paired_records', 'velocity_primary_mean')}))
