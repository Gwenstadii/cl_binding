"""Plot the complete single-M screen and a disclosed post-right arrival example."""
import argparse
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def plot(root):
    audit = json.loads((root / 'audit.json').read_text())
    cfg = json.loads((root / 'registered_config.json').read_text())
    rows = audit['per_condition']
    sites = cfg['sites']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1, 1.5]})
    columns = [(w, g) for w in (120, 400) for g in ('left', 'front')]
    for ax, field, title in zip(axes[:2], ('box_first_correct', 'box_correct'),
                                ('First target correct', 'Ever reached requested region')):
        matrix = [[next(r[field] for r in rows if r['condition'] == 'cross_same' and r['site'] == j
                        and (r['window'], r['requested_goal']) == (w, g)) for w, g in columns] for j in sites]
        ax.imshow(matrix, vmin=0, vmax=4, cmap='Blues', aspect='auto')
        for y in range(8):
            for x in range(4):
                ax.text(x, y, f'{matrix[y][x]}/4', ha='center', va='center')
        ax.set(xticks=range(4), xticklabels=['120 L', '120 F', '400 L', '400 F'],
               yticks=range(8), yticklabels=sites, ylabel='CA source index (single M)', title=title)
    ax = axes[2]
    keys = [('FT4', 'native', 0), ('FT4', 'full_old_response', 120),
            ('FT4', 'full_old_response', 400), ('R0', 'native', 0)]
    for offset, goal, color in [(-.18, 'left', '#287c8e'), (.18, 'front', '#bf773c')]:
        values = [next(r['box_first_correct'] for r in rows if (r['model'], r['condition'], r['window']) == key
                       and r['requested_goal'] == goal) for key in keys]
        bars = ax.bar(np.arange(4) + offset, values, width=.34, color=color, label=goal.title())
        ax.bar_label(bars, labels=[f'{v}/4' for v in values], padding=3)
    ax.set(xticks=range(4), xticklabels=['FT4', 'Old policy\nfirst 120', 'Old policy\nall 400', 'R0'],
           ylim=(0, 4.7), yticks=range(5), title='Positive controls: first target correct')
    ax.legend(loc='upper left', frameon=False)
    fig.suptitle('420 episodes; 2 development starts x 2 noise seeds per instruction / condition\nCorrect-source M only in heatmaps; wrong/random controls retained in results.csv', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .89))
    fig.savefig(root / 'all_sites_and_controls.png', dpi=170, bbox_inches='tight')
    plt.close(fig)

    chosen = [r for r in audit['episode_results'] if r['requested_goal'] == 'front' and r['init_index'] == 1
              and r['noise_seed'] == 2026102802 and
              ((r['condition'] == 'native' and r['model'] in ('FT4', 'R0'))
               or (r['condition'] == 'cross_same' and r['site'] == 5 and r['window'] == 400)
               or (r['condition'] == 'full_old_response' and r['window'] == 120))]
    traces = {r['episode_id']: [] for r in chosen}
    with gzip.open(root / 'evaluation/events.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line)
            if r['episode_id'] in traces and r['step']:
                traces[r['episode_id']].append(r)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharex=True, sharey=True)
    chosen.sort(key=lambda r: ['native', 'cross_same', 'full_old_response'].index(r['condition'])
                + (10 if r['model'] == 'R0' else 0))
    for ax, r in zip(axes, chosen):
        for goal, site in cfg['movement_geometry']['sites'].items():
            sizes = [sum(abs(site['rotation'][i][j]) * site['half_size'][j] for j in range(3)) for i in (0, 1)]
            cx, cy = site['centre'][:2]
            ax.add_patch(Rectangle((cx-sizes[0], cy-sizes[1]), 2*sizes[0], 2*sizes[1], fill=False, edgecolor='.6'))
            ax.text(cx, cy, goal, ha='center', va='center', color='.45')
        trace = traces[r['episode_id']]
        points = np.array([t['object_positions']['black_book_1_pos'] for t in trace])
        ax.plot(points[:, 0], points[:, 1], color='.7', linewidth=.8)
        lifted = points[:, 2] - points[0, 2] > .03
        scatter = ax.scatter(points[lifted, 0], points[lifted, 1], c=np.arange(1, 401)[lifted],
                             cmap='viridis', s=7, vmin=1, vmax=400)
        for goal, step in r['box_times'].items():
            if step is not None:
                p = points[step-1]
                ax.annotate(f'{goal}: {step}', p[:2], xytext=(8, 8), textcoords='offset points', fontsize=9,
                            arrowprops={'arrowstyle': '-', 'color': '.3'})
        title = 'R0 native' if r['model'] == 'R0' else {'native': 'FT4 native', 'cross_same': 'Old M @ site 5, all 400',
                                                       'full_old_response': 'Old policy first 120'}[r['condition']]
        ax.set(title=title, xlabel='World X (m)', aspect='equal')
    axes[0].set_ylabel('World Y (m)')
    fig.suptitle('Same Front request, init 1, seed 2026102802: book-centre paths\nPost-hoc example: site 5 reaches Front at 324, after Right at 106; not first-choice recovery', fontsize=12)
    fig.tight_layout(rect=(0, 0, .94, .87))
    cax = fig.add_axes((.955, .2, .012, .5))
    fig.colorbar(scatter, cax=cax, label='Simulation step (lift >3 cm)')
    fig.savefig(root / 'front_after_right_example.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    plot(parser.parse_args().root)
