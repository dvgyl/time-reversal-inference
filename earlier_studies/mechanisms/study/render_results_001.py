"""Render declared outcomes and complete summary tables after the study."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    'correlated_reversible_delays': 'Reversible source, delays',
    'directed_ring_identity': 'Directed ring, identity',
    'directed_ring_delays': 'Directed ring, delays',
    'directed_ring_attenuated': 'Directed ring, attenuation',
    'directed_ring_noisy': 'Directed ring, added noise',
    'directed_ring_response_zero': 'Directed ring, response zero',
    'finite_ma_nonzero_cycle': 'Finite-covariance example'}
METHODS = [('lag', 'Lag'), ('bridge', 'Cumulative'), ('intersection', 'Intersection'), ('supplied', 'Supplied')]


def write_csv(path, rows):
    columns = list(rows[0])
    with open(path, 'x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def cycle_figure(groups, output):
    selected = [group for group in groups if group['experiment'] == 'cycle']
    fig, ax = plt.subplots(figsize=(9.1, 6.5), layout='constrained')
    indices, left = np.arange(len(selected)), np.zeros(len(selected))
    colors = [('REJECT', 'Reject', '#165f91'), ('DO_NOT_REJECT', 'Do not reject', '#cbd5df'),
              ('ABSTAIN', 'Abstain', '#d79a32'), ('ERROR', 'Execution error', '#b53338')]
    for key, label, color in colors:
        values = np.array([group['statuses'].get(key, 0)/group['planned'] for group in selected])
        ax.barh(indices, values, left=left, color=color, label=label, height=.68)
        left += values
    ax.set_yticks(indices, [LABELS[group['cell']]+'\nN = '+format(group['n'], ',') for group in selected])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, .25, .5, .75, 1.])
    ax.set_xlabel('Fraction of all declared replicates')
    for index, group in enumerate(selected):
        values = [group['statuses'].get(key, 0) for key, _, _ in colors]
        ax.text(1.035, index, ' / '.join(map(str, values)), va='center', fontsize=8.5, clip_on=False)
    ax.text(1.035, -1., 'R / N / A / E', fontsize=8, clip_on=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -0.09), ncol=2, frameon=False, fontsize=9)
    fig.savefig(output/'cycle_outcomes.pdf', metadata={'Title': 'Prospective cycle-test outcomes'})
    fig.savefig(output/'cycle_outcomes.png', dpi=160)
    plt.close(fig)


def source_label(cell):
    index, regime = cell.split('_', 1)
    phi = {'p0': '0.8', 'p1': '0.97', 'p2': '0.995'}[index]
    name = {'reversible': 'Reversible', 'weak_delay': 'Weak delay', 'strong_delay': 'Strong delay'}[regime]
    return '$\\phi='+phi+'$  '+name


def source_figure(groups, output):
    long = max(group['n'] for group in groups if group['experiment'] == 'source')
    selected = [group for group in groups if group['experiment'] == 'source' and group['n'] == long]
    cells = list(dict.fromkeys(group['cell'] for group in selected))
    mapping = {(group['cell'], group['method']): group for group in selected}
    fig, axes = plt.subplots(1, 3, figsize=(11., 5.8), sharey=True, layout='constrained')
    for ax, (event, title) in zip(axes, [('finite_endpoint', 'Finite upper endpoint'), ('usable', 'Usable bank test'), ('rejection', 'Rejection')]):
        counts = np.array([[mapping[cell, method][event]['successes'] for method, _ in METHODS] for cell in cells])
        planned = np.array([[mapping[cell, method]['planned'] for method, _ in METHODS] for cell in cells])
        missing = np.array([[mapping[cell, method][event]['missing'] for method, _ in METHODS] for cell in cells])
        ax.imshow(counts/planned, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(4), [label for _, label in METHODS], rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(cells)), [source_label(cell) for cell in cells], fontsize=9)
        for i in range(len(cells)):
            for j in range(4):
                label = str(counts[i, j])+'/'+str(planned[i, j])
                if missing[i, j]:
                    label += '*'
                ax.text(j, i, label, ha='center', va='center', fontsize=9,
                        color='white' if counts[i, j]/planned[i, j] > .5 else '#152735')
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(cells), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5)
        ax.tick_params(which='minor', bottom=False, left=False)
    fig.suptitle('New paired source comparisons at N = '+format(long, ','), fontsize=12)
    fig.savefig(output/'source_mechanisms.pdf', metadata={'Title': 'Prospective source-confidence and bank outcomes'})
    fig.savefig(output/'source_mechanisms.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analysis, output = Path(args.analysis), Path(args.output)
    output.mkdir()
    summary = json.loads((analysis/'summary.json').read_text())
    groups = summary['groups']
    rows = []
    for group in groups:
        for event in ('rejection', 'usable', 'finite_endpoint', 'source_coverage'):
            if group.get(event) is None:
                continue
            values = group[event]
            rows.append(dict(experiment=group['experiment'], cell=group['cell'], n=group['n'], method=group['method'],
                             event=event, planned=values['planned'], defined=values['defined'], successes=values['successes'],
                             missing=values['missing'], possible_count_low=values['possible_count'][0], possible_count_high=values['possible_count'][1],
                             confidence=values['confidence'], interval_low=values['interval'][0], interval_high=values['interval'][1]))
    write_csv(output/'event_summary.csv', rows)
    paired = []
    for item in summary['paired']:
        paired.append({key: json.dumps(value) if isinstance(value, list) else value for key, value in item.items()})
    write_csv(output/'paired_comparisons.csv', paired)
    cycle_figure(groups, output)
    source_figure(groups, output)


if __name__ == '__main__':
    main()
