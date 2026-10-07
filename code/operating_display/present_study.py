"""Display complete frozen summaries without reevaluating any method."""

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
import hashlib
import json
from pathlib import Path
import shutil
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from validate_summaries import validate_complete


LABELS = {
    'lag': 'Lag set', 'lower_cn': 'Lower-tail finite cutoff',
    'lower_cstar': 'Lower-tail analytic cutoff', 'lower_intersection': 'Lower-tail intersection',
    'two_sided': 'Two-sided set', 'two_sided_intersection': 'Two-sided intersection',
    'supplied': 'Supplied persistence', 'common': 'Common envelope',
    'diagonal': 'Diagonal envelope', 'combined_frequencies': 'Two frequencies',
    'calibrated': 'Calibrated', 'phase_ignored': 'Phase ignored',
    'baseline/independent_windows': 'Independent windows',
    'baseline/moving_block': 'Moving-block bootstrap',
    'baseline/imaginary_coherency': 'Imaginary coherency',
}
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#333333', '#E69F00']
LINE_STYLES = ('-', '--', '-.', ':', (0, (5, 1, 1, 1, 1, 1)))
MARKERS = ('o', 's', '^', 'D', 'x')
METRIC_LABELS = {'reject': 'Rejection fraction', 'nonreject': 'Nonrejection fraction',
                 'abstain': 'Abstention fraction'}


def series_style(index):
    """Keep coincident markers visible without changing data coordinates."""
    return dict(linestyle=LINE_STYLES[index], marker=MARKERS[index],
                markersize=6 - .6 * index, markerfacecolor='none',
                markeredgewidth=1, linewidth=1.1)


def overlap_note(fig):
    """Keep one compact note for all plotted series."""
    fig.text(.5, .012,
             'Overlapping curves have equal fractions. All listed series are drawn.',
             ha='center', va='bottom', fontsize=7.5)


def digest(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class Count:
    cell: str
    method: str
    n: int
    reject: int
    nonreject: int
    abstain: int
    lower: float
    upper: float
    lower_text: str = ''
    upper_text: str = ''

    @classmethod
    def parse(cls, row, expected):
        result = cls(row['cell'], row['method'], int(row['n']),
                     int(row['reject']), int(row['nonreject']), int(row['abstain']),
                     float(row['rejection_lower']), float(row['rejection_upper']),
                     str(row['rejection_lower']), str(row['rejection_upper']))
        if result.n != expected or min(result.reject, result.nonreject, result.abstain) < 0:
            raise ValueError('A count row has an invalid denominator or negative count.')
        if result.reject + result.nonreject + result.abstain != result.n:
            raise ValueError('A count row omits or duplicates an outcome.')
        fraction = result.reject / result.n
        if abs(float(row['rejection_fraction']) - fraction) > 1e-14:
            raise ValueError('The rejection fraction disagrees with its count.')
        if not 0 <= result.lower <= fraction <= result.upper <= 1:
            raise ValueError('A rejection interval does not contain the observed fraction.')
        return result


def outward_interval(value):
    """Format saved endpoints outward without a float conversion."""
    lower = Decimal(value.lower_text) if value.lower_text else Decimal.from_float(value.lower)
    upper = Decimal(value.upper_text) if value.upper_text else Decimal.from_float(value.upper)
    quantum = Decimal('.001')
    return (format(lower.quantize(quantum, rounding=ROUND_FLOOR), '.3f'),
            format(upper.quantize(quantum, rounding=ROUND_CEILING), '.3f'))


def count_map(registry, rows):
    cells = {cell['id']: cell for cell in registry['cells']}
    result = {}
    for row in rows:
        key = (row['cell'], row['method'])
        if row['cell'] not in cells or key in result:
            raise ValueError('A count row has an unknown condition or duplicate key.')
        result[key] = Count.parse(row, cells[row['cell']]['replicates'])
    if {key[0] for key in result} != cells.keys():
        raise ValueError('The summary does not cover every declared condition.')
    required = {'source': ('lag', 'two_sided', 'two_sided_intersection', 'supplied'),
                'cycle': ('common', 'diagonal', 'combined_frequencies'),
                'brownian': ('calibrated', 'phase_ignored'),
                'phase_boundary': ('calibrated', 'phase_ignored')}
    for cell in cells.values():
        for method in required[cell['family']]:
            if (cell['id'], method) not in result:
                raise ValueError('A declared primary method is missing.')
    return result


def save_figure(fig, output, name, fixture):
    if fixture:
        fig.text(.5, 1.065, 'DETERMINISTIC DISPLAY FIXTURE. NO SCIENTIFIC RESULTS.',
                 ha='center', va='bottom', color='#990000', fontsize=8.5)
    fig.savefig(output / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(output / (name + '.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def curve(ax, cells, method, counts, metric, color, label=None, linestyle='-', style_index=0):
    cells = sorted(cells, key=lambda cell: cell['N'])
    values = [counts[(cell['id'], method)] for cell in cells]
    x = [cell['N'] for cell in cells]
    y = np.array([getattr(value, metric) / value.n for value in values])
    style = series_style(style_index)
    if style_index == 0:
        style['linestyle'] = linestyle
    if metric == 'reject':
        error = np.array([[fraction - value.lower for fraction, value in zip(y, values)],
                          [value.upper - fraction for fraction, value in zip(y, values)]])
        ax.errorbar(x, y, yerr=error, capsize=2, color=color, label=label, **style)
    else:
        ax.plot(x, y, color=color, label=label, **style)
    ax.set_xscale('log', base=2)
    ax.set_xlim(min(x) / 1.18, max(x) * 1.18)
    ax.set_ylim(-.04, 1.04)
    ax.grid(alpha=.18)


def operating_figures(registry, counts, output, fixture):
    cells = registry['cells']
    source = [c for c in cells if c['id'].startswith('source_199_200_strong_')]
    methods = ('lag', 'two_sided', 'two_sided_intersection', 'supplied')
    fig, axes = plt.subplots(3, 4, figsize=(7, 6.7), sharex=True, sharey=True)
    for column, method in enumerate(methods):
        for row, metric in enumerate(('reject', 'nonreject', 'abstain')):
            ax = axes[row, column]
            curve(ax, source, method, counts, metric, COLORS[column])
            if column == 0:
                ax.set_ylabel({'reject': 'Rejection fraction', 'nonreject': 'Nonrejection fraction',
                               'abstain': 'Abstention fraction'}[metric])
            if row == 0:
                ax.set_title(textwrap.fill(LABELS[method], 16), fontsize=9)
            if row == 2:
                ax.set_xlabel('Record length N')
        if method == 'two_sided':
            axes[0, column].plot([524289], [.90], marker='_', color='black', markersize=15)
    fig.suptitle('Persistent source: persistence = 0.995,\nstrong one-step delay', y=1.02, fontsize=10)
    fig.tight_layout()
    save_figure(fig, output, 'source_operating', fixture)

    fig, axes = plt.subplots(3, 3, figsize=(7, 6.4), sharex=True, sharey=True)
    for column, method in enumerate(('common', 'diagonal', 'combined_frequencies')):
        for row, metric in enumerate(('reject', 'nonreject', 'abstain')):
            ax = axes[row, column]
            for index, (prefix, color, label) in enumerate((('cycle_null_', COLORS[0], 'Equilibrium'),
                                                          ('cycle_rotating_', COLORS[1], 'Rotation'))):
                subset = [c for c in cells if c['id'].startswith(prefix)]
                curve(ax, subset, method, counts, metric, color, label, style_index=index)
            if column == 0:
                ax.set_ylabel(METRIC_LABELS[metric])
            if row == 0:
                ax.set_title(textwrap.fill(LABELS[method], 18), fontsize=9)
            if row == 2:
                ax.set_xlabel('Record length N')
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle('Rotational model with unequal detector responses', y=1.02, fontsize=10)
    overlap_note(fig)
    fig.tight_layout(rect=(0, .055, 1, .98), h_pad=1)
    save_figure(fig, output, 'cycle_operating', fixture)

    methods = ('calibrated', 'phase_ignored', 'baseline/independent_windows',
               'baseline/moving_block', 'baseline/imaginary_coherency')
    fig, axes = plt.subplots(3, 3, figsize=(7, 6.7), sharex=True, sharey=True)
    for column, temperature in enumerate((1, 2, 4)):
        subset = [c for c in cells if c['id'].startswith('brownian_T%d_' % temperature)]
        for row, metric in enumerate(('reject', 'nonreject', 'abstain')):
            for index, (color, method) in enumerate(zip(COLORS, methods)):
                curve(axes[row, column], subset, method, counts, metric, color, LABELS[method],
                      style_index=index)
            if column == 0:
                axes[row, column].set_ylabel(METRIC_LABELS[metric])
            if row == 0:
                axes[row, column].set_title('T2 / T1 = %d' % temperature, fontsize=9)
            if row == 2:
                axes[row, column].set_xlabel('Record length N')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(.5, .045), fontsize=7.5)
    fig.suptitle('Brownian model: detector phase and statistical assumptions', y=1.02, fontsize=10)
    overlap_note(fig)
    fig.tight_layout(rect=(0, .17, 1, .98), h_pad=1)
    save_figure(fig, output, 'brownian_operating', fixture)


def gain_figures(counts, output, fixture):
    gains = ('1/16', '1/4', '1', '4', '16', '-1')
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5), sharey=True)
    for column, regime in enumerate(('null', 'rotating')):
        cell = 'cycle_identity_' + regime
        for row, metric in enumerate(('reject', 'abstain')):
            ax = axes[row, column]
            for index, method in enumerate(('common', 'diagonal', 'inverse_transport')):
                values = [counts[(cell, 'gain_' + gain + '/' + method)] for gain in gains]
                y = [getattr(v, metric) / v.n for v in values]
                label = method.replace('_', ' ')
                ax.plot(range(6), y, color=COLORS[index], label=label, **series_style(index))
                if metric == 'reject':
                    ax.errorbar(range(6), y, yerr=[[v.reject / v.n - v.lower for v in values],
                                                 [v.upper - v.reject / v.n for v in values]],
                                fmt='none', ecolor=COLORS[index], capsize=2)
            ax.set_xticks(range(6), gains)
            ax.set_ylim(-.04, 1.04)
            ax.grid(alpha=.18)
            if column == 0:
                ax.set_ylabel(METRIC_LABELS[metric])
            if row == 0:
                ax.set_title('Equilibrium' if regime == 'null' else 'Rotation', fontsize=9)
            if row == 1:
                ax.set_xlabel('Third-channel gain')
    axes[0, 0].legend(fontsize=7.5)
    overlap_note(fig)
    fig.tight_layout(rect=(0, .055, 1, 1), h_pad=1)
    save_figure(fig, output, 'gain_controls', fixture)

    factors = ('1', '3/2', '2', '4')
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.7))
    for ax, regime in zip(axes, ('null', 'rotating')):
        values = [[counts[('cycle_identity_' + regime, 'inflation_K%s_B%s' % (ck, cb))]
                   for cb in factors] for ck in factors]
        fractions = np.array([[v.reject / v.n for v in row] for row in values])
        ax.imshow(fractions, vmin=0, vmax=1, cmap='Blues')
        for i, row in enumerate(values):
            for j, v in enumerate(row):
                lower, upper = outward_interval(v)
                ax.text(j, i, '%d/%d\n[%s,\n %s]' % (v.reject, v.n, lower, upper),
                        ha='center', va='center', fontsize=7.5,
                        color='white' if v.reject / v.n > .5 else 'black')
        ax.set_xticks(range(4), ('1', '1.5', '2', '4'))
        ax.set_yticks(range(4), ('1', '1.5', '2', '4'))
        ax.set_xlabel('Bias multiplier')
        ax.set_ylabel('Covariance multiplier')
        ax.set_title('Equilibrium' if regime == 'null' else 'Rotation', fontsize=9)
    fig.tight_layout()
    save_figure(fig, output, 'bound_inflation', fixture)


def source_figures(registry, rows, output, fixture):
    source = {(row['cell'], row['method']): row for row in rows}
    methods = ('lag', 'lower_cn', 'two_sided', 'two_sided_intersection')
    for regime in ('null', 'strong'):
        fig, axes = plt.subplots(2, 3, figsize=(7, 6.2), sharey='row')
        for column, (token, phi) in enumerate((('4_5', .8), ('97_100', .97), ('199_200', .995))):
            cells = sorted([c for c in registry['cells'] if c['id'].startswith('source_%s_%s_' % (token, regime))],
                           key=lambda c: c['N'])
            x = [(c['N'] - 1) * (1 - phi) for c in cells]
            for index, (color, method) in enumerate(zip(COLORS, methods)):
                values = [source[(c['id'], method)] for c in cells]
                coverage = [float(v['coverage']) for v in values]
                axes[0, column].errorbar(x, coverage,
                    yerr=[[float(v['coverage']) - float(v['coverage_lower']) for v in values],
                          [float(v['coverage_upper']) - float(v['coverage']) for v in values]],
                    color=color, capsize=2, label=LABELS[method], **series_style(index))
                finite = [float(v['finite_fraction']) for v in values]
                axes[1, column].plot(x, finite, color=color, **series_style(index))
            for ax in axes[:, column]:
                ax.set_xscale('log'); ax.set_ylim(-.04, 1.04); ax.grid(alpha=.18)
            axes[0, column].set_title('phi = %g' % phi, fontsize=9)
            axes[1, column].set_xlabel('(N - 1)(1 - phi)')
        axes[0, 0].set_ylabel('Source-set coverage')
        axes[1, 0].set_ylabel('Finite-endpoint fraction')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(.5, .045), fontsize=7.5)
        fig.suptitle('Source confidence: ' + ('reversible source' if regime == 'null' else 'strong delay'), y=1.02, fontsize=10)
        overlap_note(fig)
        fig.tight_layout(rect=(0, .15, 1, .98), h_pad=1)
        save_figure(fig, output, 'source_confidence_' + regime, fixture)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not (args.run / 'RUN_COMPLETE.json').is_file():
        raise ValueError('Publication requires a complete scientific run.')
    registry = json.loads(args.registry.read_text())
    metadata = json.loads((args.analysis / 'ANALYSIS.json').read_text())
    start = json.loads((args.run / 'RUN_START.json').read_text())
    expected = sum(c['replicates'] for c in registry['cells'])
    complete = json.loads((args.run / 'RUN_COMPLETE.json').read_text())
    if complete['records'] != expected:
        raise ValueError('The run completion marker has the wrong record count.')
    if metadata['registry_sha256'] != digest(args.registry) or start['registry_sha256'] != digest(args.registry):
        raise ValueError('Registry hashes disagree.')
    if metadata['records'] != expected or metadata['cells'] != len(registry['cells']) or metadata['failures'] != 0:
        raise ValueError('The analysis is incomplete.')
    rows = read_csv(args.analysis / 'cell_summary.csv')
    counts = count_map(registry, rows)
    source_rows = read_csv(args.analysis / 'source_summary.csv')
    validation = validate_complete(registry, args.analysis, rows, source_rows, metadata)
    cells = {c['id']: c for c in registry['cells']}
    args.output.mkdir(exist_ok=False)
    for name in ('cell_summary.csv', 'source_summary.csv', 'paired_comparisons.csv', 'ANALYSIS.json'):
        shutil.copyfile(args.analysis / name, args.output / name)
    write_csv(args.output / 'complete_counts_with_design.csv',
              [dict(row, family=cells[row['cell']]['family'], target=cells[row['cell']]['target'],
                    N=cells[row['cell']]['N'], parameters=json.dumps(cells[row['cell']]['parameters'], sort_keys=True))
               for row in rows])
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
    operating_figures(registry, counts, args.output, False)
    gain_figures(counts, args.output, False)
    source_figures(registry, source_rows, args.output, False)
    inputs = [args.registry, args.run / 'RUN_COMPLETE.json', args.run / 'RUN_START.json',
              args.analysis / 'ANALYSIS.json'] + sorted(args.analysis.glob('*.csv'))
    manifest = {'scope': 'Display of complete frozen summaries. No method reevaluation.',
                'records': expected, 'cells': len(cells), 'summary_validation': validation,
                'inputs': [{'path': str(p.resolve()), 'sha256': digest(p)} for p in inputs],
                'script_sha256': digest(Path(__file__)),
                'validator_sha256': digest(Path(__file__).with_name('validate_summaries.py')),
                'outputs': [{'path': p.name, 'sha256': digest(p)} for p in sorted(args.output.iterdir())]}
    (args.output / 'PRESENTATION.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
