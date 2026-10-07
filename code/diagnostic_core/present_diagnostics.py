"""Display complete descriptive summaries without evaluating an inference rule."""

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'summary_display'))
from present_study import LABELS, COLORS, digest, read_csv, write_csv, save_figure as base_save_figure
from validate_summaries import validate_complete, expected_keys

QUANTILES = ('min', 'q25', 'median', 'q75', 'max')
DECISION_METRICS = ('threshold', 'margin', 'statistic_threshold_ratio', 'source_envelope', 'q',
                    'scale_denominator', 'variance_scale', 'phase_allowance', 'sampling_radius',
                    'rounding_allowance')
SOURCE_METRICS = ('width', 'persistence_span_ratio', 'log_persistence_width',
                  'fixed_minimum_envelope', 'fixed_placement_factor',
                  'length_dependent_minimum_envelope', 'length_dependent_placement_factor')
FAILURES = ('input_numerical', 'empty_source_set', 'hull_reaching_one', 'no_finite_covariance',
            'no_positive_scale_denominator', 'available_insufficient_margin', 'rejection')
FAILURE_LABELS = ('Input or arithmetic', 'Empty set', 'Endpoint reaches one', 'No covariance bound',
                  'No positive scale bound', 'Insufficient margin', 'Rejection')
ANALYSIS_FILES = ('decisions.csv', 'source_sets.csv', 'source_summary.csv',
                  'cell_summary.csv', 'paired_comparisons.csv')
DIAGNOSTIC_FILES = ('diagnostics.jsonl.gz', 'first_failure_counts.json',
                    'source_coverage_counts.json', 'INPUT_BINDING.json')


def save_figure(fig, output, name, fixture):
    if fixture:
        fig.subplots_adjust(top=.88)
    base_save_figure(fig, output, name, fixture)


def finite(value):
    if value is None or value == '':
        return None
    x = float(value)
    if not math.isfinite(x):
        raise ValueError('A descriptive field is not finite.')
    return x


def checked_quantiles(path, summaries, metrics, count_fields):
    """Check each saved descriptive summary and retain its actual denominator."""
    values = defaultdict(lambda: defaultdict(list))
    totals = defaultdict(int)
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = row['cell'], row['method']
            totals[key] += 1
            for metric in metrics:
                value = finite(row.get(metric))
                if value is not None:
                    values[key][metric].append(value)
    summary = {(r['cell'], r['method']): r for r in summaries}
    if len(summary) != len(summaries) or summary.keys() != totals.keys():
        raise ValueError('Descriptive summary keys differ from the complete records.')
    output = []
    for key in sorted(summary):
        row = summary[key]
        if int(row['n']) != totals[key]:
            raise ValueError('A descriptive summary has an incorrect denominator.')
        for metric in metrics:
            data = values[key][metric]
            result = dict(cell=key[0], method=key[1], metric=metric,
                          n=totals[key], finite_n=len(data), missing_n=totals[key] - len(data))
            if data:
                for label, expected in zip(QUANTILES, np.quantile(data, [0, .25, .5, .75, 1])):
                    actual = finite(row.get(metric + '_' + label))
                    if actual is None or not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-13):
                        raise ValueError('A descriptive quantile differs: ' + str(key) + '/' + metric)
                    result[label] = actual
                if count_fields and int(row[metric + '_n']) != len(data):
                    raise ValueError('A descriptive metric has an incorrect finite count.')
            else:
                if any(row.get(metric + '_' + q) not in (None, '') for q in QUANTILES):
                    raise ValueError('A missing metric was given a numeric summary.')
                if count_fields and row.get(metric + '_n') not in (None, '', '0', 0):
                    raise ValueError('A missing metric has a positive finite count.')
                result.update({q: None for q in QUANTILES})
            output.append(result)
    return output


LINE_STYLES = ('-', '--', '-.', ':')
MARKERS = ('o', 's', '^', 'D')


def method_style(method):
    index = ('lag', 'lower_cn', 'two_sided', 'two_sided_intersection').index(method)
    return dict(linestyle=LINE_STYLES[index], marker=MARKERS[index],
                markersize=7-index, markerfacecolor='none')

def distribution(ax, cells, method, metric, stats, color, label):
    """Plot one median and middle-half range per condition with a finite count."""
    x, med, low, high = [], [], [], []
    for cell in sorted(cells, key=lambda c: c['N']):
        row = stats[cell['id'], method, metric]
        if row['finite_n']:
            x.append(cell['N']); med.append(row['median'])
            low.append(row['median'] - row['q25']); high.append(row['q75'] - row['median'])
    if x:
        ax.errorbar(x, med, yerr=[low, high], color=color, label=label,
                    linewidth=1, capsize=2, **method_style(method))
    else:
        ax.plot([], [], color=color, label=label, **method_style(method))
    ax.set_xscale('log', base=2)
    if cells:
        ax.set_xlim(min(c['N'] for c in cells) / 1.18, max(c['N'] for c in cells) * 1.18)
    ax.grid(alpha=.18)


def source_diagnostics(registry, numeric, output, fixture):
    stats = {(r['cell'], r['method'], r['metric']): r for r in numeric}
    cells = [c for c in registry['cells'] if c['id'].startswith('source_199_200_strong_')]
    methods = ('lag', 'lower_cn', 'two_sided', 'two_sided_intersection')
    metrics = ('persistence_span_ratio', 'fixed_placement_factor', 'statistic_threshold_ratio')
    titles = ('Persistence hull span', 'Fixed-bank placement factor', 'Selected statistic / threshold')
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
    for column, (metric, title) in enumerate(zip(metrics, titles)):
        for method, color in zip(methods, COLORS):
            distribution(axes[0, column], cells, method, metric, stats, color, LABELS[method])
            ordered = sorted(cells, key=lambda c: c['N'])
            rows = [stats[c['id'], method, metric] for c in ordered]
            axes[1, column].plot([c['N'] for c in ordered],
                                [r['finite_n'] / r['n'] for r in rows],
                                color=color, **method_style(method))
        axes[0, column].set_title(title)
        if column == 2:
            axes[0, column].set_yscale('symlog', linthresh=1)
            upper = max(1.1, axes[0, column].get_ylim()[1])
            axes[0, column].set_ylim(0, upper)
        else:
            axes[0, column].set_yscale('log')
        axes[1, column].set_xscale('log', base=2)
        axes[1, column].set_ylim(-.04, 1.04)
        axes[1, column].set_xlabel('Record length N')
        axes[1, column].grid(alpha=.18)
    axes[0, 2].axhline(1, color='black', linestyle=':', linewidth=.8)
    axes[0, 0].set_ylabel('Median and middle half')
    axes[1, 0].set_ylabel('Fraction with saved finite value')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='lower center', bbox_to_anchor=(.5, -.035), fontsize=8)
    fig.tight_layout()
    save_figure(fig, output, 'source_hull_bank_threshold', fixture)

    metrics = ('phase_allowance', 'sampling_radius', 'rounding_allowance', 'scale_denominator')
    titles = ('Phase allowance', 'Sampling radius', 'Recorder allowance', 'Scale denominator')
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ax, metric, title in zip(axes.flat, metrics, titles):
        for method, color in zip(methods, COLORS):
            distribution(ax, cells, method, metric, stats, color, LABELS[method])
        ax.set_title(title)
        ax.set_xlabel('Record length N')
        ax.set_ylabel('Median and middle half')
        # Zero recorder allowances are valid, so this display keeps a linear axis.
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='lower center', bbox_to_anchor=(.5, -.035), fontsize=8)
    fig.tight_layout()
    save_figure(fig, output, 'source_threshold_components', fixture)


def source_confidence_weak(registry, rows, output, fixture):
    source = {(row['cell'], row['method']): row for row in rows}
    methods = ('lag', 'lower_cn', 'two_sided', 'two_sided_intersection')
    for regime in ('weak',):
        fig, axes = plt.subplots(2, 3, figsize=(10, 5.7), sharey='row')
        for column, (token, phi) in enumerate((('4_5', .8), ('97_100', .97), ('199_200', .995))):
            cells = sorted([c for c in registry['cells'] if c['id'].startswith('source_%s_%s_' % (token, regime))],
                           key=lambda c: c['N'])
            x = [(c['N'] - 1) * (1 - phi) for c in cells]
            for color, method in zip(COLORS, methods):
                values = [source[(c['id'], method)] for c in cells]
                coverage = [float(v['coverage']) for v in values]
                axes[0, column].errorbar(x, coverage,
                    yerr=[[float(v['coverage']) - float(v['coverage_lower']) for v in values],
                          [float(v['coverage_upper']) - float(v['coverage']) for v in values]],
                    color=color, capsize=2, label=LABELS[method], **method_style(method))
                axes[1, column].plot(x, [float(v['finite_fraction']) for v in values],
                                     color=color, **method_style(method))
            for ax in axes[:, column]:
                ax.set_xscale('log'); ax.set_ylim(-.04, 1.04); ax.grid(alpha=.18)
            axes[0, column].set_title('phi = %g' % phi)
            axes[1, column].set_xlabel('(N - 1)(1 - phi)')
        axes[0, 0].set_ylabel('Source-set coverage')
        axes[1, 0].set_ylabel('Finite-endpoint fraction')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(.5, -.035), fontsize=8)
        fig.suptitle('Source confidence: ' + 'weak delay', y=.95 if fixture else 1.02)
        fig.tight_layout()
        save_figure(fig, output, 'source_confidence_' + regime, fixture)


def failure_counts(path, summaries, registry):
    cells = {c['id']: c for c in registry['cells']}
    expected = {(r['cell'], r['method']): r for r in summaries
                if cells[r['cell']]['family'] == 'source' and not r['method'].startswith('baseline/')}
    result = defaultdict(dict)
    for row in json.loads(path.read_text()):
        key, category = (row['cell'], row['method']), row['first_failure']
        count = row['count']
        if key not in expected or category not in FAILURES or category in result[key]:
            raise ValueError('A failure row has an unknown or repeated key.')
        if type(count) is not int or count < 0:
            raise ValueError('A failure count is invalid.')
        result[key][category] = count
    for key, counts in result.items():
        if counts.keys() != set(FAILURES) or sum(counts.values()) != int(expected[key]['n']):
            raise ValueError('Failure categories omit a declared outcome.')
        if counts['rejection'] != int(expected[key]['reject']):
            raise ValueError('Failure rejection counts differ from analysis decisions.')
        if counts['available_insufficient_margin'] != int(expected[key]['nonreject']):
            raise ValueError('Available failure counts differ from analysis decisions.')
    if result.keys() != expected.keys():
        raise ValueError('Failure counts omit a declared source method.')
    return result


def failure_figure(registry, failures, output, fixture):
    cells = sorted([c for c in registry['cells'] if c['id'].startswith('source_199_200_strong_')],
                   key=lambda c: c['N'])
    fig, axes = plt.subplots(1, 4, figsize=(11, 4.5), sharey=True)
    colors = ('#777777', '#B3B3B3', '#E69F00', '#CC79A7', '#D55E00', '#56B4E9', '#009E73')
    hatches = ('///', '...', '\\\\', 'xx', '++', 'oo', '--')
    for ax, method in zip(axes, ('lag', 'lower_cn', 'two_sided', 'two_sided_intersection')):
        bottom = np.zeros(len(cells))
        for category, label, color, hatch in zip(FAILURES, FAILURE_LABELS, colors, hatches):
            values = np.array([failures[c['id'], method][category] / c['replicates'] for c in cells])
            ax.bar(range(len(cells)), values, bottom=bottom, label=label, color=color, hatch=hatch, edgecolor='black', linewidth=.25)
            bottom += values
        ax.set_xticks(range(len(cells)), [str(c['N']) for c in cells], rotation=90, fontsize=7)
        ax.set_title(LABELS[method]); ax.set_xlabel('Record length N')
    axes[0].set_ylabel('Fraction of all declared records')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='lower center', bbox_to_anchor=(.5, -.13), fontsize=8)
    fig.tight_layout()
    save_figure(fig, output, 'source_first_failure', fixture)


def split_method_setting(method, prefix):
    """Keep the complete base method when a setting contains a fraction."""
    setting, base = method[len(prefix):].split('/', 1)
    if prefix == 'gain_' and base.partition('/')[0].lstrip('-').isdigit():
        denominator, base = base.split('/', 1)
        setting += '/' + denominator
    return setting, base


def label_method(method):
    if method in LABELS:
        return LABELS[method]
    labels = dict(inverse_transport='Inverse gain transport',
                  two_sided_adaptive_bank='Two-sided set, length-dependent bank',
                  maximum_scale='Calibrated, maximum variance ceiling',
                  old_operator='Calibrated, factor-two operator allowance',
                  missing_bound='Covariance bound not supplied',
                  missing_response='Relative phase bound not supplied')
    if method in labels:
        return labels[method]
    for suffix, label in (('_maximum_scale', 'maximum variance ceiling'),
                          ('_old_operator', 'factor-two operator allowance')):
        if method.endswith(suffix):
            return label_method(method[:-len(suffix)]) + ', ' + label
    if method.startswith('numerical/'):
        _, variant, base = method.split('/', 2)
        name = variant.replace('depth_', 'dyadic depth ').replace('precision_', 'arithmetic bits ')
        return label_method(base) + ', ' + name
    for prefix in ('fractional_', 'recorder_', 'gain_'):
        if method.startswith(prefix):
            setting, base = split_method_setting(method, prefix)
            label = {'fractional_': 'fractional grid bits ', 'recorder_': 'recorder bits ',
                     'gain_': 'gain '}[prefix]
            return label_method(base) + ', ' + label + setting
    if method.startswith('inflation_K'):
        k, b = method[len('inflation_K'):].split('_B')
        return 'Diagonal envelope, covariance factor ' + k + ', bias factor ' + b
    if method.startswith('underestimated_'):
        _, component, factor = method.split('_', 2)
        return 'Diagonal envelope, ' + ('covariance' if component == 'K' else 'bias') + ' factor ' + factor
    if method.startswith(('two_sided_shape_', 'shape_')):
        prefix = 'two_sided_shape_' if method.startswith('two_sided_shape_') else 'shape_'
        return ('Two-sided set' if prefix == 'two_sided_shape_' else 'Calibrated') + ', shape factor ' + method[len(prefix):]
    return method.replace('baseline/', '').replace('_', ' ')


def table_pages(rows):
    """Give wrapped text enough space and retain the original row order."""
    page, units = [], 0
    for row in rows:
        height = max(str(value).count('\n') + 1 for value in row) + 1
        if page and (units + height > 44 or len(page) == 12):
            yield page
            page, units = [], 0
        page.append((row, height))
        units += height
    if page:
        yield page


def table_pdf(path, title, headers, rows, fixture, note):
    """Keep each row readable and repeat column labels on each page."""
    with PdfPages(path) as pdf:
        for page in table_pages(rows):
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ax.axis('off')
            ax.set_title(title, loc='left', fontsize=13, pad=18)
            widths = ([0.22, .06, .28, .12, .15, .17] if len(headers) == 6 else
                      [.14, .05, .13, .05, .075, .075, .085, .085, .07, .095, .14]
                      if len(headers) == 11 else None)
            table = ax.table(cellText=[r for r, height in page], colLabels=headers,
                             colWidths=widths,
                             cellLoc='left', colLoc='left', loc='upper left', bbox=[0, .07, 1, .9])
            table.auto_set_font_size(False); table.set_fontsize(8)
            header_height = max(h.count('\n') + 1 for h in headers) + 1
            for column in range(len(headers)):
                table[0, column].set_height(header_height)
                for index, (row, height) in enumerate(page, start=1):
                    table[index, column].set_height(height)
            fig.text(.07, .025, textwrap.fill(note, 150), fontsize=8)
            if fixture:
                fig.text(.5, .98, 'DETERMINISTIC DISPLAY FIXTURE. NO SCIENTIFIC RESULTS.',
                         ha='center', color='#990000', fontsize=9)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def condition_label(cell):
    p = cell['parameters']
    if cell['family'] == 'source':
        regime = 'reversible' if p['delay'] == 0 else ('strong delay' if p['coupling'] == '9/10' else 'weak delay')
        return 'phi %s, %s, %s' % (p['phi'], regime, p['condition'].replace('_', ' '))
    if cell['family'] == 'phase_boundary':
        return 'Phase-tolerance ratio ' + p['population_tolerance_ratio']
    if cell['family'] == 'brownian':
        return 'Brownian temperature ratio ' + p['temperatures'][1]
    return cell['id'].replace('cycle_', '').replace('_', ' ')


def test_target(method):
    local = method.rsplit('/', 1)[-1]
    if local in ('independent_windows_phase_adjusted', 'moving_block_phase_adjusted'):
        return 'Observed lag contrast within plug-in phase tolerance'
    if local in ('independent_windows', 'moving_block'):
        return 'Zero observed lag contrast'
    if local == 'imaginary_coherency':
        return 'Zero observed imaginary coherency'
    if local == 'phase_ignored':
        return 'Zero observed two-channel contrast'
    return 'Source reversibility under stated assumptions'


def assumption_variant(method):
    """Find an assumption variant through declared method wrappers."""
    while True:
        if method.startswith('numerical/'):
            method = method.split('/', 2)[-1]
        elif method.startswith(('fractional_', 'recorder_')):
            prefix = 'fractional_' if method.startswith('fractional_') else 'recorder_'
            _, method = split_method_setting(method, prefix)
        elif method.startswith('gain_'):
            _, method = split_method_setting(method, 'gain_')
        else:
            return method.startswith(('underestimated_', 'two_sided_shape_', 'shape_'))


def supplement_tables(registry, summaries, pairs, output, fixture):
    cells = {c['id']: c for c in registry['cells']}
    groups = defaultdict(list)
    for row in summaries:
        method, cell = row['method'], cells[row['cell']]
        groups['complete'].append(row)
        if method.startswith('numerical/'):
            groups['numerical'].append(row)
        if method.startswith(('fractional_', 'recorder_')):
            groups['recorders'].append(row)
        violation = cell['family'] == 'source' and cell['parameters']['condition'] != 'gaussian'
        if violation or 'under_' in cell['target'] or assumption_variant(method):
            groups['model_boundaries'].append(row)
        if cell['family'] == 'phase_boundary':
            groups['near_boundary'].append(row)
    for name, values in groups.items():
        write_csv(output / (name + '_counts.csv'), values)
        if name == 'complete':
            continue
        rendered = []
        for r in values:
            c = cells[r['cell']]
            rendered.append([textwrap.fill(condition_label(c), 28), c['N'],
                             textwrap.fill(label_method(r['method']), 34),
                             '%s/%s/%s' % (r['reject'], r['nonreject'], r['abstain']),
                             '[%.3f, %.3f]' % (float(r['rejection_lower']), float(r['rejection_upper'])),
                             textwrap.fill(test_target(r['method']), 22)])
        table_pdf(output / (name + '_counts.pdf'), name.replace('_', ' ').capitalize(),
                  ['Condition', 'N', 'Method', 'R / NR / A', '95% rejection\ninterval', 'Nominal test target'],
                  rendered, fixture, 'R, rejection; NR, nonrejection; A, abstention. Intervals use all records. '
                  'The nominal target does not give a size guarantee. Violated source, covariance, response, or error assumptions remove that guarantee. '
                  'The baseline comparators have no finite-record source-null guarantee here.')
    # Select declared comparisons by names alone. Retain all exact paired rows separately.
    selected = []
    for row in pairs:
        a, b = row['first'], row['second']
        targets = ('gain_', 'inflation_', 'numerical/', 'fractional_', 'recorder_')
        if any(m.startswith(targets) for m in (a, b)):
            for variant, reference in ((a, b), (b, a)):
                local = variant.rsplit('/', 1)[-1]
                match = ((variant.startswith('gain_') and
                          reference == ('common' if local == 'common' else 'diagonal'))
                         or (variant.startswith('inflation_') and reference == 'diagonal')
                         or (variant.startswith('numerical/') and variant.split('/', 2)[-1] == reference)
                         or (variant.startswith(('fractional_', 'recorder_')) and variant.split('/', 1)[-1] == reference))
                if match:
                    reverse = variant == b
                    selected.append(dict(row, variant=variant, reference=reference,
                        variant_difference=-float(row['difference']) if reverse else float(row['difference']),
                        variant_lower=-float(row['upper']) if reverse else float(row['lower']),
                        variant_upper=-float(row['lower']) if reverse else float(row['upper']))); break
    write_csv(output / 'declared_paired_controls.csv', selected)
    paired_displays(registry, selected, output, fixture)
    return {name: len(values) for name, values in groups.items()} | {'selected_pairs': len(selected)}


def paired_displays(registry, selected, output, fixture):
    cells = {c['id']: c for c in registry['cells']}
    with PdfPages(output / 'paired_control_changes.pdf') as pdf:
        for start in range(0, len(selected), 18):
            values = selected[start:start + 18]
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            for i, row in enumerate(values):
                difference = row['variant_difference']
                ax.errorbar(difference, i,
                    xerr=[[difference - row['variant_lower']], [row['variant_upper'] - difference]],
                    marker='o', color=COLORS[i % len(COLORS)], markersize=4, capsize=2)
            labels = [textwrap.fill('%s, N%s, %s minus %s' %
                (condition_label(cells[r['cell']]), cells[r['cell']]['N'],
                 label_method(r['variant']), label_method(r['reference'])), 75) for r in values]
            ax.set_yticks(range(len(values)), labels, fontsize=7)
            ax.invert_yaxis(); ax.axvline(0, color='black', linewidth=.7)
            ax.set_xlim(-1.02, 1.02); ax.set_xlabel('Paired difference in unconditional rejection fraction')
            ax.set_title('Paired decision changes with conservative 95% intervals')
            ax.grid(axis='x', alpha=.18)
            fig.tight_layout(rect=(0, 0, 1, .95 if fixture else 1))
            if fixture:
                fig.text(.5, .99, 'DETERMINISTIC DISPLAY FIXTURE. NO SCIENTIFIC RESULTS.',
                         ha='center', color='#990000', fontsize=9)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    gains = registry['cycle']['gains']
    lookup = {(r['cell'], r['variant'], r['reference']): r for r in selected}
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey=True)
    for row_index, regime in enumerate(('null', 'rotating')):
        cell = 'cycle_identity_' + regime
        for col, (method, reference) in enumerate((('common', 'common'),
                                                 ('diagonal', 'diagonal'),
                                                 ('inverse_transport', 'diagonal'))):
            ax = axes[row_index, col]
            for i, gain in enumerate(gains):
                r = lookup[cell, 'gain_' + gain + '/' + method, reference]
                value = r['variant_difference']
                ax.errorbar(value, i, xerr=[[value-r['variant_lower']], [r['variant_upper']-value]],
                            color=COLORS[col], marker='o', markersize=3, capsize=2)
            ax.axvline(0, color='black', linewidth=.7); ax.grid(axis='x', alpha=.18)
            ax.set_yticks(range(len(gains)), gains)
            ax.set_title(('Equilibrium' if regime == 'null' else 'Rotation') + ', ' +
                         label_method(method))
            ax.set_xlim(-1.02, 1.02)
            if row_index == 1:
                ax.set_xlabel('Paired rejection difference')
    axes[0, 0].set_ylabel('Gain'); axes[1, 0].set_ylabel('Gain')
    fig.tight_layout()
    save_figure(fig, output, 'paired_gain_controls', fixture)


def cycle_component_summaries(path, analysis, registry, summaries):
    cells = {c['id']: c for c in registry['cells']}
    expected = {(r['cell'], int(r['replicate']), r['method']): r['status']
                for r in read_csv(analysis / 'decisions.csv')}
    components = ('tail_bias', 'centering', 'stochastic_sqrt', 'stochastic_linear', 'recording', 'total_display')
    values = defaultdict(lambda: defaultdict(list))
    seen = set()
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            key = row['cell'], row['replicate'], row['method']
            if key not in expected or key in seen or row['status'] != expected[key]:
                raise ValueError('An extracted identity or status differs from the analysis.')
            seen.add(key)
            if cells[row['cell']]['family'] != 'cycle':
                continue
            decomposition = (row.get('cycle') or {}).get('radius_decomposition', {})
            if not decomposition.get('supported'):
                continue
            edges = decomposition.get('edges', [])
            if len(edges) != 3 or {tuple(e['edge']) for e in edges} != {(0, 1), (1, 2), (2, 0)}:
                raise ValueError('A cycle component description omits an edge.')
            for edge in edges:
                fields = dict(edge['components'], total_display=edge['total_display'])
                if fields.keys() != set(components):
                    raise ValueError('A cycle component description omits a term.')
                numbers = {name: finite(value) for name, value in fields.items()}
                if any(value is None or value < 0 for value in numbers.values()):
                    raise ValueError('A cycle radius component is missing or negative.')
                if not math.isclose(sum(numbers[name] for name in components[:-1]),
                                    numbers['total_display'], rel_tol=1e-12, abs_tol=1e-13):
                    raise ValueError('A cycle component total differs from its terms.')
                group = row['cell'], row['method'], tuple(edge['edge'])
                for name, value in numbers.items():
                    values[group][name].append(value)
    if seen != expected.keys():
        raise ValueError('The extracted rows omit an analysis decision.')
    output = []
    for row in summaries:
        if cells[row['cell']]['family'] != 'cycle':
            continue
        for edge in ((0, 1), (1, 2), (2, 0)):
            for name in components:
                data = values[row['cell'], row['method'], edge][name]
                result = dict(cell=row['cell'], method=row['method'], edge='%s-%s' % edge,
                              component=name, n=int(row['n']), finite_n=len(data),
                              missing_n=int(row['n']) - len(data),
                              scope='Descriptive Decimal50 formula values. No saved component enclosures.')
                result.update(dict(zip(QUANTILES, map(float, np.quantile(data, [0, .25, .5, .75, 1]))))
                              if data else {q: None for q in QUANTILES})
                output.append(result)
    return output


def cycle_component_table(registry, numeric, output, fixture):
    write_csv(output / 'complete_cycle_radius_components.csv', numeric)
    cells = {c['id']: c for c in registry['cells']}
    selected = [r for r in numeric if r['method'] in ('common', 'diagonal', 'combined_frequencies')]
    groups = defaultdict(dict)
    for row in selected:
        groups[row['cell'], row['method'], row['edge']][row['component']] = row
    rendered = []
    for (cid, method, edge), terms in groups.items():
        count = terms['total_display']
        values = ['%.4g' % terms[name]['median'] if terms[name]['finite_n'] else 'Missing'
                  for name in ('tail_bias', 'centering', 'stochastic_sqrt',
                               'stochastic_linear', 'recording', 'total_display')]
        rendered.append([textwrap.fill(condition_label(cells[cid]), 20), cells[cid]['N'],
            textwrap.fill(label_method(method), 14), edge, *values,
            '%s/%s' % (count['finite_n'], count['n'])])
    table_pdf(output / 'cycle_radius_components.pdf', 'Cycle radius terms',
              ['Condition', 'N', 'Method', 'Edge', 'Tail bias', 'Centering', 'Square-root\nterm',
               'Linear\nterm', 'Recorder', 'Total', 'Finite / all'],
              rendered, fixture,
              'Each value is the median of available descriptive formula values. These are not saved component enclosures. '
              'Complete term quantiles and missing-value counts are in the accompanying table.')


def checked_inputs(registry_path, analysis, presentation, diagnostics, fixture):
    manifest = json.loads((presentation / 'PRESENTATION.json').read_text())
    extraction = json.loads((diagnostics / 'EXTRACTION_COMPLETE.json').read_text())
    registry = json.loads(registry_path.read_text())
    metadata = json.loads((analysis / 'ANALYSIS.json').read_text())
    expected = sum(c['replicates'] for c in registry['cells'])
    if fixture != (registry.get('status') == 'DETERMINISTIC_FIXTURE'):
        raise ValueError('Fixture mode and registry identity differ.')
    if (extraction['status'] != 'COMPLETE' or extraction['fixture'] != fixture
            or extraction['records'] != expected or extraction['methods'] != metadata['decisions']):
        raise ValueError('Complete matching extraction is required.')
    if manifest['records'] != expected or manifest['cells'] != len(registry['cells']):
        raise ValueError('Complete validated presentation is required.')
    bound = {Path(r['path']).resolve(): r['sha256'] for r in manifest['inputs']}
    required = [registry_path, analysis / 'ANALYSIS.json', *[analysis / name for name in ANALYSIS_FILES]]
    for p in required:
        if bound.get(p.resolve()) != digest(p):
            raise ValueError('An analysis input differs from the validated presentation.')
    if (metadata['registry_sha256'] != digest(registry_path) or metadata['records'] != expected
            or metadata['cells'] != len(registry['cells']) or metadata['failures'] != 0):
        raise ValueError('Analysis identities differ.')
    outputs = {}
    for entry in extraction['output_files']:
        p = diagnostics / entry['path']
        if (entry['path'] in outputs or Path(entry['path']).name != entry['path']
                or digest(p) != entry['sha256']):
            raise ValueError('An extracted diagnostic output differs.')
        outputs[entry['path']] = entry['sha256']
    if not set(DIAGNOSTIC_FILES) <= outputs.keys():
        raise ValueError('The extraction marker omits a required output declaration.')
    extracted_inputs = json.loads((diagnostics / 'INPUT_BINDING.json').read_text())
    for path in required:
        matches = [r for r in extracted_inputs if r.get('path') == str(path.resolve())]
        if len(matches) != 1 or matches[0].get('sha256') != digest(path):
            raise ValueError('The extraction input binding differs from the analysis.')
    rows = read_csv(analysis / 'cell_summary.csv')
    source = read_csv(analysis / 'source_summary.csv')
    validation = validate_complete(registry, analysis, rows, source, metadata)
    if manifest.get('summary_validation') != validation:
        raise ValueError('The presentation validation counts differ.')
    return registry, metadata, rows, source, validation


def render(registry_path, analysis, presentation, diagnostics, output, fixture=False):
    if output.exists():
        raise FileExistsError(output)
    registry, metadata, rows, source, validation = checked_inputs(
        registry_path, analysis, presentation, diagnostics, fixture)
    numeric = checked_quantiles(analysis / 'decisions.csv', rows, DECISION_METRICS, True)
    numeric += checked_quantiles(analysis / 'source_sets.csv', source, SOURCE_METRICS, False)
    failures = failure_counts(diagnostics / 'first_failure_counts.json', rows, registry)
    cycle = cycle_component_summaries(diagnostics / 'diagnostics.jsonl.gz', analysis, registry, rows)
    output.mkdir()
    write_csv(output / 'complete_numeric_summaries.csv', numeric)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
    source_diagnostics(registry, numeric, output, fixture)
    source_confidence_weak(registry, source, output, fixture)
    failure_figure(registry, failures, output, fixture)
    table_counts = supplement_tables(registry, rows, read_csv(analysis / 'paired_comparisons.csv'), output, fixture)
    cycle_component_table(registry, cycle, output, fixture)
    inputs = [registry_path, presentation / 'PRESENTATION.json',
              diagnostics / 'EXTRACTION_COMPLETE.json', *[diagnostics / name for name in DIAGNOSTIC_FILES],
              analysis / 'ANALYSIS.json', *[analysis / name for name in ANALYSIS_FILES], Path(__file__),
              Path(__file__).resolve().parents[1] / 'summary_display/present_study.py',
              Path(__file__).resolve().parents[1] / 'summary_display/validate_summaries.py']
    receipt = dict(scope='Complete descriptive presentation. No inference or observation generation.',
                   fixture=fixture, records=metadata['records'], tables=table_counts,
                   summary_validation=validation, numeric_rows=len(numeric), cycle_component_rows=len(cycle),
                   inputs=[dict(path=str(p.resolve()), sha256=digest(p)) for p in inputs],
                   outputs=[dict(path=p.name, sha256=digest(p)) for p in sorted(output.iterdir())])
    (output / 'DISPLAY_COMPLETE.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('registry', 'analysis', 'presentation', 'diagnostics', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    render(args.registry, args.analysis, args.presentation, args.diagnostics, args.output, args.fixture)


if __name__ == '__main__':
    main()
