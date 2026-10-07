"""Check the complete summary boundary against saved decisions and declared keys."""

from collections import Counter
import csv
from functools import lru_cache
from itertools import combinations
import math

import numpy as np
from scipy.stats import beta, binomtest


BASELINES = {'baseline/' + name for name in ('independent_windows',
    'independent_windows_phase_adjusted', 'moving_block', 'moving_block_phase_adjusted',
    'imaginary_coherency')}


def recorder_cell(cell):
    p = cell['parameters']
    return ((cell['family'] == 'source' and p['condition'] == 'gaussian'
             and p['phi'] == '97/100' and cell['N'] == 65537)
        or (cell['family'] == 'cycle' and cell['id'] in ('cycle_null_131072', 'cycle_rotating_131072'))
        or (cell['family'] == 'brownian' and cell['N'] == 131073)
        or (cell['family'] == 'phase_boundary' and p['population_tolerance_ratio'] == '99/100'))


def expected_keys(registry, cell):
    family = cell['family']
    numerical = set()
    source_sets = set()
    if family == 'source':
        source_sets = set(registry['source']['methods'])
        primary = source_sets | {'two_sided_adaptive_bank'}
        primary |= {m + suffix for m in ('two_sided', 'two_sided_intersection')
                    for suffix in ('_maximum_scale', '_old_operator')}
        primary |= {'two_sided_shape_' + factor for factor in registry['ablations']['underestimated_source_shape']}
        if cell['parameters']['condition'] == 'gaussian' and cell['N'] in (16385, 65537):
            numerical = {'depth_%d' % x for x in registry['ablations']['depth']}
            numerical |= {'precision_%d' % x for x in registry['ablations']['arithmetic_bits']}
    elif family == 'cycle':
        primary = {'common', 'diagonal', 'combined_frequencies'}
        if cell['parameters'].get('gain_and_inflation'):
            primary |= {'gain_' + gain + '/' + method for gain in registry['cycle']['gains']
                        for method in ('common', 'diagonal', 'inverse_transport')}
            primary |= {'inflation_K%s_B%s' % (k, b) for k in registry['cycle']['inflation']
                        for b in registry['cycle']['inflation']}
            primary |= {'underestimated_%s_%s' % (component, factor)
                        for component in ('K', 'B') for factor in registry['cycle']['underestimation']}
    else:
        primary = set(registry['physical']['methods'])
    methods = primary | BASELINES
    sets = set(source_sets)
    for variant in numerical:
        methods |= {'numerical/' + variant + '/' + m for m in source_sets - {'supplied'}}
        sets |= {variant + '/' + m for m in source_sets - {'supplied'}}
    if recorder_cell(cell):
        variants = ['fractional_%d' % x for x in registry['ablations']['fractional_bits']]
        variants += ['recorder_%d' % x for x in registry['ablations']['recorder_bits']]
        methods |= {variant + '/' + m for variant in variants for m in primary}
        sets |= {variant + '/' + m for variant in variants for m in source_sets}
    return methods, sets


@lru_cache(maxsize=None)
def cp(k, n, confidence=.95):
    tail = (1 - confidence) / 2
    return (0. if k == 0 else float(beta.ppf(tail, k, n-k+1)),
            1. if k == n else float(beta.ppf(1-tail, k+1, n-k)))


def equal_number(value, expected, name):
    actual = float(value)
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-13):
        raise ValueError('Incorrect summary value: ' + name)


def rows(path):
    with path.open(newline='') as stream:
        yield from csv.DictReader(stream)


def flag(value):
    if value not in ('True', 'False'):
        raise ValueError('A source diagnostic flag is not a saved boolean.')
    return value == 'True'


def validate_complete(registry, analysis, summary_rows, source_summary_rows, metadata):
    cells = {c['id']: c for c in registry['cells']}
    required = {c['id']: expected_keys(registry, c) for c in cells.values()}
    decisions = {(c['id'], m): np.zeros(c['replicates'], dtype=np.uint8)
                 for c in cells.values() for m in required[c['id']][0]}
    labels = {'reject': 1, 'nonreject': 2, 'abstain': 3}
    number = 0
    for row in rows(analysis / 'decisions.csv'):
        key = row['cell'], row['method']
        if key not in decisions or row['status'] not in labels:
            raise ValueError('A saved decision has an undeclared method or status.')
        c = cells[row['cell']]
        i = int(row['replicate'])
        if not 0 <= i < c['replicates'] or decisions[key][i] != 0:
            raise ValueError('A saved decision has an invalid or duplicate replicate.')
        if row['family'] != c['family'] or row['target'] != c['target']:
            raise ValueError('A saved decision has different design metadata.')
        decisions[key][i] = labels[row['status']]
        number += 1
    if number != metadata['decisions'] or any(np.any(v == 0) for v in decisions.values()):
        raise ValueError('Saved decisions do not cover every declared replicate and method.')
    seen = set()
    for row in summary_rows:
        key = row['cell'], row['method']
        if key not in decisions or key in seen:
            raise ValueError('A count summary has an undeclared or duplicate key.')
        seen.add(key)
        state = decisions[key]
        n = len(state)
        for label, code in labels.items():
            if int(row[label]) != int(np.sum(state == code)):
                raise ValueError('A count summary disagrees with the saved decisions.')
        reject = int(row['reject'])
        available = n - int(row['abstain'])
        if int(row['n']) != n or int(row['available']) != available:
            raise ValueError('A count summary has an incorrect denominator.')
        equal_number(row['rejection_fraction'], reject/n, 'rejection fraction')
        lo, hi = cp(reject, n)
        equal_number(row['rejection_lower'], lo, 'rejection lower interval')
        equal_number(row['rejection_upper'], hi, 'rejection upper interval')
        if available:
            equal_number(row['rejection_given_available'], reject/available, 'conditional fraction')
        elif row['rejection_given_available'] != '':
            raise ValueError('An all-abstention row has a conditional rejection fraction.')
    if seen != decisions.keys():
        raise ValueError('Count summaries omit declared methods.')

    source = {(c['id'], m): {} for c in cells.values() for m in required[c['id']][1]}
    source_count = 0
    for row in rows(analysis / 'source_sets.csv'):
        key = row['cell'], row['method']
        if key not in source:
            raise ValueError('A source-set row has an undeclared key.')
        i = int(row['replicate'])
        if not 0 <= i < cells[row['cell']]['replicates'] or i in source[key]:
            raise ValueError('A source-set row has an invalid or duplicate replicate.')
        if row['status'] not in ('OUTER_HULL', 'FULL_FALLBACK', 'EMPTY_CERTIFIED'):
            raise ValueError('A source-set row has an unknown status.')
        source[key][i] = (flag(row['coverage']), flag(row['finite_endpoint']), row['status'])
        source_count += 1
    if source_count != metadata['source_sets']:
        raise ValueError('The source-set metadata count is incorrect.')
    seen = set()
    for row in source_summary_rows:
        key = row['cell'], row['method']
        if key not in source or key in seen:
            raise ValueError('A source summary has an undeclared or duplicate key.')
        seen.add(key)
        values = source[key]
        n = cells[row['cell']]['replicates']
        if len(values) != n or int(row['n']) != n:
            raise ValueError('Source coverage omits a declared outcome.')
        observed = dict(covered=sum(v[0] for v in values.values()),
                        finite_endpoint=sum(v[1] for v in values.values()),
                        empty=sum(v[2] == 'EMPTY_CERTIFIED' for v in values.values()),
                        fallback=sum(v[2] == 'FULL_FALLBACK' for v in values.values()))
        for name, value in observed.items():
            if int(row[name]) != value:
                raise ValueError('Source summary disagrees with saved inclusion events.')
        equal_number(row['coverage'], observed['covered']/n, 'coverage fraction')
        equal_number(row['finite_fraction'], observed['finite_endpoint']/n, 'finite fraction')
        lo, hi = cp(observed['covered'], n)
        equal_number(row['coverage_lower'], lo, 'coverage lower interval')
        equal_number(row['coverage_upper'], hi, 'coverage upper interval')
    if seen != source.keys():
        raise ValueError('Source summaries omit a declared method.')

    required_pairs = {(c, a, b) for c, (methods, _) in required.items()
                      for a, b in combinations(sorted(methods), 2)}
    seen = set()
    for row in rows(analysis / 'paired_comparisons.csv'):
        key = row['cell'], row['first'], row['second']
        if key not in required_pairs or key in seen:
            raise ValueError('A paired summary has an undeclared or duplicate key.')
        seen.add(key)
        a = decisions[key[0], key[1]]
        b = decisions[key[0], key[2]]
        n = len(a)
        ab = int(np.sum((a == 1) & (b != 1)))
        ba = int(np.sum((b == 1) & (a != 1)))
        for name, value in (('n', n), ('first_only', ab), ('second_only', ba),
                            ('status_disagreements', int(np.sum(a != b)))):
            if int(row[name]) != value:
                raise ValueError('Paired counts disagree with saved decisions.')
        alo, ahi = cp(ab, n, .975)
        blo, bhi = cp(ba, n, .975)
        equal_number(row['difference'], (ab-ba)/n, 'paired difference')
        equal_number(row['lower'], max(-1., alo-bhi), 'paired lower interval')
        equal_number(row['upper'], min(1., ahi-blo), 'paired upper interval')
        p = float(binomtest(ab, ab+ba, .5).pvalue) if ab+ba else 1.
        equal_number(row['discordance_pvalue'], p, 'paired sign test')
    if seen != required_pairs:
        raise ValueError('Paired summaries omit a declared comparison.')
    return dict(decisions=number, source_sets=source_count, method_cells=len(decisions),
                source_method_cells=len(source), paired_comparisons=len(required_pairs))
