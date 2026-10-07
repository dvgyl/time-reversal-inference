"""Report every planned outcome under the prospective analysis rules."""
from collections import Counter
from fractions import Fraction as Q
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta


def decode(value):
    if isinstance(value, list):
        return [decode(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {'q'}:
            return Q(int(value['q'][0]), int(value['q'][1]))
        if set(value) == {'type', 'fields'}:
            return dict(_type=value['type'], **decode(value['fields']))
        if set(value) == {'enum', 'name', 'value'}:
            return value['value']
        return {key: decode(item) for key, item in value.items()}
    return value


def read(path):
    return decode(json.loads(Path(path).read_text()))


def interval(k, n, confidence=.95):
    if not 0 <= k <= n or n <= 0:
        raise ValueError('Invalid binomial counts.')
    tail = (1-confidence)/2
    return [0. if k == 0 else float(beta.ppf(tail, k, n-k+1)),
            1. if k == n else float(beta.ppf(1-tail, k+1, n-k))]


def binary_summary(values, confidence=.95):
    known = [value for value in values if value is not None]
    if any(type(value) is not bool for value in known):
        raise ValueError('An event must be Boolean or missing.')
    successes, n = sum(known), len(values)
    missing = n-len(known)
    low, high = successes, successes+missing
    return dict(planned=n, defined=len(known), successes=successes, missing=missing,
                possible_count=[low, high], possible_fraction=[low/n, high/n],
                confidence=confidence,
                interval=[interval(low, n, confidence)[0], interval(high, n, confidence)[1]])


def quantiles(values):
    defined = [value for value in values if value is not None]
    if not defined:
        return dict(defined=0, quartiles=None, minimum=None, maximum=None)
    return dict(defined=len(defined), quartiles=np.quantile(defined, [.25, .5, .75], method='linear').tolist(),
                minimum=min(defined), maximum=max(defined))


def source_details(row, result, phi, method):
    source = result['source']
    source_type = source['_type']
    row['source_type'] = source_type
    row['source_reason'] = source.get('reason')
    hull = source_type in ('SourceHull', 'SourceOuterHull')
    fallback = source_type == 'SourceFallback'
    empty = source_type == 'SourceEmpty'
    if not (hull or fallback or empty):
        raise ValueError('Unrecognized source result '+source_type)
    row['finite_endpoint'] = bool(hull and source['upper'] < 1)
    row['source_coverage'] = None if method == 'supplied' else (False if empty else source['lower'] <= phi <= source['upper'])
    row['hull_width'] = float(source['upper']-source['lower']) if hull else None
    row['upper_gap'] = float(1-source['upper']) if hull else None
    row['upper_endpoint'] = float(source['upper']) if hull else None
    members = result['members']
    available = [member for member in members if member['_type'] == 'MemberDecision']
    row['available_members'] = len(available)
    row['usable'] = len(available) > 0
    row['member_failure_reasons'] = dict(Counter(member.get('reason', 'unknown') for member in members if member['_type'] != 'MemberDecision'))
    margins, ratios, widths = [], [], []
    for member in available:
        exact_margin = member['statistic_absolute']-member['threshold']['upper']
        if member['reject'] != (exact_margin > 0):
            raise ValueError('A saved member violates strict rejection.')
        margins.append(float(exact_margin))
        ratios.append(float(member['source_ratio']))
        widths.append(float(member['threshold']['upper']-member['threshold']['lower']))
    expected = ('REJECT' if any(member['reject'] for member in available)
                else 'DO_NOT_REJECT' if available else 'ABSTAIN')
    if result['decision'] != expected:
        raise ValueError('A saved bank violates the aggregation rule.')
    row['maximum_margin'] = max(margins) if margins else None
    row['minimum_source_ratio'] = min(ratios) if ratios else None
    row['maximum_threshold_width'] = max(widths) if widths else None


def cycle_details(row, result):
    margins, epsilons, thresholds, estimates = [], [], [], []
    decisions = []
    for frequency in result['frequencies']:
        status = frequency['status']
        decisions.append(status)
        threshold = frequency.get('threshold')
        if threshold is None:
            continue
        witness = frequency['absolute_imaginary_product']
        if status == 'reject' and not witness > threshold['upper']:
            raise ValueError('A saved cycle rejection lacks its certificate.')
        if status == 'nonreject' and not witness <= threshold['lower']:
            raise ValueError('A saved cycle nonrejection lacks its certificate.')
        margins.append(float(witness-threshold['upper']))
        epsilons.append(float(frequency['epsilon']['upper']))
        thresholds.append(float(threshold['upper']))
        estimates.append(float(witness))
    expected = ('reject' if 'reject' in decisions else 'nonreject' if all(x == 'nonreject' for x in decisions) else 'abstain')
    if result['status'] != expected:
        raise ValueError('A saved cycle result violates aggregation.')
    row['maximum_margin'] = max(margins) if margins else None
    row['epsilon_upper'] = max(epsilons) if epsilons else None
    row['threshold_upper'] = max(thresholds) if thresholds else None
    row['sample_witness'] = max(estimates) if estimates else None
    row['cycle_reason'] = result.get('refusal')
    row['usable'] = result['status'] != 'abstain'


def collect(run, registry):
    rows, expected_paths = [], set()
    for experiment in ('cycle', 'source'):
        for index, cell in enumerate(registry[experiment]['cells']):
            methods = ['cycle'] if experiment == 'cycle' else registry['source']['methods']
            for replicate in range(cell['replicates']):
                job = experiment+'_'+str(index).zfill(2)+'_'+str(replicate).zfill(3)
                for n in cell['lengths']:
                    for method in methods:
                        path = run/job/('N'+str(n)+'_'+method+'.json')
                        expected_paths.add(path)
                        row = dict(experiment=experiment, cell=cell['id'], cell_index=index,
                                   replicate=replicate, n=n, method=method, status='ERROR',
                                   rejected=None, usable=None, finite_endpoint=None,
                                   source_coverage=None, maximum_margin=None, hull_width=None,
                                   upper_gap=None, minimum_source_ratio=None, maximum_threshold_width=None)
                        if not path.exists():
                            row['error'] = 'missing_planned_output'
                        else:
                            payload = read(path)
                            if payload['job']['id'] != job or payload['method'] != method or payload['n'] != n:
                                raise ValueError('A result identity differs from its planned path.')
                            if payload['execution_status'] != 'completed':
                                row['error'] = payload.get('error_type', 'execution_error')
                            else:
                                result = payload['result']
                                if experiment == 'cycle':
                                    row['status'] = {'reject': 'REJECT', 'nonreject': 'DO_NOT_REJECT', 'abstain': 'ABSTAIN'}[result['status']]
                                    cycle_details(row, result)
                                else:
                                    row['status'] = result['decision']
                                    source_details(row, result, Q(cell['phi']), method)
                                row['rejected'] = row['status'] == 'REJECT'
                        rows.append(row)
    extras = set(run.glob('*/N*_*.json'))-expected_paths
    if extras:
        raise ValueError('Unplanned inference outputs exist: '+str(sorted(extras)))
    return rows


def summarize(rows, registry):
    groups = []
    for experiment in ('cycle', 'source'):
        for cell in registry[experiment]['cells']:
            methods = ['cycle'] if experiment == 'cycle' else registry['source']['methods']
            for n in cell['lengths']:
                for method in methods:
                    chosen = [row for row in rows if row['cell'] == cell['id'] and row['n'] == n and row['method'] == method]
                    item = dict(experiment=experiment, cell=cell['id'], n=n, method=method,
                                planned=cell['replicates'], statuses=dict(Counter(row['status'] for row in chosen)),
                                rejection=binary_summary([row['rejected'] for row in chosen]),
                                usable=binary_summary([row['usable'] for row in chosen]),
                                maximum_margin=quantiles([row['maximum_margin'] for row in chosen]))
                    if experiment == 'source':
                        item.update(finite_endpoint=binary_summary([row['finite_endpoint'] for row in chosen]),
                                    source_types=dict(Counter(row.get('source_type', 'ERROR') for row in chosen)),
                                    source_reasons=dict(Counter(row.get('source_reason') or 'none' for row in chosen)),
                                    hull_width=quantiles([row['hull_width'] for row in chosen]),
                                    upper_gap=quantiles([row['upper_gap'] for row in chosen]),
                                    minimum_source_ratio=quantiles([row['minimum_source_ratio'] for row in chosen]),
                                    maximum_threshold_width=quantiles([row['maximum_threshold_width'] for row in chosen]))
                        item['source_coverage'] = None if method == 'supplied' else binary_summary([row['source_coverage'] for row in chosen])
                    groups.append(item)
    paired = []
    for cell in registry['source']['cells']:
        for n in cell['lengths']:
            by_method = {method: {row['replicate']: row for row in rows if row['cell'] == cell['id'] and row['n'] == n and row['method'] == method}
                         for method in registry['source']['methods']}
            for method in ('bridge', 'intersection', 'supplied'):
                for event in ('finite_endpoint', 'usable', 'rejected'):
                    a = [by_method[method][rep][event] for rep in range(cell['replicates'])]
                    b = [by_method['lag'][rep][event] for rep in range(cell['replicates'])]
                    sa, sb = binary_summary(a, .975), binary_summary(b, .975)
                    defined = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
                    paired.append(dict(cell=cell['id'], n=n, method=method, reference='lag', event=event,
                                       planned=len(a), defined_pairs=len(defined),
                                       first_only=sum(x and not y for x, y in defined),
                                       lag_only=sum(y and not x for x, y in defined),
                                       possible_difference=[sa['possible_fraction'][0]-sb['possible_fraction'][1], sa['possible_fraction'][1]-sb['possible_fraction'][0]],
                                       conservative_95_interval=[max(-1., sa['interval'][0]-sb['interval'][1]), min(1., sa['interval'][1]-sb['interval'][0])]))
    return dict(groups=groups, paired=paired, accounting=dict(planned_decisions=len(rows),
                statuses=dict(Counter(row['status'] for row in rows)),
                cycle_decisions=sum(row['experiment'] == 'cycle' for row in rows),
                source_decisions=sum(row['experiment'] == 'source' for row in rows)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run, output = Path(args.run).resolve(), Path(args.output).resolve()
    output.mkdir()
    registry = read(run/'EXECUTION_START.json')['registry']
    rows = collect(run, registry)
    summary = summarize(rows, registry)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (output/'rows.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with open(output/'rows.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value for key, value in row.items()})
    print(json.dumps(summary['accounting']))


if __name__ == '__main__':
    main()
