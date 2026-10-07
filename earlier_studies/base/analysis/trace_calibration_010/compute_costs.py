"""Calculate declared trace-aware costs from the fixed calibration table."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

METHODS = ('joint', 'pair_coarse', 'separate', 'exact')
BASE_LENGTHS = (128, 512, 2048, 8192, 32768, 131072, 524288)
AR_LENGTHS = BASE_LENGTHS + (2097152, 8388608, 33554432, 134217728, 536870912, 2147483648)


def radius(rho, variance_max, variance_sum, length, log_tail):
    n = length - 1
    squared_f = (2*n + 2 - 4/n - 4/n**2) / (2*n**2)
    covariance_cap = rho * variance_max
    frobenius = min(covariance_cap * math.sqrt(squared_f),
                    2 * math.sqrt(covariance_cap * length * variance_sum) / n)
    return 2*frobenius*math.sqrt(log_tail) + 4*covariance_cap*log_tail/n


def margin(length, rho, signal, phase, first_variance, second_variance):
    maximum = max(first_variance, second_variance)
    denominator = 1-rho/length-2*math.sqrt(rho*math.log(200)/length)
    if denominator <= 0:
        return None, None
    ceiling = maximum*(1+2*math.sqrt(rho*math.log(200)/length)
                       +2*rho*math.log(200)/length)/denominator
    n = length-1
    null_radius = radius(rho, ceiling, 2*ceiling, length, math.log(2/0.03))
    alternative_radius = radius(rho, maximum, first_variance+second_variance,
                                length, math.log(1/0.02))
    bias = rho*maximum*math.sqrt(2*(4*n-2))/n**2
    return signal-2*phase*ceiling-null_radius-alternative_radius-bias, ceiling


def moments(first, second, phi):
    variances = [sum(a*b*phi**abs(i-j) for i, a in enumerate(coefficients)
                     for j, b in enumerate(coefficients)) for coefficients in (first, second)]
    def cross(lag):
        return 0.4*sum(a*b*phi**abs(lag-i+j-1)
                       for i, a in enumerate(first) for j, b in enumerate(second))
    return variances[0], variances[1], abs(cross(-1)-cross(1))


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def calculate(calibration, source_ratio, lengths, phi, original):
    first = json.loads(calibration['first_coefficients'])
    second = json.loads(calibration['second_coefficients'])
    if original:
        v1, v2, signal = 1.0, 1.0, float(calibration['signal'])
    else:
        v1, v2, signal = moments(first, second, phi)
    direct = float(calibration['phase_upper'])
    separate = float(calibration['separate_channel_phase_upper'])
    exact = float(calibration['exact_phase'])
    shape = float(calibration['shape_upper'])
    exact_shape = float(calibration['exact_shape'])
    length = int(calibration['length'])
    configurations = {'joint': (2*source_ratio*shape, direct),
                      'pair_coarse': (2*source_ratio*length, direct),
                      'separate': (2*source_ratio*length, separate),
                      'exact': (2*source_ratio*exact_shape, exact)}
    case = int(calibration['calibration_case'])
    rows = []
    passing = {name: [] for name in METHODS}
    for count in lengths:
        row = dict(calibration_case=case, phi=phi, spectral_ratio=source_ratio,
                   record_length=count, first_variance=v1, second_variance=v2,
                   signal=signal)
        for name, (rho, phase) in configurations.items():
            value, ceiling = margin(count, rho, signal, phase, v1, v2)
            passes = value is not None and value > 0
            if name == 'separate' and calibration['separate_channel_disk_failure'] == 'True':
                passes = False
            row[name+'_margin'] = value
            row[name+'_variance_ceiling'] = ceiling
            row[name+'_scale_abstention'] = value is None
            row[name+'_sufficient'] = passes
            if passes:
                passing[name].append(count)
        rows.append(row)
    summary = {name: calibration[name] for name in ['common_response', 'delta', 'length',
               'calibration_noise', 'impulse_blocks', 'calibration_observations_per_channel']}
    summary.update(calibration_case=case, phi=phi, spectral_ratio=source_ratio,
                   first_variance=v1, second_variance=v2, signal=signal,
                   phase_upper=direct, population_gap=signal-2*direct*max(v1, v2),
                   rho_upper=configurations['joint'][0])
    for name in METHODS:
        summary[name+'_first_sufficient_grid_length'] = passing[name][0] if passing[name] else ''
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--original-grid', type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    old = here.parent/'relative_calibration_009/cost_results_r2'
    calibration_path = args.calibration or old/'calibration.csv'
    old_grid_path = args.original_grid or old/'testing_grid.csv'
    calibration = read_csv(calibration_path)
    if len(calibration) != 120 or {int(x['calibration_case']) for x in calibration} != set(range(120)):
        raise ValueError('The fixed calibration table must contain all 120 unique cases.')
    previous = {(int(x['calibration_case']), float(x['spectral_ratio']), int(x['record_length'])): x
                for x in read_csv(old_grid_path)}
    if len(previous) != 2520:
        raise ValueError('The original grid must contain all 2,520 unique rows.')
    original_costs, original_grid, ar_costs, ar_grid, comparisons = [], [], [], [], []
    for case in calibration:
        for ratio in (1, 2, 4):
            summary, rows = calculate(case, ratio, BASE_LENGTHS, 0.0, True)
            original_costs.append(summary)
            original_grid.extend(rows)
            for row in rows:
                before = previous[(row['calibration_case'], float(ratio), row['record_length'])]
                for name in METHODS:
                    saved = before[name+'_margin']
                    old_margin = float(saved) if saved else None
                    new_margin = row[name+'_margin']
                    if old_margin is not None and (new_margin is None or new_margin < old_margin-1e-10):
                        raise RuntimeError('The minimum radius made a fixed-design margin worse.')
                    comparisons.append(dict(calibration_case=row['calibration_case'], spectral_ratio=ratio,
                                            record_length=row['record_length'], method=name,
                                            original_margin=old_margin, trace_margin=new_margin,
                                            improvement=new_margin-old_margin if old_margin is not None else None))
        for phi in (0.0, 0.5, 0.9):
            ratio = ((1+phi)/(1-phi))**2
            summary, rows = calculate(case, ratio, AR_LENGTHS, phi, False)
            ar_costs.append(summary)
            ar_grid.extend(rows)
    if tuple(map(len, (original_costs, original_grid, ar_costs, ar_grid, comparisons))) != (360, 2520, 360, 4680, 10080):
        raise RuntimeError('The declared design is incomplete.')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    datasets = {'original_costs.csv': original_costs, 'original_testing_grid.csv': original_grid,
                'ar_costs.csv': ar_costs, 'ar_testing_grid.csv': ar_grid,
                'original_radius_comparison.csv': comparisons}
    for name, rows in datasets.items():
        write_csv(out/name, rows)
    def counts(costs, grid):
        return dict(cases=len(costs), rows=len(grid),
                    sufficient_cases={name: sum(bool(x[name+'_first_sufficient_grid_length']) for x in costs) for name in METHODS},
                    no_population_gap_cases=sum(x['population_gap'] <= 0 for x in costs),
                    joint_scale_abstention_rows=sum(x['joint_scale_abstention'] for x in grid))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {'original': counts(original_costs, original_grid), 'ar': counts(ar_costs, ar_grid),
              'ar_by_phi': {str(phi): counts([x for x in ar_costs if x['phi'] == phi],
                                           [x for x in ar_grid if x['phi'] == phi]) for phi in (0.0, 0.5, 0.9)},
              'inputs': {str(p): digest(p) for p in [Path(__file__), calibration_path, old_grid_path]},
              'outputs': {p.name: digest(p) for p in sorted(out.glob('*.csv'))},
              'scope': 'Declared analytic sufficient bounds. Fixed calibration outputs are reused. No old simulation, calibration quantile or convergence calculation was repeated. No new observations are generated by this program.'}
    (out/'SUMMARY.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
