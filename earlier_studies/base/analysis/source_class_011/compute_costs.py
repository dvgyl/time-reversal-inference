"""Add supplied-marginal controls to all fixed010 AR design rows."""
import os
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from envelope import ar1_pair_envelope

LENGTHS = (128, 512, 2048, 8192, 32768, 131072, 524288, 2097152,
           8388608, 33554432, 134217728, 536870912, 2147483648)
OLD_METHODS = ('joint', 'pair_coarse', 'separate', 'exact')
NEW_METHODS = ('marginal_joint', 'marginal_exact', 'prewhitened_joint', 'prewhitened_exact')


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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parent
    binding = json.loads((here/'SAVED_INPUT_BINDING.json').read_text())
    if binding.get('scope') != 'portable_saved_input_integrity':
        raise ValueError('The saved-input binding has the wrong scope.')
    for relative, expected in binding['files'].items():
        if digest(root/relative) != expected:
            raise ValueError('A saved scientific input changed: '+relative)
    calibration_path = root/'relative_calibration_009/cost_results_r2/calibration.csv'
    old_costs_path = root/'trace_calibration_010/cost_results/ar_costs.csv'
    old_grid_path = root/'trace_calibration_010/cost_results/ar_testing_grid.csv'
    calibration_rows = read_csv(calibration_path)
    if len(calibration_rows) != 120:
        raise ValueError('The calibration table must contain 120 cases.')
    calibration = {int(x['calibration_case']): x for x in calibration_rows}
    if set(calibration) != set(range(120)):
        raise ValueError('The calibration case keys are incomplete.')
    old_costs = read_csv(old_costs_path)
    old_grid = read_csv(old_grid_path)
    expected_cases = {(case, phi) for case in range(120) for phi in (0, 0.5, 0.9)}
    if (len(old_costs) != 360 or len(old_grid) != 4680
            or {(int(x['calibration_case']), float(x['phi'])) for x in old_costs} != expected_cases):
        raise ValueError('The saved010 design is incomplete.')
    expected_grid = {(*key, length) for key in expected_cases for length in LENGTHS}
    if {(int(x['calibration_case']), float(x['phi']), int(x['record_length'])) for x in old_grid} != expected_grid:
        raise ValueError('The saved010 length keys are incomplete.')
    controls, costs, details = {}, [], []
    for old in old_costs:
        case, phi = int(old['calibration_case']), float(old['phi'])
        cal = calibration[case]
        first, second = (json.loads(cal[x]) for x in ('first_coefficients', 'second_coefficients'))
        if any(a < 0 for h in (first, second) for a in h):
            raise ValueError('The declared coefficients must be nonnegative.')
        v1, v2, signal = moments(first, second, phi)
        for name, value in (('first_variance', v1), ('second_variance', v2), ('signal', signal)):
            if not math.isclose(value, float(old[name]), rel_tol=1e-12, abs_tol=1e-12):
                raise ArithmeticError('The copied010 marginal moments differ.')
        e = float(cal['l2_error_upper'])
        broad = float(old['rho_upper'])
        exact = ar1_pair_envelope((first, second), (0, 0), phi)
        planned = ar1_pair_envelope((first, second), (e, e), phi,
                                    broad_rho=broad, planning=True)
        rho_exact, rho_plus = exact['rho'], planned['rho']
        if rho_exact > rho_plus+1e-10 or rho_plus > broad+1e-10:
            raise ArithmeticError('The envelope ordering fails.')
        broad_exact = 2*float(old['spectral_ratio'])*float(cal['exact_shape'])
        if rho_exact > broad_exact+1e-10:
            raise ArithmeticError('The exact class ordering fails.')
        config = {'marginal_joint': (rho_plus, float(cal['phase_upper'])),
                  'marginal_exact': (rho_exact, float(cal['exact_phase'])),
                  'prewhitened_joint': (2*float(cal['shape_upper']), float(cal['phase_upper'])),
                  'prewhitened_exact': (2*float(cal['exact_shape']), float(cal['exact_phase']))}
        white_moments = moments(first, second, 0)
        if not math.isclose(white_moments[2], float(cal['signal']), rel_tol=1e-12, abs_tol=1e-12):
            raise ArithmeticError('The prewhitened contrast differs from fixed009.')
        controls[(case, phi)] = dict(config=config, white_moments=white_moments,
                                    passing={name: [] for name in NEW_METHODS})
        row = dict(old)
        for name in NEW_METHODS:
            row[name+'_rho'] = config[name][0]
            source_moments = white_moments if name.startswith('prewhitened') else (v1, v2, signal)
            row[name+'_population_gap'] = source_moments[2]-2*config[name][1]*max(source_moments[:2])
            row[name+'_first_sufficient_grid_length'] = ''
        row['prewhitened_first_variance'], row['prewhitened_second_variance'], row['prewhitened_signal'] = white_moments
        costs.append(row)
        detail = dict(calibration_case=case, phi=phi,
                      common_response=cal['common_response'], delta=cal['delta'],
                      length=cal['length'], calibration_noise=cal['calibration_noise'],
                      impulse_blocks=cal['impulse_blocks'], l2_error_upper=e,
                      gmax=(1+phi)/(1-phi), spectral_ratio=old['spectral_ratio'],
                      broad_joint_rho=broad, broad_exact_rho=broad_exact,
                      marginal_exact_rho=rho_exact,
                      marginal_planned_rho_uncapped=planned['marginal_rho_uncapped'],
                      marginal_planned_rho=rho_plus,
                      broad_cap_used=planned['broad_cap_used'],
                      broad_only_finite_guarantee=planned['broad_only_finite_guarantee'])
        for i, channel in enumerate(planned['channels'], 1):
            for name in ('variance', 'eigenvalue_upper', 'lower_norm', 'numerator_upper', 'ratio_upper', 'status'):
                detail[f'channel_{i}_{name}'] = channel[name]
            detail[f'channel_{i}_exact_ratio'] = exact['channels'][i-1]['ratio_upper']
        details.append(detail)
    grid = []
    for old in old_grid:
        key = int(old['calibration_case']), float(old['phi'])
        control = controls[key]
        row = dict(old)
        length = int(old['record_length'])
        row['prewhitened_record_length'] = length-1
        for name, (rho, phase) in control['config'].items():
            if name.startswith('prewhitened'):
                first_variance, second_variance, signal = control['white_moments']
                value, ceiling = margin(length-1, rho, signal, phase, first_variance, second_variance)
            else:
                value, ceiling = margin(length, rho, float(old['signal']), phase,
                                         float(old['first_variance']), float(old['second_variance']))
            passes = value is not None and value > 0
            row[name+'_rho'] = rho
            row[name+'_margin'] = value
            row[name+'_variance_ceiling'] = ceiling
            row[name+'_scale_abstention'] = value is None
            row[name+'_sufficient'] = passes
            if passes:
                control['passing'][name].append(length)
            if name.startswith('marginal'):
                baseline = 'joint' if name == 'marginal_joint' else 'exact'
                previous = float(old[baseline+'_margin']) if old[baseline+'_margin'] else None
                if previous is not None and (value is None or value < previous-1e-10):
                    raise ArithmeticError('The supplied-marginal margin became worse.')
        grid.append(row)
    for row in costs:
        control = controls[(int(row['calibration_case']), float(row['phi']))]
        for name, passing in control['passing'].items():
            row[name+'_first_sufficient_grid_length'] = min(passing) if passing else ''
    for before, after in zip(old_costs, costs):
        if any(after[k] != v for k, v in before.items()):
            raise ArithmeticError('An original cost field changed.')
    for before, after in zip(old_grid, grid):
        if any(after[k] != v for k, v in before.items()):
            raise ArithmeticError('An original length field changed.')
    prewhitened_rows = {}
    for row in grid:
        key = int(row['calibration_case']), int(row['record_length'])
        values = tuple(row[name+'_margin'] for name in ('prewhitened_joint', 'prewhitened_exact'))
        if key in prewhitened_rows and values != prewhitened_rows[key]:
            raise ArithmeticError('The common-filter margin depends on phi.')
        prewhitened_rows[key] = values
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (('envelope_details.csv', details), ('class_costs.csv', costs),
                       ('class_testing_grid.csv', grid)):
        write_csv(out/name, rows)
    def counts(phi):
        subset = [x for x in costs if float(x['phi']) == phi]
        length_subset = [x for x in grid if float(x['phi']) == phi]
        envelope_subset = [x for x in details if x['phi'] == phi]
        return dict(cases=len(subset), length_rows=len(length_subset),
                    sufficient_cases={name: sum(bool(x[name+'_first_sufficient_grid_length']) for x in subset)
                                      for name in OLD_METHODS+NEW_METHODS},
                    new_scale_abstention_rows={name: sum(x[name+'_scale_abstention'] for x in length_subset) for name in NEW_METHODS},
                    new_nonpositive_gap_cases={name: sum(x[name+'_population_gap'] <= 0 for x in subset) for name in NEW_METHODS},
                    broad_only_cases=sum(x['broad_only_finite_guarantee'] for x in envelope_subset),
                    broad_cap_cases=sum(x['broad_cap_used'] for x in envelope_subset),
                    nonpositive_lower_norm_channels=sum(x[f'channel_{i}_status'] == 'nonpositive_lower_norm'
                                                        for x in envelope_subset for i in (1, 2)))
    inputs = [here/'SAVED_INPUT_BINDING.json', here/'envelope.py',
              Path(__file__), calibration_path, old_costs_path, old_grid_path]
    summary = dict(case_rows=len(costs), length_rows=len(grid),
                   by_phi={str(phi): counts(phi) for phi in (0, 0.5, 0.9)},
                   case84=[x for x in costs if int(x['calibration_case']) == 84],
                   inputs={str(p): digest(p) for p in inputs},
                   outputs={p.name: digest(p) for p in sorted(out.glob('*.csv'))},
                   scope='Fixed deterministic inputs. The supplied AR marginal class adds information. All original010 fields remain unchanged. No random observations, quantiles or old campaigns were generated.')
    (out/'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
