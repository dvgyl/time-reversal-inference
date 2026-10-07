"""Compute the declared direct relative-phase calibration costs."""
import argparse
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2, f


@dataclass(frozen=True)
class ResponsePair:
    name: str
    delta: float
    first: np.ndarray
    second: np.ndarray

    @property
    def length(self):
        return len(self.first)


def response_pairs():
    common = [('identity', [1]), ('two_tap', [1, 0.5]),
              ('no_dominant_tap', [1, 1, 0.25])]
    for (name, coefficients), delta in itertools.product(common, [0, 0.025, 0.05, 0.10]):
        first = np.convolve(coefficients, [1-delta, delta]).astype(float)
        second = np.pad(np.asarray(coefficients, dtype=float), (0, 1))
        yield ResponsePair(name, delta, first/np.linalg.norm(first),
                           second/np.linalg.norm(second))


def responses(coefficients, grid_size):
    frequencies = 2*math.pi*np.arange(grid_size)/grid_size
    values = np.exp(-1j*np.outer(frequencies, np.arange(len(coefficients)))) @ coefficients
    return frequencies, values


def product_certificate(first, second, first_radius, second_radius, weights):
    product = first*np.conjugate(second)
    error = first_radius*np.abs(second)+second_radius*np.abs(first)+first_radius*second_radius
    denominator = np.abs(product)-error
    phase = np.ones(len(product))
    positive = denominator > 0
    phase[positive] = np.minimum(1, (np.abs(product.imag[positive])+error[positive])/denominator[positive])
    return float(np.max(weights*phase)), int(np.count_nonzero(~positive))


def direct_certificate(pair, l1_error, grid_size, planning):
    frequencies, first = responses(pair.first, grid_size)
    _, second = responses(pair.second, grid_size)
    spacing = math.pi/grid_size
    positions = np.arange(pair.length)
    radii = []
    for coefficients in [pair.first, pair.second]:
        derivative = float(np.dot(positions, np.abs(coefficients)))
        if planning:
            radii.append(2*l1_error+(derivative+(pair.length-1)*l1_error)*spacing)
        else:
            radii.append(l1_error+derivative*spacing)
    weights = np.minimum(1, np.abs(np.sin(frequencies))+spacing)
    return product_certificate(first, second, radii[0], radii[1], weights)


def exact_phase(delta):
    if delta == 0:
        return 0.0
    a, b = 1-delta, delta
    aa, bb = a*a+b*b, 2*a*b
    cosine = -2*bb/(4*aa+math.sqrt(16*aa*aa-12*bb*bb))
    return b*(1-cosine*cosine)/math.sqrt(aa+bb*cosine)


def calibration_error(length, repeats, noise):
    df = length*(repeats-1)
    residual = noise*math.sqrt(chi2.ppf(0.995, df)/df)
    l2 = math.sqrt(length/repeats)*residual*math.sqrt(f.ppf(0.995, length, df))
    return math.sqrt(length)*l2, l2


def separate_channel_phase(pair, error):
    bounds = []
    failed = False
    for coefficients in [pair.first, pair.second]:
        leading = abs(float(coefficients[0]))
        trailing = float(np.sum(np.abs(coefficients[1:])))
        denominator = leading-error
        epsilon = (trailing+2*error)/denominator if denominator > 0 else math.inf
        failed = failed or epsilon >= 1
        bounds.append(epsilon)
    return min(1.0, sum(bounds)), failed


def planned_shape_bound(pair, l1_error, l2_error, grid_size):
    spacing = math.pi/grid_size
    positions = np.arange(pair.length)
    bounds = []
    fallbacks = 0
    for coefficients in [pair.first, pair.second]:
        denominator = float(np.linalg.norm(coefficients))-2*l2_error
        derivative = float(np.dot(positions, np.abs(coefficients)))
        peak = float(np.sum(coefficients))
        numerator = peak+2*l1_error+(derivative+(pair.length-1)*l1_error)*spacing
        if denominator <= 0:
            bounds.append(float(pair.length))
            fallbacks += 1
        else:
            bounds.append(min(float(pair.length), (numerator/denominator)**2))
    return max(bounds), fallbacks


def cross_covariance(pair, lag):
    value = 0.0
    for i, first in enumerate(pair.first):
        for j, second in enumerate(pair.second):
            if lag-i+j-1 == 0:
                value += first*second
    return float(0.4*value)


def radius(k, n, probability_log):
    frobenius = math.sqrt((2*n+2-4/n-4/n**2)/(2*n**2))
    return 2*k*frobenius*math.sqrt(probability_log)+4*k*probability_log/n


def power_margin(length, rho, signal, phase):
    denominator = 1-rho/length-2*math.sqrt(rho*math.log(200)/length)
    if denominator <= 0:
        return None, None
    variance_upper = (1+2*math.sqrt(rho*math.log(200)/length)+2*rho*math.log(200)/length)/denominator
    n = length-1
    bias = rho*math.sqrt(2*(4*n-2))/n**2
    threshold = 2*phase*variance_upper+radius(rho*variance_upper, n, math.log(2/0.03))
    alternative = radius(rho, n, math.log(1/0.02))+bias
    return signal-threshold-alternative, variance_upper


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    grid_size = 16384
    lengths = [128, 512, 2048, 8192, 32768, 131072, 524288]
    rows, grid, calibration, convergence = [], [], [], []
    for pair in response_pairs():
        signal = abs(cross_covariance(pair, -1)-cross_covariance(pair, 1))
        known_phase = exact_phase(pair.delta)
        known_shape = max(float(np.sum(c))**2/float(np.dot(c, c)) for c in [pair.first, pair.second])
        for error, points in itertools.product([0.1, 0.03, 0.01, 0.003, 0.001, 0], [64, 256, 1024, 4096, 16384]):
            value, failures = direct_certificate(pair, error, points, False)
            convergence.append(dict(common_response=pair.name, delta=pair.delta,
                                    coefficient_l1_error=error, grid_points=points,
                                    phase_certificate=value, exact_phase=known_phase,
                                    excess=value-known_phase, failed_product_cells=failures))
        for noise, repeats in itertools.product([0.02, 0.1], [64, 256, 1024, 4096, 16384]):
            case = len(calibration)
            l1, l2 = calibration_error(pair.length, repeats, noise)
            phase, failed_cells = direct_certificate(pair, l1, grid_size, True)
            old_phase, old_failure = separate_channel_phase(pair, l1)
            shape, fallback = planned_shape_bound(pair, l1, l2, grid_size)
            calibration.append(dict(calibration_case=case, common_response=pair.name,
                                    delta=pair.delta, length=pair.length,
                                    calibration_noise=noise, impulse_blocks=repeats,
                                    calibration_observations_per_channel=pair.length*repeats,
                                    l1_error_upper=l1, l2_error_upper=l2,
                                    phase_upper=phase, exact_phase=known_phase,
                                    separate_channel_phase_upper=old_phase,
                                    separate_channel_disk_failure=old_failure,
                                    failed_planning_product_cells=failed_cells,
                                    shape_upper=shape, exact_shape=known_shape,
                                    shape_denominator_fallbacks=fallback,
                                    signal=signal, population_gap=signal-2*phase,
                                    first_coefficients=json.dumps(pair.first.tolist()),
                                    second_coefficients=json.dumps(pair.second.tolist())))
            for spectral_ratio in [1, 2, 4]:
                rho = 2*spectral_ratio*shape
                coarse_rho = 2*spectral_ratio*pair.length
                exact_rho = 2*spectral_ratio*known_shape
                passing = {name: [] for name in ['joint', 'pair_coarse', 'separate', 'exact']}
                for length in lengths:
                    joint, upper = power_margin(length, rho, signal, phase)
                    pair_coarse, _ = power_margin(length, coarse_rho, signal, phase)
                    separate, _ = power_margin(length, coarse_rho, signal, old_phase)
                    exact, _ = power_margin(length, exact_rho, signal, known_phase)
                    values = dict(joint=joint, pair_coarse=pair_coarse, separate=separate, exact=exact)
                    for name, margin in values.items():
                        if margin is not None and margin > 0 and (name != 'separate' or not old_failure):
                            passing[name].append(length)
                    grid.append(dict(calibration_case=case, spectral_ratio=spectral_ratio,
                                     record_length=length, rho_upper=rho,
                                     variance_upper=upper, scale_abstention=joint is None,
                                     joint_margin=joint, pair_coarse_margin=pair_coarse,
                                     separate_margin=separate, exact_margin=exact,
                                     sufficient=joint is not None and joint > 0))
                rows.append(dict(calibration_case=case, common_response=pair.name,
                                 delta=pair.delta, length=pair.length,
                                 calibration_noise=noise, impulse_blocks=repeats,
                                 calibration_observations_per_channel=pair.length*repeats,
                                 spectral_ratio=spectral_ratio, phase_upper=phase,
                                 separate_channel_phase_upper=old_phase,
                                 exact_phase=known_phase, signal=signal,
                                 population_gap=signal-2*phase, rho_upper=rho,
                                 coarse_rho=coarse_rho, exact_rho=exact_rho,
                                 first_sufficient_grid_length=passing['joint'][0] if passing['joint'] else '',
                                 pair_coarse_first_grid_length=passing['pair_coarse'][0] if passing['pair_coarse'] else '',
                                 separate_first_grid_length=passing['separate'][0] if passing['separate'] else '',
                                 exact_first_grid_length=passing['exact'][0] if passing['exact'] else ''))
    if (len(calibration), len(rows), len(grid), len(convergence)) != (120, 360, 2520, 360):
        raise RuntimeError('The declared grid is incomplete.')
    for name, data in [('calibration.csv', calibration), ('costs.csv', rows),
                       ('testing_grid.csv', grid), ('convergence.csv', convergence)]:
        write_csv(out/name, data)
    source = Path(__file__).resolve().parent
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    summary = dict(calibration_cases=len(calibration), cost_cases=len(rows),
                   testing_grid_rows=len(grid), convergence_rows=len(convergence),
                   joint_sufficient_cases=sum(bool(x['first_sufficient_grid_length']) for x in rows),
                   pair_coarse_sufficient_cases=sum(bool(x['pair_coarse_first_grid_length']) for x in rows),
                   separate_sufficient_cases=sum(bool(x['separate_first_grid_length']) for x in rows),
                   exact_sufficient_cases=sum(bool(x['exact_first_grid_length']) for x in rows),
                   no_population_gap_cases=sum(x['population_gap'] <= 0 for x in rows),
                   scale_abstention_rows=sum(x['scale_abstention'] for x in grid),
                   inputs={name: sha(source/name) for name in ['compute_costs.py']},
                   outputs={p.name: sha(p) for p in sorted(out.glob('*.csv'))},
                   interpretation='Conditional sufficient bounds on a fixed grid. No random observations. The floating-point evaluation is not formal interval certification.')
    (out/'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
