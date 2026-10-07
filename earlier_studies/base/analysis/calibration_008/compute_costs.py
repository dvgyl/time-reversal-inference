"""Evaluate the prespecified calibration and testing cost grid."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

from scipy.stats import chi2, f


def phase_bound(length, repeats, noise, coefficient):
    df = length * (repeats - 1)
    residual_upper = noise * math.sqrt(chi2.ppf(0.995, df) / df)
    radius = math.sqrt(length / repeats) * math.sqrt(
        length * residual_upper**2 * f.ppf(0.995, length, df)
    )
    leading = 1 / math.sqrt(1 + coefficient**2)
    trailing = coefficient * leading
    denominators = (leading - radius, 1 - radius)
    if min(denominators) <= 0:
        return radius, 1.0, True
    epsilon = ((trailing + 2 * radius) / denominators[0],
               2 * radius / denominators[1])
    disk_failed = max(epsilon) >= 1
    return radius, min(1.0, sum(epsilon)), disk_failed


def margin(records, coefficient, phase):
    df = records - 1
    variance_factor = chi2.ppf(0.995, df) / chi2.ppf(0.005, df)
    tail_log = math.log(2 / 0.03)
    tolerance = 2 * phase
    rejection_cutoff = variance_factor * (
        tolerance
        + math.sqrt(2 * (16 + tolerance**2) * tail_log / df)
        + (4 + tolerance) * tail_log / df
    )
    signal = 0.4 / math.sqrt(1 + coefficient**2)
    p_alt = 2 + 2 * coefficient / (1 + coefficient**2)
    q_alt = 2
    power_log = math.log(1 / 0.02)
    alternative_radius = (
        math.sqrt(2 * (p_alt * q_alt + signal**2) * power_log / df)
        + (math.sqrt(p_alt * q_alt) + signal) * power_log / df
    )
    return signal - rejection_cutoff - alternative_radius


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    counts = [128, 512, 2048, 8192, 32768, 131072, 524288]
    rows = []
    grid = []
    cases = itertools.product([2, 5, 9], [0.05, 0.15], [0.02, 0.1, 0.5],
                              [16, 64, 256, 1024, 4096])
    for case, (length, coefficient, noise, repeats) in enumerate(cases):
        radius, phase, disk_failed = phase_bound(length, repeats, noise, coefficient)
        tested = [(n, margin(n, coefficient, phase)) for n in counts]
        known = [(n, margin(n, coefficient, coefficient)) for n in counts]
        passing = [n for n, value in tested if value > 0 and not disk_failed]
        known_passing = [n for n, value in known if value > 0]
        rows.append(dict(case=case, length=length, filter_a=coefficient,
                         calibration_noise=noise, impulse_blocks=repeats,
                         calibration_observations_per_channel=length*repeats,
                         coefficient_l1_radius_upper=radius,
                         phase_upper=phase, dominant_disk_failure=disk_failed,
                         positive_population_gap=(2*phase < 0.4/math.sqrt(1+coefficient**2)),
                         first_sufficient_grid_count=passing[0] if passing else '',
                         known_phase_first_grid_count=known_passing[0] if known_passing else '',
                         largest_grid_margin=tested[-1][1]))
        for (n, value), (_, control) in zip(tested, known):
            grid.append(dict(case=case, records=n, margin=value,
                             sufficient=(value > 0 and not disk_failed),
                             known_phase_margin=control))
    if len(rows) != 90 or len(grid) != 630:
        raise RuntimeError('The prespecified grid is incomplete.')
    for name, data in [('calibration_cost.csv', rows), ('testing_grid.csv', grid)]:
        with (out/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    source = Path(__file__).resolve().parent
    summary = dict(cases=len(rows), testing_grid_rows=len(grid),
                   sufficient_cases=sum(bool(x['first_sufficient_grid_count']) for x in rows),
                   disk_failures=sum(x['dominant_disk_failure'] for x in rows),
                   no_population_gap=sum(not x['positive_population_gap'] for x in rows),
                   interpretation='Sufficient counts on a fixed grid. No random data or observed rejection rates.',
                   inputs={name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                           for name in ['compute_costs.py']},
                   outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(out.glob('*.csv'))})
    (out/'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
