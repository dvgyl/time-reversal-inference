"""Evaluate the declared connected-record power bounds."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def radius(k, n, probability_log):
    frobenius = math.sqrt((2*n+2-4/n-4/n**2)/(2*n**2))
    return 2*k*frobenius*math.sqrt(probability_log)+4*k*probability_log/n


def power_margin(length, rho, coefficient, phase):
    c = 1-rho/length-2*math.sqrt(rho*math.log(200)/length)
    if c <= 0:
        return None, None
    dplus = (1+2*math.sqrt(rho*math.log(200)/length)
             +2*rho*math.log(200)/length)/c
    n = length-1
    bias = rho*math.sqrt(2*(4*n-2))/n**2
    signal = 0.4/math.sqrt(1+coefficient**2)
    cutoff = 2*phase*dplus+radius(rho*dplus, n, math.log(2/0.03))
    alternative = radius(rho, n, math.log(1/0.02))+bias
    return signal-cutoff-alternative, dplus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with (root/'cost_results/calibration_cost.csv').open(newline='') as stream:
        calibration = list(csv.DictReader(stream))
    if len(calibration) != 90:
        raise ValueError('The complete calibration grid is required.')
    lengths = [128, 512, 2048, 8192, 32768, 131072, 524288]
    rows, grid = [], []
    for case in calibration:
        for spectral_ratio in [1, 2, 4]:
            rho = 2*int(case['length'])*spectral_ratio
            phase = float(case['phase_upper'])
            coefficient = float(case['filter_a'])
            admissible = case['dominant_disk_failure'] == 'False'
            sufficient, known_sufficient = [], []
            for length in lengths:
                value, dplus = power_margin(length, rho, coefficient, phase)
                control, _ = power_margin(length, rho, coefficient, coefficient)
                passes = admissible and value is not None and value > 0
                if passes:
                    sufficient.append(length)
                if control is not None and control > 0:
                    known_sufficient.append(length)
                grid.append(dict(calibration_case=case['case'], spectral_ratio=spectral_ratio,
                                 rho=rho, record_length=length, scale_abstention=value is None,
                                 variance_upper=dplus, margin=value, sufficient=passes,
                                 known_phase_margin=control))
            rows.append(dict(calibration_case=case['case'], length=case['length'],
                             filter_a=coefficient, calibration_noise=case['calibration_noise'],
                             impulse_blocks=case['impulse_blocks'], spectral_ratio=spectral_ratio,
                             rho=rho, phase_upper=phase,
                             first_sufficient_grid_length=sufficient[0] if sufficient else '',
                             known_phase_first_grid_length=known_sufficient[0] if known_sufficient else ''))
    if len(rows) != 270 or len(grid) != 1890:
        raise RuntimeError('The declared dependent grid is incomplete.')
    for name, data in [('dependent_cost.csv', rows), ('dependent_grid.csv', grid)]:
        with (out/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    summary = dict(cases=len(rows), grid_rows=len(grid),
                   sufficient_cases=sum(bool(x['first_sufficient_grid_length']) for x in rows),
                   scale_abstention_grid_rows=sum(x['scale_abstention'] for x in grid),
                   inputs={name: sha(root/name) for name in [
                       'compute_dependent_costs.py',
                       'cost_results/calibration_cost.csv']},
                   outputs={p.name: sha(p) for p in sorted(out.glob('*.csv'))},
                   interpretation='Sufficient power bounds for one connected recording. No random draws.')
    (out/'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
