"""Analytic sufficient record counts; no random draws or measured power.

Evaluate the proved strict inequalities at 80 and 160 decimal digits, requiring
identical integer counts and checking both that count and its predecessor.
This is high-precision numerical verification, not an interval certificate.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import mpmath as mp


def count(gap, d, e):
    if gap <= 0:
        return None, None, None
    root = ((d + mp.sqrt(d*d + 4*gap*e)) / (2*gap)) ** 2
    n = int(mp.floor(root)) + 2

    def margin(records):
        nu = records - 1
        return gap - d / mp.sqrt(nu) - e / nu

    passed = margin(n)
    previous = margin(n - 1) if n > 2 else None
    if passed <= 0 or (previous is not None and previous > 0):
        raise ArithmeticError('strict count boundary failed')
    return n, passed, previous


def evaluate(digits):
    rows = []
    with mp.workdps(digits):
        alpha = beta = mp.mpf('0.05')
        for procedure in ('reflection_AB', 'positivity'):
            for signal in ('0.1', '0.2', '0.25', '0.4'):
                r = mp.mpf(signal)
                for scenario in ('exact', 'epsilon_0.01', 'half_tolerance'):
                    divisor = 4 if procedure == 'reflection_AB' else 8
                    epsilon = (mp.mpf(0) if scenario == 'exact' else
                               mp.mpf('0.01') if scenario == 'epsilon_0.01'
                               else r / (divisor * mp.pi))
                    if procedure == 'reflection_AB':
                        tolerance = (r/2 if scenario == 'half_tolerance'
                                     else 2*min(1, mp.pi*epsilon))
                        t0, t1 = mp.log(2/alpha), mp.log(2/beta)
                        d = mp.sqrt(2*(16+tolerance**2)*t0)
                        d += mp.sqrt(2*(4+r*r)*t1)
                        e = (4+tolerance)*t0 + (2-r)*t1
                        units, x = 'per_condition_two_conditions', ''
                    else:
                        tolerance = (r/2 if scenario == 'half_tolerance'
                                     else 4*min(2, mp.pi*epsilon))
                        p = 1-r/2
                        t0, t1 = mp.log(1/alpha), mp.log(1/beta)
                        d = 2*mp.sqrt((1+(1-tolerance)**2)*t0)
                        d += 2*mp.sqrt((p*p+r*r/2)*t1)
                        e = 2*t0 + (mp.sqrt(2)*p-r)*t1
                        units, x = 'one_condition', mp.nstr(mp.mpf('0.5')+r/4, 20)
                    gap = r-tolerance
                    n, passed, previous = count(gap, d, e)
                    row = dict(procedure=procedure, signal=signal, x=x,
                               timing_scenario=scenario, epsilon=mp.nstr(epsilon, 20),
                               bandwidth='pi', alpha='0.05', beta='0.05',
                               source_variance_ceilings='1;1',
                               observed_variance_ceilings='1;1',
                               tolerance=mp.nstr(tolerance, 20),
                               gap=mp.nstr(gap, 20), sufficient_n=n or '',
                               record_count_units=units,
                               status='sufficient_bound' if n else 'no_positive_gap',
                               D=mp.nstr(d, 20), E=mp.nstr(e, 20),
                               passing_margin=mp.nstr(passed, 20) if n else '',
                               previous_margin=mp.nstr(previous, 20) if previous is not None else '')
                    rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = evaluate(160)
    if rows != evaluate(80):
        raise ArithmeticError('precision comparison changed a printed result')
    args.out.mkdir(parents=True, exist_ok=False)
    csv_path = args.out/'sample_cost.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(status='ANALYTIC_COST_PASS', rows=len(rows), new_draws=0,
                  mpmath=mp.__version__, precision_checks=[80, 160],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                  nonpositive_gap_rows=sum(r['status']=='no_positive_gap' for r in rows),
                  scope='Least integer satisfying a proved sufficient inequality; '
                        'not optimal n, measured power, or interval arithmetic certification.')
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
