"""Evaluate the planned same-sample variance cost bounds without random draws."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import chi2

import argparse
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
ROOT=args.root.resolve()
OUT=args.out.resolve()
OUT.mkdir(exist_ok=False)
alpha = beta = 0.05
eta = beta_scale = 0.01
alpha_test = alpha - eta
beta_test = beta - beta_scale
rows = []
for procedure in ('positivity', 'reflection_AB'):
    for signal in (0.1, 0.2, 0.25, 0.4):
        for fraction in (0.0, 0.25, 0.5, 1.0):
            tolerance = fraction * signal
            row = dict(procedure=procedure, signal=signal,
                       tolerance_fraction=fraction, tolerance=tolerance,
                       alpha=alpha, beta=beta, eta=eta, beta_scale=beta_scale,
                       known_ceiling_n='', estimated_ceiling_n='',
                       inflation='', previous_margin='', passing_margin='',
                       record_units='one_condition' if procedure == 'positivity'
                                    else 'per_condition_two_conditions')
            if tolerance >= signal:
                row['status'] = 'no_positive_gap'
                rows.append(row)
                continue
            if procedure == 'positivity':
                t_known = math.log(1 / alpha)
                t_power_known = math.log(1 / beta)
                t_null = math.log(1 / alpha_test)
                t_power = math.log(1 / beta_test)
                p = 1 - signal / 2
                null_root = 2 * math.sqrt((1 + (1 - tolerance)**2) * t_null)
                null_linear = 2 * t_null
                alt_root = 2 * math.sqrt((p*p + signal*signal/2) * t_power)
                alt_linear = (math.sqrt(2)*p - signal) * t_power
                known_root = 2 * math.sqrt((1 + (1 - tolerance)**2) * t_known)
                known_root += 2 * math.sqrt((p*p + signal*signal/2) * t_power_known)
                known_linear = 2*t_known + (math.sqrt(2)*p - signal)*t_power_known
                variance_power_count = 2
            else:
                t_known = math.log(2 / alpha)
                t_power_known = math.log(2 / beta)
                t_null = math.log(2 / alpha_test)
                t_power = math.log(2 / beta_test)
                null_root = math.sqrt(2 * (16 + tolerance*tolerance) * t_null)
                null_linear = (4 + tolerance) * t_null
                alt_root = math.sqrt(2 * (4 + signal*signal) * t_power)
                alt_linear = (2 - signal) * t_power
                known_root = math.sqrt(2*(16+tolerance*tolerance)*t_known)
                known_root += math.sqrt(2*(4+signal*signal)*t_power_known)
                known_linear = (4+tolerance)*t_known + (2-signal)*t_power_known
                variance_power_count = 4
            gap = signal - tolerance
            root = ((known_root + math.sqrt(known_root**2 + 4*gap*known_linear))/(2*gap))**2
            row['known_ceiling_n'] = math.floor(root) + 2
            previous_margin = None
            for start in range(2, 2000002, 10000):
                n = np.arange(start, start + 10000, dtype=np.float64)
                nu = n - 1
                lower = chi2.ppf(eta / 2, nu)
                upper = chi2.ppf(1 - beta_scale / variance_power_count, nu)
                inflation = upper / lower
                null_limit = tolerance + null_root / np.sqrt(nu) + null_linear / nu
                alt_limit = alt_root / np.sqrt(nu) + alt_linear / nu
                margin = signal - inflation * null_limit - alt_limit
                if not np.isfinite(margin).all():
                    raise ArithmeticError('Nonfinite sample-count margin.')
                passing = np.flatnonzero(margin > 0)
                if len(passing):
                    k = int(passing[0])
                    before = float(margin[k-1]) if k else previous_margin
                    if before is not None and before > 0:
                        raise ArithmeticError('The preceding count already passes.')
                    lower_error = abs(float(chi2.cdf(lower[k], nu[k])) - eta/2)
                    upper_error = abs(float(chi2.cdf(upper[k], nu[k]))
                                      - (1-beta_scale/variance_power_count))
                    if max(lower_error, upper_error) > 1e-10:
                        raise ArithmeticError('Chi-square quantile round trip failed.')
                    row.update(estimated_ceiling_n=int(n[k]),
                               inflation=float(inflation[k]),
                               previous_margin=before, passing_margin=float(margin[k]),
                               quantile_roundtrip_error=max(lower_error, upper_error),
                               status='sufficient_bound')
                    break
                previous_margin = float(margin[-1])
            else:
                row['status'] = 'search_cap_reached'
            rows.append(row)
fields = list(dict.fromkeys(key for row in rows for key in row))
with (OUT / 'sample_cost.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
report = dict(status='COST_EVALUATION_COMPLETE', rows=len(rows), new_random_draws=0,
              finite_rows=sum(row['status']=='sufficient_bound' for row in rows),
              no_gap_rows=sum(row['status']=='no_positive_gap' for row in rows),
              search_cap_rows=sum(row['status']=='search_cap_reached' for row in rows),
              scipy=scipy.__version__, numpy=np.__version__,
              method='Scan every integer from 2 to the first passing count in binary64.',
              limitation='Numerical check, not a formal interval certificate or optimal sample size.',
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              csv_sha256=hashlib.sha256((OUT/'sample_cost.csv').read_bytes()).hexdigest())
(OUT/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
for row in rows:
    if row['signal']==0.4:
        print(row['procedure'], row['tolerance_fraction'], row['known_ceiling_n'],
              row['estimated_ceiling_n'])
