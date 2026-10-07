"""Independent deterministic audit of the revised robust reflection API.

Run from the repository root with .venv/bin/python. No random draws or
primary data are used. High-precision references treat every input as its
exact rational value, including Decimal and binary64 inputs.
"""
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path('analysis/student')))
from robust_reflection import robust_reflection


mp.mp.dps = 650


def exact_mp(value):
    q = Fraction(value)
    return mp.mpf(q.numerator) / q.denominator


def true_threshold(n, p):
    omega = exact_mp(p['bandwidth'])
    epsilon = exact_mp(p['skew_error'])
    vi = exact_mp(p['source_variance_i'])
    vj = exact_mp(p['source_variance_j'])
    bound = exact_mp(p['projection_variance_bound'])
    xi = exact_mp(p['covariance_discrepancy'])
    alpha = exact_mp(p['alpha'])
    ratio = mp.log(4 / alpha) / (n - 1)
    return (2 * mp.sqrt(vi * vj) * min(1, omega * epsilon) + 2 * xi
            + bound * (mp.sqrt(ratio) + ratio))


def exact_centered_covariance(data, u, v):
    # Deliberately use the centered formula, not the implementation's
    # cross-product identity.
    a = [sum((Fraction(float(row[k])) * sign for k, sign in u), Fraction())
         for row in data]
    b = [sum((Fraction(float(row[k])) * sign for k, sign in v), Fraction())
         for row in data]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (len(a) - 1)


def call(data, delta, p):
    return robust_reflection(data, 2, 3, 0, 1, delta, **p)


z = [-2., -1., 0., 1., 2.]
positive = np.zeros((5, 6))
positive[:, 4] = z
positive[:, 3] = z
negative = np.zeros((5, 6))
negative[:, 2] = z
negative[:, 5] = [-x for x in z]
base = dict(bandwidth=2., skew_error=.1, source_variance_i=1.,
            source_variance_j=1., projection_variance_bound=4.,
            covariance_discrepancy=.01, alpha=.05)
cases = [
    ('binary64', base),
    ('exact_decimal_0.3', dict(base, bandwidth=Decimal(1),
         skew_error=Decimal('0.3'), projection_variance_bound=Decimal(0),
         covariance_discrepancy=Decimal(0))),
    ('subnormal_decimal_bound', dict(base, bandwidth=Decimal(1),
         skew_error=Decimal(1), source_variance_i=Decimal('1e-400'),
         source_variance_j=Decimal('1e-400'),
         projection_variance_bound=Decimal(0),
         covariance_discrepancy=Decimal(0))),
    ('fraction_alpha', dict(base, alpha=Fraction(1, 17),
         skew_error=Fraction(1, 7), bandwidth=Fraction(3, 5),
         source_variance_i=Fraction(5, 7),
         source_variance_j=Fraction(3, 8),
         projection_variance_bound=Fraction(43, 11),
         covariance_discrepancy=Fraction(1, 31))),
    ('opposing_scales', dict(base, bandwidth=Decimal('1e200'),
         skew_error=Decimal('1e200'),
         source_variance_i=Decimal('1e200'),
         source_variance_j=Decimal('1e-200'),
         projection_variance_bound=Decimal('1e200'),
         covariance_discrepancy=Decimal('1e-200'))),
    ('tiny_alpha', dict(base, alpha=Decimal('1e-400'),
         projection_variance_bound=Fraction(2, 3))),
    ('zero_bounds', dict(base, bandwidth=0, skew_error=0,
         source_variance_i=0, source_variance_j=0,
         projection_variance_bound=0,
         covariance_discrepancy=0, alpha=Fraction(1, 2))),
]

checks = []
for label, p in cases:
    out = call(positive, 1, p)
    ref = true_threshold(len(positive), p)
    upper = exact_mp(out.threshold_upper)
    assert upper >= ref, (label, out.threshold_upper, mp.nstr(ref, 80))
    assert out.sample_covariance == exact_centered_covariance(
        positive, ((4, 1), (2, 1)), ((3, 1), (1, -1)))
    assert out.reject == (abs(out.sample_covariance) > Fraction(out.threshold_upper))
    checks.append((label, str(out.threshold_upper), out.reject))

# These are exact nonbinary rational contracts, not binary64 approximations.
assert Fraction(call(positive, 1, cases[1][1]).threshold_upper) >= Fraction(3, 5)
assert Fraction(call(positive, 1, cases[2][1]).threshold_upper) >= Fraction(2, 10**400)

small = dict(base, bandwidth=0, skew_error=0, source_variance_i=0,
             source_variance_j=0, projection_variance_bound=Decimal('.01'),
             covariance_discrepancy=0)
neg = call(negative, -1, small)
assert neg.sample_covariance == exact_centered_covariance(
    negative, ((2, 1), (0, 1)), ((5, 1), (3, -1)))
assert neg.sample_covariance < 0 and neg.reject
pos = call(positive, 1, small)
assert pos.sample_covariance > 0 and pos.reject
large = call(positive, 1, dict(small, projection_variance_bound=Decimal(100)))
assert not large.reject

constant = np.ones((4, 6))
degenerate = call(constant, 1, dict(small, projection_variance_bound=0))
assert degenerate.sample_covariance == 0 and not degenerate.reject

# Minimum supported n and an exact covariance exactly at the population
# tolerance. Conservative enclosure plus a strict comparison must abstain.
boundary = np.zeros((2, 6))
boundary[:, 4] = [-1, 1]
boundary[:, 3] = [-1, 1]
boundary_bounds = dict(base, bandwidth=1, skew_error=1,
                       source_variance_i=1, source_variance_j=1,
                       projection_variance_bound=0,
                       covariance_discrepancy=0)
at_boundary = call(boundary, 1, boundary_bounds)
assert at_boundary.sample_covariance == 2 and not at_boundary.reject
boundary[:, 3] *= 2
above_boundary = call(boundary, 1, boundary_bounds)
assert above_boundary.sample_covariance == 4 and above_boundary.reject

invalid = []
def must_fail(label, data, delta, p):
    try:
        call(data, delta, p)
    except (ValueError, TypeError, OverflowError):
        invalid.append(label)
    else:
        raise AssertionError(label + ' accepted')

must_fail('n1', positive[:1], 1, base)
must_fail('wrong_shape', positive[:, :5], 1, base)
must_fail('nan_record', np.full((5, 6), np.nan), 1, base)
must_fail('nonfinite_record', np.full((5, 6), np.inf), 1, base)
must_fail('out_of_window', positive, 2, base)
for key, value in [('bandwidth', -1), ('skew_error', float('inf')),
                   ('source_variance_i', float('nan')),
                   ('source_variance_j', Decimal('-1e-400')),
                   ('projection_variance_bound', -1),
                   ('covariance_discrepancy', -1),
                   ('alpha', 0), ('alpha', 1), ('alpha', Decimal('NaN'))]:
    must_fail(key + '_' + str(value), positive, 1, dict(base, **{key: value}))

print('REVISED_ROBUST_API_PROBES_PASS')
print('thresholds:', checks)
print('decision signs:', pos.reject, neg.reject, large.reject, degenerate.reject)
print('n2 boundary decisions:', at_boundary.reject, above_boundary.reject)
print('invalid count:', len(invalid))
