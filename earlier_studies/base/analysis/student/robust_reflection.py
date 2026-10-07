"""Reflection exclusion with supplied spectral and variance bounds.

Use iid Gaussian records and a contrast chosen independently of those records.
Bounds describe the population; this routine does not estimate or validate them.
For a finite union of candidate regions, reject the model only if every region
rejects. Multiple contrasts within a region require their own error allocation.
"""
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction

import numpy as np

from reflection_test import projection


@dataclass(frozen=True)
class ReflectionDecision:
    sample_covariance: Fraction
    threshold_upper: Decimal
    reject: bool


def _nonnegative_bound(value, name):
    try:
        value = Fraction(value)
    except (ValueError, OverflowError, TypeError) as error:
        raise ValueError(name + ' must be a finite rational number') from error
    if value < 0:
        raise ValueError(name + ' must be finite and nonnegative')
    return value


def robust_reflection(records, m, T, i, j, delta, *, bandwidth, skew_error,
                      source_variance_i, source_variance_j,
                      projection_variance_bound, alpha=0.05,
                      covariance_discrepancy=0.0, h=1):
    """Test one candidate region using a conservative threshold enclosure.

    ``delta`` is its integer anchor; ``skew_error`` bounds real relative skew
    about that anchor. ``bandwidth`` is in radians per sampling interval.
    ``projection_variance_bound`` bounds both Var(U+V) and Var(U-V).
    ``covariance_discrepancy`` optionally bounds additional cross covariance
    error at each of the two lags. Bounds and alpha accept finite integers,
    binary floats, Fractions and Decimals without rounding through binary64.
    Records are represented as binary64. No lower variance bound is needed.
    Bound arithmetic uses Decimal's default exponent limits; out-of-range
    arithmetic raises a Decimal exception and returns no decision.
    """
    u, v = projection(m, T, i, j, delta, h)
    data = np.asarray(records, dtype=float)
    if data.ndim != 2 or data.shape[1] != m * T or not np.isfinite(data).all():
        raise ValueError('finite time-major records required')
    n = len(data)
    if n < 2:
        raise ValueError('at least two iid records required')
    alpha = _nonnegative_bound(alpha, 'alpha')
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between zero and one')
    omega = _nonnegative_bound(bandwidth, 'bandwidth')
    epsilon = _nonnegative_bound(skew_error, 'skew_error')
    vi = _nonnegative_bound(source_variance_i, 'source_variance_i')
    vj = _nonnegative_bound(source_variance_j, 'source_variance_j')
    bound = _nonnegative_bound(projection_variance_bound,
                               'projection_variance_bound')
    discrepancy = _nonnegative_bound(covariance_discrepancy,
                                     'covariance_discrepancy')

    def contrast(weights):
        selected = np.flatnonzero(weights)
        return [sum((int(weights[k]) * Fraction(float(row[k]))
                     for k in selected), Fraction(0)) for row in data]

    a, b = contrast(u), contrast(v)
    covariance = (n * sum(x * y for x, y in zip(a, b)) - sum(a) * sum(b))
    covariance /= n * (n - 1)

    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_CEILING
        def upper(value):
            return Decimal(value.numerator) / Decimal(value.denominator)

        omega, epsilon, vi, vj, bound, discrepancy = (
            upper(value) for value in (omega, epsilon, vi, vj, bound, discrepancy))
        with localcontext() as lower_context:
            lower_context.rounding = ROUND_FLOOR
            alpha_lower = Decimal(alpha.numerator) / Decimal(alpha.denominator)
        # sqrt and ln are correctly rounded with half-even, independently of
        # context.rounding. One successor encloses their mathematical values.
        variance_product_root = (vi * vj).sqrt().next_plus()
        tolerance = 2 * variance_product_root * min(Decimal(1), omega * epsilon)
        tolerance += 2 * discrepancy
        log_term = (Decimal(4) / alpha_lower).ln().next_plus()
        ratio = log_term / Decimal(n - 1)
        concentration = 2 * ratio.sqrt().next_plus() + 2 * ratio
        threshold = tolerance + bound * concentration / 2
    return ReflectionDecision(covariance, threshold,
                              abs(covariance) > Fraction(threshold))
