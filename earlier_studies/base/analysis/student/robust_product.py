"""Gaussian product tests with independently supplied population bounds.

Records must be iid Gaussian, with contrasts selected independently of testing.
These routines do not estimate or validate timing, spectral or variance bounds.
Records convert to binary64 before projection and covariance calculations.
Those calculations are exact for the converted values. Threshold enclosures
round outward. Complex records are not accepted.
"""
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
from typing import Optional

import numpy as np

from reflection_test import projection


@dataclass(frozen=True)
class ProductDecision:
    sample_covariance: Fraction
    threshold_upper: Optional[Decimal]
    reject: bool
    applicable: bool = True


def _rational(value, name, nonnegative=True):
    try:
        result = Fraction(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(name + ' must be finite and rational') from error
    if nonnegative and result < 0:
        raise ValueError(name + ' must be nonnegative')
    return result


def _alpha(value):
    result = _rational(value, 'alpha')
    if not 0 < result < 1:
        raise ValueError('alpha must lie strictly between zero and one')
    return result


def _upper(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def _sqrt_upper(value):
    return Decimal(0) if not value else value.sqrt().next_plus()


def _log_upper(numerator, alpha):
    with localcontext() as lower:
        lower.rounding = ROUND_FLOOR
        denominator = Decimal(alpha.numerator) / Decimal(alpha.denominator)
    return (Decimal(numerator) / denominator).ln().next_plus()


def _covariance(records, left, right):
    raw = np.asarray(records)
    if np.iscomplexobj(raw) or (
            raw.dtype.kind == 'O' and any(np.iscomplexobj(value) for value in raw.flat)):
        raise ValueError('records must be real')
    data = np.asarray(raw, dtype=float)
    if (data.ndim != 2 or data.shape[1] != len(left)
            or len(left) != len(right) or not np.isfinite(data).all()):
        raise ValueError('finite records with the specified columns required')
    n = len(data)
    if n < 2:
        raise ValueError('at least two iid records required')
    pairs = []
    for row in data:
        exact = [Fraction(float(value)) for value in row]
        pairs.append((sum((a * b for a, b in zip(exact, left)), Fraction(0)),
                      sum((a * b for a, b in zip(exact, right)), Fraction(0))))
    covariance = n * sum((a * b for a, b in pairs), Fraction(0))
    covariance -= sum((a for a, _ in pairs), Fraction(0)) * sum(
        (b for _, b in pairs), Fraction(0))
    return covariance / (n * (n - 1)), n


def _product_decision(covariance, n, tolerance, variance_u, variance_v, alpha):
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_CEILING
        b, p, q = (_upper(x) for x in (tolerance, variance_u, variance_v))
        ratio = _log_upper(2, alpha) / Decimal(n - 1)
        radius = _sqrt_upper(2 * (p * q + b * b) * ratio)
        radius += (_sqrt_upper(p * q) + b) * ratio
        threshold = b + radius
    return ProductDecision(covariance, threshold,
                           abs(covariance) > Fraction(threshold))


def covariance_interval_test(pairs, *, covariance_tolerance,
                             variance_u_bound, variance_v_bound, alpha=Fraction(1, 20)):
    """Test |Cov(U,V)| <= b using supplied separate variance ceilings.

    Threshold: b + sqrt(2(PQ+b²)t/(n-1)) + (sqrt(PQ)+b)t/(n-1),
    where t=log(2/alpha). Means are estimated by centering across records.
    Bounds accept finite int, float, Fraction and Decimal values. Decimal
    exponent overflow or underflow exceptions produce no decision.
    """
    b = _rational(covariance_tolerance, 'covariance_tolerance')
    p = _rational(variance_u_bound, 'variance_u_bound')
    q = _rational(variance_v_bound, 'variance_v_bound')
    level = _alpha(alpha)
    covariance, n = _covariance(pairs, [Fraction(1), Fraction(0)],
                                [Fraction(0), Fraction(1)])
    return _product_decision(covariance, n, b, p, q, level)


def spectral_reflection(records, m, T, i, j, delta, *, bandwidth, skew_error,
                        source_variance_i, source_variance_j,
                        observed_variance_i, observed_variance_j,
                        alpha=Fraction(1, 20), h=1):
    """Test one band-limited candidate region at a retained integer anchor.

    For a finite union, reject the model only if every candidate rejects.
    This rule supplies no complete feasibility characterization.
    """
    omega = _rational(bandwidth, 'bandwidth')
    epsilon = _rational(skew_error, 'skew_error')
    vi = _rational(source_variance_i, 'source_variance_i')
    vj = _rational(source_variance_j, 'source_variance_j')
    di = _rational(observed_variance_i, 'observed_variance_i')
    dj = _rational(observed_variance_j, 'observed_variance_j')
    level = _alpha(alpha)
    left, right = projection(m, T, i, j, delta, h)
    covariance, n = _covariance(records, [Fraction(int(x)) for x in left],
                                [Fraction(int(x)) for x in right])
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_CEILING
        b = 2 * _sqrt_upper(_upper(vi * vj)) * _upper(min(1, omega * epsilon))
    return _product_decision(covariance, n, Fraction(b), 4 * di, 4 * dj, level)


def spectral_positivity(records, *, direction=(1, -1), sign=1, bandwidth,
                        skew_error, source_variance_1, source_variance_2,
                        observed_variance_1, observed_variance_2, alpha=Fraction(1, 20)):
    """Test the union of real-skew regions around zero and one at two times.

    Columns are Y1(0),Y2(0),Y1(1),Y2(1). Staggered U=w1Y1(1)+w2Y2(0),
    V=w1Y1(0)+w2Y2(1), W=U+sign*V satisfy Cov(U,W)>=-q in the null.
    H=(|w1|sqrt(D1)+|w2|sqrt(D2))²/4 bounds Var(V)/4. The structural
    product-mgf rule uses q+2sqrt((H²+(H-q)²)t/(n-1))+2Ht/(n-1),
    t=log(1/alpha). It abstains when its enclosed q exceeds enclosed H.
    """
    sign = _rational(sign, 'sign', nonnegative=False)
    if sign not in (-1, 1) or len(direction) != 2:
        raise ValueError('two direction entries and sign -1 or 1 required')
    w1, w2 = (_rational(x, 'direction', nonnegative=False) for x in direction)
    if not w1 and not w2:
        raise ValueError('direction must be nonzero')
    omega = _rational(bandwidth, 'bandwidth')
    epsilon = _rational(skew_error, 'skew_error')
    v1 = _rational(source_variance_1, 'source_variance_1')
    v2 = _rational(source_variance_2, 'source_variance_2')
    d1 = _rational(observed_variance_1, 'observed_variance_1')
    d2 = _rational(observed_variance_2, 'observed_variance_2')
    level = _alpha(alpha)
    left = [Fraction(0), w2, w1, Fraction(0)]
    other = [w1, Fraction(0), Fraction(0), w2]
    right = [u + sign * v for u, v in zip(left, other)]
    covariance, n = _covariance(records, left, right)
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_CEILING
        q = 4 * _upper(abs(w1 * w2)) * _sqrt_upper(_upper(v1 * v2))
        q *= _upper(min(2, omega * epsilon))
        length = _upper(abs(w1)) * _sqrt_upper(_upper(d1))
        length += _upper(abs(w2)) * _sqrt_upper(_upper(d2))
        h = length * length / 4
        # Treat these enclosures as exact, more permissive population bounds.
        q_exact, h_exact = Fraction(q), Fraction(h)
        if not h_exact or q_exact > h_exact:
            return ProductDecision(covariance, None, False, False)
        squared = h_exact * h_exact + (h_exact - q_exact) ** 2
        ratio = _log_upper(1, level) / Decimal(n - 1)
        threshold = q + 2 * _sqrt_upper(_upper(squared) * ratio) + 2 * h * ratio
    return ProductDecision(covariance, threshold,
                           covariance < -Fraction(threshold))
