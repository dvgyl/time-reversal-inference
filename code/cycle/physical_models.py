"""Generate declared sampled Gaussian physical models from supplied streams."""

from fractions import Fraction as Q

import numpy as np
from scipy.linalg import cholesky
from scipy.signal import lfilter


def brownian_covariance(t1, t2):
    t1, t2 = Q(t1), Q(t2)
    total = t1+t2
    return ((t1+total/6, -total/3), (-total/3, t2+total/6))


def brownian_transition():
    slow, fast = np.exp(-.5), np.exp(-1.5)
    return np.array([[(slow+fast)/2, (fast-slow)/2],
                     [(fast-slow)/2, (slow+fast)/2]])


def brownian_source(n, t1, t2, rng):
    if type(n) is not int or n < 2 or not 1 <= Q(t1) <= 4 or not 1 <= Q(t2) <= 4:
        raise ValueError('Use n >= 2 and temperatures in the declared range [1,4].')
    covariance = np.array(brownian_covariance(t1, t2), dtype=float)
    transition = brownian_transition()
    innovation = covariance-transition @ covariance @ transition.T
    normal = rng.standard_normal((n, 2))
    initial = cholesky(covariance, lower=True) @ normal[0]
    increments = normal[1:] @ cholesky(innovation, lower=True).T
    vectors = np.array([[1., 1.], [-1., 1.]]) / np.sqrt(2.)
    rates = np.array([np.exp(-.5), np.exp(-1.5)])
    initial_modes = initial @ vectors
    forcing = increments @ vectors
    modes = np.empty((n, 2))
    modes[0] = initial_modes
    for j in range(2):
        modes[1:, j], _ = lfilter([1.], [1., -rates[j]], forcing[:, j],
                                  zi=[rates[j] * initial_modes[j]])
    return modes @ vectors.T


def brownian_record(n, t1, t2, rng, second_tap=Q(3, 100)):
    source = brownian_source(n+1, t1, t2, rng)
    return np.column_stack((source[1:, 0]+2.,
                            source[1:, 1]+float(second_tap)*source[:-1, 1]-1.))


def stationary_ar(normals, coefficient):
    values = np.asarray(normals, dtype=float)
    phi = float(coefficient)
    if not 0 <= phi < 1 or len(values) < 1:
        raise ValueError('Use 0 <= phi < 1 and a nonempty innovation array.')
    result = np.empty_like(values)
    result[0] = values[0]
    result[1:], _ = lfilter([np.sqrt(1-phi*phi)], [1., -phi], values[1:], axis=0,
                           zi=np.asarray(phi*values[0])[None, ...])
    return result


def rotational_source(n, rng, rho=Q(1, 4), correlation=Q(1, 4), circulation=True):
    if not 0 <= Q(rho) <= Q(1, 2) or not 0 <= Q(correlation) <= Q(1, 4):
        raise ValueError('Use parameters inside the declared rotational family.')
    r = float(correlation)
    normal = rng.standard_normal((n, 4))
    innovations = np.sqrt(1-r)*normal[:, :3]+np.sqrt(r)*normal[:, 3, None]
    moving = stationary_ar(innovations, rho)
    if not circulation:
        return moving
    indices = (np.arange(3)[None, :]+np.arange(n)[:, None]) % 3
    return np.take_along_axis(moving, indices, axis=1)


def filtered_rotational_record(n, rng, filters, rho=Q(1, 4), correlation=Q(1, 4), circulation=True):
    if len(filters) != 3 or any(not h for h in filters):
        raise ValueError('Supply three nonempty finite detector filters.')
    positions = [position for h in filters for position, _ in h]
    if any(type(position) is not int or position < 0 for position in positions):
        raise ValueError('Detector tap positions must be nonnegative integers.')
    prehistory = max(positions)
    source = rotational_source(n+prehistory, rng, rho, correlation, circulation)
    output = np.empty((n, 3))
    for i, taps in enumerate(filters):
        output[:, i] = sum(float(value)*source[prehistory-position:prehistory-position+n, i]
                           for position, value in taps)
    return output+np.array([2., -1., .5])


def rotational_covariance(i, j, k, rho=Q(1, 4), correlation=Q(1, 4), circulation=True):
    if k < 0:
        return rotational_covariance(j, i, -k, rho, correlation, circulation)
    row = (i+k) % 3 if circulation else i
    return Q(rho)**k * (Q(1) if row == j else Q(correlation))


def rotational_family_bounds(n, lag, filters, rho_ceiling=Q(1, 4)):
    from cycle_diagonal import DiagonalBounds
    norms = tuple(sum(abs(Q(value)) for _, value in row) for row in filters)
    if min(norms) <= 0:
        raise ValueError('Each declared filter norm must be positive.')
    distance = max(abs(a-b) for hi in filters for hj in filters for a, _ in hi for b, _ in hj)
    if lag < distance:
        raise ValueError('The declared lag is below a filter-position difference.')
    if type(rho_ceiling) is not Q or not 0 < rho_ceiling < 1:
        raise ValueError('Supply a positive rational decay ceiling below one.')
    rho, scale = rho_ceiling, Q(3, 2)
    term = 2*rho**(lag+1-distance)/(1-rho)
    term += sum(abs(k)*rho**max(abs(k)-distance, 0) for k in range(-lag, lag+1))/n
    edges = ((0, 1), (1, 2), (2, 0))
    covariance_ceiling = scale*(1+rho)/(1-rho)
    return DiagonalBounds(tuple(covariance_ceiling*g*g for g in norms),
                          tuple(scale*norms[i]*norms[j]*term for i, j in edges),
                          'Rotational OU family: rho <= %s, r <= 1/4; declared detector norms' % rho)


def phase_boundary_record(n, b, rng):
    b = Q(b)
    if not 0 <= b < 1 or n < 2:
        raise ValueError('Use 0 <= b < 1 and at least two rows.')
    normal = rng.standard_normal(n+1)
    source = np.empty(n+1)
    source[:2] = normal[:2]
    source[2:], _ = lfilter([np.sqrt(1-float(b)**2)], [1., 0., float(b)], normal[2:],
                           zi=[-float(b)*source[0], -float(b)*source[1]])
    return np.column_stack((source[1:]+2., source[:-1]-1.))


def calibration_record(rows, coefficients, sigma, rng):
    if rows < 4 or rows % 4 or len(coefficients) != 2 or sigma < 0:
        raise ValueError('Supply two coefficients, rows divisible by four, and nonnegative noise.')
    index = np.arange(rows)
    design = np.column_stack(((-1.)**index, (-1.)**(index//2)))
    return design, design @ np.asarray(coefficients, dtype=float) + sigma*rng.standard_normal(rows)
