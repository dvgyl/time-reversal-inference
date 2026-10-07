"""Independent algebraic and statistical checks of the intake proposals.

This file reads no intake executable and writes only to stdout.
"""

from math import acos, floor, pi, sqrt

import numpy as np
from scipy.stats import t


def retained_lags(T):
    return set(range(1 - T, T))


def retained_reflection_pairs(T, delta):
    lags = retained_lags(T)
    return sorted((k, 2 * delta - k) for k in lags
                  if k < 2 * delta - k and 2 * delta - k in lags)


for T in range(2, 9):
    for delta in range(-T - 1, T + 2):
        has_pair = bool(retained_reflection_pairs(T, delta))
        assert has_pair == (abs(delta) <= T - 2), (T, delta, has_pair)
print("Reflection pair equivalence verified for T=2..8 and signed delays in [-(T+1),T+1].")

# A fixed source has a reversible, white bivariate Gaussian law with
# contemporaneous cross covariance r. Choosing a common record delay at
# random from {0,1} makes the unconditional cross moments the mixture below.
r = 0.4
mixed = {-1: 0.0, 0: r / 2, 1: r / 2, 2: 0.0}
assert mixed[-1] != mixed[1] and mixed[0] != mixed[2]
print("Random per-record delay counterexample: mixed C12(-1,0,1,2) =",
      tuple(mixed[k] for k in (-1, 0, 1, 2)))


alpha = 0.05
for n in (32, 64, 128, 1024):
    nu = n - 2
    for two_sided in (False, True):
        quantile = t.ppf(1 - alpha / (2 if two_sided else 1), nu)
        correlation_cutoff = quantile / sqrt(nu + quantile**2)
        achieved = (2 if two_sided else 1) * t.sf(quantile, nu)
        rational_target = alpha**2 if two_sided else (2 * alpha)**2
        q_bound = 1 - rational_target ** (1 / nu)
        bound_stat = sqrt(nu * q_bound / (1 - q_bound))
        conservative_size = (2 if two_sided else 1) * t.sf(bound_stat, nu)
        assert abs(achieved - alpha) < 1e-9
        assert conservative_size < alpha
        print(f"Student n={n}, two_sided={two_sided}: "
              f"r_exact={correlation_cutoff:.6f}, "
              f"size={achieved:.6f}, conservative_size={conservative_size:.6f}")


# Gaussian negative log likelihood (up to scale/constant) for a scalar
# covariance x with sample variance s is log(x)+s/x.
s = 1.0
x = 3.0
curvature = (2 * s - x) / x**3
assert curvature < 0
print(f"Covariance likelihood curvature at sample variance 1, covariance 3: {curvature:.8f}")

# If covariance matrices have unit diagonal and cross covariance +/-rho,
# their precision-matrix midpoint no longer maps to a unit-diagonal
# covariance. Thus the covariance affine constraint is nonlinear in precision.
rho = 0.5
sigma_plus = np.array([[1.0, rho], [rho, 1.0]])
sigma_minus = np.array([[1.0, -rho], [-rho, 1.0]])
precision_mid = (np.linalg.inv(sigma_plus) + np.linalg.inv(sigma_minus)) / 2
sigma_from_mid = np.linalg.inv(precision_mid)
assert not np.allclose(np.diag(sigma_from_mid), np.ones(2))
print("Precision midpoint inverse diagonal:", np.diag(sigma_from_mid))


for eta in (1.0, 0.3, 0.1, 0.03, 0.01, 0.003):
    theta = acos(1 / (1 + eta))
    # Negative signed cycle: odd d has d edges; even d has d+1 edges.
    cycle_lower = next(d for d in range(2, 1000)
                       if (d if d % 2 else d + 1) * theta >= pi - 1e-12)
    sufficient_even = 2 * (floor(pi / (2 * theta)) + 1)
    assert cycle_lower <= sufficient_even
    print(f"Singular eta={eta:g}: theta={theta:.9f}, "
          f"angular lower={cycle_lower}, constructive upper={sufficient_even}")
