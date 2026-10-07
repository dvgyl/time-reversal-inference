"""Evaluate the fixed spectral-pair determinant formula numerically."""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular, svdvals, toeplitz


def divergence(n, source_ratio=9., gap=.1):
    if n < 2 or n % 2:
        raise ValueError('Use a positive even record length.')
    a = source_ratio-1
    w = np.pi/(2*a)
    theta = gap*w/np.sin(w)
    m = n//2
    indices = np.arange(m)
    marginal = np.zeros(m)
    marginal[0] = 1.
    marginal[1:] = (a/2)*(-1.)**indices[1:]*np.sin(2*w*indices[1:])/(np.pi*indices[1:])
    covariance = toeplitz(marginal)
    lag = 2*(indices[:, None]-indices[None, :])-1
    cross = -a*np.sin(np.pi*lag/2)*np.sin(w*lag)/(np.pi*lag)
    factor = cholesky(covariance, lower=True, check_finite=False)
    first = solve_triangular(factor, cross, lower=True, check_finite=False)
    whitened = solve_triangular(factor, first.T, lower=True, check_finite=False).T
    singular = svdvals(whitened, check_finite=False)
    if not np.all(theta*singular < 1):
        raise ArithmeticError('The numerical normalized covariance is not positive.')
    value = -np.log1p(-(theta*singular)**2).sum()
    residual = np.linalg.norm(covariance-factor@factor.T, ord='fro')
    return dict(N=n, source_ratio=source_ratio, gap=gap, divergence=float(value),
                theta=float(theta), largest_singular_value=float(singular.max()),
                frobenius_square=float(2*np.dot(singular, singular)),
                cholesky_residual_frobenius=float(residual),
                entropy_needed=float(.9*np.log(19)),
                scope='Floating-point evaluation of the exact determinant expression; not an interval certificate.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--lengths', nargs='+', type=int, default=[128,512,1024,2048,3200,5214])
    arguments = parser.parse_args()
    arguments.output.mkdir(exist_ok=False)
    for n in arguments.lengths:
        result = divergence(n)
        (arguments.output/('N_%d.json' % n)).write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
