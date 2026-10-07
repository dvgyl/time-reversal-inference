#!/usr/bin/env python3
"""Exact finite-delay ball certificates; standard library only.

CLI: python3 certificate.py certificate.json
No statistical coverage is inferred from a JSON center or stored record array.
"""
from fractions import Fraction as Q
from itertools import product
from math import ceil
import json
import sys


def rational(value):
    if isinstance(value, (float, bool)):
        raise ValueError('use integers or rational strings, not floats/bools')
    return Q(value)


def matrix(rows):
    if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError('nonempty rectangular matrix required')
    if any(isinstance(v, (float, bool)) for r in rows for v in r):
        raise ValueError('use integers or rational strings, not floats/bools')
    return [[rational(v) for v in r] for r in rows]


def symmetric(a, n):
    if len(a) != n or any(len(r) != n for r in a):
        raise ValueError('wrong square dimension')
    if any(a[i][j] != a[j][i] for i in range(n) for j in range(n)):
        raise ValueError('matrix must be symmetric')


def dot(a, b):
    return sum((x*y for ar, br in zip(a, b) for x, y in zip(ar, br)), Q(0))


def gram(b):
    return [[sum((x*y for x, y in zip(r, u)), Q(0)) for u in b] for r in b]


def maps(m, T, delays):
    if type(m) is not int or type(T) is not int or m < 1 or T < 1:
        raise ValueError('m,T must be positive integers')
    if len(delays) != m or any(type(d) is not int or d < 0 for d in delays) or delays[-1] != 0:
        raise ValueError('invalid calibrated delay vector')
    q = T-1+max(delays)
    coordinates = [(k, i, j) for k in range(q+1) for i in range(m) for j in range(i, m)]
    index = {v: k for k, v in enumerate(coordinates)}
    def cell(t, i, u, j, delayed):
        lag = abs(t-u-delays[i]+delays[j]) if delayed else abs(t-u)
        return index[(lag, min(i, j), max(i, j))]
    A = [[cell(t, i, u, j, True) for u in range(T) for j in range(m)]
         for t in range(T) for i in range(m)]
    F = [[cell(t, i, u, j, False) for u in range(q+1) for j in range(m)]
         for t in range(q+1) for i in range(m)]
    return coordinates, A, F


def adjoint(mapping, z, N):
    result = [Q(0)]*N
    for cells, row in zip(mapping, z):
        for k, value in zip(cells, row):
            result[k] += value
    return result


def verify(m, T, delays, center, radius, h, B):
    coords, A, F = maps(m, T, delays)
    p = m*T
    S, h, B = matrix(center), matrix(h), matrix(B)
    symmetric(S, p)
    symmetric(h, p)
    if len(B) != len(F):
        raise ValueError('Gram factor has wrong row dimension')
    R = rational(radius)
    if R < 0:
        raise ValueError('negative confidence radius')
    M = max(Q(0), *(S[j][j] for j in range(p)))
    U = m*(M+R)
    ah, fz = adjoint(A, h, len(coords)), adjoint(F, gram(B), len(coords))
    residual = sum((abs(x-y) for x, y in zip(ah, fz)), Q(0))
    norm2 = dot(h, h)
    gap = dot(h, S)+2*R+residual*U
    return {'accepted': norm2 <= 4 and gap < 0, 'gap': str(gap),
            'residual_l1': str(residual), 'h_norm_squared': str(norm2), 'variance_upper': str(U)}


def covariance(records):
    """Exact covariance of supplied rational records, not an ideal-law claim."""
    X = matrix(records)
    n, p = len(X), len(X[0])
    if n < 2:
        raise ValueError('at least two records required')
    means = [sum((r[j] for r in X), Q(0))/n for j in range(p)]
    return [[sum(((r[i]-means[i])*(r[j]-means[j]) for r in X), Q(0))/(n-1)
             for j in range(p)] for i in range(p)]


def raw_scatter_enclosure(records, coordinate_error):
    """Conditional bound: caller must justify the raw coordinate error."""
    X, e = matrix(records), rational(coordinate_error)
    if e < 0 or len(X) < 2:
        raise ValueError('invalid record enclosure')
    n, p = len(X), len(X[0])
    mean = [sum((r[j] for r in X), Q(0))/n for j in range(p)]
    b = sum((abs(r[j]-mean[j]) for r in X for j in range(p)), Q(0))
    u = n*p*e  # sqrt(np) <= np; intentionally loose rational bound
    return (2*b*u+u*u)/(n-1)


def radius(center, a, zeta=0):
    S, a, zeta = matrix(center), rational(a), rational(zeta)
    p = len(S)
    symmetric(S, p)
    if not 0 < a <= Q(1, 4) or zeta < 0:
        raise ValueError('invalid radius parameters')
    M = max(Q(0), *(S[j][j] for j in range(p)))
    return p*a*(M+zeta)/(1-a)+zeta


def design(m, T, conditions, delays, alpha, beta, K, epsilon):
    if type(conditions) is not int or conditions < 1 or not delays:
        raise ValueError('nonempty condition and delay sets required')
    alpha, beta, K, epsilon = map(rational, (alpha, beta, K, epsilon))
    if not 0 < alpha < 1 or not 0 < beta < 1 or K <= 0 or epsilon <= 0:
        raise ValueError('invalid design parameters')
    p, t = m*T, 1
    Ls = [len(maps(m, T, d)[2]) for d in delays]
    while Q(8, 3)**t < 2*conditions*p*p/min(alpha, beta):
        t += 1
    a = min(Q(1, 4), epsilon/(32*p*K))
    C = max(p*p+3*p*L**3 for L in Ls)
    D = p*K*(2*p+3*m*C)
    J = max(p, ceil(4*D/epsilon))
    return {'t': t, 'a': str(a), 'n_min': 1+ceil(16*t/a**2), 'J': J,
            'C': C, 'zeta_max': str(min(K/4, epsilon/(16*(p+1)))),
            'grid_counts': [(4*J+1)**(p*(p+1)//2)*(2*(p+1)*J+1)**(L*L) for L in Ls]}


def grid_search(m, T, delays, center, R, J, max_checks=None):
    """Complete finite grid if max_checks=None; budget exhaustion abstains."""
    if type(J) is not int or J < 1:
        raise ValueError('positive integer mesh denominator required')
    if max_checks is not None and (type(max_checks) is not int or max_checks < 0):
        raise ValueError('invalid check budget')
    p, L = m*T, len(maps(m, T, delays)[2])
    if max_checks == 0:
        return {'status': 'unresolved_budget', 'checks': 0}
    positions = [(i, j) for i in range(p) for j in range(i, p)]
    count = 0
    for entries in product(range(-2*J, 2*J+1), repeat=len(positions)):
        h = [[Q(0)]*p for _ in range(p)]
        for (i, j), value in zip(positions, entries):
            h[i][j] = h[j][i] = Q(value, J)
        if dot(h, h) > 4:
            continue
        for vals in product(range(-(p+1)*J, (p+1)*J+1), repeat=L*L):
            if max_checks is not None and count >= max_checks:
                return {'status': 'unresolved_budget', 'checks': count}
            B = [[Q(vals[i*L+j], J) for j in range(L)] for i in range(L)]
            count += 1
            report = verify(m, T, delays, center, R, h, B)
            if report['accepted']:
                return {'status': 'excluded', 'checks': count, 'h': h, 'B': B, 'report': report}
    return {'status': 'unresolved_exhausted', 'checks': count}


def search_union(m, T, delays, centers, sample_counts, zetas,
                 alpha, beta, K, epsilon, max_checks_per_pair=None):
    """Run the complete theorem algorithm; an optional budget loses power."""
    if not centers or len(centers) != len(sample_counts) or len(centers) != len(zetas):
        raise ValueError('nonempty matching condition arrays required')
    plan = design(m, T, len(centers), delays, alpha, beta, K, epsilon)
    if any(type(n) is not int or n < plan['n_min'] for n in sample_counts):
        return {'rejected': False, 'status': 'unresolved_sample_count'}
    if any(not 0 <= rational(z) <= Q(plan['zeta_max']) for z in zetas):
        return {'rejected': False, 'status': 'unresolved_enclosure'}
    radii = [radius(S, plan['a'], z) for S, z in zip(centers, zetas)]
    found = []
    for di, d in enumerate(delays):
        for ci, (S, R) in enumerate(zip(centers, radii)):
            result = grid_search(m, T, d, S, R, plan['J'], max_checks_per_pair)
            if result['status'] == 'excluded':
                found.append({'delay_index': di, 'condition_index': ci,
                              'h': result['h'], 'B': result['B']})
                break
        else:
            return {'rejected': False, 'status': 'unresolved_search', 'certificates': found}
    return {'rejected': True, 'status': 'certified_exclusion', 'certificates': found}


def verify_union(m, T, delays, centers, radii, certificates):
    """Bind each certificate to the requested candidate and condition."""
    if not delays or not centers or len(centers) != len(radii):
        raise ValueError('nonempty matching arrays required')
    excluded = set()
    reports = []
    for item in certificates:
        di, ci = item['delay_index'], item['condition_index']
        if type(di) is not int or type(ci) is not int or not 0 <= di < len(delays) or not 0 <= ci < len(centers):
            raise ValueError('invalid candidate or condition index')
        r = verify(m, T, delays[di], centers[ci], radii[ci], item['h'], item['B'])
        reports.append(r)
        if r['accepted']:
            excluded.add(di)
    return {'rejected': len(excluded) == len(delays), 'excluded': sorted(excluded), 'reports': reports}


if __name__ == '__main__':
    with open(sys.argv[1], encoding='utf-8') as stream:
        packet = json.load(stream)
    print(json.dumps(verify_union(**packet), indent=2))
