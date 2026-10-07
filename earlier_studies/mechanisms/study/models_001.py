"""Define the prospective population models and their exact cycle bounds."""
from fractions import Fraction as Q
import numpy as np
from scipy.signal import lfilter


def cadd(a, b):
    return a[0]+b[0], a[1]+b[1]


def cmul(a, b):
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]


def cdiv(a, b):
    d = b[0]*b[0]+b[1]*b[1]
    return (a[0]*b[0]+a[1]*b[1])/d, (a[1]*b[0]-a[0]*b[1])/d


def conjugate(a):
    return a[0], -a[1]


def phase(k):
    return ((Q(1), Q(0)), (Q(0), Q(-1)),
            (Q(-1), Q(0)), (Q(0), Q(1)))[k % 4]


def source_covariance(cell, i, j, k):
    if k < 0:
        return source_covariance(cell, j, i, -k)
    if cell['family'] == 'finite_ma':
        r = Q(cell['r'])
        if k == 0:
            return Q(1) if i == j else (r if {i, j} in ({0, 1}, {1, 2}) else Q(0))
        if k == 1 and (i, j) == (0, 2):
            return r/2
        if k == 1 and (i, j) == (2, 0):
            return -r/2
        return Q(0)
    rho, r = Q(cell['rho']), Q(cell['r'])
    row = (i-k) % 3 if cell['dynamics'] == 'ring' else i
    return rho**k * (Q(1) if row == j else r)


def source_spectrum(cell, i, j):
    if cell['family'] == 'finite_ma':
        value = (Q(0), Q(0))
        for k in (-1, 0, 1):
            value = cadd(value, cmul((source_covariance(cell, i, j, k), Q(0)), phase(k)))
        return value
    rho, r = Q(cell['rho']), Q(cell['r'])
    if cell['dynamics'] == 'reversible':
        return (1-rho*rho)/(1+rho*rho)*(Q(1) if i == j else r), Q(0)
    def positive(a, b):
        numerator = (Q(0), Q(0))
        for k in range(3):
            numerator = cadd(numerator, cmul((source_covariance(cell, a, b, k), Q(0)), phase(k)))
        return cdiv(numerator, (Q(1), -rho**3))
    both = cadd(positive(i, j), conjugate(positive(j, i)))
    return both[0]-(Q(1) if i == j else r), both[1]


def filters(cell):
    return tuple(tuple((int(position), Q(value)) for position, value in h)
                 for h in cell['filters'])


def observed_covariance(cell, i, j, k):
    h = filters(cell)
    value = sum(va*vb*source_covariance(cell, i, j, k-a+b)
                for a, va in h[i] for b, vb in h[j])
    if i == j and k == 0:
        value += Q(cell['noise_variance'])
    return value


def observed_spectrum(cell, i, j):
    h = filters(cell)
    responses = []
    for row in h:
        value = (Q(0), Q(0))
        for k, coefficient in row:
            value = cadd(value, cmul((coefficient, Q(0)), phase(k)))
        responses.append(value)
    value = cmul(cmul(responses[i], source_spectrum(cell, i, j)), conjugate(responses[j]))
    if i == j:
        value = value[0]+Q(cell['noise_variance']), value[1]
    return value


def cycle_population(cell, n):
    h, lag = filters(cell), cell['lag']
    distance = max(abs(a-b) for hi in h for hj in h for a, _ in hi for b, _ in hj)
    if lag < distance:
        raise ValueError('The declared lag is below a filter-position difference.')
    s = 1+2*Q(cell['r'])
    if cell['family'] == 'finite_ma':
        if lag < distance+1:
            raise ValueError('The finite covariance support exceeds the declared lag.')
        source_ceiling = s
    else:
        rho = Q(cell['rho'])
        source_ceiling = s*(1+rho)/(1-rho)
    ceiling = source_ceiling*max(sum(abs(v) for _, v in hi)**2 for hi in h)+Q(cell['noise_variance'])
    allowances = []
    edges = ((0, 1), (1, 2), (2, 0))
    for i, j in edges:
        tail = Q(0)
        if cell['family'] != 'finite_ma':
            tail = s/(1-rho)*sum(abs(va*vb)*(rho**(lag+1-a+b)+rho**(lag+1+a-b))
                                 for a, va in h[i] for b, vb in h[j])
        weighted = sum(abs(k)*abs(observed_covariance(cell, i, j, k))
                       for k in range(-lag, lag+1))
        allowances.append(tail+weighted/n)
    spectra = tuple(observed_spectrum(cell, i, j) for i, j in edges)
    product = cmul(cmul(spectra[0], spectra[1]), spectra[2])
    return dict(K=ceiling, B=max(allowances), edge_biases=tuple(allowances),
                spectra=spectra, product=product, J=product[1],
                pairwise_lag_contrast=observed_covariance(cell, 0, 1, 1)-observed_covariance(cell, 0, 1, -1))


def stationary_ar(normals, phi):
    result = np.empty_like(normals)
    result[0] = normals[0]
    scale = np.sqrt(1-float(phi)**2)
    result[1:], _ = lfilter([scale], [1., -float(phi)], normals[1:],
                           axis=0, zi=(float(phi)*normals[0])[None, ...])
    return result


def cycle_record(cell, n, source_rng, noise_rng):
    h = filters(cell)
    prehistory = max(position for hi in h for position, _ in hi)
    length = n+prehistory
    if cell['family'] == 'finite_ma':
        r = float(Q(cell['r']))
        z = source_rng.standard_normal((length+2, 7))
        source = np.empty((length, 3))
        source[:, 0] = (np.sqrt(r)*z[2:, 0] + np.sqrt(r/2)*(z[1:-1, 2]+z[1:-1, 3])
                        + np.sqrt(1-2*r)*z[2:, 4])
        source[:, 1] = np.sqrt(r)*(z[2:, 0]+z[2:, 1])+np.sqrt(1-2*r)*z[2:, 5]
        source[:, 2] = (np.sqrt(r)*z[2:, 1] + np.sqrt(r/2)*(z[2:, 2]-z[:-2, 3])
                        + np.sqrt(1-2*r)*z[2:, 6])
    else:
        r, rho = float(Q(cell['r'])), Q(cell['rho'])
        z = source_rng.standard_normal((length, 4))
        innovations = np.sqrt(1-r)*z[:, :3]+np.sqrt(r)*z[:, 3, None]
        rotating = stationary_ar(innovations, rho)
        if cell['dynamics'] == 'ring':
            rows = np.arange(length)
            columns = (np.arange(3)[None, :]-rows[:, None]) % 3
            source = np.take_along_axis(rotating, columns, axis=1)
        else:
            source = rotating
    record = np.empty((n, 3))
    for i, hi in enumerate(h):
        record[:, i] = sum(float(v)*source[prehistory-k:prehistory-k+n, i] for k, v in hi)
    noise = noise_rng.standard_normal((n, 3))
    record += np.sqrt(float(Q(cell['noise_variance'])))*noise
    record += np.array([2., -1., .5])
    return record


def source_record(cell, n, source_rng, independent_rng, calibration_rng, calibration_rows):
    phi, coupling = Q(cell['phi']), float(Q(cell['coupling']))
    delay = cell['delay']
    prehistory = delay+1
    length = n+prehistory
    x = stationary_ar(source_rng.standard_normal((length, 1)), phi)[:, 0]
    z = stationary_ar(independent_rng.standard_normal((length, 1)), phi)[:, 0]
    h0, h1 = map(lambda value: float(Q(value)), cell['response'])
    indices = np.arange(prehistory, prehistory+n)
    second_current = coupling*x[indices-delay]+np.sqrt(1-coupling**2)*z[indices]
    second_previous = coupling*x[indices-1-delay]+np.sqrt(1-coupling**2)*z[indices-1]
    record = np.column_stack((x[indices]+2., h0*second_current+h1*second_previous-1.))
    rows = np.arange(calibration_rows)
    design = np.column_stack(((-1.)**rows, (-1.)**(rows//2)))
    calibration = design @ np.array([h0, h1])+0.01*calibration_rng.standard_normal(calibration_rows)
    return record, design, calibration
