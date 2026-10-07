"""Generate one declared record from independent role-specific streams."""

from fractions import Fraction as Q

import numpy as np
from scipy.signal import lfilter

import physical_models as physical


def _ar_component(length, phi, rng, condition, burn_in):
    if condition != 'student_t_5':
        return physical.stationary_ar(rng.standard_normal((length, 1)), phi)[:, 0]
    noise = rng.standard_t(5, size=length+burn_in)*np.sqrt(3/5)
    values = lfilter([np.sqrt(1-float(phi)**2)], [1., -float(phi)], noise)
    return values[burn_in:]


def source_record(cell, streams):
    p, n = cell['parameters'], cell['N']
    phi, coupling, delay = Q(p['phi']), float(Q(p['coupling'])), p['delay']
    prehistory = delay+1
    length = n+prehistory
    x = _ar_component(length, phi, streams['source'], p['condition'], p.get('burn_in', 8192))
    z = _ar_component(length, phi, streams['independent_source'], p['condition'], p.get('burn_in', 8192))
    first = np.arange(prehistory, prehistory+n)
    current = coupling*x[first-delay]+np.sqrt(1-coupling**2)*z[first]
    previous = coupling*x[first-1-delay]+np.sqrt(1-coupling**2)*z[first-1]
    h0, h1 = map(lambda value:float(Q(value)), p['response'])
    record = np.column_stack((x[first]+2., h0*current+h1*previous-1.))
    if p['condition'].startswith('reference_snr_'):
        snr = int(p['condition'].rsplit('_', 1)[1])
        record[:, 0] += streams['measurement_error'].standard_normal(n)/np.sqrt(snr)
    return record


def generate_record(cell, streams):
    p, n, family = cell['parameters'], cell['N'], cell['family']
    if family == 'source':
        return source_record(cell, streams)
    if family == 'brownian':
        return physical.brownian_record(n, *map(Q, p['temperatures']), streams['source'],
                                        second_tap=Q(p['response'][1]))
    if family == 'phase_boundary':
        return physical.phase_boundary_record(n, Q(p['b']), streams['source'])
    if family == 'cycle':
        filters = tuple(tuple((position, Q(value)) for position,value in row) for row in p['filters'])
        record = physical.filtered_rotational_record(n, streams['source'], filters,
                                                      Q(p['rho']), Q(p['correlation']), p['circulation'])
        if p['condition'] == 'correlated_error':
            if p['error_permutation'] != [1,2,0] or p['error_independent_from_source'] is not True:
                raise ValueError('The declared error orientation or stream independence differs.')
            error = physical.rotational_source(n, streams['measurement_error'],Q(p['error_rho']),
                                                 Q(p['error_correlation']),True)
            record += np.sqrt(float(Q(p['error_variance'])))*error
        return record
    raise ValueError('Unknown declared model family.')


def quantize_channels(record, bits=20):
    if type(bits) is not int or not 0 <= bits <= 40:
        raise ValueError('Use a fractional bit count from0 through40.')
    values = np.asarray(record, dtype=float)
    if values.ndim != 2 or values.shape[1] not in (2,3) or not np.isfinite(values).all():
        raise ValueError('Supply a finite two-channel or three-channel recording.')
    scaled = np.ldexp(values,bits)
    if np.max(np.abs(scaled)) >= 2**52:
        raise ValueError('The recording exceeds the declared unsaturated range.')
    integers = np.rint(scaled).astype(np.int64)
    return integers, Q(1,2**(bits+1))


def finite_recorder(record, bits):
    if bits not in (8,12):
        raise ValueError('Use the declared8-bit or12-bit recorder.')
    values = np.asarray(record,dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Recorder inputs must be finite.')
    step = 32/(2**bits)
    low, high = -16+step/2, 16-step/2
    indices = np.floor(np.ldexp(np.clip(values,-16,16),bits-5))
    centers = np.ldexp(indices+.5,5-bits)
    clipped = np.clip(centers,low,high)
    saturated = int(np.count_nonzero((values < -16) | (values >= 16)))
    return clipped, dict(bits=bits,range=[-16,16],step=step,saturated_values=saturated,
                         valid_error_bound=None if saturated else str(Q(16,2**bits)))
