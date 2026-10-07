"""A two-channel finite-filter test with Gaussian calibration and a trace radius."""
from dataclasses import dataclass
import hashlib
import math
import operator
import numbers
from typing import Sequence

import numpy as np
from scipy.stats import f


class InputError(ValueError):
    """The input is invalid or its configuration is not supported."""


def real_scalar(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise InputError(label+' must be a real numerical scalar.')
    result = float(value)
    if not math.isfinite(result):
        raise InputError(label+' must be finite.')
    return result


def pair(value, label):
    if not isinstance(value, (tuple, list, np.ndarray)):
        raise InputError(label+' must contain exactly two entries.')
    if isinstance(value, np.ndarray) and value.ndim != 1:
        raise InputError(label+' must contain exactly two entries.')
    if len(value) != 2:
        raise InputError(label+' must contain exactly two entries.')
    return list(value)


def finite(value, label):
    if not np.isfinite(value).all():
        raise InputError(label+' is not finite. This numerical range is not supported.')


@dataclass(frozen=True)
class CalibrationData:
    """A fixed regression design, its readout, and known coefficient positions."""

    design: np.ndarray
    response: np.ndarray
    positions: Sequence[int]
    include_intercept: bool = False


@dataclass(frozen=True)
class ErrorBudgets:
    """Fixed failure budgets for calibration, variance coverage, and both tails."""

    calibration: tuple = (0.005, 0.005)
    variance: float = 0.01
    tail: float = 0.03
    target_size: float = 0.05

    def validate(self):
        calibration = [real_scalar(v, 'Calibration failure budget') for v in pair(self.calibration, 'Calibration budgets')]
        variance, tail, size = [real_scalar(v, 'Error budget') for v in [self.variance, self.tail, self.target_size]]
        values = [*calibration, variance, tail, size]
        for value in values:
            if not 0 < value < 1:
                raise InputError('Each error budget must be between zero and one.')
        total = math.fsum([*calibration, variance, tail])
        if total > size:
            raise InputError('The failure budgets exceed the declared size bound.')
        return dict(calibration=calibration, calibration_total=math.fsum(calibration),
                    variance=variance, tail=tail, target_size=size, total=total)


def integer(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise InputError(label+' must be an integer, not a Boolean value.')
    try:
        result = operator.index(value)
    except TypeError as error:
        raise InputError(label+' must be an integer.') from error
    if abs(result) > 2**31-1:
        raise InputError(label+' is outside the supported integer range.')
    return result


def real_array(value, dimensions, label):
    try:
        original = np.asarray(value)
    except (ValueError, TypeError) as error:
        raise InputError(label+' is not a valid real numerical array.') from error
    if original.ndim != dimensions or original.dtype.kind not in 'fiu':
        raise InputError(label+' must be a real numerical array with the required dimensions.')
    try:
        array = np.array(original, dtype=np.float64, copy=True)
    except (ValueError, TypeError, OverflowError) as error:
        raise InputError(label+' cannot be represented as float64.') from error
    if not np.isfinite(array).all():
        raise InputError(label+' contains a non-finite value.')
    return array


def positions_array(values, length):
    original = np.asarray(values)
    if original.ndim != 1 or len(original) != length or original.dtype.kind not in 'iu':
        raise InputError('Supply one integer position for each coefficient.')
    positions = np.array([integer(v, 'Coefficient position') for v in original], dtype=np.int64)
    if len(set(positions.tolist())) != length:
        raise InputError('Coefficient positions must be distinct.')
    return positions


def array_sha256(array):
    """Bind the canonical float64 shape and bytes used by the calculation."""
    canonical = np.ascontiguousarray(array, dtype='<f8')
    header = str(tuple(canonical.shape)).encode('ascii')+b'\nfloat64-little-endian\n'
    return hashlib.sha256(header+canonical.tobytes()).hexdigest()


def fit_calibration(data, eta):
    """Fit ordinary least squares and its exact Gaussian F confidence ellipse."""
    if not isinstance(data, CalibrationData):
        raise InputError('Each calibration must be a CalibrationData object.')
    if data.include_intercept is not False:
        raise InputError('A calibration intercept is not supported. Supply the fixed zero-offset model.')
    design = real_array(data.design, 2, 'Calibration design')
    response = real_array(data.response, 1, 'Calibration response')
    count, length = design.shape
    if length < 1 or count <= length or len(response) != count:
        raise InputError('Calibration needs matching rows and positive residual degrees of freedom.')
    positions = positions_array(data.positions, length)
    eta = real_scalar(eta, 'Calibration budget')
    if not 0 < eta < 1:
        raise InputError('The calibration budget must be between zero and one.')
    singular = np.linalg.svd(design, compute_uv=False)
    if singular[-1] <= 0 or singular[0]/singular[-1] > 1e8:
        raise InputError('The calibration design is rank deficient or too ill conditioned.')
    coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    if rank != length:
        raise InputError('The calibration design does not have full column rank.')
    residual = response-design@coefficients
    degrees = count-length
    residual_variance = float(np.dot(residual, residual)/degrees)
    if not math.isfinite(residual_variance) or residual_variance <= 0:
        raise InputError('The positive-noise calibration model needs a positive finite residual variance.')
    quantile = float(f.ppf(1-eta, length, degrees))
    gram_minimum = float(singular[-1]**2)
    ellipse_squared = float(length*residual_variance*quantile)
    finite([quantile, gram_minimum, ellipse_squared], 'Regression confidence parameters')
    if quantile <= 0 or gram_minimum <= 0 or ellipse_squared <= 0:
        raise InputError('The regression confidence parameters are outside the supported positive numerical range.')
    error_l2 = math.sqrt(ellipse_squared/gram_minimum)
    error_l1 = math.sqrt(length)*error_l2
    numeric = [quantile, gram_minimum, ellipse_squared, error_l2, error_l1, *coefficients]
    if not all(math.isfinite(float(x)) for x in numeric) or gram_minimum <= 0:
        raise InputError('The calibration calculation produced an invalid finite bound.')
    return dict(coefficients=coefficients.tolist(), positions=positions.tolist(), rows=count,
                length=length, residual_degrees=degrees, residual_variance=residual_variance,
                minimum_gram_eigenvalue=gram_minimum, design_condition=float(singular[0]/singular[-1]),
                F_quantile=quantile, ellipse_radius_squared=ellipse_squared,
                coefficient_error_l2=error_l2, coefficient_error_l1=error_l1,
                calibration_failure_budget=float(eta), design_sha256=array_sha256(design),
                response_sha256=array_sha256(response))


def frequency_certificate(fits, anchors=(0, 0), grid_points=16384, reflection_spacing=1):
    """Bound phase and filter shape on every frequency, with grid-cell margins."""
    fits = pair(fits, 'Filter fits')
    anchors = pair(anchors, 'Anchors')
    spacing = integer(reflection_spacing, 'Reflection spacing')
    if spacing != 1:
        raise InputError('Only reflection spacing one is supported by this reference procedure.')
    points = integer(grid_points, 'Grid point count')
    if not 16 <= points <= 1048576:
        raise InputError('Use between 16 and 1048576 certificate grid points.')
    anchors = [integer(a, 'Anchor') for a in anchors]
    distance = math.pi/points
    frequencies = 2*math.pi*np.arange(points)/points
    responses, radii, shapes, details = [], [], [], []
    for index, fit in enumerate(fits):
        coefficients = real_array(fit['coefficients'], 1, 'Fitted coefficients')
        length = len(coefficients)
        if length < 1:
            raise InputError('At least one fitted coefficient is required.')
        positions = positions_array(fit['positions'], length)
        shifted = positions-anchors[index]
        if np.max(np.abs(shifted)) > 2**31-1:
            raise InputError('The shifted coefficient positions are outside the supported range.')
        e2 = real_scalar(fit['coefficient_error_l2'], 'Coefficient L2 error bound')
        e1 = real_scalar(fit['coefficient_error_l1'], 'Coefficient L1 error bound')
        if e2 < 0 or e1 < math.sqrt(length)*e2:
            raise InputError('The coefficient confidence radii are invalid.')
        response = np.exp(-1j*np.outer(frequencies, shifted))@coefficients
        derivative = float(np.sum(np.abs(shifted*coefficients)))
        finite(response, 'Fitted frequency responses')
        finite(derivative, 'Fitted derivative bound')
        cell_radius = e1+derivative*distance
        peak = float(np.max(np.abs(response)))+cell_radius
        norm = float(np.linalg.norm(coefficients))
        finite([cell_radius, peak, norm], 'Transfer and coefficient norm bounds')
        norm_floor = max(0.0, norm-e2)
        if norm_floor > 0:
            ratio = peak/norm_floor
            finite(ratio, 'Filter peak-to-norm ratio')
            shape = float(length) if ratio >= math.sqrt(length) else max(1.0, ratio**2)
        else:
            shape = float(length)
        responses.append(response)
        radii.append(cell_radius)
        shapes.append(shape)
        details.append(dict(anchor=anchors[index], shifted_positions=shifted.tolist(), derivative_bound=derivative,
                            transfer_cell_radius=cell_radius, transfer_peak_upper=peak,
                            coefficient_norm_lower=norm_floor, shape_upper=shape,
                            norm_floor_fallback=norm_floor == 0))
    product = responses[0]*np.conjugate(responses[1])
    product_error = radii[0]*np.abs(responses[1])+radii[1]*np.abs(responses[0])+radii[0]*radii[1]
    finite(product, 'Fitted transfer product')
    finite(product_error, 'Transfer product error radius')
    denominator = np.abs(product)-product_error
    finite(denominator, 'Transfer product denominator')
    ratios = np.ones(points)
    safe = denominator > 0
    numerator = np.abs(np.imag(product[safe]))+product_error[safe]
    finite(numerator, 'Transfer product numerator')
    raw_ratios = numerator/denominator[safe]
    finite(raw_ratios, 'Relative phase ratios')
    ratios[safe] = np.minimum(1.0, raw_ratios)
    weights = np.minimum(1.0, np.abs(np.sin(frequencies))+distance)
    phase = float(np.max(weights*ratios))
    if not math.isfinite(phase) or not 0 <= phase <= 1 or not all(math.isfinite(x) for x in shapes):
        raise InputError('The frequency calculation produced an invalid finite certificate.')
    return dict(phase_upper=phase, shape_upper=shapes, grid_points=points, cell_distance=distance,
                universal_product_cells=int(np.count_nonzero(~safe)), reflection_spacing=spacing,
                channels=details, scope='Complete torus. One coefficient event covers every grid cell.')


def align_recording(recording, anchors=(0, 0)):
    """Retain common base times for the observed channels Y_i(t+d_i)."""
    values = real_array(recording, 2, 'Testing recording')
    anchors = pair(anchors, 'Anchors')
    if values.shape[1] != 2:
        raise InputError('Supply one recording with exactly two channels and two anchors.')
    anchors = [integer(a, 'Anchor') for a in anchors]
    start = max(-a for a in anchors)
    stop = min(len(values)-a for a in anchors)
    retained = stop-start
    if retained < 3:
        raise InputError('Anchor alignment must retain at least three observations per channel.')
    aligned = np.column_stack([values[start+a:stop+a, i] for i,a in enumerate(anchors)])
    return aligned, dict(raw_record_length=len(values), retained_record_length=retained,
                         base_time_start=start, base_time_stop_exclusive=stop, anchors=anchors,
                         channel_slice_starts=[start+a for a in anchors],
                         channel_slice_stops_exclusive=[stop+a for a in anchors])


def reflection_statistic(aligned):
    """Use the same starts and centering for the sum and difference projections."""
    values = real_array(aligned, 2, 'Aligned recording')
    if values.shape[1] != 2 or len(values) < 3:
        raise InputError('The reflection statistic needs two channels and at least three times.')
    first = values[:-1, 0]+values[1:, 0]
    second = values[:-1, 1]-values[1:, 1]
    n = len(first)
    statistic = float(np.dot(first-first.mean(), second-second.mean())/n)
    if not math.isfinite(statistic):
        raise InputError('The reflection statistic is not finite.')
    return statistic


def trace_radius(rho, variance_ceiling, record_length, t):
    """Take the minimum Frobenius and trace bounds for spacing one."""
    length = integer(record_length, 'Retained record length')
    rho, variance_ceiling, t = [real_scalar(v, 'Radius parameter') for v in [rho, variance_ceiling, t]]
    if length < 3:
        raise InputError('The radius needs finite values and at least three retained times.')
    if rho < 1 or variance_ceiling < 0 or t <= 0:
        raise InputError('The radius envelope, variance ceiling, or tail parameter is invalid.')
    n = length-1
    norm = math.sqrt((2*n+2-4/n-4/n**2)/(2*n**2))
    frobenius = variance_ceiling*rho*norm
    trace = variance_ceiling*2*math.sqrt(2*rho*length)/n
    finite([frobenius, trace], 'Frobenius and trace branches')
    minimum = min(frobenius, trace)
    linear = 4*rho*variance_ceiling*t/n
    finite(linear, 'Linear radius term')
    radius = 2*minimum*math.sqrt(t)+linear
    if not math.isfinite(radius):
        raise InputError('The trace radius is not finite.')
    return dict(F_n=norm, operator_norm_upper=2/n, Frobenius_branch=frobenius,
                trace_branch=trace, minimum_F_bound=minimum, linear_term=linear,
                radius=radius, old_radius=2*frobenius*math.sqrt(t)+linear,
                branch='trace' if trace < frobenius else 'Frobenius', tail_parameter=t)


def run_test(calibrations, recording, source_floor, source_ceiling, error_peak_ratios,
             anchors=(0, 0), budgets=ErrorBudgets(), grid_points=16384, reflection_spacing=1):
    """Return a calibrated decision or abstention for one dependent recording."""
    calibrations = pair(calibrations, 'Calibrations')
    if not isinstance(budgets, ErrorBudgets):
        raise InputError('Supply two calibrations and an ErrorBudgets object.')
    budget = budgets.validate()
    source_floor = real_scalar(source_floor, 'Source spectral floor')
    source_ceiling = real_scalar(source_ceiling, 'Source spectral ceiling')
    if not 0 < source_floor <= 1 <= source_ceiling:
        raise InputError('The normalized source spectrum needs 0 < floor <= 1 <= ceiling.')
    error_peak_ratios = [real_scalar(v, 'Error peak ratio') for v in pair(error_peak_ratios, 'Error peak ratios')]
    for value in error_peak_ratios:
        if value < 1:
            raise InputError('Each supplied error peak ratio must be finite and at least one.')
    aligned, alignment = align_recording(recording, anchors)
    fits = [fit_calibration(data, eta) for data,eta in zip(calibrations, budget['calibration'])]
    certificate = frequency_certificate(fits, anchors, grid_points, reflection_spacing)
    source_ratio = float(source_ceiling/source_floor)
    rho = max(2*source_ratio*max(certificate['shape_upper']), *[float(x) for x in error_peak_ratios])
    length = len(aligned)
    t_variance = math.log(2/budget['variance'])
    denominator = 1-rho/length-2*math.sqrt(rho*t_variance/length)
    centered = aligned-aligned.mean(axis=0)
    sample_variances = np.mean(centered*centered, axis=0)
    statistic = reflection_statistic(aligned)
    if not math.isfinite(rho) or not math.isfinite(denominator) or not np.isfinite(sample_variances).all():
        raise InputError('The recording calculation produced an invalid finite quantity.')
    result = dict(status='ABSTAIN' if denominator <= 0 else 'DECISION',
                  abstention_reason='The same-record scale denominator is nonpositive.' if denominator <= 0 else None,
                  decision=None, rejection=None, alignment=alignment, window_count=length-1,
                  fits=fits, certificate=certificate, source_floor=float(source_floor),
                  source_ceiling=float(source_ceiling), supplied_source_ratio=source_ratio,
                  supplied_error_peak_ratios=[float(x) for x in error_peak_ratios], rho_upper=rho,
                  scale_denominator=denominator, variance_tail_parameter=t_variance,
                  sample_variances=sample_variances.tolist(), variance_ceilings=None, common_variance_ceiling=None,
                  sample_means=aligned.mean(axis=0).tolist(),
                  statistic=statistic, statistic_absolute=abs(statistic), null_tolerance=None,
                  sampling_radius=None, threshold=None, budgets=budget,
                  recording_sha256=array_sha256(np.asarray(recording)), aligned_recording_sha256=array_sha256(aligned),
                  assumptions=[
                      'Each calibration design is fixed and has full column rank. Its readout error is spherical Gaussian with positive variance.',
                      'The finite filters and their integer positions are the same in calibration and testing. Calibration offsets are zero.',
                      'The source and observation errors are jointly stationary Gaussian and have spectral densities.',
                      'The normalized marginal source spectra obey the supplied floor and ceiling.',
                      'Observation errors are mutually orthogonal and orthogonal to the source. Their supplied peak ratios hold.',
                      'Anchors, reflection spacing, certificate grid and failure budgets are fixed before observation.',
                      'Under the tested null, the source is reversible. Calibration and testing need not be independent.',
                  ])
    if denominator <= 0:
        return result
    ceilings = sample_variances/denominator
    common = float(max(ceilings))
    radius = trace_radius(rho, common, length, math.log(2/budget['tail']))
    tolerance = 2*certificate['phase_upper']*common
    threshold = tolerance+radius['radius']
    if not np.isfinite(ceilings).all() or not math.isfinite(threshold):
        raise InputError('The threshold calculation is not finite.')
    rejection = abs(statistic) > threshold
    result.update(decision='REJECT' if rejection else 'DO_NOT_REJECT', rejection=bool(rejection),
                  variance_ceilings=ceilings.tolist(), common_variance_ceiling=common,
                  null_tolerance=tolerance, sampling_radius=radius, threshold=threshold)
    return result
