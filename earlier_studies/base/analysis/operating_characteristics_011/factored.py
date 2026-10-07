"""Evaluate a record with calibration outputs from an unchanged full call.

This module changes no calibration formula. The campaign must supply the fits
and certificate from the full reference010 call on this replicate.
"""
import copy
import math
import numpy as np
from inputs.reference import *

def evaluate_cached(fits, certificate, recording, source_floor, source_ceiling, error_peak_ratios, anchors=(0, 0), budgets=ErrorBudgets(), grid_points=16384, reflection_spacing=1):
    """Return a calibrated decision or abstention for one dependent recording."""
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
    (aligned, alignment) = align_recording(recording, anchors)
    source_ratio = float(source_ceiling / source_floor)
    rho = max(2 * source_ratio * max(certificate['shape_upper']), *[float(x) for x in error_peak_ratios])
    length = len(aligned)
    t_variance = math.log(2 / budget['variance'])
    denominator = 1 - rho / length - 2 * math.sqrt(rho * t_variance / length)
    centered = aligned - aligned.mean(axis=0)
    sample_variances = np.mean(centered * centered, axis=0)
    statistic = reflection_statistic(aligned)
    if not math.isfinite(rho) or not math.isfinite(denominator) or (not np.isfinite(sample_variances).all()):
        raise InputError('The recording calculation produced an invalid finite quantity.')
    result = dict(status='ABSTAIN' if denominator <= 0 else 'DECISION', abstention_reason='The same-record scale denominator is nonpositive.' if denominator <= 0 else None, decision=None, rejection=None, alignment=alignment, window_count=length - 1, fits=fits, certificate=certificate, source_floor=float(source_floor), source_ceiling=float(source_ceiling), supplied_source_ratio=source_ratio, supplied_error_peak_ratios=[float(x) for x in error_peak_ratios], rho_upper=rho, scale_denominator=denominator, variance_tail_parameter=t_variance, sample_variances=sample_variances.tolist(), variance_ceilings=None, common_variance_ceiling=None, sample_means=aligned.mean(axis=0).tolist(), statistic=statistic, statistic_absolute=abs(statistic), null_tolerance=None, sampling_radius=None, threshold=None, budgets=budget, recording_sha256=array_sha256(np.asarray(recording)), aligned_recording_sha256=array_sha256(aligned), assumptions=['Each calibration design is fixed and has full column rank. Its readout error is spherical Gaussian with positive variance.', 'The finite filters and their integer positions are the same in calibration and testing. Calibration offsets are zero.', 'The source and observation errors are jointly stationary Gaussian and have spectral densities.', 'The normalized marginal source spectra obey the supplied floor and ceiling.', 'Observation errors are mutually orthogonal and orthogonal to the source. Their supplied peak ratios hold.', 'Anchors, reflection spacing, certificate grid and failure budgets are fixed before observation.', 'Under the tested null, the source is reversible. Calibration and testing need not be independent.'])
    if denominator <= 0:
        return result
    ceilings = sample_variances / denominator
    common = float(max(ceilings))
    radius = trace_radius(rho, common, length, math.log(2 / budget['tail']))
    tolerance = 2 * certificate['phase_upper'] * common
    threshold = tolerance + radius['radius']
    if not np.isfinite(ceilings).all() or not math.isfinite(threshold):
        raise InputError('The threshold calculation is not finite.')
    rejection = abs(statistic) > threshold
    result.update(decision='REJECT' if rejection else 'DO_NOT_REJECT', rejection=bool(rejection), variance_ceilings=ceilings.tolist(), common_variance_ceiling=common, null_tolerance=tolerance, sampling_radius=radius, threshold=threshold)
    return result


def with_envelope(primary, rho, envelope_receipt):
    """Reuse recording summaries with a verified supplied-marginal envelope."""
    rho = real_scalar(rho, 'Supplied marginal envelope')
    if rho < 1 or rho > primary['rho_upper']:
        raise InputError('The marginal envelope must be valid and no larger than the broad envelope.')
    result = copy.deepcopy(primary)
    result['marginal_envelope'] = envelope_receipt
    result['envelope_source'] = 'Exactly supplied common AR(1) marginal spectra and coefficient event.'
    result['rho_upper'] = rho
    length = result['alignment']['retained_record_length']
    denominator = 1-rho/length-2*math.sqrt(rho*result['variance_tail_parameter']/length)
    if not math.isfinite(denominator):
        raise InputError('The marginal scale denominator is not finite.')
    result.update(scale_denominator=denominator, status='ABSTAIN' if denominator <= 0 else 'DECISION',
                  abstention_reason='The same-record scale denominator is nonpositive.' if denominator <= 0 else None,
                  decision=None, rejection=None, variance_ceilings=None, common_variance_ceiling=None,
                  null_tolerance=None, sampling_radius=None, threshold=None)
    if denominator <= 0:
        return result
    ceilings = np.asarray(result['sample_variances'])/denominator
    common = float(max(ceilings))
    radius = trace_radius(rho, common, length, math.log(2/result['budgets']['tail']))
    tolerance = 2*result['certificate']['phase_upper']*common
    threshold = tolerance+radius['radius']
    if not np.isfinite(ceilings).all() or not math.isfinite(threshold):
        raise InputError('The marginal threshold is not finite.')
    rejection = abs(result['statistic']) > threshold
    result.update(decision='REJECT' if rejection else 'DO_NOT_REJECT', rejection=bool(rejection),
                  variance_ceilings=ceilings.tolist(), common_variance_ceiling=common,
                  null_tolerance=tolerance, sampling_radius=radius, threshold=threshold)
    return result
