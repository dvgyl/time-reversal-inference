"""Apply calibrated thresholds to reusable exact source polynomials."""

from dataclasses import replace
from fractions import Fraction as Q

from calibrated_threshold import reflection_threshold, rounding_contrast_upper, rounding_variance_upper


def fit_rounded_calibration(reference, data, limits, error=Q(0)):
    """Enlarge the coefficient region for a bounded response perturbation.

    The calibration design is exact. Only the response has the supplied error.
    """
    if type(error) is not Q or error < 0:
        raise ValueError('Supply a nonnegative calibration error as a Fraction.')
    fitted = reference.fit_calibration(data, limits)
    if isinstance(fitted, reference.Unavailable) or error == 0:
        return fitted
    ar = reference.Arithmetic(limits)
    try:
        receipt = fitted.receipt
        rows = receipt['rows']
        gram_lower = receipt['gram_eigenvalue_lower']
        perturbation_norm = ar.mul(ar.sqrt(ar.point(rows)), ar.point(error))
        residual_norm = ar.add(ar.sqrt(ar.point(receipt['residual_sum_squares'])), perturbation_norm)
        residual_upper = ar.mul(residual_norm, residual_norm)
        stochastic_radius = ar.sqrt(ar.div(ar.mul(residual_upper, receipt['coefficient_upper']),
                                           ar.mul(receipt['residual_lower'], ar.point(gram_lower))))
        center_error = ar.div(perturbation_norm, ar.sqrt(ar.point(gram_lower)))
        radius = ar.add(stochastic_radius, center_error)
        shape, phase = reference.response_bounds(fitted.coefficients, fitted.positions, radius.upper, ar)
        details = dict(receipt, rounding_error=error, response_perturbation_norm=perturbation_norm,
                       coefficient_center_error=center_error, ideal_residual_upper=residual_upper,
                       rounding_arithmetic=ar.receipt())
        return replace(fitted, error_l2=radius, shape_upper=shape, phase_upper=phase, receipt=details)
    except reference.WorkLimit as failure:
        return reference.Unavailable(str(failure), ar.receipt())


def member_evaluator(source_module, *, scale_mode='geometric', operator_factor=1,
                     rounding_error=Q(0), shape_factor=Q(1)):
    """Build one prespecified calibrated bank member evaluator."""
    if type(rounding_error) is not Q or rounding_error < 0:
        raise ValueError('Supply a nonnegative recording error as a Fraction.')
    if type(shape_factor) is not Q or shape_factor <= 0:
        raise ValueError('Supply a positive shape multiplier as a Fraction.')
    base = source_module.base
    reference = base.reference

    def evaluate(coefficient, polynomials, source, calibration, members, limits):
        ar = reference.Arithmetic(limits)
        try:
            length = polynomials.original_length-1
            if length < 3:
                return base.Unavailable('too_few_transformed_observations', ar.receipt())
            ratio = reference.source_ratio(source.lower, source.upper, coefficient, ar)
            q = ar.mulq(shape_factor, ar.mulq(2, ar.mulq(ratio, max(Q(1), calibration.shape_upper))))
            q = max(Q(1), q)
            variances = tuple(ar.divq(poly.at(coefficient, ar), length) for poly in polynomials.innovation_norms)
            statistic = polynomials.reflection_statistic.at(coefficient, ar)
            transformed_error = ar.mulq(rounding_error, ar.addq(Q(1), abs(coefficient)))
            upper_variances = tuple(rounding_variance_upper(v, transformed_error, ar).upper for v in variances)
            bound, details = reflection_threshold(q, calibration.phase_upper, upper_variances,
                                                  length, members, ar, scale_mode, operator_factor)
            if bound is None:
                return base.Unavailable('nonpositive_scale_denominator', dict(coefficient=coefficient, **details, **ar.receipt()))
            perturbation = rounding_contrast_upper(4*length*variances[0], 4*length*variances[1],
                                                   length-1, (transformed_error, transformed_error), ar)
            bound = ar.add(bound, perturbation)
            details.update(variances=variances, ideal_variance_upper=upper_variances,
                           signed_statistic=statistic, rounding_contrast=perturbation,
                           recording_error=rounding_error, shape_factor=shape_factor,
                           common_filter=(-coefficient, Q(1)), normalization='unnormalized',
                           variance_failure_budget=Q(1, 100*members),
                           null_tail_failure_budget=Q(1, 40*members), **ar.receipt())
            return base.MemberDecision(coefficient, length, length-1, abs(statistic), bound,
                                        reference.strict_reject(abs(statistic), bound.upper), ratio, q, details)
        except reference.WorkLimit as failure:
            return base.Unavailable(str(failure), dict(coefficient=coefficient, **ar.receipt()))
    return evaluate


def supplied_physical_test(source_module, compiled, calibration, q, limits, *,
                           rounding_error=Q(0), scale_mode='geometric', operator_factor=1):
    """Use a physical-family covariance envelope without an AR source estimate."""
    base = source_module.base
    reference = base.reference
    if type(q) is not Q or q < 1 or type(rounding_error) is not Q or rounding_error < 0:
        raise ValueError('Supply exact q >= 1 and a nonnegative exact recording error.')
    if isinstance(calibration, reference.Unavailable):
        return calibration
    base._validate_compiled(compiled, limits)
    if not isinstance(compiled, base.CompiledRecord) or isinstance(compiled.bank_polynomials, base.PhaseFailure):
        return base.Unavailable('recording_moments_unavailable', {})
    polynomials = compiled.bank_polynomials.polynomials
    ar = reference.Arithmetic(limits)
    try:
        length = polynomials.original_length-1
        variances = tuple(ar.divq(poly.at(Q(0), ar), length) for poly in polynomials.innovation_norms)
        statistic = polynomials.reflection_statistic.at(Q(0), ar)
        upper = tuple(rounding_variance_upper(v, rounding_error, ar).upper for v in variances)
        bound, details = reflection_threshold(q, calibration.phase_upper, upper, length, 1, ar,
                                              scale_mode, operator_factor)
        if bound is None:
            return base.Unavailable('nonpositive_scale_denominator', details)
        correction = rounding_contrast_upper(4*length*variances[0], 4*length*variances[1], length-1,
                                              (rounding_error, rounding_error), ar)
        bound = ar.add(bound, correction)
        details.update(signed_statistic=statistic, variances=variances, ideal_variance_upper=upper,
                       rounding_contrast=correction, source='supplied physical-family covariance envelope',
                       recording_sha256=compiled.identity.input_sha256,
                       size_bound=Q(9, 200), **ar.receipt())
        return base.MemberDecision(Q(0), length, length-1, abs(statistic), bound,
                                    reference.strict_reject(abs(statistic), bound.upper), Q(1), q, details)
    except reference.WorkLimit as failure:
        return base.Unavailable(str(failure), ar.receipt())
