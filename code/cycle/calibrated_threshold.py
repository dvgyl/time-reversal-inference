"""Calculate a reflection threshold with separate variance ceilings."""

from fractions import Fraction as Q


def reflection_threshold(q, phase, variances, length, members, ar,
                         scale_mode='geometric', operator_factor=1):
    """Return an outward threshold and its component bounds.

    The supplied q bounds the full covariance after channel standardization.
    The two variances are exact observed centered second moments.
    """
    if type(q) is not Q or q < 1 or type(phase) is not Q or phase < 0:
        raise ValueError('Supply q >= 1 and a nonnegative phase bound as Fractions.')
    if len(variances) != 2 or any(type(v) is not Q or v < 0 for v in variances):
        raise ValueError('Supply two nonnegative centered variances as Fractions.')
    if type(length) is not int or length < 3 or type(members) is not int or members < 1:
        raise ValueError('Supply a valid retained length and bank size.')
    if scale_mode not in ('geometric', 'maximum') or operator_factor not in (1, 2):
        raise ValueError('Use a declared scale and operator-bound method.')
    variance_log = ar.log(Q(200 * members))
    tail_log = ar.log(Q(80 * members))
    ratio = ar.div(ar.point(q), ar.point(length))
    denominator = ar.sub(ar.sub(ar.point(1), ratio),
                         ar.mul(ar.point(2), ar.sqrt(ar.mul(ratio, variance_log))))
    details = dict(scale_denominator=denominator, variance_log=variance_log,
                   tail_log=tail_log, scale_mode=scale_mode,
                   operator_factor=operator_factor, observed_variances=variances)
    if denominator.lower <= 0:
        return None, details
    if scale_mode == 'geometric':
        observed_scale = ar.sqrt(ar.point(ar.mulq(variances[0], variances[1])))
    else:
        observed_scale = ar.point(max(variances))
    scale = ar.div(observed_scale, denominator)
    n = length - 1
    frobenius_square = Q(n + 1, n * n) - Q(2, n**3) - Q(2, n**4)
    covariance = ar.mul(ar.point(q), scale)
    frobenius = ar.mul(covariance, ar.sqrt(ar.point(frobenius_square)))
    mixed = ar.div(ar.mul(ar.point(operator_factor),
                         ar.sqrt(ar.mul(ar.mul(covariance, ar.point(length)),
                                        ar.mul(ar.point(2), scale)))), ar.point(n))
    interval_type = type(scale)
    minimum = interval_type(min(frobenius.lower, mixed.lower),
                             min(frobenius.upper, mixed.upper))
    sampling = ar.add(ar.mul(ar.mul(ar.point(2), minimum), ar.sqrt(tail_log)),
                      ar.div(ar.mul(ar.mul(ar.point(2 * operator_factor), covariance),
                                    tail_log), ar.point(n)))
    tolerance = ar.mul(ar.point(2 * phase), scale)
    result = ar.add(tolerance, sampling)
    details.update(variance_scale=scale, null_tolerance=tolerance,
                   sampling_radius=sampling, threshold_excess_upper=ar.q(result.width))
    return result, details


def rounding_variance_upper(observed_variance, error, ar):
    """Bound the ideal variance from a uniform channel error certificate."""
    if type(observed_variance) is not Q or type(error) is not Q or min(observed_variance, error) < 0:
        raise ValueError('Variance and recording error must be nonnegative Fractions.')
    norm = ar.add(ar.sqrt(ar.point(observed_variance)), ar.point(error))
    return ar.mul(norm, norm)


def rounding_contrast_upper(u_energy, w_energy, n, errors, ar):
    """Bound a reflection contrast error from centered projection energies."""
    if (type(u_energy) is not Q or type(w_energy) is not Q or
            type(errors) is not tuple or len(errors) != 2 or
            any(type(x) is not Q or x < 0 for x in (u_energy, w_energy, *errors)) or
            type(n) is not int or n < 2):
        raise ValueError('Supply exact nonnegative energies, two exact errors, and an integer length.')
    first = ar.mul(ar.point(2 * errors[0]), ar.sqrt(ar.point(w_energy / n)))
    second = ar.mul(ar.point(2 * errors[1]), ar.sqrt(ar.point(u_energy / n)))
    return ar.add(ar.add(first, second), ar.point(4 * errors[0] * errors[1]))
