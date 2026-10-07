"""Bound a two-channel recording with supplied AR(1) marginal spectra."""
import math
from numbers import Integral


def _finite(value, name):
    if not math.isfinite(value):
        raise ArithmeticError(f"The computed {name} is not finite.")
    return value


def _vector(values, name):
    try:
        result = tuple(float(x) for x in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real sequence.") from error
    if not result or not all(math.isfinite(x) for x in result):
        raise ValueError(f"{name} must be a nonempty finite real sequence.")
    return result


def ar1_prefilter_parameters(phi, error_kind='zero'):
    """Report the common filter for a supplied exact AR(1) parameter.

    Apply the coefficients at positions zero and one to each channel.
    An N-sample recording gives N-1 transformed samples. White errors
    change variance. These values do not assert unchanged testing power.
    """
    phi = float(phi)
    if not math.isfinite(phi) or not 0 <= phi < 1:
        raise ValueError("phi must be supplied in [0, 1).")
    if error_kind not in ('zero', 'white'):
        raise ValueError("This helper covers zero or white errors only.")
    normalization = math.sqrt(1-phi**2)
    return dict(phi=phi, positions=[0, 1],
                coefficients=[-phi/normalization, 1/normalization],
                boundary_samples_lost=1,
                mean_multiplier=(1-phi)/normalization,
                error_variance_multiplier=(1+phi**2)/(1-phi**2),
                transformed_error_peak_ratio=1.0 if error_kind == 'zero'
                else (1+phi)**2/(1+phi**2))


def ar1_pair_envelope(coefficients, radii, phi, *, positions=None,
                      error_peak_ratios=(1.0, 1.0), broad_rho=None,
                      planning=False):
    """Return an envelope under supplied marginal and error assumptions.

    Supply exactly two coefficient vectors and their l2 confidence radii.
    The common AR(1) parameter is supplied exactly. Source marginal scales
    can be unknown. Source and error spectra must satisfy the assumptions
    for the stated source class. Errors are orthogonal to each other and the source.
    The broad envelope, if supplied, must cover the same coefficient event.

    Fitted coefficients use planning=False. Declared true coefficients and
    upper confidence radii use planning=True. The latter doubles the radius.
    An unavailable result has rho=None. It does not authorize a test.
    Non-finite derived arithmetic raises ArithmeticError before a bound
    is returned. Numerically unresolved nonzero variances also raise it.
    """
    if not isinstance(planning, bool):
        raise ValueError("planning must be a Boolean.")
    try:
        phi = float(phi)
    except (TypeError, ValueError) as error:
        raise ValueError("phi must be supplied in [0, 1).") from error
    if not math.isfinite(phi) or not 0 <= phi < 1:
        raise ValueError("phi must be supplied in [0, 1).")
    if len(coefficients) != 2:
        raise ValueError("Supply exactly two coefficient vectors.")
    fitted = tuple(_vector(x, "coefficients") for x in coefficients)
    radii = _vector(radii, "radii")
    errors = _vector(error_peak_ratios, "error_peak_ratios")
    if len(radii) != 2 or min(radii) < 0:
        raise ValueError("Supply two nonnegative confidence radii.")
    if len(errors) != 2 or min(errors) < 1:
        raise ValueError("Supply two error peak ratios of at least one.")
    if positions is None:
        positions = tuple(tuple(range(len(x))) for x in fitted)
    if len(positions) != 2:
        raise ValueError("Supply two position sequences.")
    checked_positions = []
    for values, response in zip(positions, fitted):
        values = tuple(values)
        if (len(values) != len(response)
                or any(isinstance(x, bool) or not isinstance(x, Integral) for x in values)
                or len(set(values)) != len(values)):
            raise ValueError("Each response must have distinct integer positions.")
        checked_positions.append(values)
    if broad_rho is not None:
        broad_rho = float(broad_rho)
        if not math.isfinite(broad_rho) or broad_rho < max(1.0, max(errors)):
            raise ValueError("The broad envelope must be finite and cover the error ratios.")
    gmax = (1 + phi) / (1 - phi)
    channels = []
    for h, radius, locations in zip(fitted, radii, checked_positions):
        covariance = [[phi ** abs(a-b) for b in locations] for a in locations]
        variance = math.fsum(h[i] * h[j] * covariance[i][j]
                             for i in range(len(h)) for j in range(len(h)))
        # The covariance is positive definite at distinct positions.
        _finite(variance, "AR variance")
        if variance < 0:
            raise ArithmeticError("The computed AR variance is negative.")
        if variance == 0 and any(x != 0 for x in h):
            raise ArithmeticError("The nonzero response variance is unresolved.")
        eigenvalue_upper = min(gmax, max(math.fsum(row) for row in covariance))
        norm = math.sqrt(variance)
        applied_radius = _finite(radius * (2 if planning else 1), "confidence radius")
        lower_norm = _finite(norm - math.sqrt(eigenvalue_upper) * applied_radius, "lower norm")
        amplitude_upper = math.fsum(abs(x) for x in h) + math.sqrt(len(h))*applied_radius
        _finite(amplitude_upper, "amplitude upper bound")
        numerator = _finite(gmax * amplitude_upper**2, "numerator upper bound")
        if amplitude_upper != 0 and numerator == 0:
            raise ArithmeticError("The nonzero numerator is unresolved.")
        if all(x == 0 for x in h) and radius == 0:
            ratio, status = 1.0, "zero_response"
        elif lower_norm <= 0:
            ratio, status = None, "nonpositive_lower_norm"
        else:
            denominator = _finite(lower_norm**2, "squared lower norm")
            if denominator == 0:
                raise ArithmeticError("The positive lower norm is unresolved.")
            quotient = _finite(numerator/denominator, "peak ratio")
            ratio = max(1.0, quotient)
            status = "finite"
        channels.append(dict(length=len(h), positions=list(locations),
                             variance=variance, covariance_norm=norm,
                             confidence_radius=radius, applied_radius=applied_radius,
                             eigenvalue_upper=eigenvalue_upper, lower_norm=lower_norm,
                             numerator_upper=numerator, ratio_upper=ratio, status=status))
    available = all(x["ratio_upper"] is not None for x in channels)
    uncapped = max(2*max(x["ratio_upper"] for x in channels), max(errors)) if available else None
    if uncapped is not None:
        _finite(uncapped, "pair envelope")
    if uncapped is None:
        rho = broad_rho
    else:
        rho = uncapped if broad_rho is None else min(uncapped, broad_rho)
    return dict(phi=phi, planning=planning, channels=channels,
                error_peak_ratios=list(errors), broad_rho=broad_rho,
                marginal_rho_uncapped=uncapped, rho=rho,
                marginal_available=available,
                broad_only_finite_guarantee=not available and broad_rho is not None,
                broad_cap_used=broad_rho is not None and (uncapped is None or broad_rho < uncapped))
