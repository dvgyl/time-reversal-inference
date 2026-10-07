"""Compute declared descriptive comparators on the same observations."""

import numpy as np


def _validate(record, alpha, resamples):
    values = np.asarray(record, dtype=float)
    if (values.ndim != 2 or values.shape[1] < 2 or len(values) < 8 or
            not np.isfinite(values).all()):
        raise ValueError('Supply at least eight finite observations from two channels.')
    if not np.isfinite(alpha) or not 0 < alpha < 1 or type(resamples) is not int or resamples < 19:
        raise ValueError('Use 0 < alpha < 1 and at least 19 bootstrap resamples.')
    return values


def _moving_block_mean_pvalue(series, observed, rng, resamples=499):
    values = np.asarray(series, dtype=float)
    if (values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all() or
            not np.isfinite(observed) or type(resamples) is not int or resamples < 19):
        raise ValueError('Bootstrap inputs must be finite and contain at least four values.')
    n = len(values)
    block = max(2, int(np.ceil(n**(1/3))))
    blocks = int(np.ceil(n/block))
    centered = values-values.mean()
    prefix = np.concatenate(([0.], np.cumsum(centered)))
    sums = prefix[block:]-prefix[:-block]
    maxima = len(sums)
    exceedances = 0
    for start in range(0, resamples, 32):
        count = min(32, resamples-start)
        selections = rng.integers(0, maxima, size=(count, blocks))
        means = sums[selections].sum(axis=1)/(blocks*block)
        exceedances += int(np.count_nonzero(np.abs(means) >= abs(observed)))
    return dict(pvalue=(1+exceedances)/(resamples+1), exceedances=exceedances,
                resamples=resamples, block_length=block, sampled_blocks=blocks,
                target='Zero mean of the observed lag-product sequence',
                inference='Centered moving-block bootstrap; approximate and not finite-sample certified')


def lag_baselines(record, rng, *, phase=0., alpha=.05, resamples=499):
    values = _validate(record, alpha, resamples)
    if phase is not None and (not np.isfinite(phase) or not 0 <= phase <= 1):
        raise ValueError('Use a finite phase tolerance between zero and one.')
    # The same retained starts are used by the supplied-envelope procedure.
    values = values[1:, :2]
    values = values-values.mean(axis=0)
    u = values[:-1, 0]+values[1:, 0]
    w = values[:-1, 1]-values[1:, 1]
    u -= u.mean()
    w -= w.mean()
    product = u*w
    if not np.isfinite(product).all():
        raise ValueError('The lag products exceed finite floating-point range.')
    statistic = float(product.mean())
    n = len(product)
    variance_u, variance_w = float(np.mean(u*u)), float(np.mean(w*w))
    log = np.log(2/alpha)
    independent_radius = np.sqrt(variance_u*variance_w)*(np.sqrt(2*(n-1)*log)+log)/n
    marginal = np.mean(values*values, axis=0)
    phase_tolerance = float(2*phase*np.sqrt(np.prod(marginal))) if phase is not None else None
    bootstrap = _moving_block_mean_pvalue(product, statistic, rng, resamples)
    bootstrap.update(status='reject' if bootstrap['pvalue'] <= alpha else 'nonreject',
                     statistic=statistic)
    if phase is None:
        adjusted=dict(status='abstain',reason='phase_certificate_unavailable')
        independent_adjusted=dict(adjusted)
    else:
        adjusted_statistic = max(0., abs(statistic)-phase_tolerance)
        adjusted = _moving_block_mean_pvalue(product, adjusted_statistic, rng, resamples)
        adjusted.update(status='reject' if adjusted['pvalue'] <= alpha else 'nonreject',
                        statistic=statistic, phase_tolerance=phase_tolerance,
                        target='Lag contrast beyond a plug-in response-phase tolerance',
                        inference='Descriptive bootstrap with plug-in variance scale; no finite-sample size claim')
        independent_adjusted=dict(status='reject' if abs(statistic)>phase_tolerance+independent_radius else 'nonreject',
                    statistic=statistic, threshold=float(phase_tolerance+independent_radius),
                    target='Plug-in phase-adjusted lag asymmetry with independent-window calibration')
    return dict(independent_windows=dict(status='reject' if abs(statistic)>independent_radius else 'nonreject',
                                         statistic=statistic, threshold=float(independent_radius),
                                         target='Observed lag asymmetry with independent-window calibration'),
                independent_windows_phase_adjusted=independent_adjusted,
                moving_block=bootstrap, moving_block_phase_adjusted=adjusted)


def imaginary_coherency(record, rng, *, alpha=.05, segment_length=256, resamples=499):
    """Test zero imaginary coherency of observed channels with block resampling."""
    values = _validate(record, alpha, resamples)[:, :2]
    if type(segment_length) is not int or segment_length < 4 or segment_length % 4:
        raise ValueError('The segment length must be a positive multiple of four.')
    segments = len(values)//segment_length
    if segments < 4:
        return dict(status='abstain', reason='fewer_than_four_segments')
    windows = values[:segments*segment_length].reshape(segments, segment_length, 2)
    windows = windows-windows.mean(axis=1, keepdims=True)
    phase = np.exp(-2j*np.pi*(segment_length//4)*np.arange(segment_length)/segment_length)
    coefficient = np.einsum('stc,t->sc', windows, phase)
    cross = coefficient[:, 0]*np.conj(coefficient[:, 1])
    denominator = np.sqrt(np.mean(abs(coefficient[:, 0])**2)*np.mean(abs(coefficient[:, 1])**2))
    if not np.isfinite(cross).all() or not np.isfinite(denominator):
        raise ValueError('The spectral products exceed finite floating-point range.')
    if denominator == 0:
        return dict(status='abstain', reason='zero_spectral_scale')
    series = cross.imag/denominator
    value = float(series.mean())
    result = _moving_block_mean_pvalue(series, value, rng, resamples)
    result.update(status='reject' if result['pvalue'] <= alpha else 'nonreject',
                  imaginary_coherency=value, frequency='pi/2', segment_length=segment_length,
                  segments=segments, target='Zero imaginary coherency of the observed channels',
                  inference='Declared moving-block test of segment contributions; approximate')
    return result
