"""Evaluate a declared cycle grid at exact multiples of pi/4."""

from dataclasses import dataclass, replace
from fractions import Fraction as Q

import cycle_diagonal as cycle
from vendor import cycle_reference as ref
from vendor import reference_certified as arithmetic


@dataclass(frozen=True)
class OctantGrid:
    frequencies: tuple = (1, 2)

    def __post_init__(self):
        if (type(self.frequencies) is not tuple or not 1 <= len(self.frequencies) <= 4 or
                any(type(x) is not int or not 0 <= x <= 4 for x in self.frequencies) or
                len(set(self.frequencies)) != len(self.frequencies)):
            raise ValueError('Supply one to four distinct frequency indices from 0 through 4.')


def _negate(value, ar):
    return ar.sub(ar.point(0), value)


def _square(value, ar):
    lower = Q(0) if value.lower <= 0 <= value.upper else min(value.lower**2, value.upper**2)
    upper = max(value.lower**2, value.upper**2)
    return arithmetic.Interval(ar.q(lower), ar.q(upper))


def _absolute(value, ar):
    low = Q(0) if value.lower <= 0 <= value.upper else min(abs(value.lower), abs(value.upper))
    return arithmetic.Interval(ar.q(low), ar.q(max(abs(value.lower), abs(value.upper))))


def _multiply_complex(a, b, ar):
    return (ar.sub(ar.mul(a[0], b[0]), ar.mul(a[1], b[1])),
            ar.add(ar.mul(a[0], b[1]), ar.mul(a[1], b[0])))


def evaluate_grid(compiled, bounds, perturbation, grid=OctantGrid()):
    if isinstance(compiled, cycle.CompileFailure):
        return dict(status='abstain', reason=compiled.reason, frequencies=[])
    if isinstance(bounds, cycle.MissingBound):
        return dict(status='abstain', reason=bounds.reason, frequencies=[])
    if not isinstance(grid, OctantGrid) or not isinstance(compiled, cycle.CompiledCycle):
        raise TypeError('Supply compiled cycle statistics and a declared octant grid.')
    if not compiled.lag_covariances:
        return dict(status='abstain', reason='lag_covariances_unavailable', frequencies=[])
    if not isinstance(bounds, cycle.DiagonalBounds) or not isinstance(perturbation, cycle.Perturbation):
        raise TypeError('Supply validated bounds and recording errors.')
    output, receipts = {}, []
    # The radius uses only the grid size. Its phase labels do not enter the bound.
    radius_design = replace(compiled.design, frequencies=tuple(ref.QuarterTurn(i) for i in range(len(grid.frequencies))))
    for bits in compiled.design.policy.precision_bits:
        budget = ref._Budget(compiled.design.policy, 'octant_radius_%d' % bits)
        ar = arithmetic.Arithmetic(arithmetic.WorkLimits(
            max_observations=compiled.design.policy.max_record_values,
            max_input_bits=compiled.design.policy.max_integer_bits,
            max_rational_bits=compiled.design.policy.max_rational_bits,
            max_operations=compiled.design.policy.max_phase_operations,
            log_terms=48, sqrt_bits=bits))
        try:
            radii = cycle.radii_for_design(radius_design, compiled.centered_energy, bounds, perturbation, bits, budget)
            half_root = ar.div(ar.sqrt(ar.point(2)), ar.point(2))
            one, zero = ar.point(1), ar.point(0)
            phases = ((one, zero), (half_root, half_root), (zero, one), (_negate(half_root, ar), half_root),
                      (_negate(one, ar), zero), (_negate(half_root, ar), _negate(half_root, ar)),
                      (zero, _negate(one, ar)), (half_root, _negate(half_root, ar)))
            for frequency in grid.frequencies:
                if frequency in output and output[frequency]['status'] != 'abstain':
                    continue
                entries = []
                for row in compiled.lag_covariances:
                    real, imag = ar.point(0), ar.point(0)
                    for k, value in zip(range(-compiled.design.lag, compiled.design.lag+1), row):
                        phase = phases[(-frequency*k) % 8]
                        real = ar.add(real, ar.mul(ar.point(value), phase[0]))
                        imag = ar.add(imag, ar.mul(ar.point(value), phase[1]))
                    entries.append((real, imag))
                magnitudes = [ar.sqrt(ar.add(_square(z[0], ar), _square(z[1], ar))) for z in entries]
                product = _multiply_complex(_multiply_complex(entries[0], entries[1], ar), entries[2], ar)
                statistic = _absolute(product[1], ar)
                threshold = ref.RationalInterval(
                    cycle._product_radius(tuple(x.lower for x in magnitudes), tuple(x.lower for x in radii), budget),
                    cycle._product_radius(tuple(x.upper for x in magnitudes), tuple(x.upper for x in radii), budget))
                status = 'reject' if statistic.lower > threshold.upper else (
                    'nonreject' if statistic.upper <= threshold.lower else 'abstain')
                output[frequency] = dict(frequency_index=frequency, entries=entries,
                                         absolute_imaginary_product=statistic, radii=radii,
                                         threshold=threshold, status=status, precision_bits=bits)
            receipts.append(dict(radius=budget.receipt(), entries=ar.receipt()))
        except (ref._WorkLimit, arithmetic.WorkLimit) as error:
            receipts.append(dict(radius=budget.receipt(error if isinstance(error, ref._WorkLimit) else None),
                                 entries=dict(ar.receipt(), refusal=str(error))))
            return dict(status='reject' if any(x['status']=='reject' for x in output.values()) else 'abstain',
                        reason='resource_limit', detail=str(error), frequencies=list(output.values()),
                        receipts=receipts,
                        input_sha256=compiled.input_sha256, statistics_sha256=compiled.statistics_sha256)
        if all(value['status'] != 'abstain' for value in output.values()):
            break
    status = 'reject' if any(x['status']=='reject' for x in output.values()) else (
        'nonreject' if all(x['status']=='nonreject' for x in output.values()) else 'abstain')
    return dict(status=status, reason='numerical_ambiguity' if status=='abstain' else '',
                frequencies=[output[f] for f in grid.frequencies], receipts=receipts,
                input_sha256=compiled.input_sha256, statistics_sha256=compiled.statistics_sha256)
