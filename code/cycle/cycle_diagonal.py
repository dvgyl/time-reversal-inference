"""Test a spectral cycle with a full diagonal covariance envelope."""

from dataclasses import dataclass, field
from fractions import Fraction as Q
import hashlib
import json
from typing import Tuple, Union

from vendor import cycle_reference as ref


EDGES = ((0, 1), (1, 2), (2, 0))


@dataclass(frozen=True)
class CycleDesign:
    n: int
    lag: int
    frequencies: Tuple[ref.QuarterTurn, ...]
    alpha: Q
    policy: ref.NumericalPolicy = ref.NumericalPolicy()

    def reference_spec(self):
        spec = ref.CycleSpec(self.n, self.lag, self.frequencies, self.alpha,
                             ref.SuppliedBounds(Q(1), Q(0), 'Validation only'),
                             self.policy)
        ref._validate_spec(spec)
        return spec


@dataclass(frozen=True)
class DiagonalBounds:
    covariance: Tuple[Q, Q, Q]
    bias: Tuple[Q, Q, Q]
    source: str

    def __post_init__(self):
        if type(self.covariance) is not tuple or len(self.covariance) != 3:
            raise ValueError('Supply three full-envelope diagonal entries.')
        if type(self.bias) is not tuple or len(self.bias) != 3:
            raise ValueError('Supply three edge bias bounds.')
        if any(type(x) is not Q or x <= 0 for x in self.covariance):
            raise ValueError('Each covariance entry must be a positive Fraction.')
        if any(type(x) is not Q or x < 0 for x in self.bias):
            raise ValueError('Each bias bound must be a nonnegative Fraction.')
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError('State the source of the full covariance bound.')


@dataclass(frozen=True)
class MissingBound:
    reason: str


@dataclass(frozen=True)
class Perturbation:
    error: Tuple[Q, Q, Q] = (Q(0), Q(0), Q(0))
    source: str = 'Exact supplied input values'

    def __post_init__(self):
        if type(self.error) is not tuple or len(self.error) != 3:
            raise ValueError('Supply three recording error bounds.')
        if any(type(x) is not Q or x < 0 for x in self.error):
            raise ValueError('Recording errors must be nonnegative Fractions.')
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError('State the source of the recording error bound.')


@dataclass(frozen=True)
class CompiledCycle:
    design: CycleDesign
    entries: tuple
    centered_energy: Tuple[Q, Q, Q]
    receipts: tuple
    input_sha256: str
    lag_covariances: tuple = ()
    statistics_sha256: str = field(init=False)

    def __post_init__(self):
        self.design.reference_spec()
        if type(self.entries) is not tuple or len(self.entries) != len(self.design.frequencies):
            raise ValueError('Compiled entries must match the frequency grid.')
        for row in self.entries:
            if type(row) is not tuple or len(row) != 3:
                raise ValueError('Each compiled frequency needs three edge entries.')
            if any(type(z) is not ref.ExactComplex or type(z.real) is not Q or type(z.imag) is not Q for z in row):
                raise ValueError('Compiled entries must be exact complex rational values.')
        if (type(self.centered_energy) is not tuple or len(self.centered_energy) != 3 or
                any(type(x) is not Q or x < 0 for x in self.centered_energy)):
            raise ValueError('Supply three nonnegative exact centered energies.')
        if type(self.receipts) is not tuple:
            raise ValueError('Compiled work receipts must be a tuple.')
        if type(self.lag_covariances) is not tuple:
            raise ValueError('Lag covariances must be an immutable tuple.')
        if self.lag_covariances:
            if (type(self.lag_covariances) is not tuple or len(self.lag_covariances) != 3 or
                    any(type(row) is not tuple or len(row) != 2*self.design.lag+1 or
                        any(type(x) is not Q for x in row) for row in self.lag_covariances)):
                raise ValueError('Lag covariances must match the declared lag and three edges.')
        if (not isinstance(self.input_sha256, str) or len(self.input_sha256) != 64 or
                any(x not in '0123456789abcdef' for x in self.input_sha256)):
            raise ValueError('Supply the input SHA256 identity.')
        payload = dict(n=self.design.n, lag=self.design.lag, frequencies=list(self.design.frequencies),
                       alpha=str(self.design.alpha), input_sha256=self.input_sha256,
                       entries=[[[str(z.real), str(z.imag)] for z in row] for row in self.entries],
                       centered_energy=list(map(str, self.centered_energy)),
                       lag_covariances=[list(map(str, row)) for row in self.lag_covariances])
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        object.__setattr__(self, 'statistics_sha256', digest)


@dataclass(frozen=True)
class CompileFailure:
    design: CycleDesign
    reason: str
    receipts: tuple


@dataclass(frozen=True)
class EdgeDecision:
    frequency: ref.QuarterTurn
    entries: tuple
    absolute_imaginary_product: Q
    radii: tuple
    threshold: ref.RationalInterval
    margin: ref.RationalInterval
    status: str
    precision_bits: int


@dataclass(frozen=True)
class CycleDecision:
    status: str
    frequencies: tuple
    receipts: tuple
    reason: str = ''
    input_sha256: str = ''
    statistics_sha256: str = ''


def compile_cycle(record, design):
    """Compute exact recording statistics for a fixed lag and frequency grid."""
    spec = design.reference_spec()
    receipts = []
    budget = ref._Budget(design.policy, 'compile_cycle')
    try:
        budget.limit('record_values', 3 * design.n, design.policy.max_record_values)
        count = 3 * ((2 * design.lag + 2) * design.n - design.lag * (design.lag + 1))
        budget.limit('lag_products', count, design.policy.max_lag_products)
        prepared = ref._prepare(record, spec, budget)
        if prepared is None:
            return CompileFailure(design, 'nonfinite_record', (budget.receipt(refusal='nonfinite_record'),))
        lags = ref._lag_statistics(prepared, spec, budget)
        columns, powers, sums, maxima = prepared
        energies = []
        for i in range(3):
            square = budget.dot(columns[i], columns[i], 0, 0, design.n, maxima[i], maxima[i])
            numerator = budget.add(budget.mul(design.n, square), -budget.mul(sums[i], sums[i]))
            denominator = budget.mul(design.n, budget.shift(1, 2 * powers[i]))
            energies.append(budget.rational(numerator, denominator))
        entries = tuple(ref._entries(lags, spec, f, budget) for f in design.frequencies)
    except ref._WorkLimit as error:
        receipts.append(budget.receipt(error))
        return CompileFailure(design, 'resource_limit', tuple(receipts))
    receipts.append(budget.receipt())
    digest = hashlib.sha256()
    digest.update(b'binary64-hex-record-v1\n')
    for row in record:
        digest.update((' '.join(x.hex() for x in row)+'\n').encode())
    return CompiledCycle(design, entries, tuple(energies), tuple(receipts), digest.hexdigest(), lags)


def _add(a, b, budget):
    return ref.RationalInterval(budget.qadd(a.lower, b.lower), budget.qadd(a.upper, b.upper))


def _multiply(a, b, budget):
    if min(a.lower, b.lower) < 0:
        raise ValueError('Positive interval multiplication requires nonnegative endpoints.')
    return ref.RationalInterval(budget.qmul(a.lower, b.lower), budget.qmul(a.upper, b.upper))


def _point(x):
    return ref.RationalInterval(x, x)


def _radii(compiled, bounds, perturbation, bits, budget):
    return radii_for_design(compiled.design, compiled.centered_energy, bounds, perturbation, bits, budget)


def radii_for_design(design, centered_energy, bounds, perturbation, bits, budget):
    spec = design.reference_spec()
    common = ref._radius_interval(spec, bits, budget)
    deviations = tuple(ref._sqrt_interval(budget.qdiv(e, Q(design.n)), bits, budget)
                       for e in centered_energy)
    result = []
    for edge, (i, j) in enumerate(EDGES):
        scale = ref._sqrt_interval(budget.qmul(bounds.covariance[i], bounds.covariance[j]), bits, budget)
        value = _add(_point(bounds.bias[edge]), _multiply(scale, common, budget), budget)
        rounding = _add(_multiply(_point(perturbation.error[i]), deviations[j], budget),
                        _multiply(_point(perturbation.error[j]), deviations[i], budget), budget)
        rounding = _add(rounding, _point(budget.qmul(perturbation.error[i], perturbation.error[j])), budget)
        rounding = _multiply(_point(Q(2 * design.lag + 1)), rounding, budget)
        result.append(_add(value, rounding, budget))
    return tuple(result)


def _product_radius(magnitudes, errors, budget):
    total = Q(0)
    for mask in range(1, 8):
        term = Q(1)
        for i in range(3):
            term = budget.qmul(term, errors[i] if mask & (1 << i) else magnitudes[i])
        total = budget.qadd(total, term)
    return total


def _frequency_decision(frequency, entries, radii, bits, budget):
    product = ref._complex_product(ref._complex_product(entries[0], entries[1], budget), entries[2], budget)
    statistic = abs(product.imag)
    magnitudes = []
    for value in entries:
        square = budget.qadd(budget.qmul(value.real, value.real), budget.qmul(value.imag, value.imag))
        magnitudes.append(ref._sqrt_interval(square, bits, budget))
    threshold = ref.RationalInterval(
        _product_radius(tuple(x.lower for x in magnitudes), tuple(x.lower for x in radii), budget),
        _product_radius(tuple(x.upper for x in magnitudes), tuple(x.upper for x in radii), budget))
    margin = ref.RationalInterval(budget.qadd(statistic, -threshold.upper),
                                  budget.qadd(statistic, -threshold.lower))
    status = ref._decision(statistic, threshold, budget)
    return EdgeDecision(frequency, entries, statistic, radii, threshold, margin, status, bits)


def evaluate_cycle(compiled: Union[CompiledCycle, CompileFailure],
                   bounds: Union[DiagonalBounds, MissingBound],
                   perturbation=Perturbation()):
    """Apply supplied bounds to a recording without changing the test design."""
    if isinstance(compiled, CompileFailure):
        return CycleDecision('abstain', (), compiled.receipts, compiled.reason)
    if not isinstance(compiled, CompiledCycle):
        raise TypeError('Supply a compiled recording or its failure receipt.')
    if isinstance(bounds, MissingBound):
        return CycleDecision('abstain', (), (), bounds.reason, compiled.input_sha256, compiled.statistics_sha256)
    if not isinstance(bounds, DiagonalBounds) or not isinstance(perturbation, Perturbation):
        raise TypeError('Supply validated bounds and a perturbation certificate.')
    receipts, results = [], [None] * len(compiled.design.frequencies)
    for bits in compiled.design.policy.precision_bits:
        budget = ref._Budget(compiled.design.policy, 'diagonal_radius_%d' % bits)
        try:
            radii = _radii(compiled, bounds, perturbation, bits, budget)
            for i, (frequency, entries) in enumerate(zip(compiled.design.frequencies, compiled.entries)):
                if results[i] is None or results[i].status == 'abstain':
                    results[i] = _frequency_decision(frequency, entries, radii, bits, budget)
        except ref._WorkLimit as error:
            receipts.append(budget.receipt(error))
            decided = tuple(x for x in results if x is not None)
            status = 'reject' if any(x.status == 'reject' for x in decided) else 'abstain'
            return CycleDecision(status, decided, tuple(receipts), 'resource_limit', compiled.input_sha256, compiled.statistics_sha256)
        receipts.append(budget.receipt())
        if all(x.status != 'abstain' for x in results):
            break
    status = 'reject' if any(x.status == 'reject' for x in results) else (
        'nonreject' if all(x.status == 'nonreject' for x in results) else 'abstain')
    return CycleDecision(status, tuple(results), tuple(receipts),
                         'numerical_ambiguity' if status == 'abstain' else '', compiled.input_sha256, compiled.statistics_sha256)


def transport_gains(compiled, bounds, perturbation, gains):
    """Transport exact statistics and bounds under nonzero rational gains."""
    if len(gains) != 3 or any(type(x) is not Q or x == 0 for x in gains):
        raise ValueError('Supply three nonzero rational gains.')
    if not isinstance(compiled, CompiledCycle) or not isinstance(bounds, DiagonalBounds) or not isinstance(perturbation, Perturbation):
        raise TypeError('Supply compiled statistics and validated bound objects.')
    budget = ref._Budget(compiled.design.policy, 'transport_gains')
    try:
        for value in gains:
            budget.check_fraction(value)
        edge_gains = tuple(budget.qmul(gains[i], gains[j]) for i, j in EDGES)
        squares = tuple(budget.qmul(a, a) for a in gains)
        entries = tuple(tuple(ref.ExactComplex(budget.qmul(z.real, a), budget.qmul(z.imag, a))
                              for z, a in zip(row, edge_gains)) for row in compiled.entries)
        lags = tuple(tuple(budget.qmul(value, gain) for value in row)
                     for row, gain in zip(compiled.lag_covariances, edge_gains))
        energies = tuple(budget.qmul(e, a) for e, a in zip(compiled.centered_energy, squares))
        bound = DiagonalBounds(tuple(budget.qmul(k, a) for k, a in zip(bounds.covariance, squares)),
                               tuple(budget.qmul(b, abs(a)) for b, a in zip(bounds.bias, edge_gains)),
                               bounds.source + '; transported channel units')
        rounding = Perturbation(tuple(budget.qmul(e, abs(a)) for e, a in zip(perturbation.error, gains)),
                                perturbation.source + '; transported channel units')
        digest = hashlib.sha256(json.dumps(dict(input_sha256=compiled.input_sha256,
                                                gains=list(map(str, gains))), sort_keys=True).encode()).hexdigest()
        changed = CompiledCycle(compiled.design, entries, energies, compiled.receipts+(budget.receipt(),), digest, lags)
        return changed, bound, rounding
    except ref._WorkLimit as error:
        failure = CompileFailure(compiled.design, 'resource_limit', compiled.receipts+(budget.receipt(error),))
        return failure, MissingBound('Gain transport exceeded its work limit'), perturbation
