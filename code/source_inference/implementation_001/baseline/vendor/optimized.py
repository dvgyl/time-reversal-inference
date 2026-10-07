"""Use exact shared recording polynomials with the frozen statistical rule."""
from dataclasses import dataclass
from fractions import Fraction as Q
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Optional, Tuple, Union

REFERENCE_SHA256 = 'b99bf999b31dd4f0760c6e20866251a59b530275413479a5c2e6dcd7ba6d9d9f'
_original_reference_path = Path(__file__).with_name('reference_certified.py')
_reference_path = Path(__file__).with_name('reference_certified.py')
if not _reference_path.exists():
    _reference_path = _original_reference_path
if hashlib.sha256(_reference_path.read_bytes()).hexdigest() != REFERENCE_SHA256:
    raise RuntimeError('The frozen reference source identity changed.')
_allowed_reference_paths = {_reference_path.resolve(), _original_reference_path.resolve()}


def _verified_module(module):
    filename = getattr(module, '__file__', None)
    if not filename:
        return False
    path = Path(filename).resolve()
    return path in _allowed_reference_paths and hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA256


_module_name = '_estimated_source_012_frozen_reference_001'
_public_reference = sys.modules.get('certified')
if _public_reference is not None and _verified_module(_public_reference):
    reference = _public_reference
    if _module_name in sys.modules and sys.modules[_module_name] is not reference:
        raise RuntimeError('Two different frozen reference module instances are already loaded.')
    sys.modules[_module_name] = reference
elif _module_name in sys.modules:
    reference = sys.modules[_module_name]
    if not _verified_module(reference):
        raise RuntimeError('The cached reference module has an unknown identity.')
else:
    _spec = importlib.util.spec_from_file_location(_module_name, _reference_path)
    reference = importlib.util.module_from_spec(_spec)
    sys.modules[_module_name] = reference
    _spec.loader.exec_module(reference)
if _public_reference is None:
    sys.modules['certified'] = reference

Arithmetic = reference.Arithmetic
BankPlan = reference.BankPlan
BankResult = reference.BankResult
CalibrationData = reference.CalibrationData
InputError = reference.InputError
Interval = reference.Interval
MemberDecision = reference.MemberDecision
Quadratic = reference.Quadratic
SourceEmpty = reference.SourceEmpty
SourceFallback = reference.SourceFallback
SourceHull = reference.SourceHull
Unavailable = reference.Unavailable
WorkLimit = reference.WorkLimit
WorkLimits = reference.WorkLimits


@dataclass(frozen=True)
class PreparedRecord:
    values: Tuple[Tuple[Q, Q], ...]
    common_dyadic_exponent: Optional[int]
    input_sha256: str
    reference_input_sha256: str
    receipt: dict


@dataclass(frozen=True)
class RawMoments:
    channel_sums: Tuple[Union[int, Q], Union[int, Q]]
    channel_squared_sums: Tuple[Union[int, Q], Union[int, Q]]
    channel_adjacent_sums: Tuple[Union[int, Q], Union[int, Q]]
    reference_lag_two_sum: Union[int, Q]
    reflection_raw_coefficients: Tuple[Union[int, Q], ...]
    first_rows: tuple
    last_rows: tuple


@dataclass(frozen=True)
class RecordingPolynomials:
    original_length: int
    innovation_norms: Tuple[Quadratic, Quadratic]
    reference_lag: Quadratic
    reflection_statistic: Quadratic
    input_sha256: str
    reference_input_sha256: str
    receipt: dict


class MomentArithmetic:
    def __init__(self, ar, exponent):
        self.ar = ar
        self.exponent = exponent
        self.integer = exponent is not None
        self.maximum_integer_bits = 0
        self.denominators = None
        if self.integer:
            for bits in (exponent+1, 2*exponent+1):
                if bits > ar.limits.max_rational_bits:
                    raise WorkLimit('dyadic_denominator_bits')
            self.tick()
            first = self.check(1 << exponent)
            self.tick()
            second = self.check(1 << (2*exponent))
            self.denominators = (first, second)

    def tick(self):
        self.ar.operations += 1
        if self.ar.operations > self.ar.limits.max_operations:
            raise WorkLimit('max_operations')

    def check(self, value):
        bits = abs(value).bit_length()
        self.maximum_integer_bits = max(self.maximum_integer_bits, bits)
        self.ar.maximum_integer_bits = self.maximum_integer_bits
        if bits > self.ar.limits.max_rational_bits:
            raise WorkLimit('max_integer_bits')
        return value

    def convert(self, value):
        if not self.integer:
            return self.ar.q(value)
        self.tick()
        shift = self.exponent-(value.denominator.bit_length()-1)
        if shift < 0 or value.denominator & (value.denominator-1):
            raise InputError('The prepared record is not dyadic at the declared exponent.')
        bits = abs(value.numerator).bit_length()+shift if value.numerator else 0
        if bits > self.ar.limits.max_rational_bits:
            raise WorkLimit('converted_integer_bits')
        return self.check(value.numerator << shift)

    def add(self, left, right):
        if not self.integer:
            return self.ar.addq(left, right)
        self.tick()
        return self.check(left+right)

    def sub(self, left, right):
        if not self.integer:
            return self.ar.q(left-right)
        self.tick()
        return self.check(left-right)

    def mul(self, left, right):
        if not self.integer:
            return self.ar.mulq(left, right)
        self.tick()
        return self.check(left*right)

    def fraction(self, value, degree):
        if not self.integer:
            return self.ar.q(value)
        return self.ar.q(Q(value, self.denominators[degree-1]))


def _prepare(record, ar):
    size = reference.count(record, ar.limits.max_observations, 'recording_rows')
    values = []
    exponent, dyadic = 0, True
    for i in range(size):
        if reference.count(record[i], 2, 'recording_channels') != 2:
            raise InputError('Each recording row must contain two channels.')
        row = (ar.input(record[i][0]), ar.input(record[i][1]))
        for value in row:
            if value.denominator & (value.denominator-1):
                dyadic = False
            else:
                exponent = max(exponent, value.denominator.bit_length()-1)
        values.append(row)
    values = tuple(values)
    return PreparedRecord(values, exponent if dyadic else None,
                          reference.digest(x for row in values for x in row),
                          reference.digest(row[0] for row in values), ar.receipt())


def prepare_record(record, limits=WorkLimits()):
    return _prepare(record, Arithmetic(limits))


def _raw_moments(values, ring):
    zero = ring.convert(Q(0))
    sums, squares, adjacent = [zero, zero], [zero, zero], [zero, zero]
    lag_two, coefficients = zero, [zero, zero, zero]
    first = []
    older, previous = None, None
    for supplied in values:
        current = tuple(ring.convert(x) for x in supplied)
        if len(first) < 2:
            first.append(current)
        for j in range(2):
            sums[j] = ring.add(sums[j], current[j])
            squares[j] = ring.add(squares[j], ring.mul(current[j], current[j]))
            if previous is not None:
                adjacent[j] = ring.add(adjacent[j], ring.mul(previous[j], current[j]))
        if older is not None:
            lag_two = ring.add(lag_two, ring.mul(older[0], current[0]))
            u = ring.add(previous[0], current[0])
            v = ring.add(older[0], previous[0])
            r = ring.sub(previous[1], current[1])
            s = ring.sub(older[1], previous[1])
            coefficients[0] = ring.add(coefficients[0], ring.mul(u, r))
            coefficients[1] = ring.sub(coefficients[1], ring.add(ring.mul(u, s), ring.mul(v, r)))
            coefficients[2] = ring.add(coefficients[2], ring.mul(v, s))
        older, previous = previous, current
    return RawMoments(tuple(sums), tuple(squares), tuple(adjacent), lag_two,
                      tuple(coefficients), tuple(first), (older, previous))


def _polynomials(record, ar, accumulation):
    if not isinstance(record, PreparedRecord):
        raise InputError('Supply a PreparedRecord.')
    size = reference.count(record.values, ar.limits.max_observations, 'recording_rows')
    if size < 3:
        raise InputError('Recording polynomials need at least three rows.')
    if accumulation not in ('auto', 'rational', 'dyadic'):
        raise InputError('Use auto, rational, or dyadic accumulation.')
    if accumulation == 'dyadic' and record.common_dyadic_exponent is None:
        raise InputError('Forced dyadic accumulation requires dyadic inputs.')
    exponent = record.common_dyadic_exponent if accumulation != 'rational' else None
    ring = MomentArithmetic(ar, exponent)
    raw = _raw_moments(record.values, ring)
    sums = tuple(ring.fraction(x, 1) for x in raw.channel_sums)
    squares = tuple(ring.fraction(x, 2) for x in raw.channel_squared_sums)
    adjacent = tuple(ring.fraction(x, 2) for x in raw.channel_adjacent_sums)
    lag_two = ring.fraction(raw.reference_lag_two_sum, 2)
    coefficients = tuple(ring.fraction(x, 2) for x in raw.reflection_raw_coefficients)
    first = tuple(tuple(ring.fraction(x, 1) for x in row) for row in raw.first_rows)
    last = tuple(tuple(ring.fraction(x, 1) for x in row) for row in raw.last_rows)
    n, m = size-1, size-2
    norms = []
    totals = []
    for j in range(2):
        x, z = ar.q(sums[j]-first[0][j]), ar.q(sums[j]-last[1][j])
        totals.append((x, -z))
        constant = ar.q(squares[j]-ar.mulq(first[0][j], first[0][j])-ar.divq(ar.mulq(x, x), n))
        linear = ar.q(-2*adjacent[j]+ar.divq(ar.mulq(2*x, z), n))
        square = ar.q(squares[j]-ar.mulq(last[1][j], last[1][j])-ar.divq(ar.mulq(z, z), n))
        norms.append(Quadratic(constant, linear, square))
    raw_lag = Quadratic(ar.q(adjacent[0]-ar.mulq(first[0][0], first[1][0])),
                        ar.q(-squares[0]+ar.mulq(first[0][0], first[0][0])+ar.mulq(last[1][0], last[1][0])-lag_two),
                        ar.q(adjacent[0]-ar.mulq(last[0][0], last[1][0])))
    total_square = reference.product_linear(totals[0], totals[0], ar)
    endpoints = (ar.addq(first[1][0], last[1][0]), ar.q(-first[0][0]-last[0][0]))
    lag = reference.combine(raw_lag, total_square, Q(-n-1, n*n), ar)
    lag = reference.combine(lag, reference.product_linear(totals[0], endpoints, ar), Q(1, n), ar)
    u = ar.q(2*sums[0]-2*first[0][0]-first[1][0]-last[1][0])
    v = ar.q(2*sums[0]-first[0][0]-last[0][0]-2*last[1][0])
    r, s = ar.q(first[1][1]-last[1][1]), ar.q(first[0][1]-last[0][1])
    raw_statistic = Quadratic(*(ar.divq(x, m) for x in coefficients))
    means = reference.product_linear((u, -v), (r, -s), ar)
    statistic = reference.combine(raw_statistic, means, Q(-1, m*m), ar)
    receipt = dict(engine='dyadic' if ring.integer else 'rational', common_dyadic_exponent=exponent,
                   maximum_integer_bits=ring.maximum_integer_bits, **ar.receipt())
    return RecordingPolynomials(size, tuple(norms), lag, statistic, record.input_sha256,
                                record.reference_input_sha256, receipt)


def recording_polynomials(record, limits=WorkLimits(), accumulation='auto'):
    return _polynomials(record, Arithmetic(limits), accumulation)


def _source(polynomials, radius, ar):
    n = polynomials.original_length-1
    denominator, lag = polynomials.innovation_norms[0], polynomials.reference_lag
    if denominator == Quadratic(Q(0), Q(0), Q(0)):
        return SourceFallback('identically_zero_innovation_denominator', ar.receipt())
    mu = Q(-1, n)
    q1 = reference.combine(lag, denominator, ar.q(-mu-radius.upper), ar)
    q2 = reference.combine(Quadratic(Q(0), Q(0), Q(0)), lag, Q(-1), ar)
    q2 = reference.combine(q2, denominator, ar.q(mu-radius.upper), ar)
    kept, unresolved, violation, receipt = reference.enclose_polynomials(q1, q2, ar)
    if not kept:
        receipt.update(ar.receipt())
        return SourceEmpty(radius, receipt)
    dmin = denominator.range(Interval(Q(0), Q(1)), ar).lower
    excess = ar.divq(violation, dmin) if dmin > 0 else None
    extra = ar.mulq(2, ar.addq(radius.width, excess)) if excess is not None else None
    receipt.update(denominator_minimum=dmin, polynomial_violation_upper=violation,
                   q1=q1, q2=q2, input_sha256=polynomials.reference_input_sha256, **ar.receipt())
    return SourceHull(min(c.lower for c in kept), max(c.upper for c in kept), radius, denominator, lag,
                      tuple(kept), tuple(unresolved), excess, extra, receipt)


def _member(coefficient, polynomials, source, calibration, members, limits):
    ar = Arithmetic(limits)
    try:
        length = polynomials.original_length-1
        if length < 3:
            return Unavailable('too_few_transformed_observations', ar.receipt())
        ratio = reference.source_ratio(source.lower, source.upper, coefficient, ar)
        q = ar.mulq(2, ar.mulq(ratio, max(Q(1), calibration.shape_upper)))
        variances = tuple(ar.divq(poly.at(coefficient, ar), length) for poly in polynomials.innovation_norms)
        statistic = polynomials.reflection_statistic.at(coefficient, ar)
        bound, details = reference.threshold(q, calibration.phase_upper, max(variances), length, members, ar)
        if bound is None:
            return Unavailable('nonpositive_scale_denominator', dict(coefficient=coefficient, **details, **ar.receipt()))
        details.update(variances=variances, signed_statistic=statistic,
                       common_filter=(-coefficient, Q(1)), normalization='unnormalized',
                       normalized_quadratic_divisor=ar.q(1-coefficient*coefficient),
                       variance_failure_budget=Q(1, 100*members), null_tail_failure_budget=Q(1, 40*members), **ar.receipt())
        return MemberDecision(coefficient, length, length-1, abs(statistic), bound,
                              reference.strict_reject(abs(statistic), bound.upper), ratio, q, details)
    except WorkLimit as error:
        return Unavailable(str(error), dict(coefficient=coefficient, **ar.receipt()))


def _finish(decision, source, calibration, members, receipt, phases):
    receipt['phase_operations'] = phases.copy()
    receipt['total_executed_operations'] = sum(phases[k] for k in ('input','shared_moments','source','calibration'))+sum(phases['members'])
    return BankResult(decision, source, calibration, tuple(members), receipt)


def run_bank(plan, record, calibration, accumulation='auto'):
    if not isinstance(plan, BankPlan):
        raise InputError('Construct BankPlan before reading the recording.')
    if accumulation not in ('auto','rational','dyadic'):
        raise InputError('Use auto, rational, or dyadic accumulation.')
    phases = dict(input=0, shared_moments=0, source=0, calibration=0, members=[0 for _ in plan.coefficients])
    receipt = dict(bank=plan.coefficients, source_budget=plan.source_budget,
                   calibration_budget=plan.calibration_budget, variance_budget=plan.variance_budget,
                   tail_budget=plan.tail_budget, size_bound=Q(1, 20),
                   power_status='POWER_UNVERIFIED', statistic_loss=Q(0), accumulation=accumulation,
                   reference_sha256=REFERENCE_SHA256,
                   assumptions='Supplied common Gaussian AR(1) family; nonzero scalar reference; nonzero finite second response; zero testing errors; zero-mean spherical Gaussian calibration error; fixed full-rank design; no calibration intercept; exact-input arithmetic scope.')
    ar = Arithmetic(plan.limits)
    try:
        prepared = _prepare(record, ar)
        receipt.update(record_length=len(prepared.values), recording_sha256=prepared.input_sha256, **ar.receipt())
    except WorkLimit as error:
        phases['input'] = ar.operations
        missing = Unavailable(str(error), ar.receipt())
        return _finish('ABSTAIN', SourceFallback(str(error), ar.receipt()), missing,
                       [missing for _ in plan.coefficients], receipt, phases)
    phases['input'] = ar.operations
    if accumulation == 'dyadic' and prepared.common_dyadic_exponent is None:
        raise InputError('Forced dyadic accumulation requires dyadic inputs.')
    source_ar = Arithmetic(plan.limits)
    moment_ar = Arithmetic(plan.limits)
    polynomials = None
    try:
        radius = reference.acceptance_radius(len(prepared.values)-1, source_ar)
        if radius is None:
            source = SourceFallback('small_record_or_nonpositive_radius_denominator', source_ar.receipt())
        else:
            try:
                polynomials = _polynomials(prepared, moment_ar, accumulation)
                receipt['shared_moments'] = polynomials.receipt
            except WorkLimit as error:
                source = SourceFallback(str(error), dict(failed_phase='shared_moments', **source_ar.receipt()))
                receipt['shared_moments'] = dict(failure=str(error), maximum_integer_bits=getattr(moment_ar, 'maximum_integer_bits', 0), **moment_ar.receipt())
            else:
                source = _source(polynomials, radius, source_ar)
    except WorkLimit as error:
        source = SourceFallback(str(error), source_ar.receipt())
    phases['source'], phases['shared_moments'] = source_ar.operations, moment_ar.operations
    if not isinstance(source, SourceHull) or source.upper >= 1:
        missing = Unavailable('no_finite_source_certificate', {})
        return _finish('ABSTAIN', source, Unavailable('not_run', {}),
                       [missing for _ in plan.coefficients], receipt, phases)
    fitted = reference.fit_calibration(calibration, plan.limits)
    phases['calibration'] = fitted.receipt.get('operations', 0)
    if isinstance(fitted, Unavailable):
        return _finish('ABSTAIN', source, fitted, [fitted for _ in plan.coefficients], receipt, phases)
    members = tuple(_member(a, polynomials, source, fitted, len(plan.coefficients), plan.limits) for a in plan.coefficients)
    phases['members'] = [member.receipt.get('operations', 0) for member in members]
    decisions = [member for member in members if isinstance(member, MemberDecision)]
    decision = 'REJECT' if any(member.reject for member in decisions) else ('DO_NOT_REJECT' if decisions else 'ABSTAIN')
    return _finish(decision, source, fitted, members, receipt, phases)
