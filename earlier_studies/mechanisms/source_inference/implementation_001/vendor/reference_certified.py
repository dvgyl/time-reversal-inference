"""Certify the restricted source calculation for exact supplied numbers."""
from dataclasses import dataclass
from fractions import Fraction as Q
import hashlib
import math
from typing import Optional, Sequence, Tuple, Union


class InputError(ValueError):
    """The supplied input does not satisfy the API contract."""


class WorkLimit(ArithmeticError):
    """A declared numerical work limit was reached."""


@dataclass(frozen=True)
class Interval:
    lower: Q
    upper: Q

    def __post_init__(self):
        if self.lower > self.upper:
            raise InputError('The interval endpoints are reversed.')

    @property
    def width(self):
        return self.upper-self.lower


@dataclass(frozen=True)
class WorkLimits:
    max_observations: int = 1000000
    max_input_bits: int = 4096
    max_rational_bits: int = 262144
    max_operations: int = 20000000
    max_regression_columns: int = 32
    max_bank_size: int = 64
    log_terms: int = 48
    sqrt_bits: int = 80
    max_cells: int = 16383
    max_depth: int = 20

    def __post_init__(self):
        for key, value in self.__dict__.items():
            if type(value) is not int or value < (0 if key in ('max_cells', 'max_depth') else 1):
                raise InputError('Each work limit must be a valid nonnegative integer.')
        if self.sqrt_bits > self.max_rational_bits or self.log_terms > 4096:
            raise InputError('The requested precision exceeds the supported work limits.')
        if self.max_depth > self.max_rational_bits:
            raise InputError('The subdivision depth exceeds the rational bit limit.')


class Arithmetic:
    def __init__(self, limits):
        if not isinstance(limits, WorkLimits):
            raise InputError('Supply WorkLimits.')
        self.limits = limits
        self.operations = 0
        self.max_bits = 0

    def q(self, value):
        self.operations += 1
        if self.operations > self.limits.max_operations:
            raise WorkLimit('max_operations')
        value = Q(value)
        bits = max(value.numerator.bit_length(), value.denominator.bit_length())
        self.max_bits = max(self.max_bits, bits)
        if bits > self.limits.max_rational_bits:
            raise WorkLimit('max_rational_bits')
        return value

    def input(self, value):
        if type(value) not in (int, float, Q):
            raise InputError('Use finite Python int, float, or Fraction values.')
        if type(value) is float:
            if not math.isfinite(value):
                raise InputError('All input values must be finite.')
            value = Q.from_float(value)
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > self.limits.max_input_bits:
            raise WorkLimit('max_input_bits')
        return self.q(value)

    def addq(self, a, b):
        return self.q(a+b)

    def mulq(self, a, b):
        return self.q(a*b)

    def divq(self, a, b):
        return self.q(a/b)

    def sumq(self, values):
        result = Q(0)
        for value in values:
            result = self.addq(result, value)
        return result

    def point(self, value):
        value = self.q(value)
        return Interval(value, value)

    def add(self, a, b):
        return Interval(self.q(a.lower+b.lower), self.q(a.upper+b.upper))

    def neg(self, a):
        return Interval(self.q(-a.upper), self.q(-a.lower))

    def sub(self, a, b):
        return self.add(a, self.neg(b))

    def mul(self, a, b):
        values = [self.mulq(x, y) for x in (a.lower, a.upper) for y in (b.lower, b.upper)]
        return Interval(min(values), max(values))

    def div(self, a, b):
        if b.lower <= 0 <= b.upper:
            raise ArithmeticError('The divisor interval contains zero.')
        return self.mul(a, Interval(self.divq(1, b.upper), self.divq(1, b.lower)))

    def sqrt(self, value):
        if value.lower < 0:
            raise ArithmeticError('The square-root interval is negative.')
        def endpoint(x, upper):
            shift = 2*self.limits.sqrt_bits
            root = math.isqrt((x.numerator << shift)//x.denominator)
            lower = self.q(Q(root, 1 << self.limits.sqrt_bits))
            if upper and self.mulq(lower, lower) != x:
                return self.q(Q(root+1, 1 << self.limits.sqrt_bits))
            return lower
        return Interval(endpoint(value.lower, False), endpoint(value.upper, True))

    def log(self, value):
        value = self.q(value)
        if value <= 0:
            raise InputError('The logarithm argument must be positive.')
        k = value.numerator.bit_length()-value.denominator.bit_length()
        scale = Q(1 << k) if k >= 0 else Q(1, 1 << -k)
        m = self.divq(value, scale)
        if m < 1:
            k -= 1
            m = self.mulq(m, 2)
        def series(x):
            z = self.divq(self.q(x-1), self.q(x+1))
            z2 = self.mulq(z, z)
            power, total = z, Q(0)
            for j in range(self.limits.log_terms):
                total = self.addq(total, self.divq(power, 2*j+1))
                power = self.mulq(power, z2)
            lower = self.mulq(2, total)
            denominator = self.mulq(2*self.limits.log_terms+1, self.q(1-z2))
            tail = self.divq(self.mulq(2, power), denominator)
            return Interval(lower, self.addq(lower, tail))
        return self.add(series(m), self.mul(self.point(k), series(Q(2))))

    def receipt(self):
        return dict(arithmetic='exact rational with certified intervals', operations=self.operations,
                    maximum_rational_bits=self.max_bits, limits=self.limits.__dict__.copy())


def count(values, maximum, label):
    try:
        size = len(values)
    except TypeError as error:
        raise InputError(label+' must be an indexable finite sequence.') from error
    if size > maximum:
        raise WorkLimit(label+'_count')
    return size


def digest(values):
    result = hashlib.sha256()
    for value in values:
        for integer in (value.numerator, value.denominator):
            data = abs(integer).to_bytes(max(1, (abs(integer).bit_length()+7)//8), 'big')
            result.update((b'-' if integer < 0 else b'+')+str(len(data)).encode()+b':'+data)
    return result.hexdigest()


@dataclass(frozen=True)
class Quadratic:
    constant: Q
    linear: Q
    square: Q

    def at(self, x, ar):
        return ar.addq(ar.mulq(ar.addq(ar.mulq(self.square, x), self.linear), x), self.constant)

    def range(self, cell, ar):
        values = [self.at(cell.lower, ar), self.at(cell.upper, ar)]
        if self.square:
            vertex = ar.divq(-self.linear, ar.mulq(2, self.square))
            if cell.lower < vertex < cell.upper:
                values.append(self.at(vertex, ar))
        return Interval(min(values), max(values))

    def derivative_bound(self, ar):
        return max(abs(self.linear), abs(ar.addq(ar.mulq(2, self.square), self.linear)))


def combine(left, right, factor, ar):
    return Quadratic(*(ar.addq(x, ar.mulq(factor, y)) for x, y in
                       zip(left.__dict__.values(), right.__dict__.values())))


def product_linear(a, b, ar):
    return Quadratic(ar.mulq(a[0], b[0]),
                     ar.addq(ar.mulq(a[0], b[1]), ar.mulq(a[1], b[0])),
                     ar.mulq(a[1], b[1]))


def sufficient_statistics(y, ar):
    n = len(y)-1
    if n < 1:
        raise InputError('The sufficient statistics need at least two observations.')
    sx, sz = Q(0), Q(0)
    raw_d = Quadratic(Q(0), Q(0), Q(0))
    raw_l = raw_d
    previous = None
    for i in range(n):
        current = (y[i+1], -y[i])
        sx, sz = ar.addq(sx, current[0]), ar.addq(sz, current[1])
        raw_d = combine(raw_d, product_linear(current, current, ar), Q(1), ar)
        if previous is not None:
            raw_l = combine(raw_l, product_linear(previous, current, ar), Q(1), ar)
        previous = current
    total = (sx, sz)
    total_square = product_linear(total, total, ar)
    denominator = combine(raw_d, total_square, Q(-1, n), ar)
    endpoints = (ar.addq(y[1], y[-1]), ar.q(-y[0]-y[-2]))
    lag = combine(raw_l, total_square, Q(-n-1, n*n), ar)
    lag = combine(lag, product_linear(total, endpoints, ar), Q(1, n), ar)
    return denominator, lag


@dataclass(frozen=True)
class SourceHull:
    lower: Q
    upper: Q
    radius: Interval
    denominator: Quadratic
    lag_numerator: Quadratic
    retained_cells: Tuple[Interval, ...]
    unresolved_cells: Tuple[Interval, ...]
    acceptance_excess: Optional[Q]
    extra_contraction_radius: Optional[Q]
    receipt: dict
    status: str = 'CERTIFIED_HULL'


@dataclass(frozen=True)
class SourceEmpty:
    radius: Interval
    receipt: dict
    status: str = 'EMPTY_CERTIFIED'


@dataclass(frozen=True)
class SourceFallback:
    reason: str
    receipt: dict
    lower: Q = Q(0)
    upper: Q = Q(1)
    status: str = 'FULL_FALLBACK'


SourceResult = Union[SourceHull, SourceEmpty, SourceFallback]


def acceptance_radius(n, ar):
    if n < 2:
        return None
    d = n-1
    t0, t1 = ar.log(Q(400)), ar.log(Q(800))
    bottom = ar.sub(ar.point(d), ar.mul(ar.point(2), ar.sqrt(ar.mul(ar.point(d), t0))))
    if bottom.lower <= 0:
        return None
    norm2 = Q(n-3, 2)+Q(2, n*n)
    top = ar.add(ar.mul(ar.point(2), ar.sqrt(ar.mul(ar.point(norm2), t1))),
                 ar.mul(ar.point(2+Q(2, n)), t1))
    return ar.div(top, bottom)


def enclose_polynomials(q1, q2, ar):
    if ar.limits.max_cells < 1:
        raise WorkLimit('max_cells')
    pending = [(Interval(Q(0), Q(1)), 0)]
    retained, unresolved, excluded = [], [], []
    evaluated = 0
    excess = Q(0)
    limited = False
    while pending:
        cell, depth = pending.pop()
        ranges = (q1.range(cell, ar), q2.range(cell, ar))
        evaluated += 1
        if any(value.lower > 0 for value in ranges):
            excluded.append((cell, ranges))
        elif all(value.upper <= 0 for value in ranges):
            retained.append(cell)
        elif depth < ar.limits.max_depth and evaluated+len(pending)+2 <= ar.limits.max_cells:
            midpoint = ar.divq(ar.addq(cell.lower, cell.upper), 2)
            pending.extend(((Interval(midpoint, cell.upper), depth+1),
                            (Interval(cell.lower, midpoint), depth+1)))
        else:
            retained.append(cell)
            unresolved.append(cell)
            excess = max(excess, *(value.upper for value in ranges))
            limited = limited or depth < ar.limits.max_depth
    return retained, unresolved, excess, dict(cells_evaluated=evaluated,
        excluded_cells=excluded, cell_limit_reached=limited,
        unresolved_width=max((ar.q(cell.width) for cell in unresolved), default=Q(0)))


def _source(y, ar):
    n = len(y)-1
    radius = acceptance_radius(n, ar)
    if radius is None:
        return SourceFallback('small_record_or_nonpositive_radius_denominator', ar.receipt())
    denominator, lag = sufficient_statistics(y, ar)
    if denominator == Quadratic(Q(0), Q(0), Q(0)):
        return SourceFallback('identically_zero_innovation_denominator', ar.receipt())
    mu = Q(-1, n)
    q1 = combine(lag, denominator, ar.q(-mu-radius.upper), ar)
    q2 = combine(Quadratic(Q(0), Q(0), Q(0)), lag, Q(-1), ar)
    q2 = combine(q2, denominator, ar.q(mu-radius.upper), ar)
    retained, unresolved, violation, receipt = enclose_polynomials(q1, q2, ar)
    if not retained:
        receipt.update(ar.receipt())
        return SourceEmpty(radius, receipt)
    dmin = denominator.range(Interval(Q(0), Q(1)), ar).lower
    excess = ar.divq(violation, dmin) if dmin > 0 else None
    extra = ar.mulq(2, ar.addq(radius.width, excess)) if excess is not None else None
    receipt.update(denominator_minimum=dmin, polynomial_violation_upper=violation,
                   q1=q1, q2=q2, input_sha256=digest(y), **ar.receipt())
    return SourceHull(min(c.lower for c in retained), max(c.upper for c in retained), radius,
                      denominator, lag, tuple(retained), tuple(unresolved), excess, extra, receipt)


def source_confidence(record, limits=WorkLimits()):
    ar = Arithmetic(limits)
    try:
        size = count(record, limits.max_observations, 'source_observations')
        y = tuple(ar.input(record[i]) for i in range(size))
        return _source(y, ar)
    except WorkLimit as error:
        return SourceFallback(str(error), ar.receipt())


@dataclass(frozen=True)
class CalibrationData:
    design: Sequence[Sequence[Union[int, float, Q]]]
    response: Sequence[Union[int, float, Q]]
    positions: Sequence[int]


@dataclass(frozen=True)
class CalibrationCertificate:
    coefficients: Tuple[Q, ...]
    positions: Tuple[int, ...]
    error_l2: Interval
    shape_upper: Q
    phase_upper: Q
    receipt: dict


@dataclass(frozen=True)
class Unavailable:
    reason: str
    receipt: dict


def response_bounds(coefficients, positions, radius, ar):
    p = len(coefficients)
    l1 = ar.sumq(abs(x) for x in coefficients)
    norm2 = ar.sumq(ar.mulq(x, x) for x in coefficients)
    norm = ar.sqrt(ar.point(norm2))
    error1 = ar.mul(ar.sqrt(ar.point(p)), ar.point(radius)).upper
    peak = ar.addq(l1, error1)
    floor = ar.q(norm.lower-radius)
    shape = Q(p)
    if floor > 0:
        shape = min(shape, max(Q(1), ar.divq(ar.mulq(peak, peak), ar.mulq(floor, floor))))
    if 0 not in positions:
        return shape, Q(1)
    anchor = positions.index(0)
    off = ar.sumq(abs(x) for i, x in enumerate(coefficients) if i != anchor)
    off_error = ar.mul(ar.sqrt(ar.point(p-1)), ar.point(radius)).upper
    off = ar.addq(off, off_error)
    floor = ar.q(abs(coefficients[anchor])-radius-off)
    phase = min(Q(1), ar.divq(off, floor)) if floor > 0 else Q(1)
    return shape, phase


def inverse(matrix, ar):
    p = len(matrix)
    rows = [list(row)+[Q(int(i == j)) for j in range(p)] for i, row in enumerate(matrix)]
    for column in range(p):
        pivot = next((i for i in range(column, p) if rows[i][column]), None)
        if pivot is None:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [ar.divq(x, divisor) for x in rows[column]]
        for i in range(p):
            if i == column:
                continue
            factor = rows[i][column]
            rows[i] = [ar.q(x-ar.mulq(factor, y)) for x, y in zip(rows[i], rows[column])]
    return tuple(tuple(row[p:]) for row in rows)


def _fit(data, ar):
    if not isinstance(data, CalibrationData):
        raise InputError('Supply CalibrationData.')
    m = count(data.design, ar.limits.max_observations, 'calibration_rows')
    if count(data.response, ar.limits.max_observations, 'calibration_response') != m or not m:
        raise InputError('Calibration row counts must match and be positive.')
    p = count(data.positions, ar.limits.max_regression_columns, 'calibration_columns')
    if p < 1 or m <= p:
        raise InputError('Calibration needs coefficients and positive residual degrees of freedom.')
    positions = tuple(data.positions[i] for i in range(p))
    if any(type(x) is not int or abs(x).bit_length() > ar.limits.max_input_bits for x in positions):
        raise InputError('Coefficient positions must be bounded integers.')
    if len(set(positions)) != p:
        raise InputError('Coefficient positions must be distinct.')
    gram = [[Q(0) for _ in range(p)] for _ in range(p)]
    xy, yy = [Q(0) for _ in range(p)], Q(0)
    binding = hashlib.sha256()
    for i in range(m):
        if count(data.design[i], ar.limits.max_regression_columns, 'design_columns') != p:
            raise InputError('Every calibration row must match the coefficient count.')
        row = tuple(ar.input(data.design[i][j]) for j in range(p))
        y = ar.input(data.response[i])
        binding.update(digest((*row, y)).encode())
        yy = ar.addq(yy, ar.mulq(y, y))
        for j in range(p):
            xy[j] = ar.addq(xy[j], ar.mulq(row[j], y))
            for k in range(p):
                gram[j][k] = ar.addq(gram[j][k], ar.mulq(row[j], row[k]))
    inv = inverse(gram, ar)
    if inv is None:
        return Unavailable('singular_calibration_design', ar.receipt())
    fitted = tuple(ar.sumq(ar.mulq(inv[i][j], xy[j]) for j in range(p)) for i in range(p))
    rss = ar.q(yy-ar.sumq(ar.mulq(fitted[i], xy[i]) for i in range(p)))
    if rss <= 0:
        return Unavailable('nonpositive_calibration_residual', ar.receipt())
    gram_lower = ar.divq(1, max(ar.sumq(abs(x) for x in row) for row in inv))
    nu = m-p
    t = ar.log(Q(200))
    residual_lower = ar.sub(ar.point(nu), ar.mul(ar.point(2), ar.sqrt(ar.mul(ar.point(nu), t))))
    if residual_lower.lower <= 0:
        return Unavailable('nonpositive_calibration_tail_denominator', ar.receipt())
    coefficient_upper = ar.add(ar.add(ar.point(p), ar.mul(ar.point(2), ar.sqrt(ar.mul(ar.point(p), t)))), ar.mul(ar.point(2), t))
    radius = ar.sqrt(ar.div(ar.mul(ar.point(rss), coefficient_upper), ar.mul(residual_lower, ar.point(gram_lower))))
    shape, phase = response_bounds(fitted, positions, radius.upper, ar)
    receipt = dict(rows=m, columns=p, residual_degrees=nu, residual_sum_squares=rss,
                   gram_eigenvalue_lower=gram_lower, residual_lower=residual_lower,
                   coefficient_upper=coefficient_upper, failure_budget=Q(1, 100),
                   residual_failure_budget=Q(1, 200), coefficient_failure_budget=Q(1, 200),
                   calibration_sha256=binding.hexdigest(), positions=positions,
                   method='exact regression and two chi-square tail bounds', **ar.receipt())
    return CalibrationCertificate(fitted, positions, radius, shape, phase, receipt)


def fit_calibration(data, limits=WorkLimits()):
    ar = Arithmetic(limits)
    try:
        return _fit(data, ar)
    except WorkLimit as error:
        return Unavailable(str(error), ar.receipt())


@dataclass(frozen=True)
class BankPlan:
    coefficients: Tuple[Q, ...]
    limits: WorkLimits = WorkLimits()
    source_budget: Q = Q(1, 200)
    calibration_budget: Q = Q(1, 100)
    variance_budget: Q = Q(1, 100)
    tail_budget: Q = Q(1, 40)

    def __post_init__(self):
        ar = Arithmetic(self.limits)
        size = count(self.coefficients, self.limits.max_bank_size, 'bank_members')
        if size < 1:
            raise InputError('The fixed bank must be nonempty.')
        values = tuple(ar.input(self.coefficients[i]) for i in range(size))
        if any(not 0 <= x < 1 for x in values) or len(set(values)) != size:
            raise InputError('The bank needs distinct coefficients in [0,1).')
        expected = (Q(1, 200), Q(1, 100), Q(1, 100), Q(1, 40))
        actual = (self.source_budget, self.calibration_budget, self.variance_budget, self.tail_budget)
        if actual != expected or any(type(x) is not Q for x in actual):
            raise InputError('This implementation uses the four fixed rational budgets.')
        object.__setattr__(self, 'coefficients', values)


@dataclass(frozen=True)
class MemberDecision:
    coefficient: Q
    retained_length: int
    pair_count: int
    statistic_absolute: Q
    threshold: Interval
    reject: bool
    source_ratio: Q
    covariance_ratio: Q
    receipt: dict


@dataclass(frozen=True)
class BankResult:
    decision: str
    source: SourceResult
    calibration: Union[CalibrationCertificate, Unavailable]
    members: Tuple[Union[MemberDecision, Unavailable], ...]
    receipt: dict


def source_ratio(lower, upper, coefficient, ar):
    if not 0 <= lower <= upper < 1 or not 0 <= coefficient < 1:
        raise InputError('A finite source ratio needs a hull strictly below one.')
    def t(x):
        return ar.divq(ar.q(1+x), ar.q(1-x))
    high = ar.divq(t(upper), t(coefficient))
    low = ar.divq(t(coefficient), t(lower))
    return max(ar.mulq(high, high), ar.mulq(low, low))


def recording_statistics(values, ar):
    length = len(values)
    sums = [ar.sumq(row[i] for row in values) for i in range(2)]
    variances = tuple(ar.q(ar.divq(ar.sumq(ar.mulq(row[i], row[i]) for row in values), length)
                          -ar.mulq(ar.divq(sums[i], length), ar.divq(sums[i], length))) for i in range(2))
    n = length-1
    first_sum, second_sum, product_sum = Q(0), Q(0), Q(0)
    for i in range(n):
        first = ar.addq(values[i][0], values[i+1][0])
        second = ar.q(values[i][1]-values[i+1][1])
        first_sum, second_sum = ar.addq(first_sum, first), ar.addq(second_sum, second)
        product_sum = ar.addq(product_sum, ar.mulq(first, second))
    statistic = ar.q(ar.divq(product_sum, n)-ar.divq(ar.mulq(first_sum, second_sum), n*n))
    return variances, statistic


def trace_radius(k, total_variance, length, tail, ar):
    if length < 3:
        raise InputError('The trace radius needs three retained observations.')
    n = length-1
    norm2 = Q(2*n+2, 2*n*n)-Q(2, n**3)-Q(2, n**4)
    frobenius = ar.mul(k, ar.sqrt(ar.point(norm2)))
    trace = ar.div(ar.mul(ar.point(2), ar.sqrt(ar.mul(ar.mul(k, ar.point(length)), total_variance))), ar.point(n))
    minimum = Interval(min(frobenius.lower, trace.lower), min(frobenius.upper, trace.upper))
    return ar.add(ar.mul(ar.mul(ar.point(2), minimum), ar.sqrt(tail)),
                  ar.div(ar.mul(ar.mul(ar.point(4), k), tail), ar.point(n)))


def threshold(q, phase, variance, length, members, ar):
    u = ar.log(Q(200*members))
    s = ar.log(Q(80*members))
    ratio = ar.div(ar.point(q), ar.point(length))
    c = ar.sub(ar.sub(ar.point(1), ratio), ar.mul(ar.point(2), ar.sqrt(ar.mul(ratio, u))))
    if c.lower <= 0:
        return None, dict(scale_denominator=c)
    scale = ar.div(ar.point(variance), c)
    radius = trace_radius(ar.mul(ar.point(q), scale), ar.mul(ar.point(2), scale), length, s, ar)
    result = ar.add(ar.mul(ar.point(2*phase), scale), radius)
    return result, dict(scale_denominator=c, variance_scale=scale, variance_log=u, tail_log=s,
                        threshold_excess_upper=ar.q(result.width), statistic_loss=Q(0))


def strict_reject(statistic_lower, threshold_upper):
    return statistic_lower > threshold_upper


def _member(coefficient, values, source, calibration, members, limits):
    ar = Arithmetic(limits)
    try:
        transformed = tuple(tuple(ar.q(values[i+1][j]-ar.mulq(coefficient, values[i][j])) for j in range(2))
                            for i in range(len(values)-1))
        if len(transformed) < 3:
            return Unavailable('too_few_transformed_observations', ar.receipt())
        ratio = source_ratio(source.lower, source.upper, coefficient, ar)
        q = ar.mulq(2, ar.mulq(ratio, max(Q(1), calibration.shape_upper)))
        variances, statistic = recording_statistics(transformed, ar)
        bound, details = threshold(q, calibration.phase_upper, max(variances), len(transformed), members, ar)
        if bound is None:
            return Unavailable('nonpositive_scale_denominator', dict(coefficient=coefficient, **details, **ar.receipt()))
        details.update(variances=variances, signed_statistic=statistic,
                       common_filter=(-coefficient, Q(1)), normalization='unnormalized',
                       normalized_quadratic_divisor=ar.q(1-coefficient*coefficient),
                       variance_failure_budget=Q(1, 100*members),
                       null_tail_failure_budget=Q(1, 40*members), **ar.receipt())
        return MemberDecision(coefficient, len(transformed), len(transformed)-1, abs(statistic), bound,
                              strict_reject(abs(statistic), bound.upper), ratio, q, details)
    except WorkLimit as error:
        return Unavailable(str(error), dict(coefficient=coefficient, **ar.receipt()))


def run_bank(plan, record, calibration):
    if not isinstance(plan, BankPlan):
        raise InputError('Construct BankPlan before reading the recording.')
    ar = Arithmetic(plan.limits)
    receipt = dict(bank=plan.coefficients, source_budget=plan.source_budget,
                   calibration_budget=plan.calibration_budget, variance_budget=plan.variance_budget,
                   tail_budget=plan.tail_budget, size_bound=Q(1, 20),
                   power_status='POWER_UNVERIFIED', statistic_loss=Q(0),
                   assumptions='Supplied common Gaussian AR(1) family; nonzero scalar reference; nonzero finite second response; zero testing errors; zero-mean spherical Gaussian calibration error; fixed full-rank design; no calibration intercept; exact-input arithmetic scope.')
    try:
        size = count(record, plan.limits.max_observations, 'recording_rows')
        values = []
        for i in range(size):
            if count(record[i], 2, 'recording_channels') != 2:
                raise InputError('Each recording row must contain two channels.')
            values.append((ar.input(record[i][0]), ar.input(record[i][1])))
        receipt.update(record_length=size, recording_sha256=digest(x for row in values for x in row), **ar.receipt())
    except WorkLimit as error:
        missing = Unavailable(str(error), ar.receipt())
        return BankResult('ABSTAIN', SourceFallback(str(error), ar.receipt()), missing,
                          tuple(missing for _ in plan.coefficients), receipt)
    source = source_confidence(tuple(row[0] for row in values), plan.limits)
    if not isinstance(source, SourceHull) or source.upper >= 1:
        missing = Unavailable('no_finite_source_certificate', {})
        return BankResult('ABSTAIN', source, Unavailable('not_run', {}), tuple(missing for _ in plan.coefficients), receipt)
    fitted = fit_calibration(calibration, plan.limits)
    if isinstance(fitted, Unavailable):
        return BankResult('ABSTAIN', source, fitted, tuple(fitted for _ in plan.coefficients), receipt)
    members = tuple(_member(a, values, source, fitted, len(plan.coefficients), plan.limits) for a in plan.coefficients)
    decisions = [x for x in members if isinstance(x, MemberDecision)]
    decision = 'REJECT' if any(x.reject for x in decisions) else ('DO_NOT_REJECT' if decisions else 'ABSTAIN')
    return BankResult(decision, source, fitted, members, receipt)


def contraction_bound(phi, epsilon, n, radius_upper, unresolved_width, ar):
    """Bound hull distance on the checked uniform concentration event."""
    phi, epsilon, radius_upper, unresolved_width = map(ar.q, (phi, epsilon, radius_upper, unresolved_width))
    if not 0 <= phi < 1 or epsilon < 0 or n < 2 or radius_upper < 0 or not 0 <= unresolved_width <= 1:
        raise InputError('The contraction parameters are invalid.')
    vstar = ar.q(1-phi*phi)
    if epsilon > vstar/2:
        return Unavailable('uniform_denominator_condition_not_met', ar.receipt())
    excess = ar.divq(ar.mulq(unresolved_width, ar.mulq(ar.q(1+Q(1, n)+radius_upper), ar.q(2+8*epsilon))), ar.q(vstar-epsilon))
    h = ar.mulq(2, ar.q(radius_upper+excess+Q(1, n)+ar.divq(4*epsilon, vstar)))
    return dict(radius_upper=h, subdivision_acceptance_excess=excess,
                lower=max(Q(0), phi-h), upper=min(Q(1), phi+h),
                scope='Conditional on uniform concentration and the stated unresolved width.')


def planned_source_bound(phi, original_length, unresolved_width, ar):
    """Use the proved error bound with the source and contraction budgets."""
    phi = ar.q(phi)
    if not 0 <= phi < 1 or type(original_length) is not int or original_length < 3:
        raise InputError('Supply a valid true parameter and original recording length.')
    n = original_length-1
    radius = acceptance_radius(n, ar)
    if radius is None:
        return Unavailable('nonpositive_planned_source_denominator', ar.receipt())
    g = ar.divq(ar.q(1+phi), ar.q(1-phi))
    t = ar.log(Q(2400))
    first = ar.mul(ar.point(10), ar.sqrt(ar.div(ar.mul(ar.point(g), t), ar.point(n))))
    second = ar.div(ar.mul(ar.point(g), ar.add(ar.mul(ar.point(10), t), ar.point(16))), ar.point(n))
    epsilon = ar.add(first, second)
    result = contraction_bound(phi, epsilon.upper, n, radius.upper, unresolved_width, ar)
    if isinstance(result, Unavailable):
        return result
    result.update(uniform_error_upper=epsilon.upper, acceptance_radius=radius,
                  source_radius_enlargement_upper=ar.q(radius.width),
                  source_failure_budget=Q(1, 200), contraction_failure_budget=Q(1, 200),
                  power_status='POWER_UNVERIFIED')
    return result


def planned_response_bounds(true_coefficients, positions, noise_variance, calibration_certificate, ar):
    """Bound fitted response certificates on coefficient and residual events."""
    if not isinstance(calibration_certificate, CalibrationCertificate):
        raise InputError('Supply a completed calibration certificate for the fixed design.')
    info = calibration_certificate.receipt
    p, nu = info['columns'], info['residual_degrees']
    if any(getattr(ar.limits, key) != info['limits'][key] for key in ('sqrt_bits', 'log_terms')):
        raise InputError('Planning must use the fitted arithmetic precision.')
    if len(true_coefficients) != p or tuple(positions) != calibration_certificate.positions or noise_variance <= 0:
        raise InputError('The planned response must match the calibration design.')
    t = ar.log(Q(100))
    residual_upper = ar.add(ar.add(ar.point(nu), ar.mul(ar.point(2), ar.sqrt(ar.mul(ar.point(nu), t)))), ar.mul(ar.point(2), t))
    planned_radius = ar.sqrt(ar.div(ar.mul(ar.mul(ar.point(noise_variance), residual_upper), info['coefficient_upper']),
                                       ar.mul(info['residual_lower'], ar.point(info['gram_eigenvalue_lower'])))).upper
    _, phase = response_bounds(tuple(map(ar.q, true_coefficients)), tuple(positions), ar.mulq(2, planned_radius), ar)
    shape = Q(p)
    return dict(shape_upper=shape, phase_upper=phase, planned_coefficient_radius=planned_radius,
                residual_event_failure=Q(1, 100), scope='True coefficients and calibration noise variance are analysis inputs.')


def conditional_power_margin(q, phase, maximum_variance, total_variance, contrast_absolute, length, members, ar):
    """Compute a sufficient numerical margin, subject to completion conditions."""
    q, phase, maximum_variance, total_variance, contrast_absolute = map(ar.q, (q, phase, maximum_variance, total_variance, contrast_absolute))
    if q < 2 or not 0 <= phase <= 1 or maximum_variance <= 0 or not maximum_variance <= total_variance <= 2*maximum_variance or contrast_absolute < 0 or length < 3 or members < 1:
        raise InputError('The planned power parameters are invalid.')
    u = ar.log(Q(200*members))
    factor = ar.div(ar.mul(ar.point(q), u), ar.point(length))
    upper_variance = ar.mul(ar.point(maximum_variance), ar.add(ar.add(ar.point(1), ar.mul(ar.point(2), ar.sqrt(factor))), ar.mul(ar.point(2), factor)))
    null_upper, details = threshold(q, phase, upper_variance.upper, length, members, ar)
    if null_upper is None:
        return dict(status='POWER_UNVERIFIED', reason='nonpositive_planned_scale_denominator')
    alternative = trace_radius(ar.point(ar.mulq(q, maximum_variance)), ar.point(total_variance), length, ar.log(Q(100*members)), ar)
    n = length-1
    bias = ar.div(ar.mul(ar.point(ar.mulq(q, maximum_variance)), ar.sqrt(ar.point(2*(4*n-2)))), ar.point(n*n))
    total = ar.add(ar.add(null_upper, alternative), bias)
    return dict(status='CONDITIONAL_MARGIN_SUFFICIENT' if contrast_absolute > total.upper else 'CONDITIONAL_MARGIN_INSUFFICIENT',
                power_status='POWER_UNVERIFIED', required_contrast=total, supplied_contrast=contrast_absolute,
                margin_lower=ar.q(contrast_absolute-total.upper), null_threshold_upper=null_upper.upper,
                null_threshold_rounding_excess=details['threshold_excess_upper'],
                variance_upper_rounding_excess=ar.q(upper_variance.width), statistic_loss=Q(0),
                power_failure_budgets=(Q(1, 200), Q(1, 100), Q(1, 100), Q(1, 200), Q(1, 100), Q(1, 100)),
                conditions='The supplied q and phase must cover the enlarged numerical hull and planned calibration certificates. Use the same arithmetic precision as the bank. Moments use the unnormalized filter. All numerical phases must complete on the six good events. Recorder-model transfer is not proved.')
