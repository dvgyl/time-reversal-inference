"""Calculate the S17 test with exact quarter-turn phases and rational bounds."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from math import isfinite, isqrt
from typing import Literal, Optional, Sequence, Tuple


class QuarterTurn(IntEnum):
    ZERO = 0
    PI_OVER_TWO = 1
    PI = 2
    THREE_PI_OVER_TWO = 3


Status = Literal["reject", "nonreject", "abstain"]
Refusal = Literal["missing_bound", "nonfinite_record", "resource_limit", "numerical_ambiguity"]


@dataclass(frozen=True)
class SuppliedBounds:
    covariance_ceiling: Optional[Fraction]
    tail_bias: Optional[Fraction]
    source_note: str


@dataclass(frozen=True)
class NumericalPolicy:
    precision_bits: Tuple[int, ...] = (80, 160, 320)
    max_integer_bits: int = 16384
    max_lag_products: int = 100_000_000
    max_rational_bits: int = 8192
    max_record_values: int = 2_000_000
    max_phase_operations: int = 500_000_000
    max_log_terms: int = 4096
    max_sqrt_shift_bits: int = 2048


@dataclass(frozen=True)
class CycleSpec:
    n: int
    lag: int
    frequencies: Tuple[QuarterTurn, ...]
    alpha: Fraction
    bounds: SuppliedBounds
    numerical_policy: NumericalPolicy = NumericalPolicy()


@dataclass(frozen=True)
class RationalInterval:
    lower: Fraction
    upper: Fraction


@dataclass(frozen=True)
class ExactComplex:
    real: Fraction
    imag: Fraction


@dataclass(frozen=True)
class PhaseReceipt:
    phase: str
    status: Literal["complete", "refused"]
    operations: int
    lag_products: int
    log_terms: int
    sqrt_calls: int
    sqrt_shift_bits: int
    max_proven_integer_bits: int
    max_rational_bits: int
    refusal: Optional[str] = None
    requested: Optional[int] = None
    limit: Optional[int] = None


@dataclass(frozen=True)
class FrequencyResult:
    frequency: QuarterTurn
    estimates: Optional[Tuple[ExactComplex, ExactComplex, ExactComplex]] = None
    absolute_imaginary_product: Optional[Fraction] = None
    epsilon: Optional[RationalInterval] = None
    threshold: Optional[RationalInterval] = None
    margin: Optional[RationalInterval] = None
    status: Status = "abstain"
    precision_bits: Optional[int] = None
    refusal: Optional[Refusal] = None

    def __post_init__(self):
        if self.absolute_imaginary_product is not None and self.estimates is None:
            raise ValueError("A product requires completed estimates.")
        if self.threshold is not None and (self.estimates is None or self.epsilon is None):
            raise ValueError("A threshold requires estimates and epsilon.")
        if self.margin is not None and (self.absolute_imaginary_product is None or self.threshold is None):
            raise ValueError("A margin requires the product and threshold.")
        if self.status in ("reject", "nonreject") and any(value is None for value in (
                self.estimates, self.absolute_imaginary_product, self.epsilon,
                self.threshold, self.margin, self.precision_bits)):
            raise ValueError("A decided frequency requires complete diagnostics.")
        if self.status in ("reject", "nonreject") and self.refusal is not None:
            raise ValueError("A decided frequency cannot have a refusal.")


@dataclass(frozen=True)
class CycleResult:
    spec: CycleSpec
    status: Status
    frequencies: Tuple[FrequencyResult, ...]
    receipts: Tuple[PhaseReceipt, ...]
    arithmetic_contract: str = "exact_dyadic_quarter_turn"
    refusal: Optional[Refusal] = None


class _WorkLimit(Exception):
    def __init__(self, name: str, requested: int, limit: int):
        self.name = name
        self.requested = requested
        self.limit = limit
        super().__init__(name)


class _Budget:
    def __init__(self, policy: NumericalPolicy, phase: str):
        self.policy = policy
        self.phase = phase
        self.operations = 0
        self.lag_products = 0
        self.log_terms = 0
        self.sqrt_calls = 0
        self.sqrt_shift_bits = 0
        self.max_proven_integer_bits = 0
        self.max_rational_bits = 0

    def limit(self, name: str, requested: int, limit: int) -> None:
        if requested > limit:
            raise _WorkLimit(name, requested, limit)

    def reserve(self, operations: int) -> None:
        self.limit("phase_operations", self.operations + operations, self.policy.max_phase_operations)

    def integer(self, value: int) -> int:
        bits = abs(value).bit_length()
        self.limit("integer_bits", bits, self.policy.max_integer_bits)
        self.max_proven_integer_bits = max(self.max_proven_integer_bits, bits)
        return value

    def bound(self, bits: int) -> None:
        self.limit("integer_bits", bits, self.policy.max_integer_bits)
        self.max_proven_integer_bits = max(self.max_proven_integer_bits, bits)

    def add(self, left: int, right: int) -> int:
        self.reserve(1)
        self.bound(max(abs(left).bit_length(), abs(right).bit_length()) + 1)
        result = left + right
        self.operations += 1
        return self.integer(result)

    def mul(self, left: int, right: int) -> int:
        self.reserve(1)
        self.bound(abs(left).bit_length() + abs(right).bit_length())
        result = left * right
        self.operations += 1
        return self.integer(result)

    def shift(self, value: int, count: int) -> int:
        self.reserve(1)
        self.bound(abs(value).bit_length() + count)
        result = value << count
        self.operations += 1
        return self.integer(result)

    def div_floor(self, left: int, right: int) -> int:
        self.reserve(1)
        self.integer(left)
        self.integer(right)
        result = left // right
        self.operations += 1
        return self.integer(result)

    def rational(self, numerator: int, denominator: int = 1) -> Fraction:
        self.reserve(1)
        self.integer(numerator)
        self.integer(denominator)
        result = Fraction(numerator, denominator)
        self.operations += 1
        return self.check_fraction(result)

    def check_fraction(self, value: Fraction) -> Fraction:
        bits = max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        self.limit("rational_bits", bits, self.policy.max_rational_bits)
        self.max_rational_bits = max(self.max_rational_bits, bits)
        self.integer(value.numerator)
        self.integer(value.denominator)
        return value

    def qadd(self, left: Fraction, right: Fraction) -> Fraction:
        numerator = self.add(self.mul(left.numerator, right.denominator),
                             self.mul(right.numerator, left.denominator))
        return self.rational(numerator, self.mul(left.denominator, right.denominator))

    def qmul(self, left: Fraction, right: Fraction) -> Fraction:
        return self.rational(self.mul(left.numerator, right.numerator),
                             self.mul(left.denominator, right.denominator))

    def qdiv(self, left: Fraction, right: Fraction) -> Fraction:
        return self.rational(self.mul(left.numerator, right.denominator),
                             self.mul(left.denominator, right.numerator))

    def compare(self, left: Fraction, right: Fraction) -> int:
        a = self.mul(left.numerator, right.denominator)
        b = self.mul(right.numerator, left.denominator)
        self.reserve(1)
        self.operations += 1
        return (a > b) - (a < b)

    def sum_integers(self, values: Sequence[int]) -> int:
        count = len(values)
        self.reserve(count)
        maximum = max((abs(v) for v in values), default=0)
        self.bound(maximum.bit_length() + count.bit_length())
        result = sum(values)
        self.operations += count
        return self.integer(result)

    def dot(self, left: Sequence[int], right: Sequence[int], start_i: int,
            start_j: int, count: int, max_i: int, max_j: int) -> int:
        self.limit("lag_products", self.lag_products + count, self.policy.max_lag_products)
        self.reserve(2 * count)
        self.bound(max_i.bit_length() + max_j.bit_length() + count.bit_length())
        result = sum(left[start_i + t] * right[start_j + t] for t in range(count))
        self.operations += 2 * count
        self.lag_products += count
        return self.integer(result)

    def receipt(self, failure: Optional[_WorkLimit] = None,
                refusal: Optional[str] = None) -> PhaseReceipt:
        return PhaseReceipt(
            phase=self.phase, status="refused" if failure or refusal else "complete",
            operations=self.operations, lag_products=self.lag_products,
            log_terms=self.log_terms, sqrt_calls=self.sqrt_calls,
            sqrt_shift_bits=self.sqrt_shift_bits,
            max_proven_integer_bits=self.max_proven_integer_bits,
            max_rational_bits=self.max_rational_bits,
            refusal=failure.name if failure else refusal,
            requested=failure.requested if failure else None,
            limit=failure.limit if failure else None,
        )


def _validate_spec(spec: CycleSpec) -> None:
    if not isinstance(spec, CycleSpec):
        raise TypeError("The specification must be a CycleSpec.")
    if type(spec.n) is not int or not 1 <= spec.n <= 2**63 - 1:
        raise ValueError("N must be a positive integer no greater than 2**63-1.")
    if type(spec.lag) is not int or not 0 <= spec.lag < spec.n:
        raise ValueError("L must be an integer with 0 <= L < N.")
    if type(spec.frequencies) is not tuple or not spec.frequencies:
        raise ValueError("The frequency grid must be a nonempty tuple.")
    if any(type(f) is not QuarterTurn for f in spec.frequencies):
        raise ValueError("Only exact QuarterTurn identifiers are supported.")
    if len(set(spec.frequencies)) != len(spec.frequencies):
        raise ValueError("The frequency grid must not contain duplicates.")
    if type(spec.alpha) is not Fraction or not 0 < spec.alpha < 1:
        raise ValueError("Alpha must be an exact Fraction with 0 < alpha < 1.")
    if not isinstance(spec.bounds, SuppliedBounds):
        raise TypeError("The bounds must be SuppliedBounds.")
    k, b = spec.bounds.covariance_ceiling, spec.bounds.tail_bias
    if k is not None and (type(k) is not Fraction or k <= 0):
        raise ValueError("K must be a positive Fraction or None.")
    if b is not None and (type(b) is not Fraction or b < 0):
        raise ValueError("B must be a nonnegative Fraction or None.")
    if not isinstance(spec.bounds.source_note, str):
        raise TypeError("The bound source note must be text.")
    if (k is not None or b is not None) and not spec.bounds.source_note.strip():
        raise ValueError("Supplied bounds require a source note.")
    p = spec.numerical_policy
    if not isinstance(p, NumericalPolicy):
        raise TypeError("The numerical policy must be NumericalPolicy.")
    if type(p.precision_bits) is not tuple or not p.precision_bits:
        raise ValueError("The precision schedule must be a nonempty tuple.")
    if any(type(v) is not int or not 1 <= v <= 1024 for v in p.precision_bits):
        raise ValueError("Each precision must be an integer between 1 and 1024.")
    if any(a >= b for a, b in zip(p.precision_bits, p.precision_bits[1:])):
        raise ValueError("The precision schedule must increase strictly.")
    for name in ("max_integer_bits", "max_lag_products", "max_rational_bits",
                 "max_record_values", "max_phase_operations", "max_log_terms",
                 "max_sqrt_shift_bits"):
        value = getattr(p, name)
        if type(value) is not int or not 1 <= value <= 2**63 - 1:
            raise ValueError(name + " must be a positive bounded integer.")


def _sqrt_interval(value: Fraction, bits: int, budget: _Budget) -> RationalInterval:
    if value < 0:
        raise ValueError("A square-root input must be nonnegative.")
    budget.check_fraction(value)
    budget.limit("sqrt_shift_bits", 2 * bits, budget.policy.max_sqrt_shift_bits)
    shifted = budget.shift(value.numerator, 2 * bits)
    integer = budget.div_floor(shifted, value.denominator)
    budget.reserve(1)
    root = isqrt(integer)
    budget.operations += 1
    budget.sqrt_calls += 1
    budget.sqrt_shift_bits = max(budget.sqrt_shift_bits, 2 * bits)
    scale = budget.shift(1, bits)
    lower = budget.rational(root, scale)
    squared = budget.mul(budget.mul(root, root), value.denominator)
    budget.reserve(1)
    exact = squared == shifted
    budget.operations += 1
    upper = lower if exact else budget.rational(budget.add(root, 1), scale)
    return RationalInterval(lower, upper)


def _log_series(z: Fraction, bits: int, budget: _Budget) -> RationalInterval:
    target = budget.rational(1, budget.shift(1, bits))
    z2 = budget.qmul(z, z)
    denominator_factor = budget.qadd(Fraction(1), -z2)
    total, power = Fraction(0), z
    j = 0
    while True:
        budget.limit("log_terms", budget.log_terms + 1, budget.policy.max_log_terms)
        term = budget.qdiv(power, Fraction(2 * j + 1))
        total = budget.qadd(total, budget.qmul(Fraction(2), term))
        budget.log_terms += 1
        power = budget.qmul(power, z2)
        j += 1
        remainder = budget.qdiv(budget.qmul(Fraction(2), power),
                                budget.qmul(Fraction(2 * j + 1), denominator_factor))
        if budget.compare(remainder, target) <= 0:
            return RationalInterval(total, budget.qadd(total, remainder))


def _log_interval(value: Fraction, bits: int, budget: _Budget) -> RationalInterval:
    if value < 1:
        raise ValueError("This logarithm routine requires an input of at least one.")
    budget.check_fraction(value)
    k = value.numerator.bit_length() - value.denominator.bit_length()
    factor = budget.shift(1, k)
    y = budget.qdiv(value, Fraction(factor))
    if budget.compare(y, Fraction(1)) < 0:
        k -= 1
        y = budget.qmul(y, Fraction(2))
    z = budget.qdiv(budget.qadd(y, Fraction(-1)), budget.qadd(y, Fraction(1)))
    a = _log_series(z, bits, budget)
    b = _log_series(Fraction(1, 3), bits, budget) if k else RationalInterval(Fraction(0), Fraction(0))
    return RationalInterval(
        budget.qadd(a.lower, budget.qmul(Fraction(k), b.lower)),
        budget.qadd(a.upper, budget.qmul(Fraction(k), b.upper)),
    )


def _radius_interval(spec: CycleSpec, bits: int, budget: _Budget) -> RationalInterval:
    k, b = spec.bounds.covariance_ceiling, spec.bounds.tail_bias
    assert k is not None and b is not None
    h = Fraction(2 * spec.lag + 1)
    n = Fraction(spec.n)
    t = _log_interval(budget.qdiv(Fraction(12 * len(spec.frequencies)), spec.alpha), bits, budget)
    ht_n = RationalInterval(budget.qdiv(budget.qmul(h, t.lower), n),
                           budget.qdiv(budget.qmul(h, t.upper), n))
    roots = (_sqrt_interval(ht_n.lower, bits, budget).lower,
             _sqrt_interval(ht_n.upper, bits, budget).upper)
    sqrt2 = _sqrt_interval(Fraction(2), bits, budget)
    bias = budget.qadd(b, budget.qdiv(budget.qmul(Fraction(3), budget.qmul(k, h)), n))
    ends = []
    for root, rt2, ht in zip(roots, (sqrt2.lower, sqrt2.upper), (ht_n.lower, ht_n.upper)):
        stochastic = budget.qadd(budget.qmul(budget.qmul(Fraction(2), k), root),
                                budget.qmul(budget.qmul(rt2, k), ht))
        ends.append(budget.qadd(bias, stochastic))
    return RationalInterval(*ends)


def _prepare(record: Sequence[Sequence[float]], spec: CycleSpec, budget: _Budget):
    if len(record) != spec.n:
        raise ValueError("The recording must have N rows.")
    integers, powers = [[], [], []], [[], [], []]
    max_powers = [0, 0, 0]
    for row in record:
        if len(row) != 3:
            raise ValueError("Each recording row must have three values.")
        for i, value in enumerate(row):
            if not isinstance(value, float):
                raise TypeError("Each recording value must be a binary64 float.")
            if not isfinite(value):
                return None
            budget.reserve(1)
            num, den = value.as_integer_ratio()
            budget.operations += 1
            budget.integer(num)
            budget.integer(den)
            p = den.bit_length() - 1
            integers[i].append(num)
            powers[i].append(p)
            max_powers[i] = max(max_powers[i], p)
    for i in range(3):
        for t, p in enumerate(powers[i]):
            integers[i][t] = budget.shift(integers[i][t], max_powers[i] - p)
    sums = tuple(budget.sum_integers(col) for col in integers)
    maxima = tuple(max((abs(v) for v in col), default=0) for col in integers)
    return integers, tuple(max_powers), sums, maxima


def _lag_statistics(prepared, spec: CycleSpec, budget: _Budget):
    columns, powers, sums, maxima = prepared
    n, lag = spec.n, spec.lag
    n2, n3 = budget.mul(n, n), budget.mul(budget.mul(n, n), n)
    output = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        values = []
        denominator = budget.mul(n3, budget.shift(1, powers[i] + powers[j]))
        for k in range(-lag, lag + 1):
            count, start_i, start_j = n - abs(k), max(0, k), max(0, -k)
            u = budget.dot(columns[i], columns[j], start_i, start_j, count, maxima[i], maxima[j])
            # The omitted segments contain at most L values each.
            vi = budget.add(sums[i], -budget.sum_integers(columns[i][:start_i]))
            vi = budget.add(vi, -budget.sum_integers(columns[i][start_i + count:]))
            vj = budget.add(sums[j], -budget.sum_integers(columns[j][:start_j]))
            vj = budget.add(vj, -budget.sum_integers(columns[j][start_j + count:]))
            cross = budget.add(budget.mul(sums[i], vj), budget.mul(sums[j], vi))
            centered = budget.add(budget.mul(n2, u), -budget.mul(n, cross))
            centered = budget.add(centered, budget.mul(count, budget.mul(sums[i], sums[j])))
            values.append(budget.rational(centered, denominator))
        output.append(tuple(values))
    return tuple(output)


def _entries(lags, spec: CycleSpec, frequency: QuarterTurn, budget: _Budget):
    output = []
    for edge in lags:
        real, imag = Fraction(0), Fraction(0)
        for k, value in zip(range(-spec.lag, spec.lag + 1), edge):
            phase = (-int(frequency) * k) % 4
            if phase in (0, 2):
                real = budget.qadd(real, value if phase == 0 else -value)
            else:
                imag = budget.qadd(imag, value if phase == 1 else -value)
        output.append(ExactComplex(real, imag))
    return tuple(output)


def _complex_product(a: ExactComplex, b: ExactComplex, budget: _Budget) -> ExactComplex:
    return ExactComplex(
        budget.qadd(budget.qmul(a.real, b.real), -budget.qmul(a.imag, b.imag)),
        budget.qadd(budget.qmul(a.real, b.imag), budget.qmul(a.imag, b.real)),
    )


def _product_radius(magnitudes, e: Fraction, budget: _Budget) -> Fraction:
    x, y, z = magnitudes
    pairs = budget.qadd(budget.qadd(budget.qmul(x, y), budget.qmul(y, z)), budget.qmul(z, x))
    total = budget.qadd(budget.qadd(x, y), z)
    e2 = budget.qmul(e, e)
    return budget.qadd(budget.qadd(budget.qmul(e, pairs), budget.qmul(e2, total)), budget.qmul(e2, e))


def _threshold(entries, epsilon: RationalInterval, bits: int, budget: _Budget) -> RationalInterval:
    magnitudes = []
    for x in entries:
        square = budget.qadd(budget.qmul(x.real, x.real), budget.qmul(x.imag, x.imag))
        magnitudes.append(_sqrt_interval(square, bits, budget))
    return RationalInterval(
        _product_radius(tuple(x.lower for x in magnitudes), epsilon.lower, budget),
        _product_radius(tuple(x.upper for x in magnitudes), epsilon.upper, budget),
    )


def _decision(q: Fraction, threshold: RationalInterval, budget: _Budget) -> Status:
    if budget.compare(q, threshold.upper) > 0:
        return "reject"
    if budget.compare(q, threshold.lower) <= 0:
        return "nonreject"
    return "abstain"


def _aggregate(spec: CycleSpec, frequencies, receipts) -> CycleResult:
    status = "reject" if any(f.status == "reject" for f in frequencies) else (
        "nonreject" if all(f.status == "nonreject" for f in frequencies) else "abstain")
    reason = next((f.refusal for f in frequencies if f.refusal is not None), None)
    return CycleResult(spec, status, tuple(frequencies), tuple(receipts), refusal=reason)


def _unavailable(spec: CycleSpec, receipts, refusal: Refusal) -> CycleResult:
    return _aggregate(spec, [FrequencyResult(f, refusal=refusal) for f in spec.frequencies], receipts)


def cycle_test(record: Sequence[Sequence[float]], spec: CycleSpec) -> CycleResult:
    """Return one certified test result for the complete declared frequency grid."""
    _validate_spec(spec)
    policy = spec.numerical_policy
    receipts = []
    budget = _Budget(policy, "preflight")
    try:
        budget.limit("record_values", 3 * spec.n, policy.max_record_values)
        products = 3 * ((2 * spec.lag + 1) * spec.n - spec.lag * (spec.lag + 1))
        budget.limit("lag_products", products, policy.max_lag_products)
        budget.limit("sqrt_shift_bits", 2 * max(policy.precision_bits), policy.max_sqrt_shift_bits)
        for value in (spec.alpha, spec.bounds.covariance_ceiling, spec.bounds.tail_bias):
            if value is not None:
                budget.check_fraction(value)
    except _WorkLimit as failure:
        receipts.append(budget.receipt(failure))
        return _unavailable(spec, receipts, "resource_limit")
    if spec.bounds.covariance_ceiling is None or spec.bounds.tail_bias is None:
        receipts.append(budget.receipt(refusal="missing_bound"))
        return _unavailable(spec, receipts, "missing_bound")
    receipts.append(budget.receipt())
    budget = _Budget(policy, "record_preparation")
    try:
        prepared = _prepare(record, spec, budget)
    except _WorkLimit as failure:
        receipts.append(budget.receipt(failure))
        return _unavailable(spec, receipts, "resource_limit")
    if prepared is None:
        receipts.append(budget.receipt(refusal="nonfinite_record"))
        return _unavailable(spec, receipts, "nonfinite_record")
    receipts.append(budget.receipt())
    budget = _Budget(policy, "lag_statistics")
    try:
        lags = _lag_statistics(prepared, spec, budget)
    except _WorkLimit as failure:
        receipts.append(budget.receipt(failure))
        return _unavailable(spec, receipts, "resource_limit")
    receipts.append(budget.receipt())
    del prepared
    results = []
    for f in spec.frequencies:
        budget = _Budget(policy, "frequency_%d_entries" % int(f))
        entries, q = None, None
        try:
            entries = _entries(lags, spec, f, budget)
            product = _complex_product(_complex_product(entries[0], entries[1], budget), entries[2], budget)
            q = abs(product.imag)
        except _WorkLimit as failure:
            receipts.append(budget.receipt(failure))
            results.append(FrequencyResult(f, estimates=entries, refusal="resource_limit"))
        else:
            receipts.append(budget.receipt())
            results.append(FrequencyResult(f, estimates=entries, absolute_imaginary_product=q))
    for bits in policy.precision_bits:
        unresolved = [i for i, result in enumerate(results) if result.status == "abstain" and result.refusal != "resource_limit"]
        if not unresolved:
            break
        budget = _Budget(policy, "radius_%d" % bits)
        try:
            epsilon = _radius_interval(spec, bits, budget)
        except _WorkLimit as failure:
            receipts.append(budget.receipt(failure))
            for i in unresolved:
                old = results[i]
                results[i] = FrequencyResult(old.frequency, old.estimates, old.absolute_imaginary_product,
                                             old.epsilon, old.threshold, old.margin, "abstain",
                                             old.precision_bits, "resource_limit")
            break
        receipts.append(budget.receipt())
        for i in unresolved:
            old = results[i]
            budget = _Budget(policy, "frequency_%d_threshold_%d" % (int(old.frequency), bits))
            threshold, margin = None, None
            try:
                threshold = _threshold(old.estimates, epsilon, bits, budget)
                q = old.absolute_imaginary_product
                margin = RationalInterval(budget.qadd(q, -threshold.upper), budget.qadd(q, -threshold.lower))
                status = _decision(q, threshold, budget)
            except _WorkLimit as failure:
                receipts.append(budget.receipt(failure))
                if old.threshold is not None:
                    results[i] = FrequencyResult(old.frequency, old.estimates, old.absolute_imaginary_product,
                                                 old.epsilon, old.threshold, old.margin, "abstain",
                                                 old.precision_bits, "resource_limit")
                else:
                    results[i] = FrequencyResult(old.frequency, old.estimates, old.absolute_imaginary_product,
                                                 epsilon, threshold, margin, "abstain", bits, "resource_limit")
            else:
                receipts.append(budget.receipt())
                results[i] = FrequencyResult(old.frequency, old.estimates, q, epsilon, threshold, margin,
                                             status, bits, "numerical_ambiguity" if status == "abstain" else None)
    return _aggregate(spec, results, receipts)
