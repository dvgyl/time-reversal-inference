"""Enclose cumulative source confidence for exact supplied observations."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class DependencyIdentity:
    optimized: str
    reference: str


DEPENDENCIES = DependencyIdentity(
    "353d48dcff4b68b98e8a33b58768a141fa2758b82afa1be3554b68d40867d19c",
    "b99bf999b31dd4f0760c6e20866251a59b530275413479a5c2e6dcd7ba6d9d9f",
)


def _load_vendor():
    directory = Path(__file__).with_name("vendor")
    for name, expected in (("optimized.py", DEPENDENCIES.optimized),
                           ("reference_certified.py", DEPENDENCIES.reference)):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("The source dependency identity changed.")
    name = "_pre014_source_optimized_v001"
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != (directory / "optimized.py").resolve():
            raise RuntimeError("The cached source dependency has a different path.")
        return module
    spec = importlib.util.spec_from_file_location(name, directory / "optimized.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


frozen = _load_vendor()
reference = frozen.reference
Arithmetic = reference.Arithmetic
BankPlan = reference.BankPlan
CalibrationData = reference.CalibrationData
InputError = reference.InputError
Interval = reference.Interval
MemberDecision = reference.MemberDecision
Quadratic = reference.Quadratic
Unavailable = reference.Unavailable
WorkLimit = reference.WorkLimit
WorkLimits = reference.WorkLimits
ZERO = Quadratic(Q(0), Q(0), Q(0))


def _probability(value):
    if type(value) is not Q or not 0 < value < 1:
        raise InputError("Supply an exact Fraction probability between zero and one.")


def _cutoff_name(value):
    if value not in ("s18_cn", "s18_cstar") or type(value) is not str:
        raise InputError("Use s18_cn or s18_cstar.")


@dataclass(frozen=True)
class LagPlan:
    alpha: Q = Q(1, 200)

    def __post_init__(self):
        _probability(self.alpha)
        if self.alpha != Q(1, 200):
            raise InputError("A standalone source plan requires error 1/200.")


@dataclass(frozen=True)
class BridgePlan:
    alpha: Q = Q(1, 200)
    cutoff: str = "s18_cn"

    def __post_init__(self):
        _probability(self.alpha)
        _cutoff_name(self.cutoff)
        if self.alpha != Q(1, 200):
            raise InputError("A standalone source plan requires error 1/200.")


@dataclass(frozen=True)
class IntersectionPlan:
    alpha_lag: Q = Q(1, 400)
    alpha_bridge: Q = Q(1, 400)
    cutoff: str = "s18_cn"

    def __post_init__(self):
        _probability(self.alpha_lag)
        _probability(self.alpha_bridge)
        _cutoff_name(self.cutoff)
        if self.alpha_lag + self.alpha_bridge != Q(1, 200):
            raise InputError("Intersection source errors must sum to 1/200.")


SourcePlan = Union[LagPlan, BridgePlan, IntersectionPlan]


@dataclass(frozen=True)
class SourceBankPlan:
    bank: BankPlan
    source: SourcePlan

    def __post_init__(self):
        if not isinstance(self.bank, BankPlan):
            raise InputError("Supply the fixed BankPlan.")
        if type(self.source) not in (LagPlan, BridgePlan, IntersectionPlan):
            raise InputError("Supply a declared source plan.")


@dataclass(frozen=True)
class SuppliedSource:
    phi: Q
    population_id: str

    def __post_init__(self):
        if type(self.phi) is not Q or not 0 <= self.phi < 1:
            raise InputError("Supply an exact Fraction persistence in [0,1).")
        if type(self.population_id) is not str or not self.population_id.strip():
            raise InputError("Supply a nonempty population identifier.")


@dataclass(frozen=True)
class PhaseFailure:
    phase: str
    reason: str
    receipt: dict


@dataclass(frozen=True)
class RecordIdentity:
    original_length: int
    input_sha256: str
    reference_input_sha256: str
    limits: WorkLimits
    accumulation: str
    dependencies: DependencyIdentity


@dataclass(frozen=True)
class BasePolynomials:
    identity: RecordIdentity
    polynomials: frozen.RecordingPolynomials
    receipt: dict


@dataclass(frozen=True)
class BridgePolynomial:
    identity: RecordIdentity
    cumulative_norm: Quadratic
    receipt: dict


@dataclass(frozen=True)
class CompiledRecord:
    identity: RecordIdentity
    bank_polynomials: Union[BasePolynomials, PhaseFailure]
    bridge_polynomial: Union[BridgePolynomial, PhaseFailure]
    preparation_receipt: dict


@dataclass(frozen=True)
class CompilationFailure:
    limits: WorkLimits
    accumulation: str
    dependencies: DependencyIdentity
    failure: PhaseFailure


CompiledResult = Union[CompiledRecord, CompilationFailure]


@dataclass(frozen=True)
class CutoffCertificate:
    method: str
    alpha: Q
    n: int
    m: int
    log_interval: Interval
    pi_interval: Interval
    cutoff_interval: Interval
    cutoff_lower: Q
    receipt: dict


@dataclass(frozen=True)
class SourceOuterHull:
    lower: Q
    upper: Q
    retained_cells: Tuple[Interval, ...]
    unresolved_cells: Tuple[Interval, ...]
    acceptance_polynomials: Tuple[Quadratic, ...]
    lag_radius: Optional[Interval]
    bridge_cutoff: Optional[CutoffCertificate]
    receipt: dict
    status: str = "OUTER_HULL"


@dataclass(frozen=True)
class SourceEmpty:
    receipt: dict
    status: str = "EMPTY_CERTIFIED"


@dataclass(frozen=True)
class SourceFallback:
    reason: str
    receipt: dict
    lower: Q = Q(0)
    upper: Q = Q(1)
    status: str = "FULL_FALLBACK"


SourceResult = Union[SourceOuterHull, SourceEmpty, SourceFallback]


@dataclass(frozen=True)
class SourceBankResult:
    decision: str
    source: SourceResult
    calibration: Union[reference.CalibrationCertificate, Unavailable]
    members: Tuple[Union[MemberDecision, Unavailable], ...]
    receipt: dict


def _phase(ar, phase, **details):
    details.setdefault("maximum_integer_bits", getattr(ar, "maximum_integer_bits", 0))
    return dict(ar.receipt(), phase=phase, **details)


def _failure(ar, phase, reason):
    return PhaseFailure(phase, reason, _phase(ar, phase, failure=reason))


def _check_policy(limits, accumulation):
    if not isinstance(limits, WorkLimits):
        raise InputError("Supply WorkLimits.")
    if type(accumulation) is not str or accumulation not in ("auto", "rational", "dyadic"):
        raise InputError("Use auto, rational, or dyadic accumulation.")


def _cumulative(prepared, identity, ar):
    n = identity.original_length - 1
    exponent = prepared.common_dyadic_exponent if identity.accumulation == "dyadic" else None
    ring = frozen.MomentArithmetic(ar, exponent)
    a = b = ring.convert(Q(0))
    uaa = uab = ubb = va = vb = a
    for j in range(1, n + 1):
        a = ring.add(a, ring.convert(prepared.values[j][0]))
        b = ring.add(b, ring.convert(prepared.values[j - 1][0]))
        if j < n:
            uaa = ring.add(uaa, ring.mul(a, a))
            uab = ring.add(uab, ring.mul(a, b))
            ubb = ring.add(ubb, ring.mul(b, b))
            va = ring.add(va, ring.mul(j, a))
            vb = ring.add(vb, ring.mul(j, b))
    a, b, va, vb = (ring.fraction(x, 1) for x in (a, b, va, vb))
    uaa, uab, ubb = (ring.fraction(x, 2) for x in (uaa, uab, ubb))
    nq = ar.q(n)
    n2 = ar.mulq(nq, nq)
    h = ar.divq(ar.mulq(ar.mulq(nq, ar.q(n - 1)), ar.q(2 * n - 1)), 6)

    def centered(raw, total_left, index_left, total_right, index_right):
        cross = ar.addq(ar.mulq(total_left, index_right), ar.mulq(total_right, index_left))
        last = ar.divq(ar.mulq(ar.mulq(total_left, total_right), h), n2)
        return ar.addq(ar.q(raw - ar.divq(cross, nq)), last)

    poly = Quadratic(centered(uaa, a, va, a, va),
                     ar.mulq(-2, centered(uab, a, va, b, vb)),
                     centered(ubb, b, vb, b, vb))
    receipt = _phase(ar, "bridge_moments", maximum_integer_bits=ring.maximum_integer_bits,
                     engine=identity.accumulation)
    return BridgePolynomial(identity, poly, receipt)


def compile_record(record, limits=WorkLimits(), accumulation="auto"):
    """Compile each moment family with a separate work budget."""
    _check_policy(limits, accumulation)
    ar = Arithmetic(limits)
    try:
        prepared = frozen._prepare(record, ar)
    except WorkLimit as error:
        return CompilationFailure(limits, accumulation, DEPENDENCIES,
                                  _failure(ar, "preparation", str(error)))
    if accumulation == "dyadic" and prepared.common_dyadic_exponent is None:
        raise InputError("Forced dyadic accumulation requires dyadic inputs.")
    engine = "dyadic" if accumulation != "rational" and prepared.common_dyadic_exponent is not None else "rational"
    identity = RecordIdentity(len(prepared.values), prepared.input_sha256,
                              prepared.reference_input_sha256, limits, engine, DEPENDENCIES)
    preparation = _phase(ar, "preparation", requested_accumulation=accumulation)
    base_ar, bridge_ar = Arithmetic(limits), Arithmetic(limits)
    if identity.original_length < 3:
        return CompiledRecord(identity, _failure(base_ar, "base_moments", "small_record"),
                              _failure(bridge_ar, "bridge_moments", "small_record"), preparation)
    try:
        base = frozen._polynomials(prepared, base_ar, engine)
        base = BasePolynomials(identity, base, _phase(base_ar, "base_moments", moments=base.receipt))
    except WorkLimit as error:
        base = _failure(base_ar, "base_moments", str(error))
    try:
        bridge = _cumulative(prepared, identity, bridge_ar)
    except WorkLimit as error:
        bridge = _failure(bridge_ar, "bridge_moments", str(error))
    return CompiledRecord(identity, base, bridge, preparation)


def _validate_compiled(compiled, limits):
    if isinstance(compiled, CompilationFailure):
        if compiled.limits != limits or compiled.dependencies != DEPENDENCIES:
            raise InputError("The compiled policy or dependency identity does not match.")
        return
    if not isinstance(compiled, CompiledRecord):
        raise InputError("Supply an unmodified result from compile_record.")
    identity = compiled.identity
    if identity.limits != limits or identity.dependencies != DEPENDENCIES:
        raise InputError("The compiled policy or dependency identity does not match.")
    if identity.accumulation not in ("dyadic", "rational"):
        raise InputError("The compiled accumulation path is invalid.")
    if type(identity.original_length) is not int or not 0 <= identity.original_length <= limits.max_observations:
        raise InputError("The compiled record length is invalid.")
    base = compiled.bank_polynomials
    if not isinstance(base, (BasePolynomials, PhaseFailure)):
        raise InputError("The compiled base moments have an invalid type.")
    if not isinstance(compiled.bridge_polynomial, (BridgePolynomial, PhaseFailure)):
        raise InputError("The compiled bridge moments have an invalid type.")
    for part in (base, compiled.bridge_polynomial):
        if not isinstance(part, PhaseFailure) and part.identity != identity:
            raise InputError("The compiled moment identities do not match.")
    if isinstance(base, BasePolynomials):
        p = base.polynomials
        if (p.original_length, p.input_sha256, p.reference_input_sha256) != (
                identity.original_length, identity.input_sha256, identity.reference_input_sha256):
            raise InputError("The bank polynomial identity does not match.")
        if p.receipt["engine"] != identity.accumulation:
            raise InputError("The bank accumulation path does not match.")
    if isinstance(compiled.bridge_polynomial, BridgePolynomial):
        if compiled.bridge_polynomial.receipt["engine"] != identity.accumulation:
            raise InputError("The bridge accumulation path does not match.")


def _atan_interval(x, ar):
    x = ar.q(x)
    x2 = ar.mulq(x, x)
    power, total = x, Q(0)
    for j in range(ar.limits.log_terms):
        term = ar.divq(power, 2 * j + 1)
        total = ar.addq(total, term if j % 2 == 0 else ar.q(-term))
        power = ar.mulq(power, x2)
    tail = ar.divq(power, 2 * ar.limits.log_terms + 1)
    if ar.limits.log_terms % 2 == 0:
        return Interval(total, ar.addq(total, tail))
    return Interval(ar.q(total - tail), total)


def _pi_interval(ar):
    return ar.sub(ar.mul(ar.point(16), _atan_interval(Q(1, 5), ar)),
                  ar.mul(ar.point(4), _atan_interval(Q(1, 239), ar)))


def _cutoff(n, alpha, method, ar):
    t = ar.log(ar.divq(2, alpha))
    nine_t = ar.mul(ar.point(9), t)
    ceilings = tuple(int(ar.q(-(-x.numerator // x.denominator))) for x in (nine_t.lower, nine_t.upper))
    if ceilings[0] != ceilings[1]:
        return _failure(ar, "source", "uncertain_cutoff_integer")
    m = ceilings[0]
    if n < m + 1:
        return _failure(ar, "source", "cutoff_length_condition")
    pi = _pi_interval(ar)
    pi2 = ar.mul(pi, pi)
    if pi.lower <= 0:
        return _failure(ar, "source", "nonpositive_pi_lower")
    if method == "s18_cstar":
        value = ar.div(ar.point(3), ar.mul(ar.point(ar.mulq(17, m)), pi2))
    else:
        numerator = ar.mul(ar.point(n), ar.sub(ar.point(m),
                           ar.mul(ar.point(2), ar.sqrt(ar.mul(ar.point(m), t)))))
        vn = ar.add(ar.add(ar.point(n - 1), ar.mul(ar.point(2),
                    ar.sqrt(ar.mul(ar.point(n - 1), t)))), ar.mul(ar.point(2), t))
        denominator = ar.mul(ar.mul(pi2, ar.point(ar.mulq(m, m))), vn)
        if numerator.lower <= 0 or denominator.lower <= 0:
            return _failure(ar, "source", "nonpositive_cutoff_bound")
        value = ar.div(numerator, denominator)
    if value.lower <= 0:
        return _failure(ar, "source", "nonpositive_cutoff_lower")
    return CutoffCertificate(method, alpha, n, m, t, pi, value, value.lower,
                             dict(pi_terms=ar.limits.log_terms, **_phase(ar, "source")))


def cutoff_certificate(n, alpha, method="s18_cn", limits=WorkLimits()):
    """Enclose a fixed S18 cutoff without reading observations."""
    _probability(alpha)
    _cutoff_name(method)
    if type(n) is not int or n < 0:
        raise InputError("Supply a nonnegative integer innovation count.")
    ar = Arithmetic(limits)
    try:
        ar.input(n)
        if n > limits.max_observations - 1:
            return _failure(ar, "source", "cutoff_observation_limit")
        return _cutoff(n, alpha, method, ar)
    except WorkLimit as error:
        return _failure(ar, "source", str(error))


def _lag_radius(n, alpha, ar):
    if alpha == Q(1, 200):
        return reference.acceptance_radius(n, ar)
    d = n - 1
    t0, t1 = ar.log(ar.divq(2, alpha)), ar.log(ar.divq(4, alpha))
    bottom = ar.sub(ar.point(d), ar.mul(ar.point(2), ar.sqrt(ar.mul(ar.point(d), t0))))
    if bottom.lower <= 0:
        return None
    norm2 = ar.addq(ar.divq(n - 3, 2), ar.divq(2, ar.mulq(n, n)))
    top = ar.add(ar.mul(ar.point(2), ar.sqrt(ar.mul(ar.point(norm2), t1))),
                 ar.mul(ar.point(ar.addq(2, ar.divq(2, n))), t1))
    return ar.div(top, bottom)


def _enclose(polynomials, ar):
    if ar.limits.max_cells < 1:
        raise WorkLimit("max_cells")
    pending = [(Interval(Q(0), Q(1)), 0)]
    kept, unresolved, excluded = [], [], []
    evaluated, limited = 0, False
    while pending:
        cell, depth = pending.pop()
        ranges = tuple(q.range(cell, ar) for q in polynomials)
        evaluated += 1
        if any(x.lower > 0 for x in ranges):
            excluded.append((cell, ranges))
        elif all(x.upper <= 0 for x in ranges):
            kept.append(cell)
        elif depth < ar.limits.max_depth and evaluated + len(pending) + 2 <= ar.limits.max_cells:
            mid = ar.divq(ar.addq(cell.lower, cell.upper), 2)
            pending.extend(((Interval(mid, cell.upper), depth + 1),
                            (Interval(cell.lower, mid), depth + 1)))
        else:
            kept.append(cell)
            unresolved.append(cell)
            limited = limited or depth < ar.limits.max_depth
    details = dict(cells_evaluated=evaluated, excluded_cells=tuple(excluded),
                   cell_limit_reached=limited,
                   unresolved_width=max((ar.q(x.width) for x in unresolved), default=Q(0)))
    return tuple(kept), tuple(unresolved), details


def _source_receipt(ar, plan, compiled, **details):
    return _phase(ar, "source", plan=plan, identity=compiled.identity, **details)


def source_confidence(plan, compiled, limits=WorkLimits()):
    if type(plan) not in (LagPlan, BridgePlan, IntersectionPlan):
        raise InputError("Supply a declared source plan.")
    _validate_compiled(compiled, limits)
    ar = Arithmetic(limits)
    if isinstance(compiled, CompilationFailure):
        return SourceFallback(compiled.failure.reason,
                              _phase(ar, "source", cause=compiled.failure, plan=plan))
    n = compiled.identity.original_length - 1
    if n < 2:
        return SourceFallback("small_record", _source_receipt(ar, plan, compiled))
    if isinstance(compiled.bank_polynomials, PhaseFailure):
        return SourceFallback("base_moments_unavailable",
                              _source_receipt(ar, plan, compiled, cause=compiled.bank_polynomials))
    p = compiled.bank_polynomials.polynomials
    denominator, lag = p.innovation_norms[0], p.reference_lag
    try:
        radius, cutoff, polynomials = None, None, []
        if isinstance(plan, (BridgePlan, IntersectionPlan)) and isinstance(compiled.bridge_polynomial, PhaseFailure):
            return SourceFallback("bridge_moments_unavailable", _source_receipt(
                ar, plan, compiled, cause=compiled.bridge_polynomial))
        if denominator == ZERO:
            return SourceOuterHull(Q(0), Q(1), (Interval(Q(0), Q(1)),), (), (), None, None,
                                   _source_receipt(ar, plan, compiled, degeneracy="zero_innovation_norm"))
        if isinstance(plan, (LagPlan, IntersectionPlan)):
            alpha = plan.alpha if isinstance(plan, LagPlan) else plan.alpha_lag
            radius = _lag_radius(n, alpha, ar)
            if radius is None:
                return SourceFallback("nonpositive_lag_radius_denominator", _source_receipt(ar, plan, compiled))
            mu = ar.divq(-1, n)
            polynomials.append(reference.combine(lag, denominator, ar.q(-mu - radius.upper), ar))
            q2 = reference.combine(ZERO, lag, Q(-1), ar)
            polynomials.append(reference.combine(q2, denominator, ar.q(mu - radius.upper), ar))
        if isinstance(plan, (BridgePlan, IntersectionPlan)):
            alpha = plan.alpha if isinstance(plan, BridgePlan) else plan.alpha_bridge
            cutoff = _cutoff(n, alpha, plan.cutoff, ar)
            if isinstance(cutoff, PhaseFailure):
                return SourceFallback(cutoff.reason, _source_receipt(ar, plan, compiled, cause=cutoff))
            negative_q = reference.combine(ZERO, compiled.bridge_polynomial.cumulative_norm, Q(-1), ar)
            polynomials.append(reference.combine(negative_q, denominator, ar.mulq(n, cutoff.cutoff_lower), ar))
        polynomials = tuple(polynomials)
        kept, unresolved, details = _enclose(polynomials, ar)
        receipt = _source_receipt(ar, plan, compiled, **details)
        if not kept:
            receipt.update(acceptance_polynomials=polynomials, lag_radius=radius, bridge_cutoff=cutoff)
            return SourceEmpty(receipt)
        return SourceOuterHull(min(x.lower for x in kept), max(x.upper for x in kept), kept,
                               unresolved, polynomials, radius, cutoff, receipt)
    except WorkLimit as error:
        return SourceFallback(str(error), _source_receipt(ar, plan, compiled, failure=str(error)))


def _compilation_receipts(compiled):
    if isinstance(compiled, CompilationFailure):
        return dict(preparation=compiled.failure.receipt, base_moments=None, bridge_moments=None)
    return dict(preparation=compiled.preparation_receipt,
                base_moments=compiled.bank_polynomials.receipt,
                bridge_moments=compiled.bridge_polynomial.receipt)


def _finish_bank(bank, compiled, calibration, source, source_budget, source_plan):
    receipt = dict(bank=bank.coefficients, source_plan=source_plan, source_budget=source_budget,
                   calibration_budget=bank.calibration_budget, variance_budget=bank.variance_budget,
                   tail_budget=bank.tail_budget,
                   size_bound=source_budget + bank.calibration_budget + bank.variance_budget + bank.tail_budget,
                   dependencies=DEPENDENCIES, power_status="POWER_UNVERIFIED", statistic_loss=Q(0),
                   compilation=_compilation_receipts(compiled), source=source.receipt,
                   assumptions="Supplied common Gaussian AR(1) family; nonzero scalar reference; nonzero finite second response; zero testing errors; zero-mean spherical Gaussian calibration error; fixed full-rank design; no calibration intercept; exact-input arithmetic scope.")
    if isinstance(compiled, CompilationFailure) or isinstance(compiled.bank_polynomials, PhaseFailure):
        reason = "base_moments_unavailable"
    elif not isinstance(source, SourceOuterHull) or source.upper >= 1:
        reason = "no_finite_source_certificate"
    else:
        reason = None
    if reason is not None:
        fitted = Unavailable("not_run", dict(phase="calibration", operations=0))
        members = tuple(Unavailable(reason, dict(phase="member", coefficient=a, operations=0)) for a in bank.coefficients)
    else:
        fitted = reference.fit_calibration(calibration, bank.limits)
        if isinstance(fitted, Unavailable):
            members = tuple(Unavailable(fitted.reason, dict(phase="member", coefficient=a,
                                  operations=0, calibration_failure=fitted.receipt)) for a in bank.coefficients)
        else:
            p = compiled.bank_polynomials.polynomials
            members = tuple(frozen._member(a, p, source, fitted, len(bank.coefficients), bank.limits)
                            for a in bank.coefficients)
    available = tuple(x for x in members if isinstance(x, MemberDecision))
    decision = "REJECT" if any(x.reject for x in available) else ("DO_NOT_REJECT" if available else "ABSTAIN")
    receipt["calibration"] = dict(fitted.receipt, phase="calibration")
    receipt["members"] = tuple(dict(x.receipt, phase="member", coefficient=a) for a, x in zip(bank.coefficients, members))
    return SourceBankResult(decision, source, fitted, members, receipt)


def run_compiled_bank(plan, compiled, calibration):
    if not isinstance(plan, SourceBankPlan):
        raise InputError("Supply SourceBankPlan before observations.")
    _validate_compiled(compiled, plan.bank.limits)
    source = source_confidence(plan.source, compiled, plan.bank.limits)
    return _finish_bank(plan.bank, compiled, calibration, source, Q(1, 200), plan.source)


def run_supplied_bank(plan, compiled, calibration, source):
    if not isinstance(plan, BankPlan) or not isinstance(source, SuppliedSource):
        raise InputError("Supply BankPlan and SuppliedSource before observations.")
    _validate_compiled(compiled, plan.limits)
    ar = Arithmetic(plan.limits)
    try:
        phi = ar.input(source.phi)
        singleton = Interval(phi, phi)
        hull = SourceOuterHull(phi, phi, (singleton,), (), (), None, None,
                               _phase(ar, "source", supplied_source=source))
    except WorkLimit as error:
        hull = SourceFallback(str(error), _phase(ar, "source", failure=str(error), supplied_source=source))
    return _finish_bank(plan, compiled, calibration, hull, Q(0), source)
