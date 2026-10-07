"""Provide checked analysis planning and a supplied-parameter control."""
from dataclasses import dataclass
from fractions import Fraction as Q
import hashlib
import importlib.util
from pathlib import Path
import sys

_OPT_PATH = Path(__file__).resolve().parent.parent/'code/optimized.py'
_OPT_HASH = '353d48dcff4b68b98e8a33b58768a141fa2758b82afa1be3554b68d40867d19c'
if hashlib.sha256(_OPT_PATH.read_bytes()).hexdigest() != _OPT_HASH:
    raise RuntimeError('The frozen optimized source identity changed.')
_name = '_estimated_source_012_optimized_002'
if _name in sys.modules:
    opt = sys.modules[_name]
    if Path(opt.__file__).resolve() != _OPT_PATH.resolve():
        raise RuntimeError('The cached optimized module has an unknown identity.')
else:
    _spec = importlib.util.spec_from_file_location(_name, _OPT_PATH)
    opt = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = opt
    _spec.loader.exec_module(opt)
B = opt.reference


@dataclass(frozen=True)
class CalibrationDesign:
    rows: object
    positions: object


@dataclass(frozen=True)
class DesignCertificate:
    rows: int
    columns: int
    positions: tuple
    residual_degrees: int
    gram_eigenvalue_lower: Q
    residual_lower: B.Interval
    coefficient_upper: B.Interval
    receipt: dict


def certify_design(design, limits=B.WorkLimits()):
    if not isinstance(design, CalibrationDesign):
        raise B.InputError('Supply CalibrationDesign.')
    ar = B.Arithmetic(limits)
    try:
        m = B.count(design.rows, limits.max_observations, 'calibration_rows')
        p = B.count(design.positions, limits.max_regression_columns, 'calibration_columns')
        if p < 1 or m <= p:
            raise B.InputError('The design needs coefficients and residual degrees of freedom.')
        positions = tuple(design.positions[i] for i in range(p))
        if any(type(x) is not int or abs(x).bit_length() > limits.max_input_bits for x in positions):
            raise B.InputError('Positions must be bounded integers.')
        if len(set(positions)) != p:
            raise B.InputError('Positions must be distinct.')
        gram = [[Q(0) for _ in range(p)] for _ in range(p)]
        binding = hashlib.sha256()
        for i in range(m):
            if B.count(design.rows[i], limits.max_regression_columns, 'design_columns') != p:
                raise B.InputError('Each design row must match the coefficient count.')
            row = tuple(ar.input(design.rows[i][j]) for j in range(p))
            binding.update(B.digest(row).encode())
            for j in range(p):
                for k in range(p):
                    gram[j][k] = ar.addq(gram[j][k], ar.mulq(row[j], row[k]))
        inv = B.inverse(gram, ar)
        if inv is None:
            return B.Unavailable('singular_calibration_design', ar.receipt())
        gram_lower = ar.divq(1, max(ar.sumq(abs(x) for x in row) for row in inv))
        nu = m-p
        t = ar.log(Q(200))
        residual_lower = ar.sub(ar.point(nu), ar.mul(ar.point(2), ar.sqrt(ar.mul(ar.point(nu), t))))
        if residual_lower.lower <= 0:
            return B.Unavailable('nonpositive_calibration_tail_denominator', ar.receipt())
        coefficient_upper = ar.add(ar.add(ar.point(p), ar.mul(ar.point(2), ar.sqrt(ar.mul(ar.point(p), t)))), ar.mul(ar.point(2), t))
        return DesignCertificate(m, p, positions, nu, gram_lower, residual_lower, coefficient_upper,
                                 dict(design_sha256=binding.hexdigest(), **ar.receipt()))
    except B.WorkLimit as error:
        return B.Unavailable(str(error), ar.receipt())


def plan_response(true_coefficients, noise_variance, certificate, ar):
    if not isinstance(certificate, DesignCertificate):
        raise B.InputError('Supply a completed design-only certificate.')
    if ar.receipt()['limits'] != certificate.receipt['limits']:
        raise B.InputError('Planning must use the certified design limits and precision.')
    p = B.count(true_coefficients, ar.limits.max_regression_columns, 'true_coefficients')
    if p != certificate.columns:
        raise B.InputError('True coefficients must match all design columns.')
    coefficients = tuple(ar.input(true_coefficients[i]) for i in range(p))
    noise_variance = ar.input(noise_variance)
    if noise_variance <= 0:
        raise B.InputError('Calibration noise variance must be positive.')
    nu = certificate.residual_degrees
    t = ar.log(Q(100))
    residual_upper = ar.add(ar.add(ar.point(nu), ar.mul(ar.point(2), ar.sqrt(ar.mul(ar.point(nu), t)))), ar.mul(ar.point(2), t))
    radius = ar.sqrt(ar.div(ar.mul(ar.mul(ar.point(noise_variance), residual_upper), certificate.coefficient_upper),
                            ar.mul(certificate.residual_lower, ar.point(certificate.gram_eigenvalue_lower)))).upper
    _, phase = B.response_bounds(coefficients, certificate.positions, ar.mulq(2, radius), ar)
    return dict(shape_upper=Q(p), phase_upper=phase, planned_coefficient_radius=radius,
                residual_event_failure=Q(1, 100), scope='True coefficients and calibration noise variance are analysis inputs.')


@dataclass(frozen=True)
class SuppliedPhi:
    phi: Q
    receipt: dict
    status: str = 'SUPPLIED_PARAMETER'

    @property
    def lower(self):
        return self.phi

    @property
    def upper(self):
        return self.phi


@dataclass(frozen=True)
class ControlResult:
    decision: str
    source: SuppliedPhi
    calibration: object
    members: tuple
    receipt: dict


def run_supplied_phi(plan, record, calibration, phi):
    if not isinstance(plan, B.BankPlan):
        raise B.InputError('Construct BankPlan before the control call.')
    if type(phi) is not Q or not 0 <= phi < 1:
        raise B.InputError('Supply an exact Fraction parameter in [0,1).')
    phases = dict(input=0, shared_moments=0, source=0, calibration=0, members=[0 for _ in plan.coefficients])
    receipt = dict(bank=plan.coefficients, source_budget=plan.source_budget,
                   calibration_budget=plan.calibration_budget, variance_budget=plan.variance_budget,
                   tail_budget=plan.tail_budget, size_bound=Q(1, 20), power_status='POWER_UNVERIFIED',
                   source_budget_use='reserved and unused', statistic_loss=Q(0), accumulation='auto')
    source = SuppliedPhi(phi, dict(provenance='Supplied true parameter for paired analysis.'))
    def finish(decision, fitted, members):
        receipt['phase_operations'] = phases.copy()
        receipt['total_executed_operations'] = sum(phases[k] for k in ('input','shared_moments','source','calibration'))+sum(phases['members'])
        return ControlResult(decision, source, fitted, tuple(members), receipt)
    def abstain(reason, details):
        missing = B.Unavailable(reason, details)
        return finish('ABSTAIN', missing, [missing for _ in plan.coefficients])
    ar = B.Arithmetic(plan.limits)
    try:
        ar.input(phi)
        prepared = opt._prepare(record, ar)
        receipt.update(record_length=len(prepared.values), recording_sha256=prepared.input_sha256, **ar.receipt())
    except B.WorkLimit as error:
        phases['input'] = ar.operations
        return abstain(str(error), ar.receipt())
    phases['input'] = ar.operations
    if len(prepared.values) < 4:
        return abstain('too_few_transformed_observations', {})
    moment_ar = B.Arithmetic(plan.limits)
    try:
        polynomials = opt._polynomials(prepared, moment_ar, 'auto')
    except B.WorkLimit as error:
        phases['shared_moments'] = moment_ar.operations
        details = dict(maximum_integer_bits=getattr(moment_ar, 'maximum_integer_bits', 0), **moment_ar.receipt())
        receipt['shared_moments'] = dict(failure=str(error), **details)
        return abstain(str(error), details)
    phases['shared_moments'] = moment_ar.operations
    receipt['shared_moments'] = polynomials.receipt
    fitted = B.fit_calibration(calibration, plan.limits)
    phases['calibration'] = fitted.receipt.get('operations', 0)
    if isinstance(fitted, B.Unavailable):
        return finish('ABSTAIN', fitted, [fitted for _ in plan.coefficients])
    members = tuple(opt._member(a, polynomials, source, fitted, len(plan.coefficients), plan.limits) for a in plan.coefficients)
    phases['members'] = [member.receipt.get('operations', 0) for member in members]
    available = [member for member in members if isinstance(member, B.MemberDecision)]
    decision = 'REJECT' if any(member.reject for member in available) else ('DO_NOT_REJECT' if available else 'ABSTAIN')
    return finish(decision, fitted, members)
