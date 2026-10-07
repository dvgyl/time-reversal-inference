"""Invert both cumulative-pivot tails over the complete candidate interval."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as Q
from math import isqrt
from pathlib import Path
import importlib.util
import sys

_path=Path(__file__).with_name('baseline')/'source_confidence.py'
_spec=importlib.util.spec_from_file_location('_pre016_frozen_source',_path)
base=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=base
_spec.loader.exec_module(base)
compile_record=base.compile_record
WorkLimits=base.WorkLimits
BankPlan=base.BankPlan
CalibrationData=base.CalibrationData
Quadratic=base.Quadratic
Interval=base.Interval


@dataclass(frozen=True)
class TwoSidedPlan:
    n: int
    alpha_bridge: Q
    lower_cutoff: Q
    upper_cutoff: Q
    cutoff_identity: str
    alpha_lag: Q = Q(0)
    root_bits: int = 48

    def __post_init__(self):
        if type(self.n) is not int or type(self.root_bits) is not int:
            raise ValueError('Length and root precision must be exact integers.')
        if any(type(x) is not Q for x in (self.alpha_bridge,self.alpha_lag,self.lower_cutoff,self.upper_cutoff)):
            raise ValueError('Probabilities and cutoffs must be exact Fractions.')
        if type(self.cutoff_identity) is not str:
            raise ValueError('The cutoff identity must be a string.')
        if self.alpha_bridge+self.alpha_lag != Q(1,200):
            raise ValueError('The source errors must sum to 0.005.')
        if self.alpha_bridge<=0 or self.alpha_lag<0:
            raise ValueError('Bridge error must be positive; lag error must be nonnegative.')
        if not 0<self.lower_cutoff<=self.upper_cutoff or not self.cutoff_identity:
            raise ValueError('Supply positive ordered cutoffs and their certificate identity.')
        if self.n<2 or not 8<=self.root_bits<=256:
            raise ValueError('Require n >= 2 and 8 <= root_bits <= 256.')


def _at(q,x,ar=None):
    if ar is None:return q.constant+x*(q.linear+x*q.square)
    return ar.addq(q.constant,ar.mulq(x,ar.addq(q.linear,ar.mulq(x,q.square))))


def _range(q,a,b,ar=None):
    values=[_at(q,a,ar),_at(q,b,ar)]
    if q.square:
        v=(-q.linear/(2*q.square) if ar is None else ar.divq(ar.q(-q.linear),ar.mulq(2,q.square)))
        if a<v<b: values.append(_at(q,v,ar))
    return min(values),max(values)


def _sqrt_interval(q,bits,ar):
    shifted_bits=q.numerator.bit_length()+2*bits
    if shifted_bits>ar.limits.max_rational_bits:raise base.WorkLimit('root_shift_integer_bits')
    ar.q(Q(1,1<<bits))
    k=isqrt((q.numerator << (2*bits))//q.denominator)
    ar.q(k)
    lo=ar.q(Q(k,1<<bits))
    return lo,lo if ar.mulq(lo,lo)==q else ar.q(Q(k+1,1<<bits))


def _roots(q,bits,ar):
    if not q.square:
        if q.linear:
            x=ar.divq(ar.q(-q.constant),q.linear)
            return [(x,x)]
        return []
    factor=abs(q.square)
    q=Quadratic(*(ar.divq(x,factor) for x in (q.constant,q.linear,q.square)))
    disc=ar.q(ar.mulq(q.linear,q.linear)-ar.mulq(4,ar.mulq(q.square,q.constant)))
    if disc<0:return []
    lo,hi=_sqrt_interval(disc,bits,ar)
    den=ar.mulq(2,q.square)
    roots=[]
    for a,b in [(ar.q(-q.linear-hi),ar.q(-q.linear-lo)),(ar.q(-q.linear+lo),ar.q(-q.linear+hi))]:
        roots.append(tuple(sorted((ar.divq(a,den),ar.divq(b,den)))))
    return roots


def continuous_outer_set(polynomials,bits=48,ar=None):
    """Retain every solution of q <= 0, including isolated roots."""
    if type(bits) is not int or not 8<=bits<=256:raise ValueError('Root precision must be an integer from 8 through 256.')
    ar=ar or base.Arithmetic(WorkLimits())
    polys=tuple(polynomials)
    if len(polys)>4:raise ValueError('At most four acceptance quadratics are supported.')
    for q in polys:
        if type(q) is not Quadratic or any(type(x) is not Q for x in (q.constant,q.linear,q.square)):
            raise ValueError('Use Quadratic objects with exact Fraction coefficients.')
        for x in (q.constant,q.linear,q.square):ar.q(x)
    roots=[]
    for q in polys:
        for a,b in _roots(q,bits,ar):
            if b>=0 and a<=1:roots.append((max(Q(0),a),min(Q(1),b)))
    points=sorted({Q(0),Q(1),*(x for pair in roots for x in pair)})
    kept=[];unresolved=[]
    for a,b in zip(points,points[1:]):
        inside=any(a<r and b>l for l,r in roots)
        if inside:
            if not any(_range(q,a,b,ar)[0]>0 for q in polys):
                kept.append((a,b));unresolved.append((a,b))
        elif all(_at(q,ar.divq(ar.addq(a,b),2),ar)<=0 for q in polys):kept.append((a,b))
    for x in points:
        if all(_at(q,x,ar)<=0 for q in polys):kept.append((x,x))
    merged=[]
    for a,b in sorted(kept):
        if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:merged.append((a,b))
    return tuple(Interval(*x) for x in merged),tuple(Interval(*x) for x in unresolved)


def _rounding_polynomials(d,q,n,error,ar,method='telescoping'):
    if method not in ('global','telescoping'):raise ValueError('Use global or telescoping rounding bounds.')
    dmax=max(_at(d,Q(0),ar),_at(d,Q(1),ar))
    dsqrt=ar.sqrt(ar.point(dmax)).upper
    unit=ar.mulq(ar.sqrt(ar.point(n)).upper,error)
    if method=='telescoping':bc,bl=unit,unit
    else:bc,bl=ar.mulq(2,unit),Q(0)
    def norm_error(norm,c,l):
        return Quadratic(ar.addq(ar.mulq(2,ar.mulq(norm,c)),ar.mulq(c,c)),
                         ar.addq(ar.mulq(2,ar.mulq(norm,l)),ar.mulq(2,ar.mulq(c,l))),ar.mulq(l,l))
    dp=norm_error(dsqrt,bc,bl)
    qp=None
    if q is not None:
        qmax=max(_at(q,Q(0),ar),_at(q,Q(1),ar));qsqrt=ar.sqrt(ar.point(qmax)).upper
        if method=='telescoping':
            a0=ar.mulq(ar.mulq(2,error),ar.sqrt(ar.point(n-1)).upper)
            a1=ar.mulq(Q(n,2),unit)
            ac,al=ar.addq(a0,a1),ar.q(-a1)
        else:ac,al=ar.mulq(n,unit),Q(0)
        qp=norm_error(qsqrt,ac,al)
    return dp,qp,{'rounding_error':error,'rounding_method':method,'denominator_error_polynomial':dp,
                  'numerator_error_polynomial':qp}


def source_confidence(plan,compiled,limits=WorkLimits(),*,rounding_error=Q(0),inversion="roots",rounding_method="telescoping"):
    if not isinstance(plan,TwoSidedPlan): raise ValueError('Supply TwoSidedPlan.')
    if type(rounding_error) is not Q or rounding_error<0: raise ValueError('Supply a nonnegative Fraction rounding bound.')
    if inversion not in ('roots','dyadic'):raise ValueError('Use roots or dyadic inversion.')
    base._validate_compiled(compiled,limits)
    ar=base.Arithmetic(limits)
    receipt={'method':'two_sided_continuous_quadratic','plan':plan,'root_bits':plan.root_bits,'inversion':inversion}
    if not isinstance(compiled,base.CompiledRecord):
        return base.SourceFallback('compilation_failure',receipt)
    if compiled.identity.original_length!=plan.n+1:
        raise ValueError('The cutoff length and record length differ.')
    if isinstance(compiled.bank_polynomials,base.PhaseFailure) or isinstance(compiled.bridge_polynomial,base.PhaseFailure):
        return base.SourceFallback('moments_unavailable',receipt)
    d=compiled.bank_polynomials.polynomials.innovation_norms[0]
    q=compiled.bridge_polynomial.cumulative_norm
    radius=None
    try:
        polys=[base.reference.combine(base.reference.combine(base.ZERO,q,Q(-1),ar),d,plan.n*plan.lower_cutoff,ar),
               base.reference.combine(q,d,-plan.n*plan.upper_cutoff,ar)]
        if plan.alpha_lag:
            radius=base._lag_radius(plan.n,plan.alpha_lag,ar)
            if radius is None: return base.SourceFallback('lag_denominator_nonpositive',receipt)
            lag=compiled.bank_polynomials.polynomials.reference_lag
            mu=-Q(1,plan.n)
            polys += [base.reference.combine(lag,d,-mu-radius.upper,ar),
                      base.reference.combine(base.reference.combine(base.ZERO,lag,Q(-1),ar),d,mu-radius.upper,ar)]
        if rounding_error:
            dp,qp,rounding_receipt=_rounding_polynomials(d,q,plan.n,rounding_error,ar,rounding_method)
            relax=[base.reference.combine(qp,dp,ar.mulq(plan.n,plan.lower_cutoff),ar),
                   base.reference.combine(qp,dp,ar.mulq(plan.n,plan.upper_cutoff),ar)]
            if plan.alpha_lag:
                relax += [base.reference.combine(base.ZERO,dp,ar.addq(1,abs(ar.q(-mu-radius.upper))),ar),
                          base.reference.combine(base.ZERO,dp,ar.addq(1,abs(ar.q(mu-radius.upper))),ar)]
            polys=[base.reference.combine(poly,extra,Q(-1),ar) for poly,extra in zip(polys,relax)]
            receipt.update(rounding_receipt,polynomial_relaxations=tuple(relax))
        polys=tuple(polys)
        if inversion=="roots":
            kept,unresolved=continuous_outer_set(polys,plan.root_bits,ar)
        else:
            kept,unresolved,details=base._enclose(polys,ar)
            receipt.update(details)
        receipt.update(acceptance_polynomials=polys,unresolved_cells=unresolved,
                       arithmetic=ar.receipt(),cutoff_identity=plan.cutoff_identity,
                       source_budget=plan.alpha_bridge+plan.alpha_lag)
        if not kept: return base.SourceEmpty(receipt)
        return base.SourceOuterHull(min(x.lower for x in kept),max(x.upper for x in kept),kept,
                                    unresolved,polys,radius,None,receipt)
    except base.WorkLimit as error:
        return base.SourceFallback(str(error),receipt)


def run_compiled_bank(bank,source_plan,compiled,calibration):
    source=source_confidence(source_plan,compiled,bank.limits)
    return base._finish_bank(bank,compiled,calibration,source,Q(1,200),source_plan)


def evaluate_compiled_bank(bank,compiled,source,calibration,*,member_evaluator=None,
                           fitted_calibration=None,source_plan=None,source_budget=Q(1,200)):
    """Evaluate a supplied source hull without compiling the record again.

    Callback signature: (a, polynomials, source, fitted, member_count, limits).
    Return base.MemberDecision or base.Unavailable. A supplied fitted calibration
    avoids a second calibration fit. Use this module's base.reference types.
    """
    base._validate_compiled(compiled,bank.limits)
    evaluator=member_evaluator or base.frozen._member
    if not isinstance(compiled,base.CompiledRecord) or isinstance(compiled.bank_polynomials,base.PhaseFailure):
        reason='base_moments_unavailable'
    elif not isinstance(source,base.SourceOuterHull) or source.upper>=1:
        reason='no_finite_source_certificate'
    else: reason=None
    if reason:
        fitted=base.Unavailable('not_run',{'phase':'calibration'})
        members=tuple(base.Unavailable(reason,{'coefficient':a}) for a in bank.coefficients)
    else:
        fitted=fitted_calibration if fitted_calibration is not None else base.reference.fit_calibration(calibration,bank.limits)
        if isinstance(fitted,base.Unavailable):
            members=tuple(base.Unavailable(fitted.reason,{'coefficient':a}) for a in bank.coefficients)
        else:
            members=tuple(evaluator(a,compiled.bank_polynomials.polynomials,source,fitted,
                                    len(bank.coefficients),bank.limits) for a in bank.coefficients)
    if any(not isinstance(m,(base.MemberDecision,base.Unavailable)) for m in members):
        raise TypeError('Member evaluator returned an unsupported decision type.')
    available=[m for m in members if isinstance(m,base.MemberDecision)]
    decision='REJECT' if any(m.reject for m in available) else ('DO_NOT_REJECT' if available else 'ABSTAIN')
    receipt={'source_plan':source_plan,'source_budget':source_budget,
             'calibration_budget':bank.calibration_budget,'variance_budget':bank.variance_budget,
             'tail_budget':bank.tail_budget,'size_bound':source_budget+bank.calibration_budget+bank.variance_budget+bank.tail_budget,
             'member_evaluator':evaluator.__module__+'.'+evaluator.__name__,
             'power_status':'IDEAL_POWER_REQUIRES_PROVED_CONDITIONS',
             'source':source.receipt,'calibration':fitted.receipt,
             'members':[m.receipt for m in members]}
    return base.SourceBankResult(decision,source,fitted,members,receipt)


def plan_from_certificate(path,*,alpha_lag=Q(0),root_bits=48):
    """Load a stored cutoff certificate and check both probability directions.

    The certificate is trusted output of the separately verified CDF program.
    Its full file hash is bound to each source result.
    """
    import json,hashlib
    data_bytes=Path(path).read_bytes();data=json.loads(data_bytes)
    alpha=Q(data['alpha_bridge']);low=Q(data['lower_cutoff']);high=Q(data['upper_cutoff'])
    lower_cdf=data['lower_cutoff_cdf'];upper_cdf=data['upper_cutoff_cdf']
    if Q(lower_cdf['cutoff'])!=low or Q(upper_cdf['cutoff'])!=high:
        raise ValueError('The cutoff and CDF certificate differ.')
    if lower_cdf['n']!=data['n'] or upper_cdf['n']!=data['n']:
        raise ValueError('The cutoff and CDF lengths differ.')
    if Q(lower_cdf['upper'])>alpha/2 or Q(upper_cdf['lower'])<1-alpha/2:
        raise ValueError('The CDF certificate does not cover the tail allocation.')
    return TwoSidedPlan(data['n'],alpha,low,high,hashlib.sha256(data_bytes).hexdigest(),alpha_lag,root_bits)


def lag_confidence(compiled,limits=WorkLimits(),*,alpha=Q(1,200),root_bits=48,
                   rounding_error=Q(0),inversion='roots',rounding_method='telescoping'):
    """Apply the same continuous inversion and rounding transfer to the lag set."""
    if type(alpha) is not Q or alpha!=Q(1,200):raise ValueError('Standalone lag error must be the Fraction 1/200.')
    if type(rounding_error) is not Q or rounding_error<0:raise ValueError('Supply a nonnegative Fraction rounding bound.')
    if inversion not in ('roots','dyadic'):raise ValueError('Use roots or dyadic inversion.')
    base._validate_compiled(compiled,limits);ar=base.Arithmetic(limits)
    receipt={'method':'lag_continuous_quadratic','alpha':alpha,'root_bits':root_bits,'inversion':inversion}
    if not isinstance(compiled,base.CompiledRecord) or isinstance(compiled.bank_polynomials,base.PhaseFailure):
        return base.SourceFallback('moments_unavailable',receipt)
    n=compiled.identity.original_length-1
    try:
        radius=base._lag_radius(n,alpha,ar)
        if radius is None:return base.SourceFallback('lag_denominator_nonpositive',receipt)
        moments=compiled.bank_polynomials.polynomials;d=moments.innovation_norms[0];lag=moments.reference_lag
        mu=-Q(1,n)
        factors=(ar.q(-mu-radius.upper),ar.q(mu-radius.upper))
        polys=[base.reference.combine(lag,d,factors[0],ar),
               base.reference.combine(base.reference.combine(base.ZERO,lag,Q(-1),ar),d,factors[1],ar)]
        if rounding_error:
            dp,_,rounding_receipt=_rounding_polynomials(d,None,n,rounding_error,ar,rounding_method)
            relax=[base.reference.combine(base.ZERO,dp,ar.addq(1,abs(x)),ar) for x in factors]
            polys=[base.reference.combine(q,e,Q(-1),ar) for q,e in zip(polys,relax)]
            receipt.update(rounding_receipt,polynomial_relaxations=tuple(relax))
        polys=tuple(polys)
        if inversion=='roots':kept,unresolved=continuous_outer_set(polys,root_bits,ar)
        else:
            kept,unresolved,details=base._enclose(polys,ar);receipt.update(details)
        receipt.update(acceptance_polynomials=polys,arithmetic=ar.receipt())
        if not kept:return base.SourceEmpty(receipt)
        return base.SourceOuterHull(min(x.lower for x in kept),max(x.upper for x in kept),kept,unresolved,polys,radius,None,receipt)
    except base.WorkLimit as error:
        return base.SourceFallback(str(error),receipt)


def source_confidence_baseline(plan,compiled,limits=WorkLimits(),rounding_error=Q(0),*,
                               inversion='dyadic',root_bits=48,rounding_method='telescoping'):
    """Preserve the baseline cutoff and budget, with an outward recorder transfer."""
    if type(plan) not in (base.LagPlan,base.BridgePlan,base.IntersectionPlan):
        raise ValueError('Supply a baseline LagPlan, BridgePlan, or IntersectionPlan.')
    if type(rounding_error) is not Q or rounding_error<0:raise ValueError('Supply a nonnegative Fraction rounding bound.')
    if inversion not in ('roots','dyadic'):raise ValueError('Use roots or dyadic inversion.')
    original=base.source_confidence(plan,compiled,limits)
    if isinstance(original,base.SourceFallback):return original
    if not rounding_error and inversion=='dyadic':return original
    ar=base.Arithmetic(limits)
    receipt={'method':'baseline_source_with_recorder_transfer','plan':plan,'inversion':inversion,
             'root_bits':root_bits,'baseline_source_receipt':original.receipt}
    if isinstance(original,base.SourceOuterHull):
        polys=original.acceptance_polynomials;radius=original.lag_radius;cutoff=original.bridge_cutoff
    else:
        polys=original.receipt['acceptance_polynomials'];radius=original.receipt['lag_radius'];cutoff=original.receipt['bridge_cutoff']
    if not polys:return original
    try:
        n=compiled.identity.original_length-1;d=compiled.bank_polynomials.polynomials.innovation_norms[0]
        if rounding_error:
            q=compiled.bridge_polynomial.cumulative_norm if isinstance(plan,(base.BridgePlan,base.IntersectionPlan)) else None
            dp,qp,rounding_receipt=_rounding_polynomials(d,q,n,rounding_error,ar,rounding_method)
            relax=[]
            if isinstance(plan,(base.LagPlan,base.IntersectionPlan)):
                mu=-Q(1,n)
                relax += [base.reference.combine(base.ZERO,dp,ar.addq(1,abs(ar.q(-mu-radius.upper))),ar),
                          base.reference.combine(base.ZERO,dp,ar.addq(1,abs(ar.q(mu-radius.upper))),ar)]
            if isinstance(plan,(base.BridgePlan,base.IntersectionPlan)):
                relax.append(base.reference.combine(qp,dp,ar.mulq(n,cutoff.cutoff_lower),ar))
            polys=tuple(base.reference.combine(q,e,Q(-1),ar) for q,e in zip(polys,relax))
            receipt.update(rounding_receipt,polynomial_relaxations=tuple(relax))
        if inversion=='roots':kept,unresolved=continuous_outer_set(polys,root_bits,ar)
        else:
            kept,unresolved,details=base._enclose(polys,ar);receipt.update(details)
        total_operations=ar.operations+original.receipt.get('operations',0)
        if total_operations>limits.max_operations:raise base.WorkLimit('combined_source_operations')
        receipt.update(acceptance_polynomials=polys,arithmetic=ar.receipt(),total_source_operations=total_operations)
        if not kept:return base.SourceEmpty(receipt)
        return base.SourceOuterHull(min(x.lower for x in kept),max(x.upper for x in kept),kept,unresolved,polys,radius,cutoff,receipt)
    except base.WorkLimit as error:return base.SourceFallback(str(error),receipt)
