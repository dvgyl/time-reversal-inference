"""Certify Gaussian quadratic-form probabilities with Arb integral bounds."""
from __future__ import annotations
from fractions import Fraction
from dataclasses import dataclass
from flint import arb, acb, fmpq, ctx


def ball(q):
    if isinstance(q, Fraction):
        return arb(fmpq(q.numerator, q.denominator))
    return arb(q)


def rational_endpoint(x, upper=False):
    x = x.upper() if upper else x.lower()
    m, e = x.man_exp()
    return Fraction(int(m)) * Fraction(2) ** int(e)


@dataclass(frozen=True)
class CDFCertificate:
    lower: Fraction
    upper: Fraction
    integration: str
    tail_error: str
    origin_error: str
    n: int | None = None
    cutoff: Fraction | None = None


class WorkFailure(RuntimeError):
    pass


def coefficients(n, c):
    """Use lambda/n^2-c/n. Positive rescaling preserves the sign event."""
    if not isinstance(n, int) or n < 3:
        raise ValueError('The continuous pivot requires n >= 3.')
    c = ball(c)
    return [(arb(k)/(2*n)).sin_pi() ** (-2)/(4*n*n)-c/n for k in range(1,n)]


def cdf_zero(weights, *, shift=0, truncation=4096, near_zero='1e-12',
             tolerance='1e-8', head=128, order=20, eval_limit=30000):
    """Enclose P(sum weights[k]*Z[k]^2 <= shift).

    Weights can be Arb enclosures. A nondegenerate law is required.
    A bounded result remains a certificate if the tolerance is not reached.
    Nonfinite integral bounds raise WorkFailure.
    """
    weights = [ball(a) for a in weights]
    shift = ball(shift)
    if not weights or all(a.is_zero() for a in weights):
        raise ValueError('Use the deterministic branch for a zero quadratic form.')
    T, eps, tol = ball(truncation), ball(near_zero), ball(tolerance)
    if not (T > 0 and eps > 0 and eps < T and tol > 0):
        raise ValueError('Require 0 < near_zero < truncation and tolerance > 0.')
    weights.sort(key=lambda a: float(abs(a).upper()), reverse=True)
    exact, remainder = weights[:head], weights[head:]
    max_tail = max((abs(a).upper() for a in remainder), default=arb(0))
    if 2*T*max_tail >= arb('0.75'):
        exact, remainder = weights, []
        max_tail = arb(0)
    moments=[]
    if remainder:
        powers = list(remainder)
        for j in range(1,order+1):
            moments.append(sum(powers, arb(0)))
            powers = [p*a for p,a in zip(powers,remainder)]
        absolute_remainder = sum((abs(p).upper() for p in powers),arb(0))
    else:
        absolute_remainder=arb(0)
    I=acb(0,1)

    def integrand(t, analytic):
        z=2*I*t
        value=-I*t*shift
        for a in exact:
            value -= (1-z*a).log(analytic=analytic)/2
        if remainder:
            radius = abs(z).upper()*max_tail
            if not radius < 1:
                return acb('nan')
            power=z
            for j,m in enumerate(moments,1):
                value += power*m/(2*j)
                power *= z
            error = abs(power).upper()*absolute_remainder/(2*(order+1)*(1-radius))
            value += acb(arb(0,error.upper()),arb(0,error.upper()))
        return (value.exp()-1)/t

    # For t >= T, each selected factor contributes a decreasing modulus.
    # Its logarithmic slope has magnitude at least its value at T.
    # Thus integral_T^infty |cf(t)|/t dt <= modulus(T)/slope(T).
    modulus=arb(1)
    slope=arb(0)
    for a in weights:
        low=abs(a).lower()
        if low > 0:
            u=4*T*T*low*low
            modulus *= (1+u)**arb('-0.25')
            slope += u/(2*(1+u))
    if not slope > 0:
        raise WorkFailure('No nonzero coefficient was certified.')
    tail=modulus/slope
    origin=eps*(sum((abs(a).upper() for a in weights),arb(0))+abs(shift).upper())
    # Geometric panels limit interval inflation near the origin.
    stops=[eps]
    while stops[-1] < T:
        stops.append(min(stops[-1]*4,T))
    total=acb(0)
    for a,b in zip(stops,stops[1:]):
        part=acb.integral(integrand,a,b,abs_tol=tol/(len(stops)*8),
                          rel_tol=tol/8,eval_limit=eval_limit,depth_limit=30)
        if not part.is_finite():
            raise WorkFailure('The interval integral did not return a finite enclosure.')
        total += part
    probability=arb('0.5')-total.imag/arb.pi()
    probability += arb(0,((tail+origin)/arb.pi()).upper())
    lo=max(Fraction(0),rational_endpoint(probability))
    hi=min(Fraction(1),rational_endpoint(probability,True))
    return CDFCertificate(lo,hi,str(total.imag),str(tail/arb.pi()),str(origin/arb.pi()))


def pivot_cdf(n,c,**options):
    c=Fraction(c)
    if n==2:
        p=Fraction(int(c>=Fraction(1,4)))
        return CDFCertificate(p,p,'deterministic','0','0',n,c)
    # Spectral support branches avoid unnecessarily difficult tail integration.
    a=coefficients(n,c)
    if all(x > 0 for x in a):
        return CDFCertificate(Fraction(0),Fraction(0),'support','0','0',n,c)
    if all(x < 0 for x in a):
        return CDFCertificate(Fraction(1),Fraction(1),'support','0','0',n,c)
    r=cdf_zero(a,**options)
    return CDFCertificate(r.lower,r.upper,r.integration,r.tail_error,r.origin_error,n,c)


def quantile_bracket(n,p,lower,upper,*,steps=16,**options):
    """Bisect only when the CDF enclosure proves which side contains p."""
    p,lower,upper=map(Fraction,(p,lower,upper))
    left,right=pivot_cdf(n,lower,**options),pivot_cdf(n,upper,**options)
    if not left.upper <= p <= right.lower:
        raise WorkFailure('Initial CDF bounds do not bracket the requested probability.')
    history=[left,right]
    for _ in range(steps):
        mid=(lower+upper)/2
        result=pivot_cdf(n,mid,**options)
        history.append(result)
        if result.upper <= p:
            lower=mid
        elif result.lower >= p:
            upper=mid
        else:
            break
    return lower,upper,history
