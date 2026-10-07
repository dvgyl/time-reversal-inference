"""Certify finite lower and upper recording-length conditions with Arb."""

import json
from fractions import Fraction as Q
from pathlib import Path

from flint import arb, ctx, fmpq


def ball(value):
    value = Q(value)
    return arb(fmpq(value.numerator, value.denominator))


def endpoint(value, upper=False):
    value = value.upper() if upper else value.lower()
    mantissa, exponent = value.man_exp()
    return Q(int(mantissa)) * Q(2)**int(exponent)


def record(value):
    return dict(lower=str(endpoint(value)), upper=str(endpoint(value, True)), ball=str(value))


def entropy_upper(n, source_ratio, gap):
    a = ball(source_ratio-1)
    w = arb.pi()/(2*a)
    theta = ball(gap)*w/w.sin()
    c = a*a/(1+a)**2
    d = a**4/(4*(1+a)**3)
    harmonic_end = max(1, (n-1)//2)
    defect_log = (12+4*arb(harmonic_end).log())/arb.pi()**2
    defect = min(defect_log.upper(), (arb(n)/a).upper(), (arb(n)/4).upper())
    coefficient = -(1-theta*theta*c).log()/(2*c)
    return coefficient*(c*n/a+d*defect)


def sufficient_rhs(n_observed, source_ratio):
    q = ball(source_ratio)
    n = n_observed-1
    eta = q/n_observed
    u = arb(200).log()
    t0 = (arb(200)/3).log()
    t1 = arb(50).log()
    denominator = 1-eta-2*(eta*u).sqrt()
    if not denominator > 0:
        return None
    d = (1+2*(eta*u).sqrt()+2*eta*u)/denominator
    f_square = ball(Q(n+1, n*n)-Q(2, n**3)-Q(2, n**4))
    first, second = q*f_square.sqrt(), (2*q*n_observed).sqrt()/n
    minimum = arb(0)
    if first < second:
        minimum = first
    elif second < first:
        minimum = second
    else:
        lower = min(endpoint(first), endpoint(second))
        upper = min(endpoint(first, True), endpoint(second, True))
        minimum = ball((lower+upper)/2)+arb(0, ball((upper-lower)/2))
    def radius(scale, t):
        return scale*(2*minimum*t.sqrt()+2*q*t/n)
    return radius(d, t0)+radius(arb(1), t1)+q*arb(2*(4*n-2)).sqrt()/n**2


def certify_example(source_ratio=Q(9), gap=Q(1, 10)):
    ctx.prec = 160
    needed = ball(Q(9, 10))*arb(19).log()
    low, high = 2, 1_000_000
    while low < high:
        middle = (low+high)//2
        value = entropy_upper(middle, source_ratio, gap)
        if value < needed:
            low = middle+1
        elif value > needed:
            high = middle
        else:
            raise ArithmeticError('The entropy comparison is unresolved.')
    necessary = low
    low, high = 3, 10_000_000
    while low < high:
        middle = (low+high)//2
        value = sufficient_rhs(middle, source_ratio)
        if value is not None and value < ball(gap):
            high = middle
        elif value is None or value > ball(gap):
            low = middle+1
        else:
            raise ArithmeticError('The sufficient-condition comparison is unresolved.')
    sufficient = low
    return dict(source_ratio=str(source_ratio), gap=str(gap), precision_bits=ctx.prec,
                size='1/20', power='19/20', necessary_at_least=necessary,
                entropy_upper_at_previous=record(entropy_upper(necessary-1, source_ratio, gap)),
                required_entropy=record(needed), sufficient=sufficient,
                sufficient_rhs=record(sufficient_rhs(sufficient, source_ratio)),
                previous_rhs=record(sufficient_rhs(sufficient-1, source_ratio)),
                sufficient_to_necessary_ratio=str(Q(sufficient, necessary)),
                scope='Fixed-pair necessary length and class-wide sufficient condition; ideal Gaussian recording.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    arguments.output.write_text(json.dumps(certify_example(), indent=2)+'\n')
