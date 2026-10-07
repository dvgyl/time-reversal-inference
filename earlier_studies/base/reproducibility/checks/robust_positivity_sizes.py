"""Deterministic sufficient n for independent robust covariance tests.

Only analytic Laurent--Massart inequalities are evaluated. No random draws.
"""

import math

ALPHA = BETA = 0.05
OMEGA = math.pi


def a(n, t):
    nu = n - 1
    return 2 * math.sqrt(t / nu) + 2 * t / nu


def smallest_n(gap, null_multiplier, alt_multiplier, alt_failure, tails=4):
    if gap <= 0:
        return None
    t_null = math.log(tails / ALPHA)
    t_alt = math.log(tails / alt_failure)

    def passes(n):
        return gap > null_multiplier * a(n, t_null) + alt_multiplier * a(n, t_alt)

    hi = 2
    while not passes(hi):
        hi *= 2
    lo = 1
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if passes(mid):
            hi = mid
        else:
            lo = mid
    assert hi >= 2 and passes(hi) and (hi == 2 or not passes(hi - 1))
    return hi


def smallest_n_sharp_psd(x, eps):
    """Sub-gamma sufficient count from Var(V)<=4H and null kappa>=-q."""
    h = 1.0  # observed unit coordinate ceilings imply Var(V)<=4
    q = 4 * min(2, OMEGA*eps)
    gap = 4*x - 2 - q
    if not 0 <= q <= h or gap <= 0:
        return None
    kappa = 2 - 4*x
    vu = 2 - 2*x
    vw = 4 - 4*x
    root = math.sqrt(vu*vw)
    plus = (kappa + root)/2
    minus = (root - kappa)/2
    assert plus >= 0 and minus >= 0 and abs(plus-minus-kappa) < 1e-12
    t0 = math.log(1/ALPHA)
    t1 = math.log(1/BETA)

    def passes(n):
        nu = n-1
        null_tail = 2*math.sqrt((h*h+(h-q)**2)*t0/nu)+2*h*t0/nu
        alt_tail = 2*math.sqrt((plus*plus+minus*minus)*t1/nu)+2*plus*t1/nu
        return gap > null_tail + alt_tail

    hi = 2
    while not passes(hi):
        hi *= 2
    lo = 1
    while hi-lo > 1:
        mid = (hi+lo)//2
        if passes(mid):
            hi = mid
        else:
            lo = mid
    assert passes(hi) and (hi == 2 or not passes(hi-1))
    return hi


def smallest_n_product_ab(r, eps):
    """Sub-gamma two-sided null and directed alternative sufficient count."""
    b = 2*min(1, OMEGA*eps)
    gap = r-b
    if gap <= 0:
        return None
    t0 = math.log(2/ALPHA)
    t1 = math.log(2/BETA)  # beta/2 per region, one directed failure tail

    def passes(n):
        nu = n-1
        null_tail = math.sqrt(2*(16+b*b)*t0/nu)+(4+b)*t0/nu
        alt_tail = math.sqrt(2*(4+r*r)*t1/nu)+(2-r)*t1/nu
        return gap > null_tail+alt_tail

    hi=2
    while not passes(hi):
        hi*=2
    lo=1
    while hi-lo>1:
        mid=(hi+lo)//2
        if passes(mid): hi=mid
        else: lo=mid
    assert passes(hi) and (hi==2 or not passes(hi-1))
    return hi


def main():
    # The centered 4x4 covariance of (Y1(0),Y2(0),Y1(1),Y2(1))
    # has H^2=2*x^2*I. Also verify the asserted U,W moments directly.
    for x in (0.5, 0.55, 0.6, 0.65, 0.68):
        h = [[0, x, 0, -x], [x, 0, x, 0],
             [0, x, 0, x], [-x, 0, x, 0]]
        for i in range(4):
            for j in range(4):
                square = sum(h[i][k] * h[k][j] for k in range(4))
                assert abs(square - (2 * x * x if i == j else 0)) < 1e-12
        assert 1 - math.sqrt(2) * x > 0
        g = [[h[i][j] + (1 if i == j else 0) for j in range(4)]
             for i in range(4)]
        u = [0, -1, 1, 0]
        w = [1, -1, 1, -1]

        def quad(left, right):
            return sum(left[i] * g[i][j] * right[j]
                       for i in range(4) for j in range(4))

        assert abs(quad(u, u) - (2 - 2*x)) < 1e-12
        assert abs(quad(w, w) - (4 - 4*x)) < 1e-12
        assert abs(quad(u, w) - (2 - 4*x)) < 1e-12
    print("four-coordinate covariance identities and PD ranges: PASS")
    print("alpha=beta=0.05; nu=n-1; a=2 sqrt(t/nu)+2t/nu")
    print("A/B: each region null multiplier 4, alternative multiplier 2; beta/2 per region")
    print("r,epsilon,gap,n_per_condition")
    for r in (0.1, 0.2, 0.25, 0.4):
        for eps in (0, 0.005, 0.01):
            gap = r - 2 * min(1, OMEGA * eps)
            n = smallest_n(gap, 4, 2, BETA / 2)
            print(f"{r:.2f},{eps:.3f},{gap:.9f},{n if n else 'none'}")
    print("positivity: unit observed coordinate variance ceilings; null multiplier 10")
    print("x,epsilon,tau,gap,alternative_multiplier,n")
    for x in (0.55, 0.6, 0.65, 0.68):
        for eps in (0, 0.01, 0.02):
            tau = min(2, OMEGA * eps)
            gap = 4 * (x - 0.5 - tau)
            alt_multiplier = 3 - 3 * x
            n = smallest_n(gap, 10, alt_multiplier, BETA)
            print(f"{x:.2f},{eps:.3f},{tau:.9f},{gap:.9f},{alt_multiplier:.3f},{n if n else 'none'}")
    c = 7 / 5
    c2 = c * c
    a0 = (4*c2 + 16/c2) / 2
    print("positivity refined one-sided tails: 2e^-t; fixed P=cU,Q=W/c with c=7/5")
    print(f"null multiplier A0={a0:.12f}; strict inequality; alpha=beta=.05")
    print("x,epsilon,gap,unscaled_n,rescaled_A1,rescaled_n")
    for x in (0.55, 0.6, 0.65, 0.68):
        a1 = (1-x)*(c2 + 2/c2)
        for eps in (0, 0.01, 0.02):
            tau = min(2, OMEGA * eps)
            gap = 4*(x - 0.5 - tau)
            old_n = smallest_n(gap, 10, 3-3*x, BETA, tails=2)
            new_n = smallest_n(gap, a0, a1, BETA, tails=2)
            print(f"{x:.2f},{eps:.3f},{gap:.9f},{old_n if old_n else 'none'},{a1:.9f},{new_n if new_n else 'none'}")
    print("matched SME-design table: r means A/B correlation or positivity violation 4x-2")
    print("kind,r,x,epsilon,gap,A0,A1,n")
    for r in (0.1, 0.2, 0.25, 0.4):
        for eps in (0, 0.01, r/(4*OMEGA)):
            gap = r - 2*min(1, OMEGA*eps)
            n = smallest_n(gap, 4, 2, BETA/2, tails=4)
            print(f"AB,{r:.2f},-,{eps:.9f},{gap:.9f},4,2,{n if n else 'none'}")
        x = 0.5 + r/4
        a1 = (1-x)*(c2 + 2/c2)
        for eps in (0, 0.01, r/(8*OMEGA)):
            tau = min(2, OMEGA*eps)
            gap = r - 4*tau
            n = smallest_n(gap, a0, a1, BETA, tails=2)
            print(f"PSD,{r:.2f},{x:.6f},{eps:.9f},{gap:.9f},{a0:.9f},{a1:.9f},{n if n else 'none'}")
    print("sharper PSD sub-gamma sufficient n (q<=H=1), same matched scenarios")
    print("r,x,epsilon,q,gap,n")
    for r in (0.1, 0.2, 0.25, 0.4):
        x = 0.5 + r/4
        for eps in (0, 0.01, r/(8*OMEGA)):
            q = 4*min(2, OMEGA*eps)
            gap = r-q
            n = smallest_n_sharp_psd(x, eps)
            print(f"{r:.2f},{x:.6f},{eps:.9f},{q:.9f},{gap:.9f},{n if n else 'none'}")
    print("sharper A/B product sub-gamma sufficient n, same matched scenarios")
    print("r,epsilon,b,gap,n_per_condition")
    for r in (0.1,0.2,0.25,0.4):
        for eps in (0,0.01,r/(4*OMEGA)):
            b=2*min(1,OMEGA*eps)
            gap=r-b
            n=smallest_n_product_ab(r,eps)
            print(f"{r:.2f},{eps:.9f},{b:.9f},{gap:.9f},{n if n else 'none'}")


if __name__ == "__main__":
    main()
