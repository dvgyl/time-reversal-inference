"""Independent deterministic API probes; use real SciPy when available."""
import importlib.util
import math
import sys
import types
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "analysis" / "student"
sys.path.insert(0, str(CODE))
try:
    import scipy
    from scipy import special
except ImportError:
    scipy = types.ModuleType("scipy")
    special = types.ModuleType("scipy.special")
    special.betainc = lambda *args: None
    scipy.special = special
    sys.modules["scipy"] = scipy
    sys.modules["scipy.special"] = special
    print("IMPORT_PATH scipy.stub (unused betainc only)")
else:
    print(f"IMPORT_PATH scipy.real version={scipy.__version__}")
spec = importlib.util.spec_from_file_location("robust_product", CODE / "robust_product.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("PASS", message)


def raises(call, message):
    try:
        call()
    except (ValueError, ArithmeticError, OverflowError):
        print("PASS", message)
        return
    raise AssertionError(message)


def exact_sample_cov(records, left, right):
    data = np.asarray(records, dtype=float)
    x = [sum(Fraction(float(z))*Fraction(a) for z,a in zip(row,left)) for row in data]
    y = [sum(Fraction(float(z))*Fraction(a) for z,a in zip(row,right)) for row in data]
    n=len(data)
    return (n*sum(a*b for a,b in zip(x,y))-sum(x)*sum(y))/Fraction(n*(n-1))


def decimal(x):
    x=Fraction(x)
    return Decimal(x.numerator)/Decimal(x.denominator)


def main():
    pair = [[0.,0.],[1.,1.],[2.,4.],[3.,9.]]
    result=module.covariance_interval_test(pair,covariance_tolerance=Fraction(1,10),
             variance_u_bound=Decimal("4.00000000000000000000000001"),
             variance_v_bound=Fraction(9,2),alpha=Fraction(1,20))
    check(result.sample_covariance==exact_sample_cov(pair,[1,0],[0,1]),
          "generic pair exact binary64 represented-data covariance")
    with localcontext() as ctx:
        ctx.prec=150
        b=decimal(Fraction(1,10));p=decimal(Decimal("4.00000000000000000000000001"))
        q=decimal(Fraction(9,2));t=(Decimal(40)).ln()
        expected=b+(2*(p*q+b*b)*t/Decimal(3)).sqrt()+(p*q).sqrt()*t/Decimal(3)+b*t/Decimal(3)
    check(result.threshold_upper>=expected and result.threshold_upper-expected<Decimal("1e-65"),
          "generic threshold outward and tight against 150-digit independent formula")
    boundary=module._product_decision(Fraction(result.threshold_upper),4,
             Fraction(1,10),Fraction(Decimal("4.00000000000000000000000001")),
             Fraction(9,2),Fraction(1,20))
    check(not boundary.reject,"strict generic decision at represented threshold")
    lower=module.covariance_interval_test(pair,covariance_tolerance=Fraction(1,10),
          variance_u_bound=float(Decimal("4.00000000000000000000000001")),
          variance_v_bound=Fraction(9,2),alpha=Fraction(1,20))
    check(result.threshold_upper>lower.threshold_upper,
          "Decimal scalar bound retains information beyond binary64")
    two=module.covariance_interval_test([[0,0],[2,2]],covariance_tolerance=0,
            variance_u_bound=4,variance_v_bound=4,alpha=Fraction(1,20))
    check(two.sample_covariance==2 and not two.reject,"n=2 strict decision")
    zero=module.covariance_interval_test([[5,3]]*3,covariance_tolerance=0,
            variance_u_bound=0,variance_v_bound=0,alpha=Fraction(1,20))
    check(zero.sample_covariance==0 and zero.threshold_upper==0 and not zero.reject,
          "degenerate sample and zero bounds")

    records=np.array([[0,0,1,2],[1,2,3,4],[3,0,0,4],[8,2,3,1]],dtype=float)
    base=dict(bandwidth=Fraction(1),skew_error=Fraction(1,10),
              source_variance_1=Fraction(1),source_variance_2=Fraction(1),
              observed_variance_1=Fraction(1),observed_variance_2=Fraction(1),
              alpha=Fraction(1,20))
    w1,w2=Fraction(7,5),Fraction(-3,2)
    for sign in (1,-1):
        left=[0,w2,w1,0]
        other=[w1,0,0,w2]
        right=[a+sign*b for a,b in zip(left,other)]
        got=module.spectral_positivity(records,direction=(w1,w2),sign=sign,**base)
        check(got.sample_covariance==exact_sample_cov(records,left,right),
              f"positivity staggered signed direction sign={sign}")
        scaled=module.spectral_positivity(records,direction=(2*w1,2*w2),sign=sign,**base)
        check(scaled.sample_covariance==4*got.sample_covariance and scaled.reject==got.reject,
              f"positivity fixed direction scaling sign={sign}")
        with localcontext() as ctx:
            ctx.prec=150
            q=4*decimal(abs(w1*w2))*decimal(Fraction(1,10))
            h=(decimal(abs(w1))+decimal(abs(w2)))**2/4
            t=(Decimal(20)).ln();nu=Decimal(len(records)-1)
            expected_psd=q+2*((h*h+(h-q)*(h-q))*t/nu).sqrt()+2*h*t/nu
        check(got.threshold_upper>=expected_psd and
              got.threshold_upper-expected_psd<Decimal("1e-65"),
              f"positivity threshold outward and tight sign={sign}")
    huge=np.full((10,4),2**50,dtype=float)
    for sign in (1,-1,1.0,-1.0,Decimal("1"),Decimal("-1"),
                 Fraction(1),Fraction(-1)):
        got=module.spectral_positivity(huge,direction=(Fraction(1,3),Fraction(-1,7)),
              **{**base,"sign":sign,"bandwidth":0,"skew_error":0})
        check(type(got.sample_covariance) is Fraction and
              got.sample_covariance==0 and not got.reject,
              f"large constant positivity records preserve exact zero sign={sign!r}")
    generic_huge=module.covariance_interval_test(np.full((10,2),2**50,dtype=float),
               covariance_tolerance=0,variance_u_bound=1,
               variance_v_bound=1,alpha=Fraction(1,20))
    check(type(generic_huge.sample_covariance) is Fraction and
          generic_huge.sample_covariance==0 and not generic_huge.reject,
          "large constant generic pair preserves exact zero")
    reflection_huge=module.spectral_reflection(np.full((10,6),2**50,dtype=float),
               2,3,0,1,0,bandwidth=0,skew_error=0,
               source_variance_i=1,source_variance_j=1,
               observed_variance_i=1,observed_variance_j=1,alpha=Fraction(1,20))
    check(type(reflection_huge.sample_covariance) is Fraction and
          reflection_huge.sample_covariance==0 and not reflection_huge.reject,
          "large constant reflection preserves exact zero")
    hcase=module.spectral_positivity(records,direction=(1,-1),
              **{**base,"skew_error":Fraction(1,4)})
    check(hcase.applicable,"q=H exact physical boundary remains applicable")
    too=module.spectral_positivity(records,direction=(1,-1),
              **{**base,"skew_error":Fraction(1)})
    check(not too.applicable and not too.reject and too.threshold_upper is None,
          "q>H abstention")
    z=module.spectral_positivity(np.zeros((2,4)),direction=(1,-1),
              **{**base,"source_variance_1":0,"source_variance_2":0,
                 "observed_variance_1":0,"observed_variance_2":0})
    check(z.sample_covariance==0 and not z.reject,"zero variance positivity")

    rec=np.array([[0,1,2,3,4,5],[2,4,6,8,10,12],[5,1,9,2,7,3]],dtype=float)
    ref=dict(bandwidth=Fraction(2),skew_error=Fraction(1,100),
             source_variance_i=Fraction(3,2),source_variance_j=Decimal("1.00000000000000000000001"),
             observed_variance_i=2,observed_variance_j=3,alpha=Fraction(1,20))
    for delta in (-1,0,1):
        left,right=module.projection(2,3,0,1,delta,1)
        got=module.spectral_reflection(rec,2,3,0,1,delta,**ref)
        check(got.sample_covariance==exact_sample_cov(rec,left,right),
              f"reflection time-major orientation delta={delta}")
        with localcontext() as ctx:
            ctx.prec=150
            b=2*(decimal(Fraction(3,2))*decimal(Decimal("1.00000000000000000000001"))).sqrt()*decimal(Fraction(2,100))
            p=Decimal(8);qv=Decimal(12);t=Decimal(40).ln();nu=Decimal(len(rec)-1)
            expected_ref=b+(2*(p*qv+b*b)*t/nu).sqrt()+((p*qv).sqrt()+b)*t/nu
        check(got.threshold_upper>=expected_ref and
              got.threshold_upper-expected_ref<Decimal("1e-65"),
              f"reflection threshold outward and tight delta={delta}")
    for kwargs in (
        {"covariance_tolerance":-1,"variance_u_bound":1,"variance_v_bound":1},
        {"covariance_tolerance":0,"variance_u_bound":float("nan"),"variance_v_bound":1},
        {"covariance_tolerance":0,"variance_u_bound":1,"variance_v_bound":1,"alpha":0},
        {"covariance_tolerance":0,"variance_u_bound":1,"variance_v_bound":1,"alpha":1},
    ):
        raises(lambda kw=kwargs:module.covariance_interval_test(pair,**kw),"invalid generic scalar rejected")
    raises(lambda:module.covariance_interval_test([[1,2]],covariance_tolerance=0,
            variance_u_bound=1,variance_v_bound=1),"n=1 rejected")
    raises(lambda:module.covariance_interval_test([[1,float("inf")],[2,3]],covariance_tolerance=0,
            variance_u_bound=1,variance_v_bound=1),"nonfinite record rejected")
    raises(lambda:module.spectral_positivity(records,direction=(0,0),**base),"zero direction rejected")
    raises(lambda:module.spectral_positivity(records,sign=0,**base),"invalid sign rejected")
    raises(lambda:module.spectral_reflection(rec,2,3,0,1,2,**ref),"unretained reflection rejected")
    extreme=module.covariance_interval_test(pair,covariance_tolerance=0,
            variance_u_bound=Decimal("1e100"),variance_v_bound=Decimal("1e-100"),
            alpha=Decimal("1e-100"))
    check(extreme.threshold_upper.is_finite() and not extreme.reject,
          "wide finite Decimal extremes yield finite conservative threshold")
    print("ALL ROBUST PRODUCT PROBES PASS")


if __name__=="__main__":
    main()
