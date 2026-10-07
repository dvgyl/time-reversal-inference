"""Certify finite-dimensional endpoint and simple-law converse probabilities."""
from fractions import Fraction as Q
from flint import arb, acb, acb_mat, ctx
from certified_pivot import ball,cdf_zero,rational_endpoint,WorkFailure,CDFCertificate


def difference_covariance(n,phi):
    phi=Q(phi)
    if n<2 or not 0<=phi<1:raise ValueError('Require n >= 2 and 0 <= phi < 1.')
    return acb_mat([[ball(Q(2)/(1+phi) if i==j else -(1-phi)/(1+phi)*phi**(abs(i-j)-1))
                     for j in range(n)] for i in range(n)])


def real_eigenvalues(matrix):
    eig=matrix.eig(multiple=True,algorithm='rump')
    if any(not z.is_finite() or not z.imag.contains(0) for z in eig):
        raise WorkFailure('Real eigenvalue enclosures were not obtained.')
    return [z.real for z in eig]


def endpoint_coefficients(n,phi,c):
    """Enclose eigenvalues of A_phi H_c/n^2. A_phi H_c is similar to symmetric."""
    c=Q(c)
    A=difference_covariance(n,phi)
    rows=[Q((n-i)*(n+i-1),2) for i in range(1,n+1)]
    total=Q((n-1)*n*(2*n-1),6)
    H=acb_mat([[ball((Q(n-max(i+1,j+1))-rows[i]/n-rows[j]/n+total/(n*n)
                      -n*c*(int(i==j)-Q(1,n)))/(n*n)) for j in range(n)] for i in range(n)])
    eig=real_eigenvalues(A*H)
    zero=[i for i,x in enumerate(eig) if x.contains(0)]
    if len(zero)==1:
        # H*1=0 exactly, and all other eigenvalues exclude zero.
        eig.pop(zero[0])
    return eig


def endpoint_probability(n,phi,c,**options):
    if n==2:
        difference_covariance(n,phi)
        value=Q(int(Q(c)>=Q(1,4)))
        return CDFCertificate(value,value,'deterministic','0','0',n,Q(c))
    return cdf_zero(endpoint_coefficients(n,phi,c),**options)


def np_converse(n,phi,alpha,bracket,*,steps=14,**options):
    """Enclose the known-scale NP upper envelope for invariant endpoint rules."""
    eig=real_eigenvalues(difference_covariance(n,phi))
    if any(not a>0 for a in eig):raise WorkFailure('Positive covariance eigenvalues were not certified.')
    null=[1-1/a for a in eig]
    alternative=[a-1 for a in eig]
    p=1-Q(alpha);lo,hi=map(Q,bracket)
    left=cdf_zero(null,shift=lo,**options);right=cdf_zero(null,shift=hi,**options)
    if not left.upper<=p<=right.lower:raise WorkFailure('The likelihood-ratio threshold is not bracketed.')
    history=[]
    for _ in range(steps):
        mid=(lo+hi)/2;r=cdf_zero(null,shift=mid,**options);history.append(r)
        if r.upper<=p:lo=mid
        elif r.lower>=p:hi=mid
        else:break
    altlo=cdf_zero(alternative,shift=lo,**options)
    althi=cdf_zero(alternative,shift=hi,**options)
    return {'n':n,'phi':str(phi),'alpha':str(alpha),'threshold_lower':str(lo),'threshold_upper':str(hi),
            'power_lower':str(1-althi.upper),'power_upper':str(1-altlo.lower),
            'determinant_eigenvalue_product':str(__import__('functools').reduce(lambda x,y:x*y,eig,arb(1))),
            'determinant_exact':str(1+n*(1-Q(phi))/(1+Q(phi))),
            'scope':'Known-scale simple-law NP envelope; upper bound for every shift-invariant covering rule.',
            'cdf_steps':len(history)}
