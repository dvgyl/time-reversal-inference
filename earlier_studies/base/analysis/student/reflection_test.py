"""Gaussian reflection tests for predeclared calibrated integer delays.

Records must be iid Gaussian with a common mean and covariance. Choose
contrasts before seeing test records, or
from independent training data. Each candidate's list may include different
conditions; Bonferroni combines within a candidate, IUT across candidates.
"""
import numpy as np
from fractions import Fraction
from scipy.special import betainc


def projection(m, T, i, j, delta, h=1):
    if any(type(x) is not int for x in (m,T,i,j,delta,h)):
        raise ValueError('integer dimensions, channels and lags required')
    if not (m>=2 and T>=2 and 0<=i<j<m and h>=1 and abs(delta)+h<T):
        raise ValueError('contrast not retained')
    b,d=max(delta,0),max(-delta,0)
    u=np.zeros(m*T);v=np.zeros(m*T)
    u[(b+h)*m+i]=u[b*m+i]=1
    v[(d+h)*m+j]=1;v[d*m+j]=-1
    return u,v


def contrast_pvalue(records, m, T, i, j, delta, h=1):
    u,v=projection(m,T,i,j,delta,h)
    data=np.asarray(records,dtype=float)
    if data.ndim!=2 or data.shape[1]!=m*T or not np.isfinite(data).all():
        raise ValueError('finite time-major records required')
    if len(data)<3:return 1.0

    # Exact represented-data contrasts preserve degeneracy even after severe
    # cancellation. Only the final distribution tail is numerical.
    def exact_contrast(weights):
        selected=np.flatnonzero(weights)
        return [sum((int(weights[k])*Fraction(float(row[k])) for k in selected),Fraction(0))
                for row in data]

    a,b=exact_contrast(u),exact_contrast(v)
    n=len(data)
    sa,sb=sum(a),sum(b)
    aa=n*sum(x*x for x in a)-sa*sa
    bb=n*sum(y*y for y in b)-sb*sb
    if aa<=0 or bb<=0:return 1.0
    ab=n*sum(x*y for x,y in zip(a,b))-sa*sb
    r2=ab*ab/(aa*bb)
    if not 0<=r2<=1:raise ValueError('invalid exact correlation')
    p=float(betainc((n-2)/2,.5,float(1-r2)))
    if not np.isfinite(p) or not 0<=p<=1:raise ValueError('invalid numerical tail')
    return p


def union_pvalue(candidate_pvalues):
    if not candidate_pvalues:raise ValueError('nonempty calibrated set required')
    values=[]
    for ps in candidate_pvalues:
        if any(not np.isfinite(p) or not 0<=p<=1 for p in ps):raise ValueError('invalid p-value')
        values.append(min(1.0,len(ps)*min(ps)) if ps else 1.0)
    return max(values)
