#!/usr/bin/env python3
"""Deterministic mathematical checks; no sampled records or primary data.
Exact Fraction checks validate finite witnesses, not the new general proofs.
Floating evaluations of transcendental bounds are explicitly exploratory.
"""
from fractions import Fraction as F
from itertools import combinations
from math import acos, pi, sqrt, floor, ceil, isqrt
from pathlib import Path
import hashlib, json

HERE = Path(__file__).resolve().parent

def mat(a): return [[F(x) for x in row] for row in a]
def add(a,b,scale=1): return [[x+scale*y for x,y in zip(ar,br)] for ar,br in zip(a,b)]
def mm(a,b): return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def det(a):
    if len(a)==0: return F(1)
    if len(a)==1: return a[0][0]
    return sum((-1)**j*a[0][j]*det([row[:j]+row[j+1:] for row in a[1:]]) for j in range(len(a)))
def psd(a):
    vals=[]
    for n in range(1,len(a)+1):
        for inds in combinations(range(len(a)),n):
            vals.append(det([[a[i][j] for j in inds] for i in inds]))
    return min(vals)>=0, min(vals)
def pinv2(a):
    d=det(a)
    if d: return [[a[1][1]/d,-a[0][1]/d],[-a[1][0]/d,a[0][0]/d]]
    tr=a[0][0]+a[1][1]
    if not tr: return mat([[0,0],[0,0]])
    return [[x/tr**2 for x in row] for row in a]
def rational_sqrt(x):
    a,b=isqrt(x.numerator),isqrt(x.denominator)
    assert a*a==x.numerator and b*b==x.denominator
    return F(a,b)
def completion(a,b,c):
    p=mm(mm(b,pinv2(a)),b)
    hmat=add(a,p,-1)
    z=c-p[0][1]
    h=rational_sqrt(hmat[0][0]*hmat[1][1])
    if h:
        lam=abs(hmat[0][1]-z)/h
        s=[[lam*hmat[0][0],hmat[0][1]-z],[hmat[0][1]-z,lam*hmat[1][1]]]
    else: s=hmat
    r2=add(a,s,-1)
    rr=[a,b,r2]
    toe=[[rr[abs(t-u)][i][j] for u in range(3) for j in range(2)] for t in range(3) for i in range(2)]
    assert r2[0][1]==c
    assert psd(add(a,b))[0] and psd(add(a,b,-1))[0]
    assert psd(s)[0] and psd(add([[2*x for x in row] for row in hmat],s,-1))[0]
    assert psd(toe)[0]
    return {'R2':[[str(x) for x in row] for row in r2], 'all_63_principal_minors_nonnegative':True, 'minimum_principal_minor':str(psd(toe)[1])}

out={"status":"deterministic mathematical probes only; no primary data"}
fixtures=[
 ('singular_rank_one',mat([[1,1],[1,1]]),mat([[F(1,2),F(1,2)],[F(1,2),F(1,2)]]),F(-1,4)),
 ('zero_channel',mat([[1,0],[0,0]]),mat([[F(1,2),0],[0,0]]),F(0)),
 ('zero_A',mat([[0,0],[0,0]]),mat([[0,0],[0,0]]),F(0)),
 ('paper_boundary',mat([[1,F(1,2)],[F(1,2),1]]),mat([[0,F(1,2)],[F(1,2),0]]),F(-1,2)),
]
out['boundary_completion']={name:completion(a,b,c) for name,a,b,c in fixtures}

# Exhaustively count reflection pairs; purely integer arithmetic.
for T in range(1,31):
    H=T-1
    for delta in range(-35,36):
        pairs={(k,2*delta-k) for k in range(-H,H+1) if k<2*delta-k<=H and 2*delta-k>=-H}
        assert len(pairs)==max(0,T-1-abs(delta))
out['reflection']={'T_values':30,'relative_delays_per_T':71,'all_counts_match':True}
# Verify each legal contrast's time indices and symbolic covariance cancellation.
contrast_cases=0
for T in range(2,31):
    H=T-1
    for delta in range(-H+1,H):
        for h in range(1,H-abs(delta)+1):
            b,e=max(delta,0),max(-delta,0)
            assert all(0<=x<=H for x in (b,b+h,e,e+h))
            terms={}
            for lag,coef in [(b-e,1),(b-e+h,-1),(b-e-h,1),(b-e,-1)]:
                terms[lag]=terms.get(lag,0)+coef
            terms={k:v for k,v in terms.items() if v}
            assert terms=={delta+h:-1,delta-h:1}
            contrast_cases+=1
out['reflection']['contrast_index_and_covariance_cases']=contrast_cases

# Exact coefficient-map probe of the direct polynomial construction.
m,T=4,5
H=T-1
d=[(m-i-1)*H for i in range(m)]
C={}
for i in range(m):
    for k in range(-H,H+1): C[i,i,k]=F(1) if k==0 else F(1 if k%2==0 else -1,1000)
for i in range(m):
    for j in range(i+1,m):
        for k in range(-H,H+1):
            C[i,j,k]=F((i+1)*(j+1)+k,10000)
            C[j,i,-k]=C[i,j,k]
# R stores Fourier moments (a cosine coefficient is twice this at n>0).
R={}
for i in range(m):
    for k in range(H+1): R[i,i,k]=C[i,i,k]
for i in range(m):
    for j in range(i+1,m):
        for k in range(-H,H+1):
            n=d[i]-d[j]-k
            assert n>=0 and (i,j,n) not in R
            R[i,j,n]=R[j,i,n]=C[i,j,k]
for i in range(m):
    for j in range(m):
        for k in range(-H,H+1): assert R[i,j,abs(k-d[i]+d[j])]==C[i,j,k]
margins=[C[i,i,0]-2*sum(abs(C[i,i,k]) for k in range(1,H+1))-2*sum(abs(C[i,j,k]) for j in range(m) if j!=i for k in range(-H,H+1)) for i in range(m)]
assert min(margins)>0
out['linear_delay']={'m':m,'T':T,'delays':d,'all_144_output_moments_exact':True,'floor':str(min(margins)),'source_degree':max(n for i,j,n in R)}

# Floating evaluations are not feasibility/optimality certification.
out['angular_bounds_float']=[]
for eta in [1,.3,.1,.03,.01,.003,.001,.0001,.000001]:
    theta=acos(1/(1+eta)); x=pi/theta
    candidates=[d for d in range(max(2,floor(x)-2),2*(floor(x/2)+1)+1) if (d if d%2 else d+1)*theta>=pi-1e-12]
    out['angular_bounds_float'].append({'eta':eta,'new_real_lower':x-1,'old_real_lower':sqrt(2/(eta*(1+eta))),'polynomial_upper':2*(floor(x/2)+1),'candidates_not_excluded_by_new_bound':candidates,'sqrt_eta_times_scale':sqrt(eta)*x})
out['asymptotic_constant']=pi/sqrt(2)
assert 8*F(10,11)**3-4*F(10,11)**2-4*F(10,11)+1==F(91,1331)
assert 2*121**2>158**2
out['eta_one_tenth_exact_comparison']={'cubic_at_10_over_11':'91/1331','upper_surd_square_margin':2*121**2-158**2,'minimum_common_delay_proved_in_packet':8}
(HERE/'probe_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
