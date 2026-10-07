"""Recompute post hoc Student decisions from immutable centered scatters.

Uses exact Fraction quadratic forms and the independent incomplete-beta
representation of the central Student tail. No study generator is imported.
"""

import csv
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

from scipy.special import betainc
import mpmath as mp


BASE = Path("data/original_studies")
NEW = Path("analysis/student/results")


def dot_cov(scatter, left, right):
    return sum((left[i] * right[j] * F(scatter[i][j])
                for i in range(len(left)) for j in range(len(right))
                if left[i] and right[j]), F(0))


def pearson_p(scatter, u, v, n, two_sided):
    a = dot_cov(scatter, u, u)
    b = dot_cov(scatter, v, v)
    c = dot_cov(scatter, u, v)
    if a <= 0 or b <= 0:
        return 1.0, F(0), c
    r2 = c*c/(a*b)
    assert 0 <= r2 <= 1
    # t two-sided survival = I_(1-r^2)(nu/2,1/2).
    tail2 = betainc((n-2)/2, .5, float(1-r2))
    p = tail2 if two_sided else (tail2/2 if c < 0 else 1-tail2/2)
    return p, r2, c


def verify_hash(path, expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path


rows = list(csv.DictReader((NEW/'decisions.csv').open()))
lookup = {(r['study'],r['case'],r['method']):r for r in rows}
assert len(rows)==len(lookup)==12000
counts = {}
null_counts = {}
nearest = (1.0,None,None)
max_p_difference = 0.0
for study, folder in (("two_time","compute/evaluation"),
                      ("three_time","refinement/study_results")):
    source = BASE/folder
    manifest = json.loads((source/'case_hashes.json').read_text())
    assert set(manifest) == {p.name for p in (source/'cases').glob('*.json')}
    for name, case_hash in sorted(manifest.items()):
        file = source/'cases'/name
        verify_hash(file, case_hash)
        record = json.loads(file.read_text())
        n = record['case']['n']
        if study == 'two_time':
            for method, scatter, direction, sign, N in (
                ('oracle_t',record['full_scatter'],(F(1),F(-1)),1,n),
                ('learned_t',record['holdout_scatter'],
                 tuple(F(x) for x in record['chosen']['direction']),
                 int(record['chosen']['parity_sign']),n-n//2)):
                u = [F(0),direction[1],direction[0],F(0)]
                v = [direction[0],F(0),F(0),direction[1]]
                w = [u[i]+sign*v[i] for i in range(4)]
                p,r2,c = pearson_p(scatter,u,w,N,False)
                result = lookup[(study,name,method)]
                assert result['sha256']==case_hash
                assert (p<=.05)==(result['rejected']=='True'), (study,name,method)
                assert not (result['conservative_rejected']=='True') or p<=.05
                max_p_difference=max(max_p_difference,abs(p-float(result['pvalue'])))
                nearest=min(nearest,(abs(p-.05),(study,name,method),p))
                counts[method]=counts.get(method,0)+int(p<=.05)
                if record['case']['is_null']:
                    null_counts[method]=null_counts.get(method,0)+int(p<=.05)
        else:
            ua=[F(0),F(0),F(1),F(0),F(1),F(0)]
            va=[F(0),F(-1),F(0),F(1),F(0),F(0)]
            ub=[F(1),F(0),F(1),F(0),F(0),F(0)]
            vb=va
            pA,_,_=pearson_p(record['scatters'][0],ua,va,n,True)
            pB,_,_=pearson_p(record['scatters'][1],ub,vb,n,True)
            p=max(pA,pB)
            result=lookup[(study,name,'reflection_t')]
            assert result['sha256']==case_hash
            assert (p<=.05)==(result['rejected']=='True'), (study,name)
            assert not (result['conservative_rejected']=='True') or p<=.05
            max_p_difference=max(max_p_difference,abs(p-float(result['pvalue'])))
            nearest=min(nearest,(abs(p-.05),(study,name,'reflection_t'),p))
            counts['reflection_t']=counts.get('reflection_t',0)+int(p<=.05)
            if record['case']['is_null']:
                null_counts['reflection_t']=null_counts.get('reflection_t',0)+int(p<=.05)

assert counts == {'oracle_t':2010,'learned_t':2250,'reflection_t':604}
assert null_counts == {'oracle_t':41,'learned_t':47,'reflection_t':19}
assert max_p_difference < 5e-11
mp.mp.dps=80
near_key=nearest[1]
near_record=lookup[near_key]
# Reconstruct high-precision beta tail from the nearest case's r^2, again
# reading its scatter rather than a producer statistic.
near_path=BASE/('compute/evaluation' if near_key[0]=='two_time' else 'refinement/study_results')/'cases'/near_key[1]
near_data=json.loads(near_path.read_text())
if near_key[2]=='oracle_t':
    sc=near_data['full_scatter']; dr=(F(1),F(-1)); sign=1; N=near_data['case']['n']; two=False
    u=[F(0),dr[1],dr[0],F(0)]; v=[dr[0],F(0),F(0),dr[1]]
    w=[u[i]+sign*v[i] for i in range(4)]
elif near_key[2]=='learned_t':
    sc=near_data['holdout_scatter']; dr=tuple(F(x) for x in near_data['chosen']['direction'])
    sign=int(near_data['chosen']['parity_sign']); N=near_data['case']['n']-near_data['case']['n']//2; two=False
    u=[F(0),dr[1],dr[0],F(0)]; v=[dr[0],F(0),F(0),dr[1]]
    w=[u[i]+sign*v[i] for i in range(4)]
else:
    N=near_data['case']['n']; two=True
    candidates=[]
    for sc,u,w in ((near_data['scatters'][0],[F(0),F(0),F(1),F(0),F(1),F(0)],[F(0),F(-1),F(0),F(1),F(0),F(0)]),
                   (near_data['scatters'][1],[F(1),F(0),F(1),F(0),F(0),F(0)],[F(0),F(-1),F(0),F(1),F(0),F(0)])):
        candidates.append((pearson_p(sc,u,w,N,True)[0],sc,u,w))
    _,sc,u,w=max(candidates,key=lambda x:x[0])
a=dot_cov(sc,u,u); b=dot_cov(sc,w,w); c=dot_cov(sc,u,w); x=F(1)-c*c/(a*b)
bx=mp.mpf(x.numerator)/x.denominator
tail=mp.betainc(mp.mpf(N-2)/2,mp.mpf('0.5'),0,bx,regularized=True)
high_p=tail if two else (tail/2 if c<0 else 1-tail/2)
assert abs(float(high_p)-float(near_record['pvalue']))<1e-11
print(json.dumps({'verified_decisions':len(rows),'rejections':counts,
                  'null_rejections':null_counts,
                  'max_abs_p_difference':max_p_difference,
                  'closest_key':near_key,'closest_p_beta_float':nearest[2],
                  'closest_p_beta_80dps':str(high_p),
                  'closest_distance_from_0.05':str(abs(high_p-mp.mpf('0.05')))},indent=2))
