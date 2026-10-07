"""Independent deterministic regression probes of revised reflection API."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.special import betainc


path = Path("analysis/student/reflection_test.py")
spec = importlib.util.spec_from_file_location("reflection_api_revised", path)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)

x = np.array([-3., -2., -1., -.5, .3, 1.2, 2., 3.])
y = x + np.array([.1, -.1, .1, -.1, -.1, .1, -.1, .1])
data = np.zeros((len(x), 6))
data[:,4] = x
data[:,3] = y
xc=x-x.mean(); yc=y-y.mean()
r=np.dot(xc,yc)/(np.linalg.norm(xc)*np.linalg.norm(yc))
reference=betainc((len(x)-2)/2,.5,1-r*r)

probes=[]
for sx, sy in ((1.,1.),(1e-200,1e200),(1e200,1e-200),
               (1e-307,1e307),(1e307,1e-307),
               (1e-100,1e100)):
    scaled=data.copy()
    scaled[:,[2,4]]*=sx
    scaled[:,[1,3]]*=sy
    assert np.isfinite(scaled).all()
    p=api.contrast_pvalue(scaled,2,3,0,1,1)
    assert np.isfinite(p) and 0<=p<=1
    assert abs(p-reference)<1e-14, (sx,sy,p,reference)
    probes.append((sx,sy,p))

invalid=[]
for args in ((2,3,0,0,1,1),(1,3,0,0,0,1),(2,1,0,1,0,1),
             (2,3,0,1,2,1),(2,3,0,1,1,0),(2,3,0,1,1.0,1)):
    try:
        api.contrast_pvalue(data[:2],*args)
    except ValueError:
        invalid.append(args)
    else:
        raise AssertionError((args,'invalid request accepted'))

assert api.contrast_pvalue(np.ones((2,6)),2,3,0,1,1)==1.0
assert api.contrast_pvalue(np.ones((8,6)),2,3,0,1,1)==1.0
assert api.union_pvalue([[.01,.2],[.04]])==.04
assert api.union_pvalue([[.01],[]])==1.0
for bad in ([],[[float('nan')]],[[float('inf')]],[[1.1]],[[-.1]]):
    try:
        api.union_pvalue(bad)
    except ValueError:
        pass
    else:
        raise AssertionError((bad,'bad p-values accepted'))

print('REVISED_REFLECTION_PASS')
print('reference incomplete-beta p:',reference)
print('opposite-scale probes:',probes)
print('invalid projection cases rejected:',len(invalid))
