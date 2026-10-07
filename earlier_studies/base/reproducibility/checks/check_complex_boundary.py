"""Require each robust API to reject complex records with ValueError."""
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'analysis/student'))
import robust_product as api

common = dict(bandwidth=1, skew_error=0)
cases = [
    ('covariance_interval_test', 2, lambda rows: api.covariance_interval_test(
        rows, covariance_tolerance=0, variance_u_bound=1, variance_v_bound=1)),
    ('spectral_reflection', 6, lambda rows: api.spectral_reflection(
        rows, 2, 3, 0, 1, 0, source_variance_i=1, source_variance_j=1,
        observed_variance_i=1, observed_variance_j=1, **common)),
    ('spectral_positivity', 4, lambda rows: api.spectral_positivity(
        rows, source_variance_1=1, source_variance_2=1,
        observed_variance_1=1, observed_variance_2=1, **common)),
]
results = []
for name, columns, call in cases:
    for dtype, scalar, imaginary in ((complex, complex, True),
                                     (object, complex, True),
                                     (complex, complex, False),
                                     (object, np.complex128, True),
                                     (object, np.complex128, False)):
        rows = np.array([[scalar(complex(i+j, 1+i+j if imaginary else 0))
                          for j in range(columns)] for i in range(3)], dtype=dtype)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            try:
                decision = call(rows)
            except ValueError:
                outcome = 'ValueError'
            except Exception as error:
                outcome = type(error).__name__
            else:
                outcome = 'decision returned'
            results.append(dict(api=name, dtype=str(np.dtype(dtype)), scalar=scalar.__name__,
                nonzero_imaginary=imaginary, outcome=outcome,
                warnings=[str(w.message) for w in captured]))
report = dict(source_sha256=hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest(),
    numpy=np.__version__, status='PASS' if all(r['outcome']=='ValueError' for r in results) else 'FAIL', cases=results)
print(json.dumps(report, indent=2))
sys.exit(0 if report['status']=='PASS' else 1)
