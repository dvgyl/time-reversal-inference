"""Check the three omitted-alpha API paths against the preserved source."""

from dataclasses import asdict
from fractions import Fraction
import importlib.util
import inspect
from pathlib import Path
import sys

import scipy


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "student"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        from importlib.machinery import SourceFileLoader
        spec = importlib.util.spec_from_loader(name, SourceFileLoader(name, str(path)))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


current = load("robust_product_exact_alpha_current", ROOT / "analysis" / "student" / "robust_product.py")
print(f"IMPORT_PATH scipy.real version={scipy.__version__}")

assert Fraction(0.05) > Fraction(1, 20)
print(f"PRIOR_BINARY64_ALPHA {Fraction(0.05)}")
print(f"EXACT_ALPHA {Fraction(1, 20)}")

cases = (
    ("covariance_interval_test", ([[0, 0], [1, 1], [2, 4], [3, 9]],),
     {"covariance_tolerance": Fraction(1, 10), "variance_u_bound": 4,
      "variance_v_bound": 5}),
    ("spectral_reflection", ([[0, 1, 2, 3, 4, 5], [2, 4, 6, 8, 10, 12],
                              [5, 1, 9, 2, 7, 3]], 2, 3, 0, 1, 0),
     {"bandwidth": 2, "skew_error": Fraction(1, 100),
      "source_variance_i": 4, "source_variance_j": 9,
      "observed_variance_i": 4, "observed_variance_j": 9}),
    ("spectral_positivity", ([[0, 0, 1, 2], [1, 2, 3, 4],
                              [3, 0, 0, 4], [8, 2, 3, 1]],),
     {"direction": (Fraction(7, 5), Fraction(-3, 2)), "sign": 1.0,
      "bandwidth": 1, "skew_error": Fraction(1, 10),
      "source_variance_1": 1, "source_variance_2": 1,
      "observed_variance_1": 1, "observed_variance_2": 1}),
)

for name, args, kwargs in cases:
    new = getattr(current, name)
    default = inspect.signature(new).parameters["alpha"].default
    assert type(default) is Fraction and default == Fraction(1, 20), name
    assert current._alpha(default) == Fraction(1, 20), name
    omitted = new(*args, **kwargs)
    exact = new(*args, **kwargs, alpha=Fraction(1, 20))
    assert asdict(omitted) == asdict(exact), name
    assert type(omitted.sample_covariance) is Fraction, name
    print(f"PASS {name} omitted=explicit; applicable={omitted.applicable} reject={omitted.reject}")

print("ALL EXACT DEFAULT ALPHA CHECKS PASS")
