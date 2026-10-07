"""Check complex rejection and real-input behavior in the three public APIs."""

from dataclasses import asdict
from decimal import Decimal
from fractions import Fraction
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_file_location, spec_from_loader
import inspect
from pathlib import Path
import sys
import warnings

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "student"))


def load(name, path):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        spec = spec_from_loader(name, SourceFileLoader(name, str(path)))
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


current = load("real_input_current_005", ROOT / "analysis" / "student" / "robust_product.py")
print(f"IMPORT_PATH scipy.real version={scipy.__version__} numpy={np.__version__}")

cases = (
    ("covariance_interval_test", np.array([[0, 1], [1, 2], [2, 4], [3, 9]], dtype=float), (),
     {"covariance_tolerance": Fraction(1, 10), "variance_u_bound": 4,
      "variance_v_bound": 9}),
    ("spectral_reflection", np.array([[0, 1, 2, 3, 4, 5], [2, 4, 6, 8, 10, 12],
                                      [5, 1, 9, 2, 7, 3]], dtype=float),
     (2, 3, 0, 1, 0),
     {"bandwidth": 2, "skew_error": Fraction(1, 100),
      "source_variance_i": 4, "source_variance_j": 9,
      "observed_variance_i": 4, "observed_variance_j": 9}),
    ("spectral_positivity", np.array([[0, 0, 1, 2], [1, 2, 3, 4],
                                      [3, 0, 0, 4], [8, 2, 3, 1]], dtype=float),
     (),
     {"direction": (Fraction(7, 5), Fraction(-3, 2)), "sign": 1.0,
      "bandwidth": 1, "skew_error": Fraction(1, 10),
      "source_variance_1": 1, "source_variance_2": 1,
      "observed_variance_1": 1, "observed_variance_2": 1}),
)


def rejects_complex(call):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            call()
        except ValueError as error:
            assert str(error) == "records must be real", str(error)
        else:
            raise AssertionError("complex records were accepted")


for name, real, extra, kwargs in cases:
    new = getattr(current, name)
    for function in (new,):
        default = inspect.signature(function).parameters["alpha"].default
        assert type(default) is Fraction and default == Fraction(1, 20), name

    for label, sample in (
        ("real_float", real),
        ("real_int", real.astype(int)),
        ("real_object", real.astype(object)),
    ):
        now = new(sample, *extra, **kwargs)
        assert asdict(now) == asdict(new(real, *extra, **kwargs)), (name, label)
        assert type(now.sample_covariance) is Fraction, (name, label)
        print(f"PASS REAL {name} {label}")

    for imaginary in (0, 1):
        complex_array = real.astype(complex)
        complex_array[0, 0] += 1j * imaginary
        rejects_complex(lambda d=complex_array: new(d, *extra, **kwargs))
        print(f"PASS COMPLEX {name} ndarray imag={imaginary}")

    for scalar_name, scalar in (
        ("python_complex_zero", complex(1, 0)),
        ("numpy_complex128_zero", np.complex128(1 + 0j)),
        ("numpy_complex128_nonzero", np.complex128(1 + 1j)),
    ):
        object_array = real.astype(object)
        object_array[0, 0] = scalar
        rejects_complex(lambda d=object_array: new(d, *extra, **kwargs))
        print(f"PASS COMPLEX {name} object {scalar_name}")

decimal_records = np.array([[Decimal("1.00000000000000000000000001"), 0],
                            [2, 1], [3, 2]], dtype=object)
converted_records = np.asarray(decimal_records, dtype=float)
assert converted_records[0, 0] == 1.0
now = current.covariance_interval_test(decimal_records, covariance_tolerance=0,
    variance_u_bound=4, variance_v_bound=4)
converted = current.covariance_interval_test(converted_records, covariance_tolerance=0,
    variance_u_bound=4, variance_v_bound=4)
assert asdict(now) == asdict(converted)
assert now.sample_covariance == Fraction(1)
original_change = Fraction(decimal_records[0, 0]) - 1
assert original_change > 0
assert now.sample_covariance != 1 - original_change / 2
print("PASS BINARY64 Decimal record converts before exact covariance")

print("ALL REAL INPUT CHECKS PASS")
