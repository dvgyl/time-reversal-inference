"""Accumulate bounded dyadic integer products without integer overflow."""

from dataclasses import dataclass
from fractions import Fraction as Q
import hashlib

import numpy as np

import cycle_diagonal as cycle
from vendor import cycle_reference as ref


LIMIT = (1 << 63) - 1


@dataclass(frozen=True)
class DyadicRecord:
    integers: np.ndarray
    fractional_bits: int

    def __post_init__(self):
        if not isinstance(self.integers, np.ndarray) or self.integers.dtype != np.dtype('int64'):
            raise TypeError('Supply an int64 array.')
        if self.integers.ndim != 2 or self.integers.shape[1] != 3 or len(self.integers) < 1:
            raise ValueError('Supply a nonempty array with three channels.')
        if type(self.fractional_bits) is not int or not 0 <= self.fractional_bits <= 512:
            raise ValueError('Supply a fractional bit count between 0 and 512.')
        if int(self.integers.min()) == -(1 << 63):
            raise ValueError('The minimum int64 value is outside this arithmetic contract.')
        values = np.array(self.integers, dtype=np.int64, order='C', copy=True)
        values.flags.writeable = False
        object.__setattr__(self, 'integers', values)


def exact_sum(values, maximum):
    if maximum == 0:
        return 0
    chunk = max(1, min(65536, LIMIT // maximum))
    return sum(int(np.sum(values[start:start+chunk], dtype=np.int64))
               for start in range(0, len(values), chunk))


def exact_dot(left, right, maximum_left, maximum_right):
    product = maximum_left * maximum_right
    if product == 0:
        return 0
    if product > LIMIT:
        return sum(int(x) * int(y) for x, y in zip(left, right))
    chunk = max(1, min(65536, LIMIT // product))
    return sum(int(np.dot(left[start:start+chunk], right[start:start+chunk]))
               for start in range(0, len(left), chunk))


def compile_dyadic(record, design):
    """Compile exact dyadic observations with checked int64 partial sums."""
    if not isinstance(record, DyadicRecord):
        raise TypeError('Supply a validated DyadicRecord.')
    spec = design.reference_spec()
    if len(record.integers) != design.n:
        raise ValueError('The record length must equal the declared length.')
    budget = ref._Budget(design.policy, 'compile_dyadic_cycle')
    try:
        n, lag = design.n, design.lag
        budget.limit('record_values', 3 * n, design.policy.max_record_values)
        products = 3 * ((2 * lag + 2) * n - lag * (lag + 1))
        budget.limit('lag_products', products, design.policy.max_lag_products)
        budget.reserve(2 * products + 3 * n)
        maxima = tuple(max(abs(int(record.integers[:, i].min())), abs(int(record.integers[:, i].max())))
                       for i in range(3))
        for value in maxima:
            budget.bound(2 * value.bit_length() + n.bit_length())
        columns = tuple(np.ascontiguousarray(record.integers[:, i]) for i in range(3))
        sums = tuple(budget.integer(exact_sum(column, maximum)) for column, maximum in zip(columns, maxima))
        budget.operations += 3 * n
        n2, n3 = budget.mul(n, n), budget.mul(budget.mul(n, n), n)
        scale = budget.shift(1, 2 * record.fractional_bits)
        denominator = budget.mul(n3, scale)
        lags = []
        for i, j in cycle.EDGES:
            values = []
            for k in range(-lag, lag + 1):
                count, si, sj = n - abs(k), max(0, k), max(0, -k)
                budget.reserve(2 * count)
                raw = budget.integer(exact_dot(columns[i][si:si+count], columns[j][sj:sj+count], maxima[i], maxima[j]))
                budget.operations += 2 * count
                budget.lag_products += count
                first = budget.add(sums[i], -budget.sum_integers([int(x) for x in columns[i][:si]]))
                first = budget.add(first, -budget.sum_integers([int(x) for x in columns[i][si+count:]]))
                second = budget.add(sums[j], -budget.sum_integers([int(x) for x in columns[j][:sj]]))
                second = budget.add(second, -budget.sum_integers([int(x) for x in columns[j][sj+count:]]))
                cross = budget.add(budget.mul(sums[i], second), budget.mul(sums[j], first))
                centered = budget.add(budget.mul(n2, raw), -budget.mul(n, cross))
                centered = budget.add(centered, budget.mul(count, budget.mul(sums[i], sums[j])))
                values.append(budget.rational(centered, denominator))
            lags.append(tuple(values))
        energies = []
        for column, maximum, total in zip(columns, maxima, sums):
            budget.reserve(2 * n)
            square = budget.integer(exact_dot(column, column, maximum, maximum))
            budget.operations += 2 * n
            budget.lag_products += n
            numerator = budget.add(budget.mul(n, square), -budget.mul(total, total))
            energies.append(budget.rational(numerator, budget.mul(n, scale)))
        entries = tuple(ref._entries(tuple(lags), spec, f, budget) for f in design.frequencies)
        digest = hashlib.sha256(('dyadic-int64-le-v1:%d:%d\n' % (n, record.fractional_bits)).encode())
        digest.update(record.integers.astype('<i8', copy=False).tobytes(order='C'))
        return cycle.CompiledCycle(design, entries, tuple(energies), (budget.receipt(),), digest.hexdigest(), tuple(lags))
    except ref._WorkLimit as error:
        return cycle.CompileFailure(design, 'resource_limit', (budget.receipt(error),))


def quantize(record, fractional_bits):
    """Round finite binary64 values to a fixed dyadic grid without saturation."""
    if type(fractional_bits) is not int or not 0 <= fractional_bits <= 40:
        raise ValueError('Use a fractional bit count between 0 and 40.')
    values = np.asarray(record, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError('Supply a finite three-channel array.')
    scaled = np.ldexp(values, fractional_bits)
    if np.max(np.abs(scaled)) >= 2**52:
        raise ValueError('The record exceeds the unsaturated quantization range.')
    integers = np.rint(scaled).astype(np.int64)
    return DyadicRecord(integers, fractional_bits), cycle.Perturbation(
        (Q(1, 1 << (fractional_bits + 1)),) * 3, 'Nearest dyadic rounding without saturation')
