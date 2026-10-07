"""Calculate exact population moments without statistical observations."""
from dataclasses import dataclass
from fractions import Fraction as Q


@dataclass(frozen=True)
class Population:
    phi: Q
    correlation: Q
    delay: int
    reference_scalar: Q
    response: tuple
    coefficient: Q

    def __post_init__(self):
        if not 0 <= self.phi < 1 or not -1 <= self.correlation <= 1:
            raise ValueError('The source parameters are outside the model.')
        if type(self.delay) is not int or self.reference_scalar == 0:
            raise ValueError('Supply an integer delay and nonzero reference scalar.')
        if not 0 <= self.coefficient < 1 or not self.response:
            raise ValueError('Supply a valid common filter and finite response.')
        positions = [position for position, value in self.response]
        if any(type(position) is not int for position in positions) or len(set(positions)) != len(positions):
            raise ValueError('The response needs distinct integer positions.')
        if not any(value != 0 for position, value in self.response):
            raise ValueError('The second response must be nonzero.')
        numbers = [self.phi, self.correlation, self.reference_scalar, self.coefficient]
        numbers.extend(value for position, value in self.response)
        if any(type(value) is not Q for value in numbers):
            raise ValueError('The oracle inputs must be exact rational numbers.')


def gamma(spec, lag):
    return spec.phi ** abs(lag)


def kernel(spec, lag):
    a = spec.coefficient
    return (1+a*a)*gamma(spec, lag)-a*(gamma(spec, lag+1)+gamma(spec, lag-1))


def covariance(spec, first, second, lag):
    h = spec.response
    s, c, d = spec.reference_scalar, spec.correlation, spec.delay
    if (first, second) == (1, 1):
        return s*s*kernel(spec, lag)
    if (first, second) == (2, 2):
        return sum((u*v*kernel(spec, lag-r+t) for r, u in h for t, v in h), Q(0))
    if (first, second) == (1, 2):
        return s*c*sum((v*kernel(spec, lag+d+t) for t, v in h), Q(0))
    if (first, second) == (2, 1):
        return covariance(spec, 1, 2, -lag)
    raise ValueError('The channel indices must be one or two.')


def source_covariance(spec, first, second, lag):
    if first == second:
        return gamma(spec, lag)
    return spec.correlation*gamma(spec, lag+(spec.delay if first == 1 else -spec.delay))


def effective_response(spec, channel):
    original = ((0, spec.reference_scalar),) if channel == 1 else spec.response
    terms = {}
    for position, value in original:
        terms[position-1] = terms.get(position-1, Q(0))+value
        terms[position] = terms.get(position, Q(0))-spec.coefficient*value
    return tuple(sorted(terms.items()))


def direct_covariance(spec, first, second, lag):
    left = effective_response(spec, first)
    right = effective_response(spec, second)
    return sum((u*v*source_covariance(spec, first, second, lag-r+t)
                for r, u in left for t, v in right), Q(0))


def moments(spec):
    first = covariance(spec, 1, 1, 0)
    second = covariance(spec, 2, 2, 0)
    contrast = covariance(spec, 1, 2, 1)-covariance(spec, 1, 2, -1)
    return dict(first_variance=first, second_variance=second,
                maximum_variance=max(first, second), total_variance=first+second,
                signed_contrast=contrast)


def centered_statistic_expectation(spec, transformed_length):
    if type(transformed_length) is not int or transformed_length < 3:
        raise ValueError('At least three transformed observations are required.')
    p = transformed_length-1
    def cross(lag):
        return covariance(spec, 1, 2, lag+1)-covariance(spec, 1, 2, lag-1)
    bias = sum(((p-abs(lag))*cross(lag) for lag in range(1-p, p)), Q(0))/p**2
    return cross(0)-bias


def direct_centered_expectation(spec, transformed_length):
    p = transformed_length-1
    entries = []
    for t in range(p):
        row = []
        for v in range(p):
            value = Q(0)
            for first_offset, first_weight in ((0, 1), (1, 1)):
                for second_offset, second_weight in ((0, 1), (1, -1)):
                    value += first_weight*second_weight*direct_covariance(
                        spec, 1, 2, t+first_offset-v-second_offset)
            row.append(value)
        entries.append(row)
    diagonal = sum((entries[t][t] for t in range(p)), Q(0))/p
    total = sum((value for row in entries for value in row), Q(0))/p**2
    return diagonal-total


def centered_variance_expectation(spec, channel, transformed_length):
    if type(transformed_length) is not int or transformed_length < 1:
        raise ValueError('The transformed length must be positive.')
    b = transformed_length
    mean_variance = sum(((b-abs(k))*covariance(spec, channel, channel, k)
                         for k in range(1-b, b)), Q(0))/b**2
    return covariance(spec, channel, channel, 0)-mean_variance


def required_indices(spec, original_length):
    if type(original_length) is not int or original_length < 1:
        raise ValueError('The original length must be positive.')
    low_position = min(position for position, value in spec.response)
    high_position = max(position for position, value in spec.response)
    end = original_length-1
    x = (min(0, -high_position-spec.delay), max(end, end-low_position-spec.delay))
    z = (-high_position, end-low_position)
    return dict(X=x, Z=z, shared=(min(x[0], z[0]), max(x[1], z[1])),
                transformed_length=max(0, original_length-1),
                reflection_pairs=max(0, original_length-2))
