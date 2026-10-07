"""Check sufficient population bounds without reading or generating a record."""
from dataclasses import replace
from fractions import Fraction as Q
import json
from pathlib import Path
import sys
import models_001 as models
import run_study_001 as io


sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'cycle'/'implementation_001'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'source_inference'/'implementation_001'))
import cycle_test as cycle
import source_confidence as source


def cycle_certificate(cell, n, settings):
    population = models.cycle_population(cell, n)
    parameters = dict(settings['numerical_policy'])
    parameters['precision_bits'] = tuple(parameters['precision_bits'])
    policy = cycle.NumericalPolicy(**parameters)
    spec = cycle.CycleSpec(n, cell['lag'], (cycle.QuarterTurn.PI_OVER_TWO,), Q(1, 20),
                           cycle.SuppliedBounds(population['K'], population['B'], 'declared population'), policy)
    budget = cycle._Budget(policy, 'population_power')
    epsilon_a = cycle._radius_interval(spec, 80, budget)
    epsilon_b = cycle._radius_interval(replace(spec, alpha=Q(1, 5)), 80, budget)
    magnitudes = []
    for real, imaginary in population['spectra']:
        square = budget.qadd(budget.qmul(real, real), budget.qmul(imaginary, imaginary))
        magnitudes.append(cycle._sqrt_interval(square, 80, budget).upper)
    first = cycle._product_radius(tuple(magnitudes), epsilon_b.upper, budget)
    second = cycle._product_radius(tuple(budget.qadd(x, epsilon_b.upper) for x in magnitudes), epsilon_a.upper, budget)
    upper_radius = budget.qadd(first, second)
    witness = abs(population['J'])
    margin_lower = budget.qadd(witness, -upper_radius)
    return dict(cell=cell['id'], n=n, population=population, alpha=Q(1, 20), beta=Q(1, 5),
                epsilon_alpha=epsilon_a, epsilon_beta=epsilon_b,
                sufficient_radius_upper=upper_radius, sufficient_margin_lower=margin_lower,
                sufficient_power_08=budget.compare(margin_lower, Q(0)) > 0, receipt=budget.receipt())


def source_certificate(cell, n, settings):
    limits = source.WorkLimits(**settings['limits'])
    alpha, beta = Q(1, 200), Q(1, 5)
    cutoff = source.cutoff_certificate(n-1, alpha, settings['cutoff'], limits)
    ar = source.Arithmetic(limits)
    t = ar.log(ar.divq(2, beta))
    h = ar.add(ar.add(ar.point(1), ar.mul(ar.point(2), ar.sqrt(t))), ar.mul(ar.point(2), t))
    length = n-1
    delta = 1-Q(cell['phi'])
    if not isinstance(cutoff, source.CutoffCertificate):
        return dict(cell=cell['id'], n=n, endpoint_condition=False, cutoff=cutoff)
    required = ar.div(ar.mul(ar.point(2), h), ar.point(cutoff.cutoff_lower))
    valid = length >= 4 and ar.q(length) >= ar.mulq(128, t.upper) and ar.mulq(length, delta) > required.upper
    return dict(cell=cell['id'], n=n, alpha=alpha, beta=beta, cutoff=cutoff,
                log_interval=t, h_interval=h, effective_length=length*delta,
                required_effective_length_upper=required.upper, endpoint_condition=valid,
                guaranteed_probability_if_condition=1-alpha-beta, receipt=ar.receipt())


def main():
    registry = json.loads(Path(__file__).with_name('population_registry_002.json').read_text())
    report = dict(scope='Supplied-population sufficient conditions for ideal continuous Gaussian mathematical rules. No random draw or recording input. The positive S17 row concerns the ideal cycle rule only. Bounded implementation completion and transfer to rounded generator or recorder values are unproved. The S18 condition concerns the ideal nonempty source upper endpoint only; it is not a full-bank power bound. This is not empirical size or power.',
                  cycle=[cycle_certificate(cell, n, registry['cycle']) for cell in registry['cycle']['cells'] for n in cell['lengths']],
                  source=[source_certificate(cell, n, registry['source']) for cell in registry['source']['cells'] for n in cell['lengths']])
    io.write_json(Path(sys.argv[1]), report)
    print(json.dumps(dict(cycle_available=sum(row['sufficient_power_08'] for row in report['cycle']),
                          cycle_conditions=len(report['cycle']),
                          source_endpoint_available=sum(row['endpoint_condition'] for row in report['source']),
                          source_conditions=len(report['source']))))


if __name__ == '__main__':
    main()
