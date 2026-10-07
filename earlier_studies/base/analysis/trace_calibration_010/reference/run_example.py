"""Run exactly the declared paired AR(1) example and preserve both outcomes."""
import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import scipy
from scipy.signal import lfilter

from reference import CalibrationData, ErrorBudgets, run_test


ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def stationary_path(generator, phi, length):
    initial = float(generator.standard_normal())
    innovations = generator.standard_normal(length-1)
    path = np.empty(length)
    path[0] = initial
    path[1:] = lfilter([math_sqrt(1-phi*phi)], [1, -phi], innovations, zi=[phi*initial])[0]
    return path, initial, innovations


def math_sqrt(value):
    return float(np.sqrt(value))


def filtered_record(source, coefficients, retained_length):
    """The source starts at time -3 and includes all required filter padding."""
    return sum(coefficient*source[3-position:3-position+retained_length]
               for position,coefficient in enumerate(coefficients))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'example_attempt001')
    args = parser.parse_args()
    out = args.out.resolve()
    if out != ROOT and ROOT not in out.parents:
        raise ValueError('The example output must be inside the reference directory.')
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    sources = out/'source_bytes'
    sources.mkdir()
    for source in [ROOT/'reference.py', ROOT/'run_example.py']:
        shutil.copyfile(source, sources/source.name)
    design = dict(scope='One fixed paired example. No empirical size or power estimate.',
                  calibration_seed=20261003, testing_seed=20261004, generator='NumPy PCG64',
                  common_coefficients=[1.,1.,.25,0.], coefficient_normalization='Unit coefficient norm.',
                  interpolation_delta=0., impulse_blocks_per_channel=16384, calibration_noise_standard_deviation=.02,
                  certificate_grid_points=16384, source_AR_coefficient=.5, retained_record_length=262144,
                  channel_means=[2.,-3.], source_correlation=.4, observation_error_variances=[0.,0.],
                  source_floor=1/3, source_ceiling=3., error_peak_ratios=[1.,1.], anchors=[0,0], reflection_spacing=1,
                  calibration_positions=[0,1,2,3], calibration_offset=0.,
                  budgets=dict(calibration=[.005,.005], variance=.01, tail=.03, target_size=.05),
                  reversible_source='X2(t)=0.4 X1(t)+sqrt(0.84) Z(t)',
                  alternative_source='X2(t)=0.4 X1(t-1)+sqrt(0.84) Z(t)',
                  orientation='The alternative has C12(k)=0.4 gamma(k+1) and cross spectrum 0.4 exp(+i lambda) g_phi in the paper convention.',
                  statistic='U(t)=Y1(t)+Y1(t+1); W(t)=Y2(t)-Y2(t+1); s=(U-mean(U)) dot (W-mean(W))/(N-1).',
                  pairing='Both cases use the same X1 and Z paths, the same calibration readouts, and the same channel 1 recording.',
                  random_draw_order='Calibration: all channel 1 errors, then all channel 2 errors. Testing: X1 initial value, X1 innovations, Z initial value, Z innovations.',
                  initial_and_padding='X1 and Z start at time -4 from independent N(0,1) values. Innovations give the exact stationary AR recurrence. Filtering uses times -3 through N-1.',
                  source_files={p.name:sha(p) for p in sorted(sources.iterdir())}, declared_before_execution_utc=started)
    save(out/'DESIGN.json', design)
    try:
        length = design['retained_record_length']
        blocks = design['impulse_blocks_per_channel']
        coefficients = np.array(design['common_coefficients'])
        coefficients /= np.linalg.norm(coefficients)
        regression_design = np.tile(np.eye(4), (blocks,1))
        calibration_generator = np.random.Generator(np.random.PCG64(design['calibration_seed']))
        calibration_errors = np.array([calibration_generator.normal(0.,.02,len(regression_design)) for _ in range(2)])
        calibration_responses = np.array([regression_design@coefficients+error for error in calibration_errors])
        np.savez_compressed(out/'calibration_raw.npz', design=regression_design, responses=calibration_responses,
                            errors=calibration_errors, true_coefficients=coefficients, positions=np.arange(4))
        testing_generator = np.random.Generator(np.random.PCG64(design['testing_seed']))
        phi = design['source_AR_coefficient']
        first, first_initial, first_innovations = stationary_path(testing_generator, phi, length+4)
        independent, independent_initial, independent_innovations = stationary_path(testing_generator, phi, length+4)
        second_reversible = .4*first[1:]+math_sqrt(.84)*independent[1:]
        second_alternative = .4*first[:-1]+math_sqrt(.84)*independent[1:]
        observed_first = filtered_record(first[1:], coefficients, length)+2.
        reversible = np.column_stack([observed_first, filtered_record(second_reversible, coefficients, length)-3.])
        alternative = np.column_stack([observed_first, filtered_record(second_alternative, coefficients, length)-3.])
        np.savez_compressed(out/'testing_raw.npz', first_source=first, independent_source=independent,
                            first_initial=first_initial, independent_initial=independent_initial,
                            first_innovations=first_innovations, independent_innovations=independent_innovations,
                            raw_path_time_start=-4, second_source_time_start=-3,
                            second_reversible=second_reversible, second_alternative=second_alternative,
                            reversible_recording=reversible, alternative_recording=alternative,
                            observed_channel_means=np.array([2.,-3.]))
        calibrations = [CalibrationData(regression_design, response, np.arange(4)) for response in calibration_responses]
        outcomes = {}
        for name,recording in [('reversible',reversible), ('alternative',alternative)]:
            outcomes[name] = run_test(calibrations, recording, design['source_floor'], design['source_ceiling'],
                                      design['error_peak_ratios'], anchors=(0,0), budgets=ErrorBudgets(), grid_points=16384)
        covariance = lambda k: phi**abs(k)
        filtered_variance = float(sum(a*b*covariance(i-j) for i,a in enumerate(coefficients) for j,b in enumerate(coefficients)))
        alternative_covariance = lambda k: .4*sum(a*b*covariance(k-i+j+1) for i,a in enumerate(coefficients) for j,b in enumerate(coefficients))
        diagnostic = dict(true_observed_marginal_variances=[filtered_variance,filtered_variance],
                          uncentered_reversible_reflection_contrast=0.,
                          uncentered_alternative_reflection_contrast=float(alternative_covariance(1)-alternative_covariance(-1)),
                          warning='These true-model quantities are diagnostics. The decision uses fitted calibration and the same observed recording only.')
        results = dict(design_sha256=sha(out/'DESIGN.json'), raw_input_hashes={name:sha(out/name) for name in ['calibration_raw.npz','testing_raw.npz']},
                       outcomes=outcomes, true_model_diagnostics=diagnostic,
                       elapsed_seconds=time.perf_counter()-start, started_utc=started,
                       finished_utc=datetime.now(timezone.utc).isoformat(),
                       versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
                       scope=design['scope'])
        save(out/'RESULTS.json', results)
        rows = []
        for name,result in outcomes.items():
            rows.append(dict(case=name, status=result['status'], statistic=result['statistic'],
                             phase_upper=result['certificate']['phase_upper'], rho_upper=result['rho_upper'],
                             scale_denominator=result['scale_denominator'], common_variance_ceiling=result['common_variance_ceiling'],
                             null_tolerance=result['null_tolerance'], radius=result['sampling_radius']['radius'] if result['sampling_radius'] else None,
                             threshold=result['threshold'], decision=result['decision'],
                             sample_mean_1=result['sample_means'][0], sample_mean_2=result['sample_means'][1],
                             declared_mean_1=2., declared_mean_2=-3., retained_N=result['alignment']['retained_record_length']))
        with (out/'outcomes.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        save(out/'MANIFEST.json', {str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
        print(json.dumps(dict(status='FIXED_PAIRED_EXAMPLE_COMPLETE', output=str(out), outcomes=rows,
                              result_sha256=sha(out/'RESULTS.json'), manifest_sha256=sha(out/'MANIFEST.json')), indent=2))
    except Exception:
        (out/'FAILURE.txt').write_text(traceback.format_exc())
        save(out/'FAILURE_RECEIPT.json', dict(started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
             elapsed_seconds=time.perf_counter()-start, design_sha256=sha(out/'DESIGN.json'), failure_sha256=sha(out/'FAILURE.txt'),
             preserved_files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
        raise


if __name__ == '__main__':
    main()
