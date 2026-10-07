"""Calculate fixed design quantities without random observations."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import sys
sys.dont_write_bytecode = True

import numpy as np
import scipy
from scipy.stats import beta


HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # The imported modules contain no top-level observation generation.
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(count, total, alpha):
    return [0.0 if count == 0 else float(beta.ppf(alpha/2, count, total-count+1)),
            1.0 if count == total else float(beta.ppf(1-alpha/2, count+1, total-count))]


def main():
    calibration_path = RESEARCH/'relative_calibration_009/compute_costs.py'
    trace_path = RESEARCH/'trace_calibration_010/compute_costs.py'
    envelope_path = RESEARCH/'source_class_api_011/envelope.py'
    calibration = load('oc011_calibration', calibration_path)
    trace = load('oc011_trace', trace_path)
    envelope = load('oc011_envelope', envelope_path)
    smoothing = (np.array([1, 1, .25, 0])/math.sqrt(2.0625)).tolist()
    designs = []
    for number, phi, alternative, coefficients, lengths in [
            (1, .5, False, [1.0], [2048, 8192, 32768]),
            (2, .5, True, [1.0], [2048, 8192, 32768]),
            (3, .9, False, [1.0], [2048, 8192, 32768]),
            (4, .9, True, [1.0], [2048, 8192, 32768]),
            (5, .5, False, smoothing, [16384, 65536, 262144]),
            (6, .5, True, smoothing, [16384, 65536, 262144])]:
        designs.append(dict(design_id=number, source='AR1', phi=phi,
                            hypothesis='alternative' if alternative else 'null',
                            coefficients=[coefficients, coefficients],
                            lengths=lengths))
    epsilon, damping = .1, .95
    aa, bb = 1+epsilon**2, 2*epsilon
    cosine = -2*bb/(4*aa+math.sqrt(16*aa**2-12*bb**2))
    omega = math.acos(cosine)
    phase = epsilon*(1-cosine**2)/math.sqrt(aa+bb*cosine)
    second_variance = 1+epsilon**2+2*epsilon*damping*cosine
    signed = epsilon*(damping**2*math.cos(2*omega)-1)
    ideal = 2*phase*math.sqrt(second_variance)
    common = 2*phase*max(1, second_variance)
    rotation = dict(design_id=7, source='rotation', hypothesis='null',
                    damping=damping, omega=omega,
                    coefficients=[[1.0, 0.0], [1.0, epsilon]],
                    lengths=[32768, 65536, 131072])
    designs.append(rotation)
    ratio_ideal, ratio_common = abs(signed)/ideal, abs(signed)/common
    if not .95 <= ratio_ideal <= 1 or not .95 <= ratio_common <= 1:
        raise RuntimeError('The proposed population tolerance ratios miss the declared range.')
    planning = []
    for design in designs:
        first, second = [np.array(x) for x in design['coefficients']]
        length = len(first)
        l1, l2 = calibration.calibration_error(length, 4096, .02)
        pair = calibration.ResponsePair('new011_design', 0, first, second)
        phase_upper, cells = calibration.direct_certificate(pair, l1, 16384, True)
        shape_upper, fallbacks = calibration.planned_shape_bound(pair, l1, l2, 16384)
        if design['source'] == 'rotation':
            lower = (1-damping)/(1+damping)
            upper = 1/lower
            variances = (1.0, second_variance)
            signed_contrast = signed
        else:
            phi = design['phi']
            lower, upper = (1-phi)/(1+phi), (1+phi)/(1-phi)
            v1, v2, signal = trace.moments(first, second, phi)
            variances = (v1, v2)
            signed_contrast = -signal if design['hypothesis'] == 'alternative' else 0.0
        rho_broad = 2*(upper/lower)*shape_upper
        lanes = [('primary', rho_broad, variances, abs(signed_contrast))]
        marginal = None
        if design['source'] == 'AR1':
            marginal = envelope.ar1_pair_envelope(
                [first, second], [l2, l2], design['phi'], broad_rho=rho_broad,
                positions=[list(range(length))]*2, planning=True)
            lanes.append(('marginal', marginal['rho'], variances, abs(signed_contrast)))
            v1w, v2w, signalw = trace.moments(first, second, 0.0)
            lanes.append(('prefilter', 2*shape_upper, (v1w, v2w),
                          signalw if design['hypothesis'] == 'alternative' else 0.0))
        for lane, rho, moments, signal in lanes:
            for count in design['lengths']:
                value, ceiling = trace.margin(count-(lane == 'prefilter'), rho, signal, phase_upper, *moments)
                planning.append(dict(design_id=design['design_id'], lane=lane,
                                     retained_length=count-(lane == 'prefilter'), observed_length=count,
                                     l1_error_upper=l1, l2_error_upper=l2,
                                     phase_upper=phase_upper, shape_upper=shape_upper,
                                     failed_phase_cells=cells, shape_fallbacks=fallbacks,
                                     rho_upper=rho, variances=list(moments), signal=signal,
                                     margin=value, variance_ceiling=ceiling,
                                     sufficient_95_percent_power=bool(design['hypothesis'] == 'alternative'
                                                                       and value is not None and value > 0)))
    count = 400
    raw_record_bytes = sum(count*(32*max(d['lengths'])+64) for d in designs)
    calibration_bytes = sum(count*16*4096*len(d['coefficients'][0]) for d in designs)
    report = dict(status='PROPOSED_NO_DRAWS', random_observations_generated=0,
                  rotation=dict(c_star=cosine, omega=omega, phase_exact=phase,
                                variances=[1, second_variance], signed_contrast=signed,
                                ideal_tolerance=ideal, common_variance_tolerance=common,
                                ideal_ratio=ratio_ideal, common_variance_ratio=ratio_common),
                  raw_array_budget=dict(record_and_innovation_bytes=raw_record_bytes,
                                        calibration_response_bytes=calibration_bytes,
                                        total_bytes=raw_record_bytes+calibration_bytes,
                                        total_GiB=(raw_record_bytes+calibration_bytes)/2**30,
                                        header_and_shared_input_headroom_bytes=10*2**30
                                        -raw_record_bytes-calibration_bytes),
                  interval_examples={str(k): {'pointwise_95': interval(k,count,.05),
                                             'simultaneous_57_cells': interval(k,count,.05/57)}
                                     for k in [0, 20, 200, 380, 400]},
                  planning=planning, design=designs,
                  planning_note='These are deterministic sufficient margins, not measured rejection rates. '
                  'Positive alternative margins use the existing 0.05 total power-failure budget. '
                  'These are analytic planning calculations, not study outcomes.',
                  versions=dict(numpy=np.__version__, scipy=scipy.__version__),
                  input_sha256={str(p.relative_to(RESEARCH.parent)): digest(p) for p in
                                [Path(__file__), calibration_path,
                                 trace_path, envelope_path]})
    destination = HERE/'ANALYTIC_FEASIBILITY_V002.json'
    if destination.exists():
        raise RuntimeError('Preserve the existing analytic output. Use a new attempt.')
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'output': str(destination), 'sha256': digest(destination),
                      'ideal_tolerance_ratio': ratio_ideal, 'common_tolerance_ratio': ratio_common,
                      'raw_array_GiB': report['raw_array_budget']['total_GiB']}))


if __name__ == '__main__':
    main()
