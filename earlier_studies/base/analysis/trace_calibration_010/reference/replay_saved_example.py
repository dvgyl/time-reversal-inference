"""Replay the saved pair without generating calibration or recording data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy

from reference import CalibrationData, ErrorBudgets, run_test

ROOT = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT/'example_attempt001')
    parser.add_argument('--out', type=Path, default=ROOT/'saved_replay_attempt001')
    args = parser.parse_args()
    inputs, out = args.inputs.resolve(), args.out.resolve()
    if inputs == out or inputs in out.parents or out in inputs.parents or ROOT not in out.parents:
        raise ValueError('Use separate replay outputs inside the reference directory.')
    manifest = json.loads((inputs/'MANIFEST.json').read_text())
    actual = {str(p.relative_to(inputs)):sha(p) for p in inputs.rglob('*') if p.is_file() and p!=inputs/'MANIFEST.json'}
    if actual != manifest:
        raise ValueError('The saved example inventory or input bytes changed.')
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    design = json.loads((inputs/'DESIGN.json').read_text())
    saved = json.loads((inputs/'RESULTS.json').read_text())
    with np.load(inputs/'calibration_raw.npz', allow_pickle=False) as calibration:
        calibrations = [CalibrationData(calibration['design'], response, calibration['positions']) for response in calibration['responses']]
    outcomes = {}
    declared_budget = design['budgets']
    budgets = ErrorBudgets(calibration=tuple(declared_budget['calibration']), variance=declared_budget['variance'],
                           tail=declared_budget['tail'], target_size=declared_budget['target_size'])
    with np.load(inputs/'testing_raw.npz', allow_pickle=False) as testing:
        for name,key in [('reversible','reversible_recording'), ('alternative','alternative_recording')]:
            outcomes[name] = run_test(calibrations, testing[key], design['source_floor'], design['source_ceiling'],
                                      design['error_peak_ratios'], anchors=tuple(design['anchors']), budgets=budgets,
                                      grid_points=design['certificate_grid_points'], reflection_spacing=design['reflection_spacing'])
    result = dict(status='PASS' if outcomes == saved['outcomes'] else 'FAILED', all_reported_outcomes_exactly_equal=outcomes == saved['outcomes'],
                  generated_random_observations=0, outcomes=outcomes,
                  input_manifest_sha256=sha(inputs/'MANIFEST.json'), input_results_sha256=sha(inputs/'RESULTS.json'),
                  active_reference_sha256=sha(ROOT/'reference.py'), replay_script_sha256=sha(Path(__file__)),
                  started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-start,
                  versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__))
    (out/'REPLAY.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if actual != {str(p.relative_to(inputs)):sha(p) for p in inputs.rglob('*') if p.is_file() and p!=inputs/'MANIFEST.json'}:
        raise ValueError('Saved inputs changed during replay.')
    print(json.dumps({k:v for k,v in result.items() if k!='outcomes'}, indent=2))
    if result['status'] != 'PASS':
        raise ValueError('The replay differs from the preserved initial example. See REPLAY.json.')


if __name__ == '__main__':
    main()
