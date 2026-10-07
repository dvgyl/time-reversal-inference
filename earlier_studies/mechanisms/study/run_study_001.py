"""Execute the fixed local study only after an independent verification lock."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from fractions import Fraction as Q
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import numpy as np
import scipy
import models_001 as models


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'cycle'/'implementation_001'))
sys.path.insert(0, str(ROOT/'source_inference'/'implementation_001'))


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def encode(value):
    if isinstance(value, Q):
        return {'q': [str(value.numerator), str(value.denominator)]}
    if isinstance(value, Enum):
        return {'enum': type(value).__name__, 'name': value.name, 'value': value.value}
    if is_dataclass(value):
        return {'type': type(value).__name__,
                'fields': {field.name: encode(getattr(value, field.name)) for field in fields(value)}}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError('A result mapping requires string keys.')
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError('Unsupported result value '+type(value).__name__)


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(str(path))
    temporary = path.with_name(path.name+'.writing')
    with open(temporary, 'x') as stream:
        json.dump(encode(value), stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(path)


def save_array(path, array):
    with open(path, 'xb') as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def generator(master_seed, experiment, cell, replicate, stream):
    seed = np.random.SeedSequence(master_seed, spawn_key=(experiment, cell, replicate, stream))
    result = np.random.Generator(np.random.PCG64(seed))
    return result, result.bit_generator.state


def run_cycle(job, registry, directory, record):
    import cycle_test as cycle
    cell = registry['cycle']['cells'][job['cell_index']]
    policy_values = dict(registry['cycle']['numerical_policy'])
    policy_values['precision_bits'] = tuple(policy_values['precision_bits'])
    policy = cycle.NumericalPolicy(**policy_values)
    results = []
    for n in cell['lengths']:
        name = 'N'+str(n)+'_cycle.json'
        try:
            population = models.cycle_population(cell, n)
            spec = cycle.CycleSpec(
                n=n, lag=cell['lag'],
                frequencies=(cycle.QuarterTurn(registry['cycle']['frequency_quarter_turn']),),
                alpha=Q(registry['cycle']['alpha']),
                bounds=cycle.SuppliedBounds(population['K'], population['B'],
                                            'registry_001/'+cell['id']+'/models_001'),
                numerical_policy=policy)
            result = cycle.cycle_test(record[:n].tolist(), spec)
            payload = dict(job=job, n=n, method='cycle', population=population,
                           execution_status='completed', result=result)
        except Exception as error:
            payload = dict(job=job, n=n, method='cycle', execution_status='error',
                           error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        write_json(directory/name, payload)
        results.append(dict(path=name, sha256=digest(directory/name), execution_status=payload['execution_status']))
    return results


def run_source(job, registry, directory, record, design, calibration_values):
    import source_confidence as source
    cell = registry['source']['cells'][job['cell_index']]
    limits = source.WorkLimits(**registry['source']['limits'])
    bank = source.BankPlan(tuple(Q(value) for value in registry['source']['bank']), limits=limits)
    calibration = source.CalibrationData(design.tolist(), calibration_values.tolist(),
                                         tuple(registry['source']['calibration_positions']))
    plans = {
        'bridge': source.SourceBankPlan(bank, source.BridgePlan(Q(1, 200), registry['source']['cutoff'])),
        'intersection': source.SourceBankPlan(bank, source.IntersectionPlan(Q(1, 400), Q(1, 400), registry['source']['cutoff']))}
    supplied = source.SuppliedSource(Q(cell['phi']), 'registry_001/'+cell['id'])
    results = []
    for n in cell['lengths']:
        values = record[:n].tolist()
        compiled = None
        compilation_error = None
        try:
            compiled = source.compile_record(values, limits, accumulation='auto')
        except Exception as error:
            compilation_error = dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        for method in registry['source']['methods']:
            name = 'N'+str(n)+'_'+method+'.json'
            try:
                if method == 'lag':
                    result = source.frozen.run_bank(bank, values, calibration, accumulation='auto')
                elif compilation_error is not None:
                    payload = dict(job=job, n=n, method=method, execution_status='error', phase='compile', **compilation_error)
                    write_json(directory/name, payload)
                    results.append(dict(path=name, sha256=digest(directory/name), execution_status='error'))
                    continue
                elif method == 'supplied':
                    result = source.run_supplied_bank(bank, compiled, calibration, supplied)
                else:
                    result = source.run_compiled_bank(plans[method], compiled, calibration)
                payload = dict(job=job, n=n, method=method, execution_status='completed', result=result)
            except Exception as error:
                payload = dict(job=job, n=n, method=method, execution_status='error',
                               error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            write_json(directory/name, payload)
            results.append(dict(path=name, sha256=digest(directory/name), execution_status=payload['execution_status']))
    return results


def run_job(job, registry, output):
    directory = Path(output)/job['id']
    directory.mkdir()
    write_json(directory/'START.json', dict(job=job, started_utc=utc()))
    cell = registry[job['experiment']]['cells'][job['cell_index']]
    stream_count = 2 if job['experiment'] == 'cycle' else 3
    experiment_index = 0 if job['experiment'] == 'cycle' else 1
    streams, states = [], []
    for index in range(stream_count):
        rng, state = generator(registry['master_seed'], experiment_index, job['cell_index'], job['replicate'], index)
        streams.append(rng)
        states.append(state)
    write_json(directory/'GENERATOR_STATES.json', dict(initial_states=states, cell=cell))
    try:
        n = max(cell['lengths'])
        if job['experiment'] == 'cycle':
            record = models.cycle_record(cell, n, *streams)
            save_array(directory/'record.npy', record)
            inputs = ['record.npy']
        else:
            record, design, calibration = models.source_record(cell, n, *streams,
                                                               registry['source']['calibration_rows'])
            save_array(directory/'record.npy', record)
            save_array(directory/'calibration_design.npy', design)
            save_array(directory/'calibration_response.npy', calibration)
            inputs = ['record.npy', 'calibration_design.npy', 'calibration_response.npy']
        write_json(directory/'INPUTS.json', dict(files=[dict(path=name, bytes=(directory/name).stat().st_size,
                                                            sha256=digest(directory/name)) for name in inputs]))
        if job['experiment'] == 'cycle':
            results = run_cycle(job, registry, directory, record)
        else:
            results = run_source(job, registry, directory, record, design, calibration)
        completion = dict(job=job, completed_utc=utc(), execution_status='completed', results=results,
                          inference_errors=sum(row['execution_status'] == 'error' for row in results))
    except Exception as error:
        completion = dict(job=job, completed_utc=utc(), execution_status='error',
                          error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    write_json(directory/'COMPLETE.json', completion)
    return dict(job=job['id'], execution_status=completion['execution_status'],
                inference_errors=completion.get('inference_errors'), sha256=digest(directory/'COMPLETE.json'))


def verify_lock(path):
    lock = json.loads(Path(path).read_text())
    if lock['status'] != 'VERIFIED_FOR_PROSPECTIVE_EXECUTION':
        raise ValueError('The study lacks its required verification lock.')
    for item in lock['files']:
        target = ROOT/item['path']
        if target.stat().st_size != item['bytes'] or digest(target) != item['sha256']:
            raise ValueError('A frozen study input changed: '+item['path'])
    registry_path = ROOT/lock['registry']
    registry = json.loads(registry_path.read_text())
    if registry['version'] != 'FROZEN_001':
        raise ValueError('The registry is not frozen.')
    if platform.python_version() != lock['runtime']['python'] or np.__version__ != lock['runtime']['numpy'] or scipy.__version__ != lock['runtime']['scipy']:
        raise ValueError('The runtime differs from the verified runtime.')
    return lock, registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    lock, registry = verify_lock(args.lock)
    output = Path(args.output).resolve()
    output.mkdir()
    jobs = []
    for experiment in ('cycle', 'source'):
        for index, cell in enumerate(registry[experiment]['cells']):
            for replicate in range(cell['replicates']):
                jobs.append(dict(id=experiment+'_'+str(index).zfill(2)+'_'+str(replicate).zfill(3),
                                 experiment=experiment, cell_index=index, cell_id=cell['id'], replicate=replicate))
    write_json(output/'EXECUTION_START.json', dict(started_utc=utc(), lock_sha256=digest(args.lock),
                                                 runtime=lock['runtime'], jobs=jobs, registry=registry))
    results = []
    with ProcessPoolExecutor(max_workers=registry['execution']['max_worker_processes']) as executor:
        pending = {executor.submit(run_job, job, registry, str(output)): job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            try:
                result = future.result()
            except Exception as error:
                result = dict(job=job['id'], execution_status='worker_error', error_type=type(error).__name__, error=str(error))
            results.append(result)
            if len(results) % 8 == 0 or len(results) == len(jobs):
                print(json.dumps(dict(completed_jobs=len(results), planned_jobs=len(jobs))), flush=True)
    write_json(output/'EXECUTION_COMPLETE.json', dict(completed_utc=utc(), planned_jobs=len(jobs), results=results))


if __name__ == '__main__':
    main()
