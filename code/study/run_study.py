"""Run only a sealed, independently checked prospective registry."""

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import fields, is_dataclass
from fractions import Fraction as Q
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import scipy

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'cycle'))
sys.path.insert(0,str(ROOT/'source_inference/implementation_001'))

import physical_models
import generation
import evaluate_methods as methods
import baselines


def encode(value):
    if isinstance(value,Q):
        return {'__fraction__':str(value)}
    if is_dataclass(value):
        return dict(__type__=type(value).__name__,**{field.name:encode(getattr(value,field.name)) for field in fields(value)})
    if isinstance(value,dict):
        return {str(key):encode(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return [encode(item) for item in value]
    if isinstance(value,np.ndarray):
        return encode(value.tolist())
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,Path):
        return str(value)
    return value


def write_json(path,value):
    if path.exists():
        raise FileExistsError(path)
    text=json.dumps(encode(value),sort_keys=True,allow_nan=False,separators=(',',':'))+'\n'
    temporary=path.with_name(path.name+'.writing')
    if temporary.exists():
        raise FileExistsError(temporary)
    if path.suffix=='.gz':
        with gzip.open(temporary,'wt',encoding='utf-8') as output:
            output.write(text)
    else:
        temporary.write_text(text)
    os.replace(temporary,path)


def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def stream(master,cell,replicate,role):
    key=json.dumps([cell,replicate,role],separators=(',',':')).encode()
    words=np.frombuffer(hashlib.sha256(key).digest(),dtype='<u4').astype(np.uint64).tolist()
    sequence=np.random.SeedSequence([int(master),*words])
    return np.random.Generator(np.random.PCG64DXSM(sequence))


def recorder_cell(cell):
    p=cell['parameters']
    return ((cell['family']=='source' and p['condition']=='gaussian' and p['phi']=='97/100' and cell['N']==65537) or
            (cell['family']=='cycle' and cell['id'] in ('cycle_null_131072','cycle_rotating_131072')) or
            (cell['family']=='brownian' and cell['N']==131073) or
            (cell['family']=='phase_boundary' and p['population_tolerance_ratio']=='99/100'))


def _evaluate(cell,integers,bits,error,calibration,registry,cutoffs):
    if cell['family']=='cycle':
        return methods.evaluate_cycle(cell,integers,bits,error,registry),None
    design,response,calibration_error=calibration
    if cell['family']=='source':
        return methods.evaluate_source(cell,integers,bits,error,design,response,calibration_error,registry,cutoffs)
    return methods.evaluate_physical(cell,integers,bits,error,design,response,calibration_error,registry),None


def run_job(arguments):
    registry,cell,replicate,run_directory,cutoff_directory,freeze_identity=arguments
    job=Path(run_directory)/('%s__%04d'%(cell['id'],replicate))
    job.mkdir(exist_ok=False)
    write_json(job/'START.json',dict(cell=cell['id'],replicate=replicate,
                                    utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=freeze_identity))
    streams={}
    try:
        disk_guard(Path(run_directory),cell)
        streams={role:stream(registry['execution']['master_seed'],cell['id'],replicate,role)
                 for role in registry['execution']['roles']}
        before={role:rng.bit_generator.state for role,rng in streams.items()}
        write_json(job/'STREAMS_BEFORE.json',before)
        write_json(job/'CELL.json',cell)
        raw=generation.generate_record(cell,streams)
        np.savez_compressed(job/'record_binary64.npz',record=raw)
        bits=registry['execution']['primary_fractional_bits']
        integers,error=generation.quantize_channels(raw,bits)
        np.savez_compressed(job/'record_dyadic.npz',integers=integers,fractional_bits=np.array(bits))
        calibration=(None,None,Q(0))
        if cell['family'] in ('source','brownian'):
            coefficients=list(map(lambda x:float(Q(x)),cell['parameters']['response']))
            design,response=physical_models.calibration_record(registry['execution']['calibration_rows'],coefficients,
                float(Q(registry['execution']['calibration_noise_sd'])),streams['calibration'])
            np.savez_compressed(job/'calibration_binary64.npz',design=design,response=response)
            rounded,calibration_error=generation.quantize_channels(np.column_stack((response,response)),bits)
            exact_response=[Q(int(value),2**bits) for value in rounded[:,0]]
            np.savez_compressed(job/'calibration.npz',design=design,response_binary64=response,
                                response_integers=rounded[:,0],fractional_bits=np.array(bits))
            calibration=(design.astype(int).tolist(),exact_response,calibration_error)
        primary,compiled=_evaluate(cell,integers,bits,error,calibration,registry,Path(cutoff_directory))
        write_json(job/'PRIMARY.json.gz',primary)
        observed=np.ldexp(integers.astype(float),-bits)
        fitted=primary.get('calibration')
        phase=float(fitted.phase_upper) if hasattr(fitted,'phase_upper') else None
        comparators=baselines.lag_baselines(observed,streams['bootstrap'],phase=phase,
                                           resamples=registry['baselines']['bootstrap_resamples'])
        comparators['imaginary_coherency']=baselines.imaginary_coherency(observed,streams['bootstrap'],
            resamples=registry['baselines']['bootstrap_resamples'],
            segment_length=registry['baselines']['imaginary_coherency_segment_length'])
        write_json(job/'BASELINES.json',comparators)
        p=cell['parameters']
        if cell['family']=='source' and p['condition']=='gaussian' and cell['N'] in (16385,65537):
            variants=methods.numerical_variants(compiled,error,registry,Path(cutoff_directory),*calibration)
            write_json(job/'NUMERICAL_VARIANTS.json.gz',variants)
        if recorder_cell(cell):
            for resolution in registry['ablations']['fractional_bits']:
                changed,allowance=generation.quantize_channels(raw,resolution)
                result,_=_evaluate(cell,changed,resolution,allowance,calibration,registry,Path(cutoff_directory))
                write_json(job/('FRACTIONAL_%d.json.gz'%resolution),result)
            for resolution in registry['ablations']['recorder_bits']:
                changed,receipt=generation.finite_recorder(raw,resolution)
                if receipt['saturated_values']:
                    result=dict(status='abstain',reason='recorder_saturation_without_error_certificate')
                else:
                    exponent=resolution-4
                    rounded=np.rint(np.ldexp(changed,exponent)).astype(np.int64)
                    result,_=_evaluate(cell,rounded,exponent,Q(receipt['valid_error_bound']),calibration,registry,Path(cutoff_directory))
                write_json(job/('RECORDER_%d.json.gz'%resolution),dict(recorder=receipt,result=result))
        write_json(job/'STREAMS_AFTER.json',{role:rng.bit_generator.state for role,rng in streams.items()})
        files=[dict(path=path.name,sha256=file_hash(path),bytes=path.stat().st_size) for path in sorted(job.iterdir()) if path.is_file()]
        write_json(job/'COMPLETE.json',dict(cell=cell['id'],replicate=replicate,
                                           utc=datetime.now(timezone.utc).isoformat(),files=files))
        return dict(cell=cell['id'],replicate=replicate,status='complete')
    except Exception as failure:
        write_json(job/'FAILED.json',dict(cell=cell['id'],replicate=replicate,error=repr(failure),
                                         traceback=traceback.format_exc(),
                                         stream_states={role:rng.bit_generator.state for role,rng in streams.items()}))
        return dict(cell=cell['id'],replicate=replicate,status='failed',error=repr(failure))


def disk_guard(directory,cell):
    channels=3 if cell['family']=='cycle' else 2
    # Reserve raw and integer arrays for both workers without assuming compression.
    required=32*cell['N']*channels+64*1024**2
    if shutil.disk_usage(directory).free < required+30*1024**3:
        raise RuntimeError('The disk reserve is below 30 GiB. No further draw is permitted.')


def storage_preflight(directory,registry,done):
    values=sum(cell['N']*(3 if cell['family']=='cycle' else 2)*
               sum((cell['id'],replicate) not in done for replicate in range(cell['replicates']))
               for cell in registry['cells'])
    raw_and_dyadic=16*values
    auxiliary_allowance=16*1024**3
    reserve=30*1024**3
    free=shutil.disk_usage(directory).free
    receipt=dict(remaining_scalar_values=values,uncompressed_primary_array_bytes=raw_and_dyadic,
                 auxiliary_allowance_bytes=auxiliary_allowance,reserve_bytes=reserve,free_bytes=free,
                 note='Auxiliary allowance is a planning estimate. Per-draw guards remain active.')
    if free<raw_and_dyadic+auxiliary_allowance+reserve:
        raise RuntimeError('The complete remaining study exceeds the local storage planning allowance.')
    return receipt


def completed_jobs(directory,registry,freeze_identity):
    expected={('%s__%04d'%(cell['id'],replicate)):(cell['id'],replicate)
              for cell in registry['cells'] for replicate in range(cell['replicates'])}
    complete=set()
    for job in sorted(directory.iterdir()):
        if not job.is_dir():
            continue
        if job.name not in expected:
            raise ValueError('An unregistered job directory is present: '+job.name)
        if not (job/'COMPLETE.json').exists() or (job/'FAILED.json').exists():
            raise ValueError('An incomplete or failed draw is preserved and cannot be rerun: '+job.name)
        receipt=json.loads((job/'COMPLETE.json').read_text())
        start=json.loads((job/'START.json').read_text())
        cell,replicate=expected[job.name]
        if (receipt['cell'],receipt['replicate'])!=(cell,replicate) or start['freeze_sha256']!=freeze_identity:
            raise ValueError('A completed job has a different identity: '+job.name)
        listed={entry['path'] for entry in receipt['files']}
        actual={path.name for path in job.iterdir() if path.is_file()}-{'COMPLETE.json'}
        if listed!=actual:
            raise ValueError('A completed job file list changed: '+job.name)
        for entry in receipt['files']:
            if Path(entry['path']).name!=entry['path'] or file_hash(job/entry['path'])!=entry['sha256']:
                raise ValueError('A completed job file changed: '+job.name)
        complete.add((cell,replicate))
    return complete


def verify_freeze(registry_path,freeze_path,verification_path):
    registry=json.loads(registry_path.read_text())
    freeze=json.loads(freeze_path.read_text())
    verification=json.loads(verification_path.read_text())
    if registry.get('status')!='SEALED_FOR_FRESH_STUDY':
        raise ValueError('The registry is not sealed.')
    if freeze['registry_sha256']!=file_hash(registry_path):
        raise ValueError('The registry identity changed.')
    if verification.get('verdict')!='PASS' or verification.get('freeze_sha256')!=file_hash(freeze_path):
        raise ValueError('The independent pre-run verification does not match this freeze.')
    for entry in freeze['files']:
        path=ROOT/entry['path']
        if file_hash(path)!=entry['sha256']:
            raise ValueError('A frozen code or cutoff identity changed: '+entry['path'])
    identifiers=[cell['id'] for cell in registry['cells']]
    if len(set(identifiers))!=len(identifiers):
        raise ValueError('Study cell identifiers must be unique.')
    return registry


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--verification',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cutoffs',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    arguments=parser.parse_args()
    registry=verify_freeze(arguments.registry,arguments.freeze,arguments.verification)
    if arguments.workers not in (1,2):
        raise ValueError('Use one or two local workers.')
    if arguments.cutoffs.resolve()!=(ROOT/'source_inference/records/cutoffs_001').resolve():
        raise ValueError('Use the frozen cutoff directory.')
    total=sum(cell['replicates'] for cell in registry['cells'])
    if arguments.validate_only:
        print(json.dumps(dict(status='validated',cells=len(registry['cells']),records=total)),flush=True)
        return
    freeze_identity=file_hash(arguments.freeze)
    identity=dict(registry_sha256=file_hash(arguments.registry),
        freeze_sha256=file_hash(arguments.freeze),verification_sha256=file_hash(arguments.verification),
        workers=arguments.workers,utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
        numpy=np.__version__,scipy=scipy.__version__,records=total)
    if arguments.resume:
        if (arguments.output/'RUN_COMPLETE.json').exists():
            raise ValueError('A completed study cannot be resumed.')
        original=json.loads((arguments.output/'RUN_START.json').read_text())
        for key in ('registry_sha256','freeze_sha256','verification_sha256','python','numpy','scipy','records'):
            if identity[key]!=original[key]:
                raise ValueError('The resumed run identity differs: '+key)
        done=completed_jobs(arguments.output,registry,freeze_identity)
        sequence=len(list(arguments.output.glob('RESUME_*.json')))+1
        write_json(arguments.output/('RESUME_%03d.json'%sequence),identity)
    else:
        arguments.output.mkdir(exist_ok=False)
        write_json(arguments.output/'RUN_START.json',identity)
        done=set()
    sequence=len(list(arguments.output.glob('STORAGE_*.json')))+1
    write_json(arguments.output/('STORAGE_%03d.json'%sequence),storage_preflight(arguments.output,registry,done))
    completed=len(done)
    failure=None
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for cell in registry['cells']:
            pending=iter(rep for rep in range(cell['replicates']) if (cell['id'],rep) not in done)
            active={}
            while True:
                while failure is None and len(active)<arguments.workers:
                    replicate=next(pending,None)
                    if replicate is None:
                        break
                    try:
                        disk_guard(arguments.output,cell)
                        future=pool.submit(run_job,(registry,cell,replicate,arguments.output,arguments.cutoffs,freeze_identity))
                        active[future]=replicate
                    except Exception as error:
                        failure=dict(cell=cell['id'],replicate=replicate,reason='dispatch_failure',error=repr(error))
                if not active:
                    break
                finished,_=wait(active,return_when=FIRST_COMPLETED)
                for future in finished:
                    replicate=active.pop(future)
                    try:
                        result=future.result()
                    except Exception as error:
                        result=dict(cell=cell['id'],replicate=replicate,status='failed',error=repr(error))
                    completed+=1
                    print(json.dumps(dict(completed=completed,total=total,**result)),flush=True)
                    if result['status']=='failed':
                        failure=dict(reason='execution_failure',**result)
            if failure is not None:
                sequence=len(list(arguments.output.glob('RUN_STOPPED_*.json')))+1
                write_json(arguments.output/('RUN_STOPPED_%03d.json'%sequence),dict(completed=completed,**failure))
                raise RuntimeError('The study stopped. All opened draws and failures are preserved.')
    write_json(arguments.output/'RUN_COMPLETE.json',dict(records=completed,utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    main()
