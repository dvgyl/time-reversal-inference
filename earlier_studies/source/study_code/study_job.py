"""Run one locked replicate and preserve each completed method call."""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
import adapters as A
from generator import draw_locked_job
from study_common import BANK,CELLS,DESIGN,LENGTHS,REPLICATES,array_hash,read,save

B=A.B
ROOT=Path(__file__).resolve().parent


def verify_lock():
    path=ROOT/'OBSERVATION_LOCK.json'
    lock=read(path)
    if lock.get('observation_authorized') is not True:
        raise RuntimeError('The observation lock has not been authorized.')
    for item in lock['files']:
        if hashlib.sha256((ROOT.parent/item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('A locked study input changed: '+item['path'])
    return lock


def diagnostics(result,cell,length,population):
    source=result.source
    if isinstance(source,B.SourceHull):
        source_fields=dict(source_width=source.upper-source.lower,source_covers_phi=source.lower<=cell['phi']<=source.upper,
                           source_radius_width=source.radius.width,source_acceptance_excess=source.acceptance_excess)
    elif isinstance(source,B.SourceFallback):
        source_fields=dict(source_width=B.Q(1),source_covers_phi=True,source_radius_width=None,source_acceptance_excess=None)
    elif isinstance(source,B.SourceEmpty):
        source_fields=dict(source_width=None,source_covers_phi=False,source_radius_width=None,source_acceptance_excess=None)
    else:
        source_fields=dict(source_width=B.Q(0),source_covers_phi=None,source_radius_width=None,source_acceptance_excess=None)
    fitted=result.calibration
    cal_fields=dict(calibration_squared_error=None,calibration_radius_upper=None,calibration_covers_coefficients=None)
    if isinstance(fitted,B.CalibrationCertificate):
        squared=sum(((x-y)**2 for x,y in zip(fitted.coefficients,cell['response'])),B.Q(0))
        cal_fields=dict(calibration_squared_error=squared,calibration_radius_upper=fitted.error_l2.upper,
                        calibration_covers_coefficients=squared<=fitted.error_l2.upper**2)
    members=[]
    for coefficient,member in zip(BANK,result.members):
        if not isinstance(member,B.MemberDecision):
            members.append(dict(available=False,reason=member.reason))
            continue
        threshold=member.threshold.upper
        contrast=abs(population[(cell['cell_id'],length,coefficient)]['signed_contrast'])
        members.append(dict(available=True,strict_margin=member.statistic_absolute-threshold,
                            observed_threshold_ratio=member.statistic_absolute/threshold if threshold else None,
                            nominal_population_threshold_ratio=contrast/threshold if threshold else None,
                            threshold_excess=member.receipt['threshold_excess_upper'],statistic_loss=member.receipt['statistic_loss']))
    return dict(source_status=source.status,calibration_status='CERTIFIED' if isinstance(fitted,B.CalibrationCertificate) else fitted.reason,
                members=members,**source_fields,**cal_fields)


def execute(cell_id,replicate,destination):
    lock=verify_lock()
    if not 0<=cell_id<len(CELLS) or not 0<=replicate<REPLICATES:
        raise ValueError('The job is outside the declared plan.')
    cell=CELLS[cell_id]
    static=read(ROOT/'STATIC_PLANNING.json')
    population={(row['cell']['cell_id'],row['length'],row['coefficient']):row['population'] for row in static['rows']}
    record,response,generation=draw_locked_job(cell,replicate,lock)
    np.save(destination/'record.npy',record,allow_pickle=False)
    np.save(destination/'calibration.npy',response,allow_pickle=False)
    save(destination/'GENERATION.json',dict(cell=cell,replicate=replicate,generation=generation,
        record_file_sha256=hashlib.sha256((destination/'record.npy').read_bytes()).hexdigest(),
        calibration_file_sha256=hashlib.sha256((destination/'calibration.npy').read_bytes()).hexdigest()))
    calibration=B.CalibrationData(DESIGN,response.tolist(),(0,1))
    plan=B.BankPlan(BANK)
    native=record.tolist()
    if any(type(x) is not float for row in native for x in row) or any(type(x) is not float for x in calibration.response):
        raise TypeError('Numerical inputs must use native Python floats.')
    calls=[]
    for length in LENGTHS:
        values=native[:length]
        prefix_hash=array_hash(record[:length])
        for lane in ('primary','control'):
            started=time.perf_counter()
            result=A.opt.run_bank(plan,values,calibration) if lane=='primary' else A.run_supplied_phi(plan,values,calibration,cell['phi'])
            seconds=time.perf_counter()-started
            raw_hash=save(destination/f'RAW_N_{length}_{lane}.json',result)
            call=dict(complete=True,cell_id=cell_id,replicate=replicate,length=length,lane=lane,
                      result=result,diagnostics=diagnostics(result,cell,length,population),whole_call_seconds=seconds,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      prefix_byte_hash=prefix_hash,raw_result_sha256=raw_hash,
                      source_status=result.source.status,calibration_input_hash=generation['calibration_response_hash'])
            digest=save(destination/f'N_{length}_{lane}.json',call)
            calls.append(dict(length=length,lane=lane,sha256=digest,decision=result.decision))
            save(destination/'CALL_CHECKPOINT.json',calls)
    save(destination/'JOB_RESULT.json',dict(status='COMPLETED',cell_id=cell_id,replicate=replicate,calls=calls,
                                          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

if __name__=='__main__':
    destination=Path(sys.argv[3])
    try:
        execute(int(sys.argv[1]),int(sys.argv[2]),destination)
    except Exception as error:
        save(destination/'JOB_FAILURE.json',dict(exception=repr(error),traceback=traceback.format_exc()))
        raise
