"""Check committed study calls before execution completion or analysis."""
import hashlib
from pathlib import Path
import numpy as np
from study_common import BANK,CELLS,DESIGN,LENGTHS,MASTER_SEED,array_hash,read
from fractions import Fraction as Q
from dataclasses import asdict
from adapters import B

EXPECTED=tuple((length,lane) for length in LENGTHS for lane in ('primary','control'))
FAILURES=('FAILED','TIMEOUT','INTERRUPTED','CONTRACT_FAILURE')

class ContractError(ValueError):
    pass


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition,message):
    if not condition: raise ContractError(message)


def failure(path,error):
    path=Path(path)
    return dict(path=str(path),reason=str(error),sha256=file_hash(path) if path.is_file() else None)


def inputs(directory,cell_id,replicate):
    generation=read(directory/'GENERATION.json')
    declared=dict(generation['cell'])
    declared['response']=tuple(declared['response'])
    require(declared==CELLS[cell_id] and generation['replicate']==replicate,'Generation job identity mismatch.')
    record_path,cal_path=directory/'record.npy',directory/'calibration.npy'
    require(file_hash(record_path)==generation['record_file_sha256'],'Recording file hash mismatch.')
    require(file_hash(cal_path)==generation['calibration_file_sha256'],'Calibration file hash mismatch.')
    record=np.load(record_path,allow_pickle=False)
    calibration=np.load(cal_path,allow_pickle=False)
    require(record.shape==(LENGTHS[-1],2) and record.dtype==np.dtype('float64'),'Recording shape or type mismatch.')
    require(calibration.shape==(128,) and calibration.dtype==np.dtype('float64'),'Calibration shape or type mismatch.')
    require(bool(np.isfinite(record).all()) and bool(np.isfinite(calibration).all()),'Nonfinite study input.')
    info=generation['generation']
    require(array_hash(record)==info['record_hash'] and array_hash(calibration)==info['calibration_response_hash'],'Input byte hash mismatch.')
    require(info['master_seed']==MASTER_SEED,'Master seed mismatch.')
    require(tuple(map(tuple,info['spawn_keys']))==tuple((cell_id,replicate,s) for s in range(3)),'Seed stream identity mismatch.')
    require(info['generator_algorithm']=='PCG64' and len(info['initial_generator_states'])==3,'Generator state receipt mismatch.')
    require(all(state['bit_generator']=='PCG64' for state in info['initial_generator_states']),'Generator state algorithm mismatch.')
    return dict(prefix_hashes={n:array_hash(record[:n]) for n in LENGTHS},calibration_hash=info['calibration_response_hash'])


def checkpoint(directory):
    entries=read(directory/'CALL_CHECKPOINT.json')
    require(isinstance(entries,list),'The call checkpoint must be a list.')
    keys=[(entry['length'],entry['lane']) for entry in entries]
    require(len(set(keys))==len(keys) and all(key in EXPECTED for key in keys),'Duplicate or unknown checkpoint call.')
    require(tuple(keys)==EXPECTED[:len(keys)],'Call checkpoint order mismatch.')
    return {(entry['length'],entry['lane']):entry for entry in entries}


def validate_call(directory,cell_id,replicate,length,lane,commit,input_info):
    path=directory/f'N_{length}_{lane}.json'
    require(file_hash(path)==commit['sha256'],'Committed call hash mismatch.')
    call=read(path)
    require(isinstance(call,dict),'Invalid call object.')
    require(call.get('complete') is True,'The call is incomplete.')
    require((call['cell_id'],call['replicate'],call['length'],call['lane'])==(cell_id,replicate,length,lane),'Call identity mismatch.')
    require(call['prefix_byte_hash']==input_info['prefix_hashes'][length],'Call recording prefix hash mismatch.')
    require(call['calibration_input_hash']==input_info['calibration_hash'],'Call calibration hash mismatch.')
    raw_path=directory/f'RAW_N_{length}_{lane}.json'
    require(file_hash(raw_path)==call['raw_result_sha256'],'Raw result hash mismatch.')
    require(read(raw_path)==call['result'],'Call and raw result disagree.')
    result=call['result']
    require(isinstance(result,dict),'Invalid result object.')
    require(isinstance(result['source'],dict) and isinstance(result['receipt'],dict),'Invalid source or result receipt.')
    require(isinstance(result['members'],list) and all(isinstance(member,dict) for member in result['members']),'Invalid result member list.')
    require(result['decision'] in ('REJECT','DO_NOT_REJECT','ABSTAIN'),'Unknown statistical decision.')
    require(commit['decision']==result['decision'],'Committed decision mismatch.')
    require(len(result['members'])==len(BANK),'Member count mismatch.')
    receipt=result['receipt']
    require(tuple(receipt['bank'])==BANK,'Bank coefficient mismatch.')
    for key,value in (('source_budget',Q(1,200)),('calibration_budget',Q(1,100)),('variance_budget',Q(1,100)),('tail_budget',Q(1,40))):
        require(receipt[key]==value,'Probability budget mismatch: '+key)
    require(receipt['power_status']=='POWER_UNVERIFIED','Power status mismatch.')
    require(receipt['record_length']==length,'Recording length receipt mismatch.')
    require(receipt.get('limits')==asdict(B.WorkLimits()),'Arithmetic limits mismatch.')
    phases=receipt['phase_operations']
    require(set(phases)=={'input','shared_moments','source','calibration','members'} and len(phases['members'])==len(BANK),'Phase receipt shape mismatch.')
    operations=[phases[k] for k in ('input','shared_moments','source','calibration')]+phases['members']
    require(all(type(x) is int and x>=0 for x in operations) and sum(operations)==receipt['total_executed_operations'],'Operation receipt mismatch.')
    if lane=='control': require(result['source']['phi']==CELLS[cell_id]['phi'],'Supplied parameter mismatch.')
    require(call['source_status']==result['source']['status'],'Source status mismatch.')
    require(call['source_status'] in (('CERTIFIED_HULL','EMPTY_CERTIFIED','FULL_FALLBACK') if lane=='primary' else ('SUPPLIED_PARAMETER',)),'Source type mismatch.')
    available=[]
    for coefficient,member in zip(BANK,result['members']):
        if 'reject' not in member:
            require(isinstance(member.get('reason'),str) and isinstance(member.get('receipt'),dict),'Unavailable member receipt mismatch.')
            continue
        require(member['coefficient']==coefficient,'Member coefficient mismatch.')
        require(member['retained_length']==length-1 and member['pair_count']==length-2,'Retained length mismatch.')
        threshold=member['threshold']
        require(type(member['reject']) is bool and 0<=threshold['lower']<=threshold['upper'] and member['statistic_absolute']>=0,'Invalid member decision.')
        require(member['receipt']['limits']==asdict(B.WorkLimits()),'Member arithmetic limits mismatch.')
        require(member['reject']==(member['statistic_absolute']>threshold['upper']),'Strict rejection comparison mismatch.')
        available.append(member)
    expected='REJECT' if any(member['reject'] for member in available) else ('DO_NOT_REJECT' if available else 'ABSTAIN')
    require(result['decision']==expected,'Bank OR decision mismatch.')
    if available and lane=='primary': require(result['source']['status']=='CERTIFIED_HULL','Available primary member lacks a finite source hull.')
    validate_diagnostics(call,cell_id,lane)
    diagnostics=call['diagnostics']
    require(diagnostics['source_status']==call['source_status'],'Diagnostic source status mismatch.')
    require(len(diagnostics['members'])==len(BANK),'Diagnostic member count mismatch.')
    require(all(item['available']==('reject' in member) for item,member in zip(diagnostics['members'],result['members'])),'Diagnostic member availability mismatch.')
    return call


def rational(value, name, optional=False, nonnegative=False):
    require((optional and value is None) or type(value) is Q,'Invalid rational field: '+name)
    if value is not None and nonnegative: require(value>=0,'Negative field: '+name)


def interval(value, name):
    require(isinstance(value,dict),'Invalid interval: '+name)
    rational(value['lower'],name+'.lower',nonnegative=True)
    rational(value['upper'],name+'.upper',nonnegative=True)
    require(value['lower']<=value['upper'],'Reversed interval: '+name)


def validate_diagnostics(call,cell_id,lane):
    result=call['result']
    source=result['source']
    diagnostics=call['diagnostics']
    require(isinstance(diagnostics,dict),'Invalid diagnostic object.')
    require(type(diagnostics['source_covers_phi']) in (bool,type(None)),'Invalid source coverage diagnostic.')
    require(type(diagnostics['calibration_covers_coefficients']) in (bool,type(None)),'Invalid calibration coverage diagnostic.')
    for key in ('source_width','source_radius_width','source_acceptance_excess','calibration_squared_error','calibration_radius_upper'):
        rational(diagnostics[key],key,optional=True,nonnegative=True)
    if source['status']=='CERTIFIED_HULL':
        rational(source['lower'],'source.lower',nonnegative=True)
        rational(source['upper'],'source.upper',nonnegative=True)
        require(source['lower']<=source['upper']<=1,'Invalid source endpoints.')
        interval(source['radius'],'source.radius')
        rational(source['acceptance_excess'],'source.acceptance_excess',optional=True,nonnegative=True)
        width=source['upper']-source['lower']
        coverage=source['lower']<=CELLS[cell_id]['phi']<=source['upper']
        radius_width=source['radius']['upper']-source['radius']['lower']
        excess=source['acceptance_excess']
    else:
        width=Q(1) if source['status']=='FULL_FALLBACK' else (Q(0) if lane=='control' else None)
        coverage=True if source['status']=='FULL_FALLBACK' else (None if lane=='control' else False)
        radius_width=excess=None
    require((diagnostics['source_width'],diagnostics['source_covers_phi'],diagnostics['source_radius_width'],diagnostics['source_acceptance_excess'])==(width,coverage,radius_width,excess),'Source diagnostics disagree with the source result.')
    calibration=result['calibration']
    require(isinstance(calibration,dict),'Invalid calibration result.')
    squared=radius=covered=None
    if 'coefficients' in calibration:
        require(len(calibration['coefficients'])==2 and tuple(calibration['positions'])==(0,1),'Calibration coefficient structure mismatch.')
        for value in calibration['coefficients']: rational(value,'calibration.coefficient')
        interval(calibration['error_l2'],'calibration.error_l2')
        squared=sum(((x-y)**2 for x,y in zip(calibration['coefficients'],CELLS[cell_id]['response'])),Q(0))
        radius=calibration['error_l2']['upper'];covered=squared<=radius**2
        calibration_status='CERTIFIED'
    else:
        require(isinstance(calibration['reason'],str) and isinstance(calibration['receipt'],dict),'Invalid unavailable calibration.')
        calibration_status=calibration['reason']
    require(diagnostics['calibration_status']==calibration_status,'Calibration status diagnostic mismatch.')
    require((diagnostics['calibration_squared_error'],diagnostics['calibration_radius_upper'],diagnostics['calibration_covers_coefficients'])==(squared,radius,covered),'Calibration diagnostics disagree with the result.')
    require(isinstance(diagnostics['members'],list) and len(diagnostics['members'])==len(BANK),'Invalid diagnostic member list.')
    for index,(item,member) in enumerate(zip(diagnostics['members'],result['members'])):
        require(isinstance(item,dict) and type(item['available']) is bool,'Invalid member availability diagnostic.')
        available='reject' in member
        require(item['available']==available,'Member availability diagnostic mismatch.')
        if not available:
            require(isinstance(item['reason'],str) and item['reason']==member['reason'],'Invalid unavailable member reason.')
            continue
        require('coefficients' in calibration,'Available member lacks a calibration certificate.')
        for key in ('statistic_absolute','source_ratio','covariance_ratio'):
            rational(member[key],'member.'+key,nonnegative=True)
        interval(member['threshold'],'member.threshold')
        receipt=member['receipt']
        rational(receipt['signed_statistic'],'member.signed_statistic')
        require(member['statistic_absolute']==abs(receipt['signed_statistic']),'Signed and absolute statistics disagree.')
        require(isinstance(receipt['variances'],list) and len(receipt['variances'])==2,'Invalid variance list.')
        for value in receipt['variances']:rational(value,'member.variance',nonnegative=True)
        for key in ('threshold_excess_upper','statistic_loss'):rational(receipt[key],'member.'+key,nonnegative=True)
        for key in ('observed_threshold_ratio','nominal_population_threshold_ratio','strict_margin','threshold_excess','statistic_loss'):
            rational(item[key],'diagnostic.'+key,optional=key.endswith('_ratio'),nonnegative=key!='strict_margin')
        threshold=member['threshold']['upper']
        ratio=member['statistic_absolute']/threshold if threshold else None
        require(item['observed_threshold_ratio']==ratio and item['strict_margin']==member['statistic_absolute']-threshold,'Invalid realized-threshold diagnostic.')
        require(item['threshold_excess']==receipt['threshold_excess_upper'] and item['statistic_loss']==receipt['statistic_loss'],'Invalid arithmetic-loss diagnostic.')


def validate_completed_job(directory,cell_id,replicate):
    directory=Path(directory)
    job=read(directory/'JOB_RESULT.json')
    require(job['status']=='COMPLETED' and (job['cell_id'],job['replicate'])==(cell_id,replicate),'Completed job identity mismatch.')
    entries=job['calls']
    keys=[(entry['length'],entry['lane']) for entry in entries]
    require(len(keys)==len(EXPECTED) and len(set(keys))==len(EXPECTED) and set(keys)==set(EXPECTED),'Completed job call set mismatch.')
    commits=checkpoint(directory)
    require(set(commits)==set(EXPECTED),'Completed checkpoint call set mismatch.')
    for entry in entries: require(entry==commits[(entry['length'],entry['lane'])],'Job and checkpoint disagree.')
    info=inputs(directory,cell_id,replicate)
    calls={key:validate_call(directory,cell_id,replicate,*key,commits[key],info) for key in EXPECTED}
    for length in LENGTHS:
        first,second=calls[(length,'primary')],calls[(length,'control')]
        require(first['result']['receipt'].get('recording_sha256')==second['result']['receipt'].get('recording_sha256'),'Paired exact recording identities disagree.')
    return job,calls


def load_calls(directory,cell_id,replicate):
    directory=Path(directory)
    issues=[failure(path,'Uncommitted atomic write.') for path in directory.glob('*.pending')]
    process_path=directory/'PROCESS.json'
    try:
        process=read(process_path) if process_path.exists() else dict(status='UNEXECUTED')
        status=process['status']
        if process_path.exists(): require((process['cell_id'],process['replicate'])==(cell_id,replicate),'Process job identity mismatch.')
        require(status in ('COMPLETED','UNEXECUTED',*FAILURES),'Unknown process status.')
    except (OSError,ValueError,KeyError,TypeError,ArithmeticError) as error:
        status='CONTRACT_FAILURE';issues.append(failure(process_path,error))
    if status=='COMPLETED':
        try:
            _,calls=validate_completed_job(directory,cell_id,replicate)
            return calls,issues
        except (OSError,ValueError,KeyError,TypeError,IndexError,ArithmeticError) as error:
            issues.append(failure(directory/'JOB_RESULT.json',error))
            for path in directory.glob('N_*.json'):
                try: read(path)
                except (OSError,ValueError,TypeError,ArithmeticError) as call_error: issues.append(failure(path,call_error))
            status='CONTRACT_FAILURE'
    commits={}
    info=None
    try:
        if (directory/'CALL_CHECKPOINT.json').exists():
            commits=checkpoint(directory)
            info=inputs(directory,cell_id,replicate)
    except (OSError,ValueError,KeyError,TypeError,IndexError,ArithmeticError) as error:
        issues.append(failure(directory/'CALL_CHECKPOINT.json',error))
        commits={}
    calls={}
    for key in EXPECTED:
        path=directory/f'N_{key[0]}_{key[1]}.json'
        if key in commits:
            try:
                calls[key]=validate_call(directory,cell_id,replicate,*key,commits[key],info)
                continue
            except (OSError,ValueError,KeyError,TypeError,IndexError,ArithmeticError) as error:
                issues.append(failure(path,error))
        elif path.exists():
            issues.append(failure(path,'A call file has no valid checkpoint commitment.'))
        calls[key]=dict(missing_status='ERROR' if status in FAILURES or path.exists() or key in commits else 'UNEXECUTED')
    return calls,issues
