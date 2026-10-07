"""Execute the fixed two-worker study after its observation lock."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from study_common import CELLS,REPLICATES,read,save
from contracts import validate_completed_job

ROOT=Path(__file__).resolve().parent
PYTHON=sys.executable
THREAD_KEYS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def run(attempt_name):
    lock_path=ROOT/'OBSERVATION_LOCK.json'
    lock=read(lock_path)
    if lock.get('observation_authorized') is not True:
        raise RuntimeError('The observation lock has not been authorized.')
    for item in lock['files']:
        if hashlib.sha256((ROOT.parent/item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('A locked study input changed: '+item['path'])
    attempt=ROOT/'study_attempts'/attempt_name
    attempt.mkdir(parents=True,exist_ok=False)
    source=attempt/'source'
    source.mkdir()
    for item in lock['files']:
        path=ROOT.parent/item['path']
        if path.parent==ROOT: shutil.copy2(path,source/path.name)
    shutil.copy2(lock_path,source/lock_path.name)
    planned=[dict(job_id=f'r{replicate:03d}_c{cell["cell_id"]:02d}',cell_id=cell['cell_id'],replicate=replicate,status='UNEXECUTED')
             for replicate in range(REPLICATES) for cell in CELLS]
    save(attempt/'PLANNED_JOBS.json',planned)
    for row in planned: (attempt/'jobs'/row['job_id']).mkdir(parents=True)
    env=os.environ.copy()
    env.update({key:'1' for key in THREAD_KEYS})
    started=time.monotonic()
    running={}
    next_job=0
    stop=None
    completed_bytes=0
    if shutil.disk_usage(attempt).free<4*2**30: stop='initial_free_storage_below_4GiB'
    while running or (next_job<len(planned) and stop is None):
        now=time.monotonic()
        if now-started>=21600 and stop is None: stop='wave_time_limit'
        for job_id,work in list(running.items()):
            process=work['process']
            status=None
            if now-started>=21600 and process.poll() is None:
                try: process.kill()
                except ProcessLookupError: pass
                status='INTERRUPTED'
            elif now-work['start']>=120 and process.poll() is None:
                try: process.kill()
                except ProcessLookupError: pass
                status='TIMEOUT'
            code=process.poll()
            if code is None and status is None: continue
            if code is None: code=process.wait()
            work['stdout'].close();work['stderr'].close()
            row=planned[work['index']]
            destination=attempt/'jobs'/job_id
            status=status or ('COMPLETED' if code==0 else 'FAILED')
            if status=='COMPLETED':
                try:
                    result,_=validate_completed_job(destination,row['cell_id'],row['replicate'])
                    if result['peak_rss_bytes']>512*2**20 and stop is None: stop='peak_memory_above_512MiB'
                except Exception as error:
                    status='CONTRACT_FAILURE';row['contract_error']=repr(error)
            row.update(status=status,returncode=code,total_child_seconds=time.monotonic()-work['start'])
            save(destination/'PROCESS.json',row)
            completed_bytes+=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
            if status!='COMPLETED' and stop is None: stop='job_'+status.lower()
            del running[job_id]
            save(attempt/'WAVE_CHECKPOINT.json',dict(jobs=planned,stop_reason=stop,elapsed_seconds=time.monotonic()-started))
            print(row,flush=True)
        recorded_bytes=completed_bytes
        candidates=list(attempt.glob('*'))+list(source.rglob('*'))
        for job_id in running: candidates.extend((attempt/'jobs'/job_id).rglob('*'))
        for path in candidates:
            try:
                if path.is_file(): recorded_bytes+=path.stat().st_size
            except FileNotFoundError:
                pass
        if recorded_bytes>4*2**30 and stop is None: stop='recorded_output_above_4GiB'
        if shutil.disk_usage(attempt).free<2**30 and stop is None: stop='free_storage_below_1GiB'
        while stop is None and len(running)<2 and next_job<len(planned):
            if time.monotonic()-started>=21600:
                stop='wave_time_limit'
                break
            row=planned[next_job]
            destination=attempt/'jobs'/row['job_id']
            command=[PYTHON,str(ROOT/'study_job.py'),str(row['cell_id']),str(row['replicate']),str(destination)]
            stdout=(destination/'stdout.txt').open('w')
            stderr=(destination/'stderr.txt').open('w')
            launch=time.monotonic()
            save(destination/'LAUNCH.json',dict(command=command,thread_environment={key:env[key] for key in THREAD_KEYS},
                 timeout_seconds=120,launch_index=next_job,wave_elapsed_seconds=launch-started))
            try:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
            except OSError as error:
                stdout.close();stderr.close()
                row.update(status='FAILED',launch_error=repr(error))
                save(destination/'PROCESS.json',row)
                stop='job_launch_failure'
                next_job+=1
                save(attempt/'WAVE_CHECKPOINT.json',dict(jobs=planned,stop_reason=stop,elapsed_seconds=time.monotonic()-started))
                break
            row['status']='RUNNING'
            running[row['job_id']]=dict(process=process,start=launch,index=next_job,stdout=stdout,stderr=stderr)
            next_job+=1
        if running: time.sleep(0.2)
    for row in planned:
        if row['status']=='UNEXECUTED':
            row['reason']=stop or 'not_launched'
            save(attempt/'jobs'/row['job_id']/'PROCESS.json',row)
    save(attempt/'WAVE_RESULT.json',dict(jobs=planned,stop_reason=stop,total_seconds=time.monotonic()-started,
         completed_job_output_bytes=completed_bytes,last_measured_recorded_bytes=recorded_bytes if 'recorded_bytes' in locals() else 0,observation_lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest()))

if __name__=='__main__': run(sys.argv[1])
