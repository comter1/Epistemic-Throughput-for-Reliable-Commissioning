"""A detached remote process writes real trainer exit and GPU telemetry to shared storage."""
from pathlib import Path
import argparse
import fcntl
import os
import platform
import subprocess
import sys
import time
from cgx.provenance import read_json,atomic_json,now,require_protocol_lock,verify_hashes
from cgx.config import allocated_cpu_count


def launch(request_file):
    request=read_json(request_file);destination=Path(request['destination']);status=destination/'remote_status.json'
    if status.exists() and read_json(status).get('state') in ('starting','running','passed'):
        raise RuntimeError('Remote job already launched; inspect existing status instead of duplicating it')
    record=dict(state='starting',job_signature=request['job_signature'],hostname=platform.node(),gpu_uuid=request['gpu_uuid'],started_at=now())
    atomic_json(status,record)
    try:
        with (destination/'remote_supervisor.log').open('a') as stream:
            process=subprocess.Popen([sys.executable,'-m','cgx.runtime.remote_worker','--request',str(request_file)],
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,cwd=request['source_directory'],env=os.environ.copy())
        # The child owns future status updates, including its own PID.
        print({'supervisor_pid':process.pid,'status_file':str(status)},flush=True)
    except BaseException as exc:
        record.update(state='failed',exit_code=1,failure_reason=str(exc));atomic_json(status,record);raise


def run(request_file):
    request=read_json(request_file);root=Path(request['root']);destination=Path(request['destination']);job=request['job']
    status_file=destination/'remote_status.json';status=read_json(status_file);started=time.monotonic();process=None;peak=0
    with (destination/'remote_supervisor.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            require_protocol_lock(root);verify_hashes(job['prerequisite_files'])
            occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=15)
            if any(row.split(',')[0].strip()==request['gpu_uuid'] for row in occupied.splitlines()):raise RuntimeError('Remote GPU occupied; no unrelated process interrupted')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=request['gpu_uuid'],CGX_DATA_ROOT=str(root),
                PYTHONPATH=str(Path(request['source_directory'])/'src'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                CGX_EVAL_WORKERS=str(request['evaluation_workers']),CGX_FEATURE_READERS=str(request['feature_readers']),PYTHONUNBUFFERED='1')
            with (destination/'stdout.log').open('a') as out,(destination/'stderr.log').open('a') as err:
                process=subprocess.Popen(job['command'],cwd=request['source_directory'],env=env,stdout=out,stderr=err,start_new_session=True)
                status.update(state='running',supervisor_pid=os.getpid(),pid=process.pid,allocated_cpu_count=allocated_cpu_count(),command=job['command'])
                atomic_json(status_file,status)
                while process.poll() is None:
                    try:
                        usage=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
                        for row in usage.splitlines():
                            pid,mb=row.replace(' ','').split(',')
                            if int(pid)==process.pid:peak=max(peak,int(mb)*1024**2)
                    except (ValueError,subprocess.SubprocessError):pass
                    status.update(heartbeat=now(),peak_gpu_memory_bytes=peak,elapsed_seconds=time.monotonic()-started);atomic_json(status_file,status)
                    time.sleep(5)
            status.update(state='passed' if process.returncode==0 else 'failed',exit_code=process.returncode)
        except BaseException as exc:
            status.update(state='failed',exit_code=1,failure_reason=f'{type(exc).__name__}: {exc}')
        finally:
            status.update(finished_at=now(),elapsed_seconds=time.monotonic()-started,peak_gpu_memory_bytes=peak)
            atomic_json(status_file,status)
    if status['state']!='passed':raise RuntimeError(status.get('failure_reason','Remote trainer failed'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--launch',action='store_true');a=p.parse_args()
    (launch if a.launch else run)(a.request)
