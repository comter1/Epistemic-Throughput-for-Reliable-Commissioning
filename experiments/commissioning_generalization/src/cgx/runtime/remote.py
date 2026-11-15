"""Launch one detached GPU worker over SSH on a verified shared filesystem."""
from pathlib import Path
import os
import shlex
import subprocess
import time
from cgx.config import EXPERIMENT
from cgx.provenance import read_json,atomic_json,file_sha256,hash_json,now


def registered_workers():
    value=os.environ.get('CGX_REMOTE_WORKERS_FILE')
    if not value or not Path(value).exists():return []
    registry=read_json(value);workers=registry.get('workers',[])
    for worker in workers:
        report=read_json(worker['hardware_report'])
        if (report.get('state')!='passed' or not report.get('actual_gpu_forward_backward_index_add')
                or file_sha256(worker['hardware_report'])!=worker['hardware_report_hash']
                or report.get('gpu_uuid')!=worker['gpu_uuid'] or not worker.get('shared_filesystem_verified')):
            raise ValueError('Unqualified remote worker')
        if not Path(worker['identity_file']).is_file():raise ValueError('Remote SSH identity file missing')
    if len({w['gpu_uuid'] for w in workers})!=len(workers):raise ValueError('Duplicate remote GPU')
    return workers


def launch_remote(worker,job,root,destination):
    root=Path(root);destination=Path(destination);status_file=destination/'remote_status.json'
    signature=hash_json(job)
    if status_file.exists():
        old=read_json(status_file)
        if old['job_signature']!=signature:raise ValueError('Remote job identity changed')
        if old['state'] in ('starting','running','passed'):return old
    request={'job':job,'job_signature':signature,'root':str(root),'destination':str(destination),
             'gpu_uuid':worker['gpu_uuid'],'source_directory':str(EXPERIMENT),'requested_at':now(),
             'evaluation_workers':min(16,worker['allocated_cpu_count']//2),
             'feature_readers':4}
    request_file=destination/'remote_request.json';atomic_json(request_file,request)
    command=['env','PYTHONPATH='+str(EXPERIMENT/'src'),'PYTHONDONTWRITEBYTECODE=1',
             worker['python'],'-m','cgx.runtime.remote_worker','--launch','--request',str(request_file)]
    ssh=['ssh','-p',str(worker['port']),'-i',worker['identity_file'],'-o','IdentitiesOnly=yes',
         '-o','BatchMode=yes','-o','ConnectTimeout=15',worker['host'],shlex.join(command)]
    result=subprocess.run(ssh,text=True,capture_output=True,timeout=45)
    atomic_json(destination/'remote_launch.json',{'command':ssh,'time':now(),'exit_code':result.returncode,
        'stdout':result.stdout,'stderr':result.stderr,'worker':{k:v for k,v in worker.items() if k!='identity_file'}})
    if result.returncode and not status_file.exists():raise RuntimeError(f'Remote launch failed: {result.stderr.strip()}')
    if not status_file.exists():raise RuntimeError('Remote launcher did not record its process')
    return read_json(status_file)


def remote_progress(destination):
    return read_json(Path(destination)/'remote_status.json')
