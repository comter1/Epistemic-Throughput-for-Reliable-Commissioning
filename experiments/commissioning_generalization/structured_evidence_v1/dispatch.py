"""Single-owner remote jobs across the available GPU/CPU slots.

This scheduler changes neither samples nor training updates. Failed scientific
jobs remain failed for inspection; no automatic retries or result pruning.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from cgx.config import EXPERIMENT
from cgx.provenance import atomic_json, read_json, file_sha256


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def locked(path):
    with path.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def initialize(run, stage, tasks_file):
    directory = Path(run) / 'dispatch' / stage
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'queue.json'
    with locked(path):
        if path.exists():
            raise ValueError('Existing dispatch queue is preserved')
        tasks = read_json(tasks_file)
        assert len({t['id'] for t in tasks}) == len(tasks)
        assert all(isinstance(t['argv'], list) and t['argv'] for t in tasks)
        data = dict(created_at=now(), tasks_file_sha256=file_sha256(tasks_file),
            scheduler_sha256=file_sha256(__file__),
            jobs={t['id']: dict(state='pending', task=t) for t in tasks})
        atomic_json(path, data)
    return data


def claim(path, owner):
    with locked(path):
        data = read_json(path)
        if data['scheduler_sha256'] != file_sha256(__file__):
            raise ValueError('Dispatcher changed after task queue initialization')
        for name, job in data['jobs'].items():
            if job['state'] == 'pending':
                job.update(state='running', owner=owner, started_at=now())
                atomic_json(path, data)
                return name, job['task']
    return None


def worker(run, stage, node, gpu, workers, max_tasks=0):
    run = Path(run)
    path = run / 'dispatch' / stage / 'queue.json'
    owner = dict(node=str(node), hostname=socket.gethostname(), gpu=gpu,
        cpu_workers=workers, pid=os.getpid())
    # Do not appropriate a card occupied by another job. Each GPU has one
    # scheduler worker, and jobs execute serially within that slot.
    used = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if int(used.strip()) > 1024:
        raise RuntimeError(f'GPU {gpu} currently occupied ({used.strip()} MiB); preserve other jobs')
    completed = 0
    while not max_tasks or completed < max_tasks:
        selected = claim(path, owner)
        if selected is None:
            break
        name, task = selected
        substitutions = dict(device=f'cuda:{gpu}', gpu=gpu, workers=workers, run_dir=str(run))
        command = [str(arg).format(**substitutions) for arg in task['argv']]
        environment = dict(os.environ, PYTHONPATH='.:src', OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        environment.update({k: str(v).format(**substitutions) for k,v in task.get('env', {}).items()})
        started = time.monotonic()
        log = run / 'logs' / f'{stage}_{name}.log'
        with log.open('a') as handle:
            process = subprocess.Popen(command, cwd=EXPERIMENT, env=environment,
                stdout=handle, stderr=subprocess.STDOUT)
            exit_code = process.wait()
        elapsed = time.monotonic()-started
        result = dict(exit_code=exit_code, finished_at=now(), wall_seconds=elapsed,
            argv=command, child_pid=process.pid, log_path=str(log),
            state='passed' if exit_code == 0 else 'failed')
        with locked(path):
            data=read_json(path)
            if data['jobs'][name]['owner'] != owner:
                raise RuntimeError('Job ownership changed during execution')
            data['jobs'][name].update(result)
            atomic_json(path, data)
        print(json.dumps(dict(job=name, owner=owner, **result)), flush=True)
        completed += 1
        if exit_code:
            raise RuntimeError(f'{name} failed with exit {exit_code}; inspect preserved log before retry')
    return completed


if __name__ == '__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    i=s.add_parser('init');i.add_argument('--run-dir',required=True);i.add_argument('--stage',required=True)
    i.add_argument('--tasks-file',required=True)
    w=s.add_parser('worker');w.add_argument('--run-dir',required=True);w.add_argument('--stage',required=True)
    w.add_argument('--node',required=True);w.add_argument('--gpu',type=int,required=True)
    w.add_argument('--workers',type=int,required=True);w.add_argument('--max-tasks',type=int,default=0)
    a=p.parse_args()
    if a.command=='init':
        result=initialize(a.run_dir,a.stage,a.tasks_file);print(len(result['jobs']),'queued jobs')
    else:
        print(worker(a.run_dir,a.stage,a.node,a.gpu,a.workers,a.max_tasks),'jobs completed')
