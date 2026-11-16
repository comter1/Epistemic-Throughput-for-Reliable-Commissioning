"""CPU-only evaluation queue, including hosts whose GPU is occupied.

The existing GPU dispatcher is immutable after its training queue was sealed.
This adapter shares its atomic claiming protocol without acquiring a GPU.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from cgx.config import EXPERIMENT
from cgx.provenance import atomic_json, file_sha256, read_json
from .dispatch import initialize as initialize_base, locked, claim, now


def initialize(run, stage, tasks_file):
    initialize_base(run, stage, tasks_file)
    path = Path(run) / 'dispatch' / stage / 'queue.json'
    with locked(path):
        data = read_json(path)
        assert all(job['state'] == 'pending' for job in data['jobs'].values())
        data['cpu_scheduler_sha256'] = file_sha256(__file__)
        atomic_json(path, data)
    return len(data['jobs'])


def worker(run, stage, node, slot, workers):
    run = Path(run)
    path = run / 'dispatch' / stage / 'queue.json'
    assert read_json(path)['cpu_scheduler_sha256'] == file_sha256(__file__)
    owner = dict(node=str(node), hostname=socket.gethostname(), slot=slot,
                 cpu_workers=workers, pid=os.getpid(), device='cpu')
    completed = 0
    while True:
        selected = claim(path, owner)
        if selected is None:
            break
        name, task = selected
        substitutions = dict(device='cpu', workers=workers, run_dir=str(run))
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
        result = dict(exit_code=exit_code, finished_at=now(),
            wall_seconds=time.monotonic()-started, argv=command,
            child_pid=process.pid, log_path=str(log), state='passed' if exit_code == 0 else 'failed')
        with locked(path):
            data = read_json(path)
            assert data['jobs'][name]['owner'] == owner
            data['jobs'][name].update(result)
            atomic_json(path, data)
        print(json.dumps(dict(job=name, owner=owner, **result)), flush=True)
        completed += 1
        if exit_code:
            raise RuntimeError(f'{name} failed; inspect {log} before any explicit retry')
    return completed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    setup = sub.add_parser('init')
    setup.add_argument('--run-dir', required=True)
    setup.add_argument('--stage', required=True)
    setup.add_argument('--tasks-file', required=True)
    launch = sub.add_parser('worker')
    launch.add_argument('--run-dir', required=True)
    launch.add_argument('--stage', required=True)
    launch.add_argument('--node', required=True)
    launch.add_argument('--slot', type=int, required=True)
    launch.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    result = initialize(args.run_dir, args.stage, args.tasks_file) if args.command == 'init' else \
        worker(args.run_dir, args.stage, args.node, args.slot, args.workers)
    print(result, 'jobs', flush=True)
