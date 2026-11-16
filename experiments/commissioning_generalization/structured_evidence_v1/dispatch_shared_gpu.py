"""Co-schedule independent fits on GPUs occupied only by this experiment.

Measured CPU-bound fits leave most GPU compute unused. Each process retains its
own original optimizer, batch, RNG, checkpoint selection and deadlines. The
separately sealed primary dispatcher remains unchanged.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from cgx.config import EXPERIMENT
from cgx.provenance import atomic_json, file_sha256, read_json
from .dispatch import locked, claim, now


def verify_occupants(run, gpu):
    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    uuid, free = next(csv.reader(row.splitlines()))
    if int(free.strip()) < 20000:
        raise RuntimeError('Insufficient GPU memory headroom for another independent fit')
    listing = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    occupants = []
    for fields in csv.reader(listing.splitlines()):
        if len(fields) != 2 or fields[0].strip() != uuid.strip():
            continue
        pid = int(fields[1].strip()); process = Path('/proc') / str(pid)
        try:
            arguments = (process / 'cmdline').read_bytes().decode().split('\0')
            uid = process.stat().st_uid
        except FileNotFoundError:
            continue
        if uid != os.getuid() or 'structured_evidence_v1.policy_training' not in arguments or \
                'fit' not in arguments or str(run) not in arguments:
            raise RuntimeError(f'GPU {gpu} has a process outside this experiment: {pid}')
        occupants.append(pid)
    return dict(gpu=gpu, free_mib=int(free.strip()), verified_same_run_fit_pids=occupants)


def worker(run, node, gpu, slot, workers=8):
    run = Path(run)
    directory = run / 'dispatch/policies'
    allocation = read_json(directory / 'shared_gpu_allocation.json')
    assert allocation['implementation_sha256'] == file_sha256(__file__)
    assert dict(node=str(node), gpu=gpu, slot=slot, cpu_workers=workers) in allocation['allowed_workers']
    queue = directory / 'queue.json'
    owner = dict(node=str(node), hostname=socket.gethostname(), gpu=gpu,
                 slot=slot, cpu_workers=workers, pid=os.getpid(), shared_same_experiment_gpu=True)
    completed = 0
    while True:
        occupancy = verify_occupants(run, gpu)
        selected = claim(queue, owner)
        if selected is None:
            break
        name, task = selected
        substitutions = dict(device=f'cuda:{gpu}', gpu=gpu, workers=workers, run_dir=str(run))
        command = [str(arg).format(**substitutions) for arg in task['argv']]
        environment = dict(os.environ, PYTHONPATH='.:src', OMP_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        environment.update({k: str(v).format(**substitutions) for k,v in task.get('env', {}).items()})
        started = time.monotonic()
        log = run / 'logs' / f'policies_{name}.log'
        with log.open('a') as handle:
            process = subprocess.Popen(command, cwd=EXPERIMENT, env=environment,
                stdout=handle, stderr=subprocess.STDOUT)
            with locked(queue):
                data = read_json(queue); assert data['jobs'][name]['owner'] == owner
                data['jobs'][name].update(child_pid=process.pid, argv=command,
                    log_path=str(log), startup_occupancy=occupancy)
                atomic_json(queue, data)
            exit_code = process.wait()
        result = dict(exit_code=exit_code, finished_at=now(),
            wall_seconds=time.monotonic()-started, state='passed' if exit_code == 0 else 'failed')
        with locked(queue):
            data = read_json(queue); assert data['jobs'][name]['owner'] == owner
            data['jobs'][name].update(result); atomic_json(queue, data)
        print(json.dumps(dict(job=name, owner=owner, **result)), flush=True)
        completed += 1
        if exit_code:
            raise RuntimeError(f'{name} failed; inspect preserved log before any explicit retry')
    return completed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--slot', type=int, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    print(worker(args.run_dir, args.node, args.gpu, args.slot, args.workers), 'jobs completed', flush=True)
