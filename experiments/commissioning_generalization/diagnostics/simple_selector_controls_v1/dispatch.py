"""Local SSH coordination only; all experiment computation runs on remote CPUs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


EXPERIMENT = Path(__file__).resolve().parents[2]
PYTHON = '/home/leiyo/work/cgx_env/bin/python'
SLOTS = ((2379, 40), (2982, 40), (2982, 40),
         (2404, 40), (2404, 40), (2404, 40), (2404, 40))


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def remote_command(arguments):
    env = ['env', 'PYTHONPATH=.:src', 'OPENBLAS_NUM_THREADS=1',
           'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'CUDA_VISIBLE_DEVICES=']
    return 'cd ' + shlex.quote(str(EXPERIMENT)) + ' && exec ' + shlex.join(env + [PYTHON] + arguments)


def run(source_root, sed_run, run_dir, phase, pilot=False):
    root = Path(run_dir).resolve(); label = phase + ('_pilot32' if pilot else '_matrix')
    if pilot and phase != 'validation':
        raise ValueError('Only the preselected source validation subset is a throughput pilot')
    slots = ((2379, 24),) if pilot else SLOTS
    directory = root/'dispatch'/label; directory.mkdir(parents=True, exist_ok=True)
    if (directory/'launch.json').exists():
        raise RuntimeError('Existing dispatch preserved; inspect its task markers before explicit recovery')
    tasks = []; children = []
    for shard, (port, workers) in enumerate(slots):
        arguments = ['-m', 'diagnostics.simple_selector_controls_v1.evaluation', 'evaluate',
                     '--source-root', str(source_root), '--sed-run', str(sed_run),
                     '--run-dir', str(root), '--phase', phase, '--workers', str(workers),
                     '--shard', str(shard), '--shards', str(len(slots))]
        if pilot:
            arguments += ['--case-subset', str(root/'validation/pilot_cases.json')]
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                   '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6',
                   '-p', str(port), 'leiyo@ssh.cloud.sdu.dk', remote_command(arguments)]
        tasks.append(dict(shard=shard, shards=len(slots), port=port, workers=workers,
                          command=command, log=str(directory/f'shard{shard}.log')))
    launch = dict(state='launching', created_at=now(), phase=phase, pilot=pilot,
                  local_role='SSH coordination only', cpu_workers=sum(w for _, w in slots),
                  per_worker_threads=1, no_automatic_failed_cell_retry=True, tasks=tasks)
    save(directory/'launch.json', launch)
    for task in tasks:
        stream = open(task['log'], 'x')
        process = subprocess.Popen(task['command'], stdout=stream, stderr=subprocess.STDOUT)
        task.update(local_ssh_pid=process.pid, started_at=now())
        children.append((process, stream, task))
        save(directory/'launch.json', launch)
    while children:
        remaining = []
        for process, stream, task in children:
            code = process.poll()
            if code is None:
                remaining.append((process, stream, task))
            else:
                stream.close(); task.update(exit_code=code, finished_at=now())
                print(json.dumps({k:task[k] for k in ('shard','port','exit_code','finished_at')}), flush=True)
        children = remaining
        save(directory/'status.json', dict(state='running' if children else 'finished',
             updated_at=now(), active_shards=len(children), tasks=tasks))
        if children:
            time.sleep(15)
    passed = all(t['exit_code'] == 0 for t in tasks)
    result = dict(state='passed' if passed else 'failed_preserved_for_inspection',
                  finished_at=now(), tasks=tasks, cpu_workers=sum(w for _, w in slots))
    save(directory/'complete.json', result)
    if not passed:
        raise RuntimeError('A remote shard failed; logs retained and no automatic retry performed')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True); parser.add_argument('--sed-run', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--phase', required=True, choices=('validation','primary','transfer'))
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.source_root, args.sed_run, args.run_dir, args.phase, args.pilot)), flush=True)


if __name__ == '__main__':
    main()
