"""Prepare a fresh run with the source-qualified controller and profiling inputs.

This replication convenience wrapper was added after the original formal run.
It does not replace or change any of that run's sealed scientific code. The
original run's equivalent preparation commands and failures remain in its archive.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import shlex
import sys
import time
from .protocol import draft, EXPERIMENT
from .storage import write_json, PhysicsLedger, file_hash, digest


def prepare_on_node(root):
    from .physics import PayloadPlant, Dynamics, DEFAULT_CONTROLLER
    from .trajectories import action_library, original_task, workload, workload_spec
    from .worker import read_json, noise_stream
    root = Path(root)
    if (root / 'protocol_locked.yaml').exists():
        raise RuntimeError('Preparation must never modify a sealed run; use run.py to resume it.')
    if not (root / 'protocol_draft.yaml').exists():
        draft(root)
    (root / 'configs').mkdir(exist_ok=True)
    config_files = {'controller.json': DEFAULT_CONTROLLER, 'test_library.json': action_library(),
                    'workload_spec.json': workload_spec()}
    for name, data in config_files.items():
        path = root / 'configs' / name
        if path.exists() and read_json(path) != data:
            raise RuntimeError('Existing configuration differs; refusing to replace ' + str(path))
        write_json(path, data)
    profile_path = root / 'resources/profile20.json'
    if profile_path.exists():
        if len(read_json(profile_path)['rows']) != 20:
            raise RuntimeError('Incomplete existing profile must be inspected before reusing it')
        return
    case = read_json(root / 'private/source.json')[0]
    graph = read_json(root / 'configs/healthy_graph.json')
    actions = [original_task(case['task_seed']), *action_library(),
               *[workload(918050 + i) for i in range(9)]]
    plant = PayloadPlant(case['true_payload'], case['true_payload'], noise_stream(case['case_id'], 5),
                         graph=graph, ledger=PhysicsLedger(root), controller=DEFAULT_CONTROLLER)
    rows = []
    for index, action in enumerate(actions):
        started = time.perf_counter()
        block = plant.execute(action, f'replication_profile/{index}/{action["id"]}', kind='source_qualification')
        paths = block.save(root / 'source_validation/profile20_replication' / f'{index:02}_{action["id"]}')
        compressed = sum(Path(path).stat().st_size for path in paths.values())
        rows.append({'action': action['id'], 'physical_seconds': block.physical_seconds,
                     'wall_seconds_with_write': time.perf_counter() - started,
                     'bytes_compressed': compressed, 'private_metrics': block.private['metrics'], 'paths': paths})
        if block.private['metrics']['hard_abort']:
            write_json(root / 'checks/replication_profile_failure.json', {'rows': rows, 'passed': False})
            raise RuntimeError('Nominal replication profile hard-aborted; retained evidence requires investigation')
    write_json(profile_path, {'rows': rows, 'total_simulated_seconds': sum(r['physical_seconds'] for r in rows),
                              'total_wall_seconds_with_write': sum(r['wall_seconds_with_write'] for r in rows),
                              'total_bytes_compressed': sum(r['bytes_compressed'] for r in rows),
                              'source_case_selection': 'first outcome-independent source manifest entry',
                              'controller_sha256': digest(DEFAULT_CONTROLLER)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--run-all', action='store_true')
    parser.add_argument('--on-node', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(); root = args.run_root.resolve()
    if args.on_node:
        prepare_on_node(root)
        return
    root.mkdir(parents=True, exist_ok=True)
    command = ['env', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
               'CUDA_VISIBLE_DEVICES=', 'PYTHONPATH=.:src', 'CGX_DATA_ROOT=/work/Users/leiyo/cgx_data_v2',
               '/home/leiyo/work/cgx_env/bin/python', '-m', 'payload_change_v1.prepare',
               '--run-root', str(root), '--on-node']
    remote = 'cd ' + shlex.quote(str(EXPERIMENT)) + ' && ' + shlex.join(command)
    subprocess.run(['ssh', '-o', 'BatchMode=yes', '-p', '2379', 'leiyo@ssh.cloud.sdu.dk', remote], check=True)
    if args.run_all:
        from .run import STAGES, stage
        for name in STAGES:
            stage(root, name)


if __name__ == '__main__':
    main()
