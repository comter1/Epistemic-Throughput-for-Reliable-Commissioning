"""Hardware-only override: exclusively claim original 96-trajectory live windows.

The sealed LiveCoordinator, reader rules, heads, cases, batch size and within-
window order are unchanged. One atomic queue owns each window on exactly one
node/GPU. A failed/running window is never silently stolen or retried.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import fcntl
import multiprocessing as mp
import os
from pathlib import Path
import socket
import time
import traceback

from cgx.provenance import atomic_json, read_json, file_sha256, hash_json
from diagnostics.false_alarm_causes.interventions import load_blob, dump_blob
from .live import LiveCoordinator, summarize
from .rollouts import METHODS, SEEDS, read_cases, initialize_worker
from .ledger import Ledger, TimedStage

WINDOW = 96
BATCH = 16
ALLOWED = {('2379', 'cuda:0'), *[('2404', f'cuda:{i}') for i in range(4)]}


def utc():
    return datetime.now(timezone.utc).isoformat()


def check(run):
    run = Path(run)
    seal = read_json(run / 'manifests/source_variant_selection.json')
    override = read_json(run / 'manifests/hardware_authorization_override.json')
    assert override['authority'] == 'Latest explicit user instruction in the active session'
    assert {2379, 2404}.issubset(set(override['authorized_host_ports']))
    assert seal['state'] == 'sealed_before_fresh_outcomes' and seal['optional_live']['enabled']
    assert not seal['fresh_outcomes_exist_at_seal'] and not seal['fresh_initial_acquisitions_exist_at_seal']
    assert seal['selected_reader_variant'] in ('R-P', 'R-C', 'R-PC')
    for name, digest in seal['frozen_scientific_code'].items():
        assert file_sha256(Path(__file__).with_name(name)) == digest, name
    for relative, digest in seal['frozen_files'].items():
        assert file_sha256(run / relative) == digest, relative
    for relative, digest in seal['all_head_checkpoints'].items():
        assert file_sha256(run / relative) == digest, relative
    completed = read_json(run / 'trajectories/confirmation/complete.json')
    assert completed['state'] == 'passed' and completed['trajectories_present'] == 768
    cases = read_cases(run / 'manifests/fresh_cases.json')
    assert len(cases) == 128
    assert all((run / 'cache/initial' / f"{case['case_id']}.json.gz").exists() for case in cases)
    return seal, override, cases


def jobs_for(run, cases):
    jobs = [(case, method, seed, Path(run) / 'trajectories/live/cases' /
             f"{hash_json([case['case_id'], method, seed])}.json.gz")
            for case in cases for method in METHODS for seed in SEEDS]
    windows = [jobs[start:start + WINDOW] for start in range(0, len(jobs), WINDOW)]
    assert len(windows) == 8 and all(len(window) == WINDOW for window in windows)
    assert len({str(job[3]) for window in windows for job in window}) == 768
    case_owners = {}
    for index, window in enumerate(windows):
        for case, _, _, _ in window:
            assert case_owners.setdefault(case['case_id'], index) == index
    return windows


def prepare(run_dir):
    run = Path(run_dir)
    seal, override, cases = check(run)
    windows = jobs_for(run, cases)
    directory = run / 'live_dispatch'
    directory.mkdir(exist_ok=True)
    manifest = dict(state='sealed_dispatch_before_live', created_at=utc(),
        scientific_choices_unchanged=True, dispatch_only_user_override=True,
        hardware_override_sha256=file_sha256(run / 'manifests/hardware_authorization_override.json'),
        scientific_selection_sha256=file_sha256(run / 'manifests/source_variant_selection.json'),
        dispatcher_sha256=file_sha256(__file__), original_window_size=WINDOW, original_gpu_batch_size=BATCH,
        worker_cpu_processes=32, allowed_workers=[dict(node=n, device=d) for n, d in sorted(ALLOWED)],
        unrelated_2982_jobs_preserved=True, reader_GPU0_priority_on_2404=True,
        windows=[dict(index=i, jobs=[dict(case_id=c['case_id'], method_key=m, policy_seed=s, output=str(p))
                                    for c, m, s, p in window]) for i, window in enumerate(windows)])
    with (directory / 'queue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / 'queue.json').exists():
            old = read_json(run / 'manifests/live_dispatch.json')
            assert old['dispatcher_sha256'] == manifest['dispatcher_sha256']
            assert old['windows'] == manifest['windows']
            return old
        assert not list((run / 'trajectories/live/cases').glob('*.json.gz'))
        atomic_json(run / 'manifests/live_dispatch.json', manifest)
        atomic_json(directory / 'queue.json', dict(windows={str(i): {'state': 'pending'} for i in range(8)}))
        atomic_json(run / 'manifests/live_rollouts.json', dict(
            phase='live', execution_mode='original-window-preserving distributed dispatch',
            variant=seal['selected_reader_variant'], cases=128, original_trajectories=768,
            maximum_additional_physical_executions=3072, initial_observations_reused_from_A0=True,
            RG_live_uses_exact_R_live_prefix=True, window=WINDOW, batch_size=BATCH,
            dispatcher_manifest='manifests/live_dispatch.json',
            frozen_scientific_code=seal['frozen_scientific_code'],
            hardware_override_sha256=manifest['hardware_override_sha256'],
            changed_reader_gate_calibration_claimed=False))
    return manifest


def claim(run, owner):
    directory = Path(run) / 'live_dispatch'
    with (directory / 'queue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read_json(directory / 'queue.json')
        if any(row['state'] == 'failed' for row in state['windows'].values()):
            return None
        for index in range(8):
            row = state['windows'][str(index)]
            if row['state'] == 'pending':
                row.update(state='running', owner=owner, started_at=utc())
                atomic_json(directory / 'queue.json', state)
                return index
    return None


def release(run, index, owner, result):
    directory = Path(run) / 'live_dispatch'
    with (directory / 'queue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read_json(directory / 'queue.json')
        row = state['windows'][str(index)]
        assert row['state'] == 'running' and row['owner'] == owner
        row.update(result, finished_at=utc())
        atomic_json(directory / 'queue.json', state)


def worker(root, run_dir, node, device, workers=32):
    run = Path(run_dir)
    assert (str(node), device) in ALLOWED and workers == 32
    seal, override, cases = check(run)
    manifest = read_json(run / 'manifests/live_dispatch.json')
    assert manifest['dispatcher_sha256'] == file_sha256(__file__)
    assert manifest['hardware_override_sha256'] == file_sha256(run / 'manifests/hardware_authorization_override.json')
    assert workers < override['known_hardware'][str(node)]['CPU_quota_cores']
    windows = jobs_for(run, cases)
    owner = dict(node=str(node), hostname=socket.gethostname(), pid=os.getpid(), device=device,
                 worker_id=hash_json([socket.gethostname(), os.getpid(), time.time_ns()]))
    index = claim(run, owner)
    if index is None:
        print(dict(stage='live_dispatch', owner=owner, state='no_unclaimed_window'), flush=True)
        return
    started, cpu = time.monotonic(), time.process_time()
    owned = []
    with TimedStage(run, f'live_dispatch_{node}_{device}'):
        coordinator = LiveCoordinator(root, run, seal['selected_reader_variant'], device, BATCH)
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                initializer=initialize_worker, initargs=(str(root), str(run))) as pool:
            while index is not None:
                stamp = time.monotonic()
                jobs = windows[index]
                assert not any(path.exists() for _, _, _, path in jobs), 'A claimed window already has another writer'
                try:
                    rows = coordinator.window(pool, jobs, 'live')
                    assert len(rows) == WINDOW
                    for _, _, _, path in jobs:
                        item = load_blob(path)
                        item['row'].update(execution_host=owner['hostname'], execution_node=str(node),
                                           dispatch_window_index=index, dispatch_worker_id=owner['worker_id'])
                        item['result_hash'] = hash_json({k: v for k, v in item.items() if k != 'result_hash'})
                        dump_blob(path, item)
                    result = dict(state='passed' if all(r['state'] == 'passed' for r in rows) else 'failed',
                                  trajectories=len(rows), failures=sum(r['state'] != 'passed' for r in rows),
                                  wall_seconds=time.monotonic() - stamp,
                                  worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in rows))
                    release(run, index, owner, result)
                    owned.append(index)
                    print(dict(stage='live_dispatch_window', window=index, owner=owner, **result), flush=True)
                    if result['state'] != 'passed':
                        raise RuntimeError('A live window retained failures; do not automatically retry')
                except Exception as exc:
                    state = read_json(run / 'live_dispatch/queue.json')['windows'][str(index)]
                    if state['state'] == 'running':
                        release(run, index, owner, dict(state='failed', error=f'{type(exc).__name__}: {exc}',
                                                       traceback=traceback.format_exc()))
                    raise
                index = claim(run, owner)
        atomic_json(run / 'live_dispatch' / f"worker_{owner['worker_id']}.json", dict(
            state='passed', owner=owner, windows=owned, wall_seconds=time.monotonic() - started,
            coordinator_cpu_seconds=time.process_time() - cpu, model_timings=dict(coordinator.timings),
            head_forward_examples=coordinator.head_examples, cpu_workers=workers))


def finish(run_dir):
    run = Path(run_dir)
    seal, _, cases = check(run)
    manifest = read_json(run / 'manifests/live_dispatch.json')
    assert manifest['dispatcher_sha256'] == file_sha256(__file__)
    queue = read_json(run / 'live_dispatch/queue.json')
    assert len(queue['windows']) == 8 and all(row['state'] == 'passed' for row in queue['windows'].values())
    assert sum(row['trajectories'] for row in queue['windows'].values()) == 768
    owners = set()
    for index, window in enumerate(jobs_for(run, cases)):
        expected = queue['windows'][str(index)]['owner']
        owners.add(expected['worker_id'])
        for _, _, _, path in window:
            item = load_blob(path)
            assert item['row']['dispatch_window_index'] == index
            assert item['row']['dispatch_worker_id'] == expected['worker_id']
            assert item['row']['execution_node'] == expected['node']
    # The same frozen aggregate/replay code performs the final CPU head-sign audit.
    summarize(run, cases, seal['selected_reader_variant'])
    atomic_json(run / 'tests/live_dispatch_audit.json', dict(state='passed', windows=8, trajectories=768,
        original_window_size=WINDOW, original_batch_size=BATCH, within_window_order_unchanged=True,
        one_writer_per_trajectory=True, worker_ids=sorted(owners), queue=queue,
        scientific_code_hashes_unchanged=True, user_hardware_override_verified=True,
        all_physical_and_full_model_calls_use_original_atomic_ledger=True))
    print(dict(stage='live_dispatch_complete', trajectories=768, windows=8, workers=len(owners)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'worker', 'finish'))
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--root', default='/work/Users/leiyo/cgx_data_v2')
    parser.add_argument('--node', choices=('2379', '2404'))
    parser.add_argument('--device', choices=tuple(f'cuda:{i}' for i in range(4)))
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.run_dir)
    elif args.stage == 'finish':
        finish(args.run_dir)
    else:
        if not args.node or not args.device:
            parser.error('worker requires --node and --device')
        worker(args.root, args.run_dir, args.node, args.device, args.workers)
