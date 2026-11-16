"""Resumable evaluator jobs. Only public observations cross to estimation/design."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
from pathlib import Path
import time
import traceback
import numpy as np
from .storage import digest, file_hash, write_json, PhysicsLedger


def read_json(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=2)
def context(root, split):
    root = Path(root)
    cases = {r['case_id']: r for r in read_json(root / 'private' / f'{split}.json')}
    graph = read_json(root / 'configs/healthy_graph.json')
    controller_path = root / 'configs/controller.json'
    controller = read_json(controller_path) if controller_path.exists() else None
    return cases, graph, controller


def input_signature(root, split):
    root = Path(root)
    package = Path(__file__).parent
    files = [package / name for name in ('physics.py', 'trajectories.py', 'cases.py')]
    files += [root / 'private' / f'{split}.json', root / 'configs/healthy_graph.json']
    if (root / 'configs/controller.json').exists():
        files.append(root / 'configs/controller.json')
    return digest({str(p): file_hash(p) for p in files})


def noise_stream(case_id, noise_seed):
    # Evaluator RNG address only: IDs never enter estimator/design numeric features.
    return int(digest(['payload-measurements', case_id, int(noise_seed)])[:15], 16)


def source_job(job):
    from .physics import PayloadPlant
    from .trajectories import original_task, action_library
    from .estimator import calibration_record, recalibrate_source_case
    root = Path(job['run_root']); cid = job['case_id']; signature = job['input_signature']
    output = root / 'source_validation/source_records' / f'{cid}.json'
    calibration_hash = file_hash(Path(__file__).parent / 'estimator.py')
    if output.exists() and read_json(output).get('input_signature') == signature:
        cached = read_json(output)
        if cached.get('calibration_code_sha256') != calibration_hash:
            cached['calibration'] = recalibrate_source_case(root, cid)
            cached['calibration_code_sha256'] = calibration_hash
            write_json(output, cached)
        return {'case_id': cid, 'status': 'cached'}
    cases, graph, controller = context(str(root), 'source'); case = cases[cid]
    ledger = PhysicsLedger(root)
    start = time.perf_counter(); logs, blocks = [], []
    plant = PayloadPlant(case['true_payload'], case['configured_payload'], noise_stream(cid, 5),
                         graph=graph, ledger=ledger, controller=controller)
    actions = [original_task(case['task_seed']), *action_library()]
    for index, action in enumerate(actions):
        block = plant.execute(action, f'source/{cid}/{index}/{action["id"]}', kind='source_collection')
        path = root / 'raw/source' / cid / f'block_{index:02}_{action["id"]}'
        paths = block.save(path)
        logs.append({**block.public, 'execution_id': block.metadata['execution_id']})
        blocks.append({'test_id': action['id'], 'physical_seconds': block.physical_seconds,
                       'public_metrics': block.private['public_metrics'],
                       'private_metrics': block.private['metrics'], 'paths': paths,
                       'wall_seconds': block.metadata['wall_seconds'],
                       'bytes_uncompressed': block.metadata['bytes_uncompressed']})
        if block.private['metrics']['hard_abort']:
            # Source-only calibration may start a fresh declared episode after a failure.
            # All formal online methods instead stop their identification on hard abort.
            persistent_bias = plant._bias.copy()  # evaluator keeps the same physical sensor
            plant = PayloadPlant(case['true_payload'], case['configured_payload'],
                                 noise_stream(cid, 500 + index), graph=graph, ledger=ledger, controller=controller)
            plant._bias[:] = persistent_bias
            blocks[-1]['source_only_restart_after_abort'] = True
    calibration = calibration_record(logs, case['true_payload'])
    result = {'case_id': cid, 'family_id': case['family_id'], 'split': 'source',
              'input_signature': signature, 'calibration': calibration, 'blocks': blocks,
              'calibration_code_sha256': calibration_hash,
              'physical_seconds': sum(b['physical_seconds'] for b in blocks),
              'wall_seconds': time.perf_counter() - start,
              'completed': True, 'source_only_calibration_not_online_method': True}
    write_json(output, result)
    return {'case_id': cid, 'status': 'completed', 'seconds': result['physical_seconds'],
            'wall_seconds': result['wall_seconds']}


def dispatch_one(job):
    try:
        if job['kind'] == 'source':
            return source_job(job)
        if job['kind'] == 'source_coverage':
            from .estimator import source_coverage_case
            root = Path(job['run_root'])
            output = root / 'source_validation/source_coverage' / (job['case_id'] + '.json')
            if output.exists() and read_json(output).get('input_signature') == job['input_signature']:
                return {'case_id': job['case_id'], 'status': 'cached'}
            parameters = read_json(root / 'configs/estimator_parameters_before_coverage.json')
            rows = source_coverage_case(root, job['case_id'], parameters)
            write_json(output, {'case_id': job['case_id'], 'input_signature': job['input_signature'], 'prefixes': rows})
            return {'case_id': job['case_id'], 'status': 'completed'}
        if job['kind'] == 'commission':
            from .commissioning import commissioning_job
            return commissioning_job(job)
        raise ValueError('unknown job kind')
    except Exception:
        error = {'job': job, 'error': traceback.format_exc(), 'failed_at': time.time()}
        root = Path(job['run_root'])
        write_json(root / 'failures' / (digest(job) + '.json'), error)
        return {'status': 'failed', 'case_id': job.get('case_id'), 'error': error['error']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    jobs = read_json(args.jobs)
    start = time.time(); results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(dispatch_one, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            print(json.dumps({'finished': len(results), 'total': len(jobs), **result}), flush=True)
    failures = [r for r in results if r['status'] == 'failed']
    write_json(args.result, {'started_at': start, 'finished_at': time.time(), 'workers': args.workers,
                             'jobs': len(jobs), 'failed': len(failures), 'results': results,
                             'exit_code': 1 if failures else 0})
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
