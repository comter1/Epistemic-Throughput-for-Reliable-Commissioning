"""Unified resumable entry point; large computations run on three authorized nodes."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from .storage import digest, file_hash, write_json, PhysicsLedger
from .protocol import EXPERIMENT, code_hashes, draft, seal, verify_seal
from .cases import MAIN_METHODS, REFERENCES

PORTS = [2379, 2982, 2404]
WORKERS = [40, 80, 160]
PYTHON = '/home/leiyo/work/cgx_env/bin/python'
STAGES = ['audit', 'qualify-source', 'fit', 'validate', 'seal', 'evaluate', 'report', 'package']


def stage_signature(root, name):
    """Resume only when the stage's concrete inputs still match."""
    package = Path(__file__).parent
    scientific = ['physics.py', 'trajectories.py', 'cases.py', 'estimator.py', 'design.py',
                  'commissioning.py', 'worker.py', 'protocol.py', 'storage.py']
    selected = {'audit': ['protocol.py', 'cases.py', 'integrity.py'],
                'qualify-source': ['physics.py', 'trajectories.py', 'estimator.py',
                                   'estimator_checks.py', 'physical_checks.py', 'integrity.py'],
                'fit': ['physics.py', 'trajectories.py', 'cases.py', 'estimator.py', 'design.py', 'worker.py'],
                'report': ['reporting.py', 'integrity.py'], 'package': ['packaging.py']}.get(name, scientific)
    files = [package / item for item in selected]
    if name in ('fit', 'validate', 'seal', 'evaluate'):
        files += [root / 'configs/controller.json', root / 'configs/test_library.json', root / 'configs/healthy_graph.json']
    if name in ('validate', 'seal', 'evaluate'):
        files += [root / 'configs/estimator_parameters.json', root / 'configs/design_bank.json', root / 'configs/fixed_schedules.json']
    if name in ('evaluate', 'report', 'package'):
        files += [root / 'protocol_locked.yaml']
    return digest({str(p): file_hash(p) if p.exists() else None for p in files})


def read(path):
    return json.loads(Path(path).read_text())


def remote_jobs(root, name, jobs, workers=None):
    """Weighted dispatch; subprocess completion markers retain each actual exit code."""
    root = Path(root); workers = workers or WORKERS
    shards = [[] for _ in PORTS]
    for i, job in enumerate(jobs):
        # Pick smallest assigned job/CPU ratio for balanced short independent cases.
        choice = min(range(3), key=lambda k: len(shards[k]) / workers[k])
        shards[choice].append(job)
    records = []
    generation = str(time.time_ns())
    for port, nworkers, shard in zip(PORTS, workers, shards):
        if not shard:
            continue
        prefix = root / 'jobs' / f'{name}_{generation}_{port}'
        job_file = prefix.with_suffix('.jobs.json'); result = prefix.with_suffix('.result.json')
        log = prefix.with_suffix('.log'); exit_file = prefix.with_suffix('.exit')
        write_json(job_file, shard)
        command = ['env', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
                   'CUDA_VISIBLE_DEVICES=', 'PYTHONPATH=.:src', 'CGX_DATA_ROOT=/work/Users/leiyo/cgx_data_v2',
                   PYTHON, '-m', 'payload_change_v1.worker', '--jobs', str(job_file),
                   '--workers', str(min(nworkers, len(shard))), '--result', str(result)]
        script = ('cd ' + shlex.quote(str(EXPERIMENT)) + '\n' + shlex.join(command) + '\n'
                  + 'status=$?\n' + 'printf "%s\\n" "$status" > ' + shlex.quote(str(exit_file)) + '\n'
                  + 'exit "$status"\n')
        script_file = prefix.with_suffix('.sh'); script_file.write_text(script)
        launch = 'nohup bash ' + shlex.quote(str(script_file)) + ' > ' + shlex.quote(str(log)) + ' 2>&1 < /dev/null &'
        subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-p', str(port),
                        'leiyo@ssh.cloud.sdu.dk', launch], check=True, timeout=30)
        records.append({'port': port, 'workers': min(nworkers, len(shard)), 'jobs': len(shard),
                        'result': str(result), 'log': str(log), 'exit_file': str(exit_file),
                        'command_script': str(script_file), 'script_sha256': file_hash(script_file)})
    write_json(root / 'commands' / f'{name}_{generation}.json', records)
    start = time.time()
    while not all(Path(record['exit_file']).exists() for record in records):
        completed = sum(Path(record['exit_file']).exists() for record in records)
        print(json.dumps({'stage': name, 'finished_nodes': completed, 'total_nodes': len(records),
                          'wall_seconds': round(time.time()-start, 1)}), flush=True)
        time.sleep(10)
    failures = [r for r in records if Path(r['exit_file']).read_text().strip() != '0']
    if failures:
        raise RuntimeError('Remote jobs failed; retained logs: ' + ', '.join(r['log'] for r in failures))
    return [read(r['result']) for r in records]


def do_fit(root):
    from .worker import input_signature
    from .estimator import fit_parameters, fit_uncertainty_inflation
    signature = input_signature(root, 'source')
    cases = read(root / 'private/source.json')
    jobs = [{'kind': 'source', 'run_root': str(root), 'case_id': c['case_id'],
             'input_signature': signature} for c in cases]
    remote_jobs(root, 'source', jobs)
    rows = [read(root / 'source_validation/source_records' / f'{c["case_id"]}.json') for c in cases]
    if any(row['input_signature'] != signature or not row['completed'] for row in rows):
        raise RuntimeError('source collection incomplete or stale')
    parameters = fit_parameters([row['calibration'] for row in rows])
    write_json(root / 'configs/estimator_parameters_before_coverage.json', parameters)
    coverage_signature = digest({'parameters': parameters, 'estimator_code': file_hash(Path(__file__).parent / 'estimator.py')})
    coverage_jobs = [{'kind': 'source_coverage', 'run_root': str(root), 'case_id': c['case_id'],
                      'input_signature': coverage_signature} for c in cases]
    remote_jobs(root, 'source_coverage', coverage_jobs)
    coverage = [read(root / 'source_validation/source_coverage' / f'{c["case_id"]}.json')['prefixes'] for c in cases]
    parameters, coverage_summary = fit_uncertainty_inflation(coverage, parameters)
    write_json(root / 'source_validation/source_uncertainty_calibration.json', coverage_summary)
    write_json(root / 'configs/estimator_parameters.json', parameters)
    from .design import build_design_bank, fixed_schedules
    from .trajectories import action_library, workload
    public_work_seeds = [918041, 918042, 918043, 918044]
    bank = build_design_bank(action_library(), [workload(seed) for seed in public_work_seeds], parameters)
    write_json(root / 'configs/design_bank.json', bank)
    write_json(root / 'configs/public_work_grid.json', {'seeds': public_work_seeds,
               'reference_sampling_hz': 4, 'not_private_audit_instances': True})
    write_json(root / 'configs/fixed_schedules.json', fixed_schedules(bank, parameters))
    write_json(root / 'source_validation/source_summary.json', {
        'cases': len(rows), 'physical_seconds': sum(r['physical_seconds'] for r in rows),
        'sum_worker_wall_seconds': sum(r['wall_seconds'] for r in rows),
        'source_calibration_labels_only': True, 'input_signature': signature})


def do_validate(root):
    signature = digest({'code': code_hashes(), 'estimator': file_hash(root / 'configs/estimator_parameters.json'),
                        'schedules': file_hash(root / 'configs/fixed_schedules.json')})
    cases = read(root / 'private/validation.json')
    jobs = [{'kind': 'commission', 'run_root': str(root), 'split': 'validation',
             'case_id': c['case_id'], 'noise_seed': 7, 'method': method,
             'thresholds': [.02, .05, .10], 'input_signature': signature}
            for method in MAIN_METHODS for c in cases]
    remote_jobs(root, 'validation', jobs)
    candidates = []
    for threshold in (.02, .05, .10):
        rows = [read(p) for p in sorted((root / 'runs/validation' / f'threshold_{threshold:.2f}').glob('*/*.json'))]
        if len(rows) != 256:
            raise RuntimeError(f'Expected 256 validation rows for threshold {threshold}; got {len(rows)}')
        released = [r for r in rows if r['released_status'].startswith('pass_')]
        false = sum(not r['all_audits_pass'] for r in released)
        correct = len(released) - false
        candidates.append({'threshold': threshold, 'runs': len(rows), 'released': len(released),
                           'false_releases': false, 'correct_releases': correct,
                           'false_release_rate': false / len(released) if released else None,
                           'mean_online_physical_seconds': sum(r['physical_test_seconds'] + r['verification_seconds'] for r in rows) / len(rows)})
    # Empty release cannot beat a valid releasing configuration merely by defining 0/0 as zero.
    eligible = [r for r in candidates if r['correct_releases'] > 0] or candidates
    best = min(eligible, key=lambda r: (r['false_release_rate'] if r['false_release_rate'] is not None else 1.,
                                        -r['correct_releases'], r['mean_online_physical_seconds'], r['threshold']))
    write_json(root / 'source_validation/threshold_selection.json', {
        'selected_threshold': best['threshold'], 'candidates': candidates,
        'selection_order': 'false releases / released, successful release coverage, physical time, smaller threshold',
        'test_outcomes_used': False, 'shared_acquisition_histories': True,
        'independent_audits_per_configuration': 4, 'validation_seed': 7})


def do_evaluate(root):
    lock = verify_seal(root)
    cases = read(root / 'private/test.json')
    priority = ['original_task_once', 'fixed_four_pose', 'true_payload_settings_reference']
    remaining = ['fixed_mixed_suite', 'adaptive_task_checks', 'keep_existing_settings']
    for phase, methods in [('test_priority', priority), ('test_remaining', remaining)]:
        jobs = [{'kind': 'commission', 'run_root': str(root), 'split': 'test', 'case_id': c['case_id'],
                 'noise_seed': seed, 'method': method, 'thresholds': [lock['selected_stop_threshold']],
                 'input_signature': digest(lock)}
                for method in methods for c in cases for seed in (11, 23, 37)]
        remote_jobs(root, phase, jobs)
    verify_seal(root)


def stage(root, name, force=False):
    marker = root / 'stages' / (name + '.json')
    signature = stage_signature(root, name)
    if (marker.exists() and read(marker).get('completed')
            and read(marker).get('input_signature') == signature and not force):
        print(f'{name}: already completed', flush=True)
        return
    started = datetime.now(timezone.utc).isoformat()
    record = {'stage': name, 'started_utc': started, 'command': shlex.join(sys.argv),
              'input_code_hashes': code_hashes(), 'input_signature': signature, 'completed': False}
    write_json(marker, record)
    try:
        if name == 'audit':
            if not (root / 'protocol_draft.yaml').exists():
                draft(root)
            from .integrity import protected_snapshot
            if not (root / 'checks/protected_before.json').exists():
                protected_snapshot(root)
            for document in ('IMPLEMENTATION_AUDIT.md', 'CONTROLLER_DECISION.md'):
                if not (root / document).exists():
                    source = Path(__file__).parent / 'documents' / document
                    (root / document).write_text(source.read_text())
        elif name == 'qualify-source':
            from .physical_checks import run_checks
            from .estimator_checks import run_checks as estimator_checks
            run_checks(root)
            estimator_checks(root)
            from .integrity import legacy_regression
            if not (root / 'checks/legacy_regression.json').exists():
                legacy_regression(root)
            for filename in ('physical_checks.json', 'estimator_checks.json', 'legacy_regression.json'):
                if not read(root / 'checks' / filename)['passed']:
                    raise RuntimeError('Engineering check failed: ' + filename)
        elif name == 'fit':
            do_fit(root)
        elif name == 'validate':
            do_validate(root)
        elif name == 'seal':
            seal(root)
        elif name == 'evaluate':
            do_evaluate(root)
        elif name == 'report':
            from .reporting import report
            report(root)
            from .integrity import check_protected
            if not check_protected(root)['passed']:
                raise RuntimeError('Protected files changed')
        elif name == 'package':
            from .packaging import package
            package(root)
        else:
            raise ValueError(name)
    except BaseException as error:
        record.update(exit_code=1, error=str(error), finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(marker, record)
        raise
    record.update(completed=True, exit_code=0, finished_utc=datetime.now(timezone.utc).isoformat(),
                  physics_ledger=PhysicsLedger(root).summary())
    write_json(marker, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=STAGES + ['all'])
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--force-stage', action='store_true')
    args = parser.parse_args()
    root = args.run_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    for name in STAGES if args.stage == 'all' else [args.stage]:
        stage(root, name, args.force_stage)


if __name__ == '__main__':
    main()
