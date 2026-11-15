"""One independent process per GPU UUID with checked training inputs and outputs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time
import threading
from cgx.config import EXPERIMENT, allocated_cpu_count
from cgx.provenance import atomic_json, file_sha256, hash_json, now, read_json, require_protocol_lock, verify_hashes


def _selected_outputs(marker_path, completion):
    """A success JSON alone never proves a trained checkpoint exists or is intact."""
    marker_path = Path(marker_path)
    selected = completion.get('selected_checkpoint')
    if not isinstance(selected, str) or not selected:
        raise ValueError('Trainer completion lacks selected checkpoint')
    checkpoint = (marker_path.parent / selected).resolve()
    if not checkpoint.is_relative_to(marker_path.parent.resolve()):
        raise ValueError('Selected checkpoint escapes its training directory')
    expected = completion.get('selected_checkpoint_hash')
    if not checkpoint.is_file() or file_sha256(checkpoint) != expected:
        raise ValueError(f'Selected checkpoint missing or changed: {checkpoint}')
    validation = marker_path.parent / 'validation.json'
    if not validation.is_file():
        raise ValueError(f'Trainer validation evidence missing: {validation}')
    return {str(p): file_sha256(p) for p in (marker_path, checkpoint, validation)}


def _phase_inputs(root, lock, phase):
    """Training's own identity is checked against the prepared common source corpus."""
    root = Path(root)
    feature_path = root / 'training/features_complete.json'
    features = read_json(feature_path)
    if features.get('state') != 'passed' or features['identity']['protocol_lock_hash'] != hash_json(lock):
        raise ValueError('Prepared replay does not match the protocol seal')
    manifest = root / 'training/feature_hashes.jsonl'
    if file_sha256(manifest) != features['manifest_sha256']:
        raise ValueError('Prepared feature manifest changed')
    replays = features['identity']['replays']
    expected = {'protocol_lock_hash': hash_json(lock), 'protocol_hash': lock['config_hash'],
                'replay_hash': replays['train']}
    prerequisites = {str(feature_path): file_sha256(feature_path), str(manifest): file_sha256(manifest)}
    if phase == 'reader_training':
        expected['validation_replay_hash'] = replays['validation']
    elif phase == 'policy_training':
        score_path = root / 'training/scores_complete.json'
        scores = read_json(score_path)
        if (scores.get('state') != 'passed' or scores['identity']['protocol_lock_hash'] != hash_json(lock)
                or scores['identity']['replay_hash'] != replays['train']):
            raise ValueError('Cross-fitted scores do not match prepared source replay')
        score_db = root / 'training/scores.sqlite'
        if file_sha256(score_db) != scores['output_hash']:
            raise ValueError('Cross-fitted score database changed')
        reader_path = root / 'models/readers/final/complete.json'
        reader = read_json(reader_path)
        protocol = read_json(root/'software/protocol.json')
        reader_identity = {**expected, 'reader': 'final', 'excluded_fold': None,
                           'seed': protocol['seeds']['training'][0], 'validation_replay_hash': replays['validation']}
        if (reader.get('state') != 'passed' or reader.get('technical_smoke')
                or any(reader.get('identity', {}).get(k) != v for k, v in reader_identity.items())):
            raise ValueError('Final reader identity differs from the policy source corpus')
        from cgx.training.stopping import validate_completion
        validate_completion(reader,read_json(reader_path.parent/'validation.json'),protocol['training'],protocol['training']['predictor_update_cap'])
        prerequisites.update(_selected_outputs(reader_path, reader))
        prerequisites.update({str(score_path): file_sha256(score_path), str(score_db): scores['output_hash']})
        expected.update(reward_hash=scores['output_hash'], reader_hash=reader['selected_checkpoint_hash'])
    else:
        raise ValueError(f'Unregistered queued training phase: {phase}')
    return expected, prerequisites


def build_jobs(root, config, phase='policies', out=None):
    root = Path(root)
    lock = require_protocol_lock(root, config)
    if phase not in ('policies', 'readers'):
        raise ValueError('Unknown registered training phase')
    training_phase = 'policy_training' if phase == 'policies' else 'reader_training'
    expected, prerequisites = _phase_inputs(root, lock, training_phase)
    rows = []
    if phase == 'policies':
        for method in config['methods']['learned']:
            for seed in config['seeds']['training']:
                rows.append({'job_id': f'{method}_seed{seed}', 'phase': training_phase, 'method': method, 'seed': seed,
                             'state': 'not_started', 'input_hashes': {'protocol_lock': hash_json(lock)},
                             'expected_identity': {**expected, 'method': method, 'seed': seed},
                             'prerequisite_files': prerequisites,
                             'maximum_updates': config['training']['policy_update_cap'],
                             'completion_marker': str(root/'models/policies'/f'{method}_seed{seed}'/'complete.json'),
                             'command': [sys.executable, '-m', 'cgx.cli', 'fit-policy', '--data-root', str(root),
                                         '--method', method, '--seed', str(seed), '--resume']})
    else:
        for name in [*[f'crossfit_{i}' for i in range(config['training']['crossfit_folds'])], 'final', 'independent']:
            rows.append({'job_id': f'reader_{name}', 'phase': training_phase, 'reader': name,
                         'state': 'not_started', 'input_hashes': {'protocol_lock': hash_json(lock)},
                         'expected_identity': {**expected, 'reader': name,
                                               'seed': config['seeds']['training'][0]+(1009 if name=='independent' else 0),
                                               'excluded_fold': int(name.split('_')[1]) if name.startswith('crossfit_') else None},
                         'prerequisite_files': prerequisites,
                         'maximum_updates': config['training']['predictor_update_cap'],
                         'completion_marker': str(root/'models/readers'/name/'complete.json'),
                         'command': [sys.executable, '-m', 'cgx.cli', 'fit-readers', '--data-root', str(root), '--only', name, '--resume']})
    target = Path(out or root/f'jobs/{phase}.jsonl')
    for row in rows:
        row['command'] += ['--config',str(root/'software/protocol.json')]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return {'jobs': len(rows), 'path': str(target), 'hash': file_sha256(target), 'launched': 0}


def gpu_uuids(indices):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True, timeout=10)
    mapping = dict(line.replace(' ', '').split(',') for line in output.strip().splitlines())
    result = []
    for index in indices:
        value = str(index)
        if value in mapping:
            result.append(mapping[value])
        elif value in mapping.values():
            result.append(value)
        else:
            raise ValueError(f'Unknown physical GPU: {value}')
    if not result or len(set(result)) != len(result):
        raise ValueError('GPU allocation must contain distinct physical devices')
    return result


def _checked_completion(job, lock):
    marker = Path(job['completion_marker'])
    completion = read_json(marker)
    if completion.get('state') != 'passed' or completion.get('technical_smoke'):
        raise ValueError('Trainer lacks a formal completion')
    from cgx.training.stopping import validate_completion
    protocol=read_json(marker.parents[3]/'software/protocol.json')
    validate_completion(completion,read_json(marker.parent/'validation.json'),protocol['training'],job['maximum_updates'])
    expected = {'protocol_lock_hash': hash_json(lock), 'protocol_hash': lock['config_hash'], **job['expected_identity']}
    if any(completion.get('identity', {}).get(k) != v for k, v in expected.items()):
        raise ValueError('Trainer completion identity differs from queued inputs')
    verify_hashes(job['prerequisite_files'])
    return completion, _selected_outputs(marker, completion)


def run_queue(root, jobs_path, gpus=(0, 1), resume=False):
    root = Path(root)
    lock = require_protocol_lock(root)
    protocol = read_json(root/'software/protocol.json')
    soft_cap = protocol['compute']['gpu_hours_initial_soft_cap']
    jobs = [json.loads(line) for line in Path(jobs_path).read_text().splitlines() if line.strip()]
    if len({x['job_id'] for x in jobs}) != len(jobs):
        raise ValueError('Duplicate job IDs')
    # Expensive shared source verification occurs once per queue invocation.
    phase_inputs = {}
    for job in jobs:
        if job['input_hashes']['protocol_lock'] != hash_json(lock):
            raise ValueError('Job manifest protocol does not match')
        cap_key = 'predictor_update_cap' if job['phase'] == 'reader_training' else 'policy_update_cap'
        if job['maximum_updates'] != protocol['training'][cap_key]:
            raise ValueError('Job update cap differs from the sealed protocol')
        if Path(job['job_id']).name != job['job_id'] or job['job_id'] in ('.', '..'):
            raise ValueError('Invalid job ID')
        if job['phase'] not in phase_inputs:
            phase_inputs[job['phase']] = _phase_inputs(root, lock, job['phase'])
        expected, prerequisites = phase_inputs[job['phase']]
        if any(job.get('expected_identity', {}).get(k) != v for k, v in expected.items()):
            raise ValueError('Queued training identity no longer matches prepared data')
        if job.get('prerequisite_files') != prerequisites:
            raise ValueError('Queued prerequisite hashes changed')
    pending_jobs = []
    for job in jobs:
        status_file = root/'jobs'/job['job_id']/'status.json'
        if status_file.exists():
            previous = read_json(status_file)
            if previous.get('signature') != hash_json(job):
                raise ValueError(f"Changed job cannot resume: {job['job_id']}")
            if resume and previous.get('state') == 'passed':
                _, outputs = _checked_completion(job, lock)
                if previous.get('output_hashes') != outputs:
                    raise ValueError(f"Completed job outputs changed: {job['job_id']}")
                continue
            if not resume:
                raise FileExistsError(status_file)
        pending_jobs.append(job)
    # Artifact-only resume needs no CUDA discovery or occupancy query.
    if not pending_jobs:
        return {'state': 'passed', 'jobs': len(jobs), 'launched': 0, 'devices': []}
    uuids = gpu_uuids(gpus)
    from .remote import registered_workers,launch_remote,remote_progress
    remotes={worker['gpu_uuid']:worker for worker in registered_workers()}
    if set(uuids)&set(remotes):raise ValueError('GPU registered both locally and remotely')
    uuids += list(remotes)
    required=protocol['compute'].get('gpus')
    if required is not None and len(uuids)!=required:
        raise ValueError(f'Registered {required} GPU slots required, discovered {len(uuids)}; check the remote worker registry')
    pending = iter(pending_jobs)
    guard = threading.Lock()
    failures = []
    active = {}

    def usage_report():
        with guard:
            completed = 0.; unclosed = []
            for path in (root/'jobs').glob('*/status.json'):
                if path.parent.name in active:
                    continue
                record = read_json(path)
                for index, attempt in enumerate(record.get('attempts', [])):
                    if attempt.get('state') == 'running' and 'elapsed_seconds' not in attempt:
                        unclosed.append(f'{path.parent.name}:attempt{index}')
                if record.get('state') == 'running':
                    unclosed.append(path.parent.name)
                else:
                    completed += record.get('cumulative_elapsed_seconds', record.get('elapsed_seconds', 0.))
            running = sum(time.monotonic()-v['began']+v['previous_elapsed'] for v in active.values())
            seconds = completed+running
            atomic_json(root/'reports/gpu_allocation_usage.json', {
                'updated_at': now(), 'gpu_hours_completed_jobs': completed/3600,
                'gpu_hours_active_jobs': running/3600, 'gpu_hours_total': seconds/3600,
                'initial_soft_cap_gpu_hours': soft_cap, 'soft_cap_exceeded': seconds>soft_cap*3600,
                'unclosed_job_attempts': unclosed, 'usage_incomplete': bool(unclosed),
                'action': 'report_only_no_seed_or_method_pruning',
                'boundary': 'allocated job wall time including validation; not GPU kernel active time'})

    def worker(uuid):
        lock_path = root/'jobs'/f'{uuid}.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a') as device_lock:
            fcntl.flock(device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while True:
                with guard:
                    job = next(pending, None)
                if job is None:
                    return
                destination = root/'jobs'/job['job_id']
                destination.mkdir(parents=True, exist_ok=True)
                status_file = destination/'status.json'
                previous = read_json(status_file) if status_file.exists() else {}
                attempts = list(previous.get('attempts', []))
                if previous:
                    attempts.append({k: v for k, v in previous.items() if k != 'attempts'})
                previous_elapsed = previous.get('cumulative_elapsed_seconds', previous.get('elapsed_seconds', 0.))
                status = dict(job, signature=hash_json(job), started_at=now(), state='running', device_uuid=uuid,
                              attempts=attempts, allocation={'gpus': 1, 'blas_threads': 1, 'allocated_cpu_count': allocated_cpu_count()},
                              stdout_path=str(destination/'stdout.log'), stderr_path=str(destination/'stderr.log'))
                if uuid in remotes:
                    status['execution_host']=remotes[uuid]['host']
                    status['allocation']['allocated_cpu_count']=remotes[uuid]['allocated_cpu_count']
                    status['remote_hardware_report_hash']=remotes[uuid]['hardware_report_hash']
                atomic_json(status_file, status)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CGX_DATA_ROOT=str(root), OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
                began = time.monotonic(); peak = 0; proc = None; remote_code=None
                with guard:
                    active[job['job_id']] = {'began': began, 'previous_elapsed': previous_elapsed}
                try:
                    require_protocol_lock(root)
                    verify_hashes(job['prerequisite_files'])
                    if uuid in remotes:
                        remote=launch_remote(remotes[uuid],job,root,destination);last_usage=0.
                        while remote['state'] in ('starting','running'):
                            status.update(remote_status_path=str(destination/'remote_status.json'),remote_pid=remote.get('pid'))
                            atomic_json(status_file,status)
                            peak=max(peak,remote.get('peak_gpu_memory_bytes',0))
                            if time.monotonic()-last_usage>=30:usage_report();last_usage=time.monotonic()
                            if time.time()-(destination/'remote_status.json').stat().st_mtime>180:
                                raise RuntimeError('Remote heartbeat missing for180s; preserve the remote process and inspect it before retry')
                            time.sleep(5);remote=remote_progress(destination)
                        remote_code=remote.get('exit_code');peak=max(peak,remote.get('peak_gpu_memory_bytes',0))
                        if remote['state']!='passed' or remote_code!=0:raise RuntimeError(f"Remote trainer failed: {remote.get('failure_reason',remote_code)}")
                    else:
                        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True, timeout=10)
                        if any(line.split(',')[0].strip() == uuid for line in occupied.splitlines()):
                            raise RuntimeError(f'GPU {uuid} is occupied; no unrelated process was interrupted')
                        with (destination/'stdout.log').open('a') as out, (destination/'stderr.log').open('a') as err:
                            proc = subprocess.Popen(job['command'], env=env, cwd=EXPERIMENT, stdout=out, stderr=err)
                            status['pid'] = proc.pid
                            atomic_json(status_file, status)
                            last_usage = 0.
                            while proc.poll() is None:
                                try:
                                    usage = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=10)
                                    for row in usage.splitlines():
                                        pid, mb = row.replace(' ', '').split(',')
                                        if int(pid) == proc.pid:
                                            peak = max(peak, int(mb)*1024**2)
                                except (ValueError, subprocess.SubprocessError):
                                    pass
                                if time.monotonic()-last_usage >= 30:
                                    usage_report(); last_usage = time.monotonic()
                                time.sleep(2)
                        if proc.returncode != 0:
                            raise RuntimeError(f'Training process exited {proc.returncode}')
                    completion, outputs = _checked_completion(job, lock)
                    status.update(state='passed', actual_updates=completion['actual_updates'],
                                  selected_checkpoint=completion['selected_checkpoint'], output_hashes=outputs, failure_reason=None)
                except Exception as exc:
                    if proc is not None and proc.poll() is None:
                        proc.terminate()
                        try: proc.wait(timeout=10)
                        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                    status.update(state='failed', output_hashes={}, failure_reason=f'{type(exc).__name__}: {exc}')
                    with guard: failures.append(job['job_id'])
                finally:
                    elapsed = time.monotonic()-began
                    status.update(finished_at=now(), elapsed_seconds=elapsed, cumulative_elapsed_seconds=previous_elapsed+elapsed,
                                  exit_code=proc.returncode if proc is not None else remote_code, peak_gpu_memory_bytes=peak)
                    atomic_json(status_file, status)
                    with guard: active.pop(job['job_id'], None)
                    usage_report()
    with ThreadPoolExecutor(max_workers=len(uuids)) as pool:
        list(pool.map(worker, uuids))
    if failures:
        raise RuntimeError(f'Failed jobs retained for inspection: {failures}')
    return {'state': 'passed', 'jobs': len(jobs), 'launched': len(pending_jobs), 'devices': uuids}
