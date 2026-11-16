"""Seal benchmark identity and protect previous runs before SED-1.0 work."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
from cgx.config import EXPERIMENT, load_config
from cgx.provenance import atomic_json, file_sha256, read_json


def record_source_amendment(source_root, run_dir):
    source, run = Path(source_root), Path(run_dir)
    original = read_json(source / 'software/source_files.json')
    changed = []
    for relative, expected in original.items():
        path = EXPERIMENT / relative
        actual = file_sha256(path) if path.is_file() else None
        if actual != expected:
            changed.append(dict(path=relative, original_sha256=expected, current_sha256=actual))
    added = [str(p.relative_to(EXPERIMENT)) for p in (EXPERIMENT / 'src').rglob('*.py')
             if str(p.relative_to(EXPERIMENT)) not in original]
    # Earlier user-authorized method-display renaming changed reporting/docs,
    # not simulation, model architecture, features, training, or rollout code.
    allowed = ('docs/', 'src/cgx/reporting/')
    if any(not item['path'].startswith(allowed) for item in changed):
        raise ValueError('Unaccounted scientific source change since original main run')
    if any(not name.startswith('src/cgx/reporting/') for name in added):
        raise ValueError('Unaccounted new original-runtime module')
    from cgx.provenance import hash_json
    lock = read_json(source / 'protocol_lock.json')
    assert hash_json(load_config()) == lock['config_hash']
    record = dict(state='passed', original_protocol_lock_sha256=file_sha256(source / 'protocol_lock.json'),
        changed_since_original_seal=changed, added_since_original_seal=added,
        original_scientific_runtime_unchanged=True, active_config_exactly_matches_original=True,
        source_for_new_experiment_commit=read_json(run / 'source_checks.json')['base_commit'],
        interpretation='New namespace explicitly uses the committed source after the previously authorized display-name changes. The original protocol lock is preserved; the full-tree hash difference is not silently ignored.')
    atomic_json(run / 'manifests/source_version_amendment.json', record)
    return record


def prepare(source_root, run_dir, request, workers=32):
    source = Path(source_root).resolve()
    run = Path(run_dir).resolve()
    if (run / 'source_checks.json').exists():
        raise RuntimeError('Existing experiment is preserved; use its recorded resume interfaces')
    for d in ('manifests', 'provenance', 'commands', 'logs', 'tests', 'models/readers',
              'training', 'replay', 'evaluation', 'public', 'results', 'figures'):
        (run / d).mkdir(parents=True, exist_ok=True)
    shutil.copy2(request, run / 'REQUEST.txt')
    config = load_config()
    atomic_json(run / 'base_config.json', config)
    # Copies of sealed metadata are read-only inputs to this experiment. The
    # original public blob store is accessed through OverlayStore, never linked
    # as a writable output store.
    for path in (source / 'manifests').iterdir():
        if path.is_file():
            shutil.copy2(path, run / 'manifests' / path.name)
    asset = run / 'assets'
    if not asset.exists():
        asset.symlink_to((source / 'assets').resolve(), target_is_directory=True)
    repo = EXPERIMENT.parents[1]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=repo).decode().split('\0')
    original_source = [repo / name for name in tracked if name and (repo / name).is_file()]
    frozen = read_json(source / 'models/frozen.json')
    protected = set(original_source)
    protected.update(source / name for name in frozen['files'])
    for dirname in ('manifests', 'results'):
        protected.update(p for p in (source / dirname).rglob('*') if p.is_file())
    protected.update(p for p in (source / 'training').glob('*') if p.is_file() and p.suffix != '.lock')
    protected.update([source / 'private/private_cases.jsonl', source / 'protocol_lock.json',
                      source / 'models/selection.json', source / 'models/frozen.json'])
    protected = sorted(p for p in protected if p.is_file())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashes = dict(zip(map(str, protected), pool.map(file_sha256, protected)))
    atomic_json(run / 'provenance/protected_before.json', hashes)
    (run / 'provenance/git_start_status.txt').write_text(subprocess.check_output(
        ['git', 'status', '--short', '--branch'], cwd=repo, text=True))
    source_inventory = source / 'results/evaluation_input_hashes.jsonl'
    main_items = [json.loads(line) for line in source_inventory.read_text().splitlines() if line]
    def check_main(item):
        return file_sha256(source / item['path']) == item['sha256']
    with ThreadPoolExecutor(max_workers=workers) as pool:
        checked = list(pool.map(check_main, main_items))
    if not all(checked):
        raise ValueError('Previous main-evaluation archive differs from its recorded hashes')
    shutil.copy2(source_inventory, run / 'provenance/original_evaluation_input_hashes.jsonl')
    replay_counts = {}
    for split in ('train', 'validation'):
        path = source / 'training' / f'replay_{split}.sqlite'
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True) as db:
            replay_counts[split] = dict(cases=db.execute('SELECT COUNT(*) FROM cases').fetchone()[0],
                transitions=db.execute('SELECT COUNT(*) FROM transitions').fetchone()[0])
    protocol = {
        'protocol_id': 'SED-1.0', 'state': 'design_fixed_before_new_outcomes',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'base_commit': commit,
        'original_data_root': str(source), 'run_directory': str(run),
        'unchanged_benchmark_config_sha256': file_sha256(EXPERIMENT / 'configs/protocol.json'),
        'seeds': config['seeds'], 'splits': config['splits'],
        'methods': config['methods']['main_learned_evaluation'] + ['full_recording_feasible'],
        'execution_budgets': [1, 2, 4], 'full_recording_budget': 4,
        'main_expected_rows': 147200, 'main_physical_cases': 3200,
        'selector_training': config['training'], 'selector_model_architectures': config['models'],
        'source_replay': replay_counts,
        'reader_model': {
            'family': 'component-local learned likelihoods with analytically marginalized shared latents',
            'clock_latent': 'one persistent offset per public clock group across all acquisitions',
            'message_latent': 'shared message epoch, equality-only opaque token',
            'fault_delay': 'case-persistent learned three-component Gaussian mixture',
            'fault_gain': 'case-persistent learned six-component Gaussian mixture',
            'parameters': 'fit original TRAIN public observations and allowed fault labels only',
            'crossfit': 'final and three original topology-exclusion folds',
            'missing_without_values': 'explicit missing/timeout metadata, no fault-likelihood contribution',
            'count_or_action_likelihood': False, 'numeric_equality_deduplication': False,
            'grid_search': False, 'test_time_updates': False,
        },
        'gates': {
            'reader_replay': 'manifests/reader_replay_authorized.json after final/OOF fits and invariance controls',
            'policy_training': 'manifests/policy_training_authorized.json after complete old-trajectory reader replay',
            'main': 'all readers/policies and validation choices sealed before new main evaluation',
        },
        'comparisons': ['original_as_deployed', 'original_GNN_reread', 'R-C', 'Structured Evidence Diagnoser'],
        'statistics': {'bootstrap_replicates': 2000, 'bootstrap_seed': config['seeds']['bootstrap'],
            'unit': 'average policy seeds within physical case, then configurations and stratified cases',
            'healthy_false_alarm': 'argmax target != healthy conditional on true healthy',
            'fault_miss': 'argmax target == healthy conditional on true faulty',
            'joint_accuracy': 'exact target key, separately all cases and fault-only',
            'nll': 'true target log loss in bits', 'brier': 'sum multiclass squared probability error',
            'ece': 'ten fixed equal-width maximum-probability bins',
            'no_test_based_selection': True},
        'hardware': {'preference': 'all available authorized remote GPUs and CPUs including 2379',
            'authorized_ports': [2379, 2982, 2404],
            'initial_available_gpu_slots': [{'port': p, 'gpu': g} for p, gs in [(2404, range(4)), (2982, range(2))] for g in gs],
            'cpu_quota_cores': {'2379': 48, '2982': 96, '2404': 192},
            'initial_2379': 'occupied by unrelated hajimi jobs; recheck and use when available',
            'unrelated_jobs_must_be_preserved': True, 'local_compute': 'coordination and editing only'},
        'resource_accounting': {'logical_acquisition_costs_unchanged': True,
            'physical_cache_hits_separate': True, 'source_generation': 'reuse unchanged original replay',
            'new_data_namespace_only': True, 'previous_V2_diagnostic_caps': 'not applied to this new full-benchmark task'},
    }
    atomic_json(run / 'protocol.json', protocol)
    checks = dict(state='passed', original_data_root=str(source), base_commit=commit,
        protected_files=len(hashes), previous_evaluation_artifacts_verified=len(checked),
        previous_evaluation_inventory_sha256=file_sha256(source_inventory), source_replay=replay_counts,
        protocol_sha256=file_sha256(run / 'protocol.json'), request_sha256=file_sha256(run / 'REQUEST.txt'),
        manifest_hashes={str(p.relative_to(run)): file_sha256(p) for p in (run / 'manifests').iterdir() if p.is_file()},
        no_original_artifacts_written=True)
    atomic_json(run / 'source_checks.json', checks)
    print(json.dumps(checks), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', required=True); p.add_argument('--run-dir', required=True)
    p.add_argument('--request', required=True); p.add_argument('--workers', type=int, default=32)
    args = p.parse_args(); prepare(args.source_root, args.run_dir, args.request, args.workers)
