"""Audit a completed extension and create one archive below 500 million bytes.

No acquisition, inference, model fitting, or test-case selection happens here.
The durable reservation counter is the authority for fresh physical attempts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import difflib
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

import pandas as pd

from cgx.provenance import atomic_json, file_sha256, now, read_json

NAMESPACE = Path(__file__).resolve().parent
EXPERIMENT = NAMESPACE.parents[1]
REPOSITORY = EXPERIMENT.parents[1]
MAXIMUM_BYTES = 500_000_000


def _passed(path, state='passed'):
    value = read_json(path)
    if value.get('state') != state:
        raise ValueError(f'Required artifact is not {state}: {path}')
    return value


def _hashes(base, hashes, workers=16):
    def check(item):
        name, digest = item
        path = Path(name) if Path(name).is_absolute() else base / name
        return name, path.is_file() and file_sha256(path) == digest
    with ThreadPoolExecutor(max_workers=workers) as pool:
        bad = [name for name, good in pool.map(check, hashes.items()) if not good]
    if bad:
        raise ValueError(f'Missing or changed artifacts ({len(bad)}): {bad[:12]}')
    return len(hashes)


def _phase(run, phase):
    directory = run / ('validation' if phase == 'validation' else 'evaluation')
    value = _passed(directory / f'phase_{phase}_complete.json')
    expected = {'validation': 16896, 'primary': 32000, 'transfer': 32000, 'all': 64000}[phase]
    if value['rows'] != expected or value['expected'] != expected:
        raise ValueError(f'Wrong phase denominator: {phase}')
    _hashes(directory, value['outputs_sha256'])
    return value


def validation_views(run):
    """Named views of sealed source rows; never new selection observations."""
    frames, inputs = [], {}
    for mode in ('online', 'completed_read'):
        path = run / 'validation' / f'validation_per_case_{mode}.parquet'
        frame = pd.read_parquet(path)
        frame['result_mode'] = mode
        frame['derived_view_not_independent_trial'] = True
        frames.append(frame)
        inputs[str(path.relative_to(run))] = file_sha256(path)
    frame = pd.concat(frames, ignore_index=True)
    outputs = {}
    for method, name, expected in [('fixed_template_validation', 'fixed_templates_per_case', 24576),
                                   ('interface_rule_se', 'heuristic_stop_options_per_case', 9216)]:
        selected = frame[frame.method == method]
        if len(selected) != expected:
            raise ValueError(f'Validation derived view denominator mismatch: {name}')
        path = run / 'validation' / (name + '.parquet')
        selected.to_parquet(path, index=False)
        outputs[str(path.relative_to(run))] = dict(rows=len(selected), sha256=file_sha256(path))
    record = dict(state='passed', original_source_rows_per_mode=16896,
        new_acquisition_inference_or_selection=False, duplicate_views_not_independent_observations=True,
        input_hashes=inputs, outputs=outputs)
    atomic_json(run / 'validation/derived_candidate_views.json', record)
    return record


def resource_usage(run):
    ledger = read_json(run / 'resources/fresh_attempt_budget.json')
    if ledger['cap'] != 190000 or not 0 <= ledger['robot_execution_attempts'] <= 190000:
        raise ValueError('Fresh physical-attempt cap invalid or exceeded')
    journal = run / 'resources/fresh_attempt_reservations.jsonl'
    journal_robots = journal_calibrations = journal_entries = 0
    sequences = set()
    for line in journal.open():
        entry = json.loads(line)
        if entry['sequence'] in sequences:
            raise ValueError('Repeated durable reservation sequence')
        sequences.add(entry['sequence'])
        journal_entries += 1
        journal_robots += entry['robots']
        journal_calibrations += entry['calibrations']
    if journal_robots > ledger['robot_execution_attempts'] or journal_calibrations > ledger['calibration_attempts']:
        raise ValueError('Reservation journal exceeds durable counter')
    if journal_entries > ledger['physical_operations']:
        raise ValueError('Reservation journal has more operations than counter')
    frames = []
    for phase, directory in [('validation', 'validation'), ('all', 'evaluation')]:
        path = run / directory / f'{phase}_per_case_online.parquet'
        frame = pd.read_parquet(path)
        frame['work_phase'] = phase
        frames.append(frame)
    logical = pd.concat(frames, ignore_index=True)
    # Every alias retains the original accounting; take it once per canonical
    # rollout, even when a different logical alias happened to populate cache.
    fields = ['new_robot_execution_attempts', 'new_calibration_attempts',
              'actual_new_physical_seconds', 'physical_cache_hits',
              'imported_physical_cache_hits', 'extension_physical_cache_hits']
    available = [c for c in fields if c in logical]
    for column in available:
        logical[column] = pd.to_numeric(logical[column], errors='coerce')
        if logical.groupby('execution_source_key')[column].nunique(dropna=True).max() > 1:
            raise ValueError(f'Alias changed canonical work accounting: {column}')
    canonical = logical.groupby('execution_source_key', as_index=False)[available].first()
    sums = {c: float(canonical[c].sum()) for c in available}
    attributed = sums.get('new_robot_execution_attempts', 0.)
    if attributed > ledger['robot_execution_attempts']:
        raise ValueError('Canonical rollout attempts exceed the global reservation counter')
    totals = dict(state='passed', created_at=now(),
        authoritative_fresh_attempt_ledger=str(run / 'resources/fresh_attempt_budget.json'),
        robot_execution_attempt_cap=ledger['cap'],
        all_extension_reserved_robot_attempts=ledger['robot_execution_attempts'],
        all_extension_reserved_calibration_attempts=ledger['calibration_attempts'],
        all_extension_reserved_physical_operations=ledger['physical_operations'],
        journal_robot_attempts=journal_robots, journal_calibration_attempts=journal_calibrations,
        journal_operations=journal_entries,
        counter_only_operations=ledger['physical_operations'] - journal_entries,
        counter_commits_before_journal=True, failed_interrupted_qualification_attempts_not_refunded=True,
        validation_logical_rows=16896, new_test_logical_rows=64000, imported_test_logical_rows=147200,
        canonical_rollouts_across_validation_and_test=len(canonical),
        canonical_accounting_known_rows={c: int(canonical[c].notna().sum()) for c in available},
        canonical_accounting_sums=sums,
        reserved_robot_attempts_outside_attributed_canonical_rows=ledger['robot_execution_attempts'] - attributed,
        unattributed_reason='Includes engineering qualifications, failed/interrupted work and rows with unavailable accounting; not assumed to be completed executions',
        imported_new_physical_attempts=0, imported_new_inference_calls=0,
        new_selector_training_steps=0, new_reader_fits=0, new_GPU_training_jobs=0,
        logical_costs_preserved_on_cache_hits=True,
        cross_run_wall_time='descriptive only; hardware/load/cache/concurrency were not controlled')
    atomic_json(run / 'resources/resource_totals.json', totals)
    canonical.to_parquet(run / 'resources/canonical_rollout_work.parquet', index=False)
    launches = []
    for path in sorted((run / 'dispatch').glob('*/launch.json')):
        item = read_json(path)
        launches.append(dict(path=str(path.relative_to(run)), phase=item.get('phase'),
            pilot=item.get('pilot'), cpu_workers=item.get('cpu_workers'),
            ports=sorted({task['port'] for task in item.get('tasks', [])})))
    text = ['# Resource usage',
        f"The durable counter reserved **{ledger['robot_execution_attempts']:,} / 190,000 robot-execution attempts**, "
        f"{ledger['calibration_attempts']:,} calibration attempts and {ledger['physical_operations']:,} physical operations. "
        'These counts include source validation, engineering qualifications and failed/interrupted attempts. A reservation is not proof of a successful execution.',
        f'Validation has 16,896 logical rows; new tests have 64,000; imported tests have 147,200. '
        f'Validation and test rows resolve to {len(canonical):,} unique canonical rollouts. '
        'Identical plan aliases and one-pass budget references do not add independent physical trials. '
        'Full_once methods were executed at B=1; the B=2/4 entries are verified feasible references.',
        f'Attributed canonical rollout accounting contains {attributed:,.0f} fresh robot attempts. '
        f"The remaining {ledger['robot_execution_attempts']-attributed:,.0f} reserved attempts include qualification work, "
        'interrupted/failed work or unavailable row accounting. The reservation ledger is authoritative; missing attribution is not converted to zero.',
        'Imported predictions and their historical costs required no fresh robot acquisition, selector inference or diagnosis here. '
        'Logical bytes/executions/calibrations remain charged on physical cache hits. '
        'Selection, diagnosis and end-to-end wall time remain separate; cross-run differences are descriptive because hardware load and cache/concurrency differ.',
        'No selector network was trained, no diagnoser was refitted and no GPU training job was started. '
        'The shared diagnoser was fitted in the preceding study. Available machines were used for remote CPU simulation and diagnosis.',
        'Recorded dispatches (worker counts are configured concurrent CPU processes, not measured CPU utilization):',
        '```json', json.dumps(launches, indent=2), '```',
        'Exact ledger, reservation journal, canonical work and remote commands are in resources/, dispatch/ and commands/. '
        'Per-method logical resource differences and uncertainty are in reports/resources_at_each_budget.csv and reports/paired_differences.csv.']
    (run / 'RESOURCE_USAGE.md').write_text('\n\n'.join(text) + '\n')
    return totals


def evidence_index(run, workers=32):
    """Exact external-file resolution for every new saved evidence history."""
    inventory = _passed(run / 'source_inventory.json')
    roots = [run, Path(inventory['sed_run']), Path(inventory['source_root'])]
    references = {}
    def add(kind, address):
        if not address or (kind, address) in references:
            return
        found = next((root / 'public' / kind / (address + '.json.gz') for root in roots
                      if (root / 'public' / kind / (address + '.json.gz')).is_file()), None)
        if found is None:
            raise FileNotFoundError((kind, address))
        references[kind, address] = dict(kind=kind, content_address=address, resolved_path=str(found))
    for phase, directory in [('validation', 'validation'), ('all', 'evaluation')]:
        path = run / directory / f'{phase}_acquired_history_index.parquet'
        frame = pd.read_parquet(path)
        for row in frame.itertuples():
            if row.outer_failure_no_history:
                continue
            packed = json.loads(row.packed_history_json)
            add('histories', row.history_hash)
            add('graphs', packed['graph_ref'])
            add('tasks', packed['task_ref'])
            for ref in packed['observation_refs']:
                add('observations', ref)
            name = ('event_provenance', row.history_hash + '/' + row.provenance_hash)
            references[name] = dict(kind=name[0], content_address=row.provenance_hash,
                history_hash=row.history_hash, resolved_path=row.provenance_path)
    def fingerprint(row):
        path = Path(row['resolved_path'])
        return dict(row, stored_bytes=path.stat().st_size, stored_sha256=file_sha256(path))
    path = run / 'provenance/new_evidence_files.jsonl.gz'
    with ThreadPoolExecutor(max_workers=workers) as pool, gzip.open(path, 'wt') as stream:
        for row in pool.map(fingerprint, references.values()):
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    old_index = Path(inventory['sed_run']) / 'evidence_export/new_main/blobs.jsonl.gz'
    shutil.copy2(old_index, run / 'provenance/imported_evidence_files.jsonl.gz')
    record = dict(state='passed', created_at=now(), new_evidence_unique_files=len(references),
        all_new_validation_and_test_evidence_resolved=True,
        measurements_in_archive=False, checkpoints_in_archive=False,
        new_evidence_index_sha256=file_sha256(path),
        imported_evidence_index_sha256=file_sha256(old_index),
        imported_index_original=str(old_index),
        stores_in_resolution_order=[str(p / 'public') for p in roots],
        new_logical_cells=str(run / 'evaluation/cases/{cell_key}.json'),
        validation_logical_cells=str(run / 'validation/cases/{cell_key}.json'),
        canonical_rollouts=str(run / 'rollouts/{execution_source_key[0:2]}/{execution_source_key}.json'),
        imported_logical_cells=str(Path(inventory['sed_run']) / 'evaluation/main/cases/{cell_key}.json'),
        checkpoints='Exact paths/hashes in source_inventory.json and original frozen manifest; external only')
    atomic_json(run / 'provenance/external_store_references.json', record)
    return record


def _snapshot_code(run):
    destination = run / 'code'
    destination.mkdir(exist_ok=True)
    patch = []
    for path in sorted(NAMESPACE.glob('*')):
        if path.suffix not in ('.py', '.md') or not path.is_file():
            continue
        shutil.copy2(path, destination / path.name)
        rel = str(path.relative_to(REPOSITORY))
        base = subprocess.run(['git', 'show', 'HEAD:' + rel], cwd=REPOSITORY,
                              capture_output=True, text=True)
        before = base.stdout.splitlines(keepends=True) if base.returncode == 0 else []
        patch.extend(difflib.unified_diff(before, path.read_text().splitlines(keepends=True),
            fromfile='a/' + rel if before else '/dev/null', tofile='b/' + rel))
    (destination / 'implementation.patch').write_text(''.join(patch))
    for directory in ('src', 'structured_evidence_v1', 'configs', 'docs', 'schemas'):
        source = EXPERIMENT / directory
        if source.is_dir():
            shutil.copytree(source, run / 'reference_source' / directory, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    for name in ('pyproject.toml', 'requirements-spec.txt', 'requirements.lock.txt', 'AGENTS.md'):
        if (EXPERIMENT / name).is_file():
            shutil.copy2(EXPERIMENT / name, run / 'reference_source' / name)
    git = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, text=True).strip(),
               branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPOSITORY, text=True).strip(),
               status=subprocess.check_output(['git', 'status', '--short'], cwd=REPOSITORY, text=True),
               source_snapshot_time=now())
    atomic_json(run / 'provenance/delivery_git_state.json', git)


def audit(run_dir, workers=32):
    run = Path(run_dir).resolve()
    started = time.monotonic()
    inventory = _passed(run / 'source_inventory.json')
    protected = read_json(run / 'provenance/protected_before.json')
    protected_count = _hashes(run, protected, workers)
    _hashes(run, {row['path']: row['sha256'] for row in inventory['inputs'].values()}, workers)
    design = _passed(run / 'provenance/validation_design_seal.json', 'sealed_before_validation')
    _hashes(NAMESPACE, design['code_hashes'], workers)
    for name, field in [('protocol.json', 'protocol_sha256'), ('source_inventory.json', 'source_inventory_sha256')]:
        if file_sha256(run / name) != design[field]:
            raise ValueError(f'Validation design changed: {name}')
    selection = _passed(run / 'selection_manifest.json', 'sealed')
    if selection['reader_hash'] != inventory['reader_hash'] or not selection['new_test_cells_absent_at_seal']:
        raise ValueError('Selection changed the shared reader or lacked pretest isolation')
    _hashes(run, selection['validation_input_hashes'], workers)
    _hashes(run, selection['selection_output_hashes'], workers)
    for name, digest in selection['code_hashes'].items():
        path = Path(name) if Path(name).is_absolute() else (EXPERIMENT / name if '/' in name else NAMESPACE / name)
        if file_sha256(path) != digest:
            raise ValueError(f'Selected runtime changed: {path}')
    phases = {p: _phase(run, p) for p in ('validation', 'primary', 'transfer', 'all')}
    if phases['all']['physical_cases'] != 3200 or not phases['all']['one_pass_cap_views_verified']:
        raise ValueError('Case support or one-pass identity controls missing')
    execution_audit = _passed(run / 'evaluation/all_execution_integrity_audit.json')
    if execution_audit['physical_cases'] != 3200 or not execution_audit['all_64000_unique_logical_rows_present']:
        raise ValueError('All-matrix execution audit is incomplete')
    imported = _passed(run / 'evaluation/import_complete.json')
    if imported['rows'] != 147200 or imported['physical_acquisitions'] != 0 or imported['reader_calls'] != 0:
        raise ValueError('Imported counts or work changed')
    _hashes(run / 'evaluation', imported['output_hashes'], workers)
    for name in ('policy_controls', 'runner_controls', 'one_pass_cap_views', 'reporting_engineering', 'imported_port_extraction'):
        _passed(run / 'tests' / (name + '.json'))
    engineering = read_json(run / 'tests/reporting_engineering.json')
    if engineering['implementation_sha256'] != file_sha256(NAMESPACE / 'reporting.py'):
        raise ValueError('Reporting controls must be rerun after implementation changes')
    report = _passed(run / 'reports/complete.json')
    if report['new_executed_rows'] != 64000 or report['imported_rows'] != 147200:
        raise ValueError('Report omitted planned rows')
    if report['bootstrap_repetitions'] != 2000 or not report['within_case_seed_averaging']:
        raise ValueError('Registered paired inference changed')
    _hashes(run, report['input_hashes'], workers)
    _hashes(run, report['output_hashes'], workers)
    figures = _passed(run / 'figures/complete.json')
    if figures['required_figure_families'] != 4 or figures['standalone_tex'] != figures['standalone_pdfs']:
        raise ValueError('Required figure/source pairs incomplete')
    if figures['input_report_sha256'] != file_sha256(run / 'reports/complete.json'):
        raise ValueError('Figures use a different report')
    _hashes(run / 'figures', figures['files'], workers)
    text = (run / 'FINDINGS_CN.md').read_text()
    if not all(f'## {i}.' in text for i in range(1, 9)):
        raise ValueError('Findings must answer all eight questions')
    totals = resource_usage(run)
    derived = validation_views(run)
    index = evidence_index(run, workers)
    _snapshot_code(run)
    _write_readme(run)
    result = dict(state='passed', completed_at=now(), elapsed_seconds=time.monotonic()-started,
        protected_files_unchanged=protected_count, reader_hash=inventory['reader_hash'],
        all_planned_denominators_retained=True, new_test_rows=64000, imported_test_rows=147200,
        source_validation_rows=16896, new_models_trained=0, no_new_diagnoser_fitting=True,
        fresh_robot_attempts=totals['all_extension_reserved_robot_attempts'],
        evidence_files_indexed=index['new_evidence_unique_files'],
        latex_status=figures['latex_status'], raw_evidence_and_checkpoints_external=True,
        implementation_sha256=file_sha256(__file__),
        artifacts={name: file_sha256(run / name) for name in (
            'reports/complete.json', 'figures/complete.json', 'selection_manifest.json',
            'evaluation/all_execution_integrity_audit.json',
            'resources/resource_totals.json', 'provenance/external_store_references.json',
            'validation/derived_candidate_views.json', 'FINDINGS_CN.md')})
    atomic_json(run / 'provenance/final_audit.json', result)
    return result


def _write_readme(run):
    (run / 'README.md').write_text(
        '# CGX-SIMPLE-SELECTORS-V1\n\n'
        'Read FINDINGS_CN.md first. All methods share the same frozen, previously fitted Structured Evidence diagnoser. '
        'This extension trains no selector and refits no diagnoser. It adds six simple inspection policies to the original '
        'Panda commissioning benchmark and imports five learned selectors plus the original four-execution full-recording reference.\n\n'
        'The archive contains all 64,000 new and 147,200 imported test rows in both online and completed-reading forms. '
        'Online retains the original deadlines. Completed reading uses exactly the saved evidence and never recovers '
        'earlier abandoned acquisition. Failures, fallbacks and unresolved rereads remain in every denominator. '
        'Deterministic methods have a null seed; Random/learned seeds are averaged within physical case for paired inference. '
        'One-full-execution methods have one actual B=1 row and are shown as verified feasible references at B=2/4.\n\n'
        'reports/normalized_per_case.parquet retains every probability vector and row provenance. '
        'evaluation/ holds original-format new/imported probabilities, acquired-history indexes and action traces. '
        'validation/ contains all predefined candidate rows, selected settings and configuration-balanced selection summaries. '
        'The four figure families include vector PDFs, previews, standalone PGFPlots sources and exact plotted CSV data. '
        'figures/README.md records whether TeX compilation was available.\n\n'
        'code/ contains the extension implementation and a patch; reference_source/ contains frozen dependencies. '
        'Run commands, runtime seals, hashes and tests are in commands/, dispatch/, provenance/ and tests/. '
        'RESOURCE_USAGE.md separates logical acquisition charges from actual reused/new work and preserves the global attempt ledger.\n\n'
        '**Large raw measurements, simulator caches, per-cell envelopes and checkpoints are outside the ZIP.** '
        'provenance/external_store_references.json provides store resolution rules and original paths. '
        'provenance/new_evidence_files.jsonl.gz and imported_evidence_files.jsonl.gz contain exact raw-file references and hashes. '
        'source_inventory.json identifies the frozen model/checkpoint files; nothing omitted is fabricated from a summary. '
        'These indexes require access to the original storage for full physical replay.\n\n'
        'DELIVERY_FILE_INDEX.json lists every included file with byte count and SHA-256; '
        'DELIVERY_OMISSIONS.json records external bulky categories and any excluded duplicate artifacts. '
        'The package is required to be strictly below 500,000,000 bytes.\n')


def package(run_dir, output=None):
    run = Path(run_dir).resolve()
    audit_record = _passed(run / 'provenance/final_audit.json')
    _hashes(run, audit_record['artifacts'])
    target = Path(output) if output else run.parent / f'Simple_Selector_Controls_{run.name}_GPT_Pro.zip'
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f'Existing delivery preserved: {target}')
    excluded = []
    selected = []
    import_marker = _passed(run / 'evaluation/import_complete.json')
    redundant_imports = {
        'imported_action_profile.parquet': 'Per-case fields are joined into normalized_per_case and imported prediction tables; exact source hashes remain in import_complete.json',
        'imported_observation_ports.parquet': 'Intermediate per-observation port extraction; final per-case acquired-port coverage is retained, and raw-file paths/hashes are in imported_evidence_files.jsonl.gz',
    }
    import pyarrow.parquet as pq
    profile_columns = set(pq.read_schema(run / 'evaluation/imported_action_profile.parquet').names)
    normalized_columns = set(pq.read_schema(run / 'reports/normalized_per_case.parquet').names)
    if not profile_columns.issubset(normalized_columns):
        raise ValueError(f'Cannot omit action profile: unique fields absent from normalized results: {profile_columns-normalized_columns}')
    for path in sorted(run.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(run)
        parts = rel.parts
        # Root objects, compact reports, audit indexes and source snapshots only.
        include = len(parts) == 1 and path.suffix in ('.json', '.md')
        if parts[0] in ('code', 'reference_source', 'reports', 'figures', 'tests', 'provenance', 'commands', 'resources', 'dispatch'):
            include = True
        if parts[0] in ('evaluation', 'validation') and len(parts) == 2:
            include = True
        if path.suffix in ('.lock', '.tmp', '.pyc') or '__pycache__' in parts:
            include = False
        if parts[0] == 'tests' and len(parts) > 1 and parts[1] == 'qualification_records':
            include = False
        if parts[0] == 'evaluation' and path.name in redundant_imports:
            include = False
        # Full tables supersede intermediate phase copies. Primary report is
        # retained to demonstrate that it preceded transfer evaluation.
        if parts[0] == 'evaluation' and path.name.startswith(('primary_per_case_', 'transfer_per_case_',
                'primary_acquired_', 'transfer_acquired_', 'primary_action_', 'transfer_action_',
                'all_per_case_', 'all_acquired_', 'all_action_')):
            include = False
        if len(parts) > 1 and parts[:2] == ('reports', 'primary') and path.name == 'normalized_per_case.parquet':
            include = False
        if path.name in ('DELIVERY_FILE_INDEX.json', 'DELIVERY_OMISSIONS.json') or path == target:
            include = False
        if include:
            selected.append(path)
        elif parts[0] not in ('public', 'cache', 'rollouts') and not ('cases' in parts):
            excluded.append(dict(path=str(path), relative_path=str(rel), bytes=path.stat().st_size,
                sha256=import_marker['output_hashes'].get(path.name) if path.name in redundant_imports else None,
                reason=redundant_imports.get(path.name, 'Duplicate/intermediate table, bulky qualification envelope or transient runtime file; retained in original run')))
    required = ['evaluation/per_case_online.parquet', 'evaluation/per_case_completed_read.parquet',
        'evaluation/imported_learned_rows.parquet', 'evaluation/imported_completed_read.parquet',
        'evaluation/acquired_history_index.parquet', 'evaluation/action_traces.jsonl.gz',
        'reports/normalized_per_case.parquet', 'validation/selected_settings.json',
        'validation/fixed_templates_per_case.parquet', 'validation/heuristic_stop_options_per_case.parquet',
        'FINDINGS_CN.md', 'RESOURCE_USAGE.md', 'README.md', 'provenance/final_audit.json',
        'provenance/new_evidence_files.jsonl.gz', 'provenance/imported_evidence_files.jsonl.gz']
    names = {str(path.relative_to(run)) for path in selected}
    if not set(required).issubset(names):
        raise ValueError(f'Delivery missing required files: {set(required)-names}')
    omissions = dict(raw_measurements_and_checkpoints_external=True,
        external_categories=[dict(path=str(run / name), reason=reason) for name, reason in (
            ('public', 'Raw compressed measurements/graphs/histories; exact file hashes in evidence indexes'),
            ('cache', 'Exact physical simulator result cache; not needed to analyze compact predictions'),
            ('rollouts', 'Canonical envelopes; histories/action traces/predictions supplied compactly'),
            ('evaluation/cases', 'Per-cell envelopes; cell paths/hashes in acquired_history_index'),
            ('validation/cases', 'Per-candidate envelopes; compact candidate outputs supplied'))],
        previous_models_and_raw_stores='See source_inventory.json and provenance/external_store_references.json',
        excluded_run_files=excluded)
    atomic_json(run / 'DELIVERY_OMISSIONS.json', omissions)
    selected.append(run / 'DELIVERY_OMISSIONS.json')
    index = {str(path.relative_to(run)): dict(bytes=path.stat().st_size, sha256=file_sha256(path)) for path in selected}
    atomic_json(run / 'DELIVERY_FILE_INDEX.json', dict(files=index, generated_at=now(),
        index_self_hash_excluded=True, byte_limit=MAXIMUM_BYTES, raw_measurements_and_checkpoints_external=True))
    selected.append(run / 'DELIVERY_FILE_INDEX.json')
    temporary = target.with_suffix('.partial.zip')
    if temporary.exists():
        raise FileExistsError(f'Partial archive preserved: {temporary}')
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in selected:
            archive.write(path, str(path.relative_to(run)))
    size = temporary.stat().st_size
    if size >= MAXIMUM_BYTES:
        raise ValueError(f'Archive is {size} bytes, over the strict limit; partial archive retained for inspection: {temporary}')
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC integrity test failed')
        if len(archive.namelist()) != len(selected):
            raise ValueError('Archive file count mismatch')
    temporary.rename(target)
    result = dict(state='passed', completed_at=now(), path=str(target), bytes=size,
        maximum_exclusive_bytes=MAXIMUM_BYTES, sha256=file_sha256(target), files=len(selected),
        crc_verified=True, per_case_probability_vectors_included_for_all_methods=True,
        raw_evidence_and_checkpoints_external=True)
    atomic_json(run / 'provenance/delivery.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('audit', 'package', 'all'))
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.command in ('audit', 'all'):
        print(json.dumps(audit(args.run_dir, args.workers)), flush=True)
    if args.command in ('package', 'all'):
        print(json.dumps(package(args.run_dir, args.output)), flush=True)
