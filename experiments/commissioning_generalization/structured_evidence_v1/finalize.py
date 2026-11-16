"""Verify preservation and package the completed, auditable analysis outputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import zipfile
from cgx.config import EXPERIMENT
from cgx.provenance import atomic_json, file_sha256, read_json, now
from .gates import verify_readers


REQUIRED_QUESTIONS = (
    'Does the structured diagnoser eliminate sensitivity to duplicated evidence?',
    'Does it correctly distinguish new measurements from repeated storage?',
    'Does it correctly preserve shared-error uncertainty?',
    'Does it reduce the healthy false alarms previously caused by continued investigation?',
    'Does adaptive investigation still degrade as B increases?',
    'Does Replan now beat, match, or still lose to Plan-Once?',
    'Does topology generalization remain?',
    'Which remaining errors are caused by diagnosis, and which by acquisition selection?',
)


def report_artifacts(run_dir):
    """Seal derived delivery files after all report producers have finished."""
    run = Path(run_dir)
    report = run / 'reports/REPORT.md'
    predictions = run / 'reports/main_per_case_predictions.parquet'
    if not report.is_file() or not predictions.is_file():
        raise ValueError('Final report and full per-case prediction table are required')
    content = report.read_text()
    for number, question in enumerate(REQUIRED_QUESTIONS, 1):
        title = re.escape(f'{number}. **{question}**')
        if len(re.findall(r'^' + title, content, flags=re.MULTILINE)) != 1:
            raise ValueError(f'Required report question {number} must appear exactly once')
    # The audit and delivery markers live outside reports, avoiding a hash cycle.
    paths = {report, predictions}
    paths.update(path for path in (run / 'reports').rglob('*')
                 if path.is_file() and path.suffix in ('.csv', '.json', '.parquet'))
    return {str(path.relative_to(run)): file_sha256(path) for path in sorted(paths)}


def verify_report_artifacts(run_dir, final_audit):
    expected = final_audit.get('report_artifact_hashes')
    if not expected or final_audit.get('required_report_questions') != list(REQUIRED_QUESTIONS):
        raise ValueError('Final audit lacks the report seal; rerun the final audit')
    if report_artifacts(run_dir) != expected:
        raise ValueError('Report files changed after the final audit; delivery is blocked')


def audit(run_dir, workers=32):
    run = Path(run_dir)
    original = read_json(run / 'provenance/protected_before.json')
    def check(item):
        path, expected = item
        actual = file_sha256(path) if Path(path).is_file() else None
        return None if actual == expected else dict(path=path, expected=expected, actual=actual)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        differences = [item for item in pool.map(check, original.items()) if item is not None]
    source = Path(read_json(run / 'source_checks.json')['original_data_root'])
    inventory = [json.loads(line) for line in
                 (run / 'provenance/original_evaluation_input_hashes.jsonl').read_text().splitlines() if line]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        changed_main = [item for item in pool.map(check,
            ((str(source / item['path']), item['sha256']) for item in inventory)) if item is not None]
    seals = {}
    verify_readers(run)
    for relative in ('models/readers/frozen.json', 'models/frozen.json'):
        seal = read_json(run / relative)
        errors = [name for name, expected in seal['files'].items()
                  if not (run / name).is_file() or file_sha256(run / name) != expected]
        for name, expected in seal.get('runtime_files', {}).items():
            if file_sha256(EXPERIMENT / name) != expected:
                errors.append(name)
        for name, expected in seal.get('scientific_code', {}).items():
            if file_sha256(EXPERIMENT / 'structured_evidence_v1' / name) != expected:
                errors.append(name)
        seals[relative] = dict(sha256=file_sha256(run / relative), changed=errors)
    completed = {}
    for relative in ('replay/complete.json', 'invariance/complete.json',
                     'evaluation/main/complete.json', 'reports/main_complete.json',
                     'runtime_audit/complete.json', 'error_audit/old_histories/complete.json',
                     'error_audit/new_main/complete.json',
                     'evidence_export/old_histories/complete.json',
                     'evidence_export/new_main/complete.json', 'figures/complete.json'):
        item = read_json(run / relative)
        assert item['state'] == 'passed', relative
        completed[relative] = file_sha256(run / relative)
    replay = read_json(run / 'replay/complete.json')
    assert replay['per_case_predictions_sha256'] == file_sha256(run / 'replay/per_case_predictions.parquet')
    assert replay['invariance_complete_sha256'] == file_sha256(run / 'invariance/complete.json')
    assert replay['reader_frozen_sha256'] == file_sha256(run / 'models/readers/frozen.json')
    report = read_json(run / 'reports/main_complete.json')
    assert report['original_sha256'] == file_sha256(source / 'results/per_case_results.parquet')
    assert report['new_sha256'] == file_sha256(run / 'evaluation/main/per_case_results.parquet')
    training = read_json(run / 'manifests/policy_training_authorized.json')
    assert training['reader_replay_complete_sha256'] == file_sha256(run / 'replay/complete.json')
    assert training['new_scores_complete_sha256'] == file_sha256(run / 'training/scores_complete.json')
    assert training['new_scores_sqlite_sha256'] == file_sha256(run / 'training/scores.sqlite')
    for relative, expected in training['policy_runtime_files'].items():
        assert file_sha256(EXPERIMENT / relative) == expected, relative
    for relative, expected in training['preflight_tests'].items():
        assert file_sha256(run / relative) == expected, relative
    validation_audit_path = run / 'reports/validation_timeout_audit_refreshed.json'
    validation_audit = read_json(validation_audit_path)
    assert validation_audit['state'] == 'passed' and validation_audit['source_validation']
    assert len(validation_audit['completed_models']) == 15
    assert validation_audit['models_still_training'] == 0
    assert validation_audit['initial_audit_unchanged']
    assert file_sha256(validation_audit['initial_audit_path']) == validation_audit['initial_audit_sha256']
    for relative, expected in validation_audit['inputs_sha256'].items():
        assert file_sha256(run / relative) == expected, relative
    completed['reports/validation_timeout_audit_refreshed.json'] = file_sha256(validation_audit_path)
    runtime = read_json(run / 'runtime_audit/complete.json')
    assert runtime['official_results_unchanged'] and runtime['full_matrix_denominator_unchanged']
    assert runtime['implementation_sha256'] == file_sha256(EXPERIMENT / 'structured_evidence_v1/runtime_audit.py')
    assert runtime['declaration_sha256'] == file_sha256(run / 'manifests/reader_deadline_audit_predeclared.json')
    for relative, expected in runtime['output_hashes'].items():
        assert file_sha256(run / 'runtime_audit' / relative) == expected, relative
    for cohort in ('old_histories', 'new_main'):
        directory = run / 'error_audit' / cohort
        footprint = read_json(directory / 'complete.json')
        assert footprint['all_contexts'] == 147200 and footprint['all_wrong_contexts_retained']
        assert footprint['correct_and_wrong_full_denominators_preserved']
        assert footprint['models_and_policies_unchanged'] and footprint['physical_acquisitions'] == 0
        for relative, expected in footprint['output_hashes'].items():
            assert file_sha256(directory / relative) == expected, relative
    replay_inventory = read_json(run / 'replay/inventory.json')
    for cohort, expected_contexts, planned_contexts in (
            ('old_histories', replay_inventory['all_prediction_jobs'], replay_inventory['contexts']),
            ('new_main', 147200, 147200)):
        directory = run / 'evidence_export' / cohort
        export = read_json(directory / 'complete.json')
        assert export['expected_contexts'] == expected_contexts
        assert export['planned_prediction_contexts'] == planned_contexts
        assert export['all_contexts_retained'] and export['all_dependency_locators_verified']
        assert export['no_new_measurement_identities'] and export['no_reader_calls']
        assert export['physical_acquisitions'] == 0 and export['output_rows']['failures'] == 0
        assert export['implementation_sha256'] == file_sha256(
            EXPERIMENT / 'structured_evidence_v1/evidence_export.py')
        for path, expected in export['input_hashes'].items():
            assert file_sha256(path) == expected, path
        for relative, expected in export['output_hashes'].items():
            assert file_sha256(directory / relative) == expected, relative
    report_hashes = report_artifacts(run)
    figures = sorted(p.name for p in (run / 'figures').glob('*.png'))
    figure_seal = read_json(run / 'figures/complete.json')
    assert figure_seal['required_figure_families'] == 8
    assert len(figure_seal['files']) == len(figure_seal['png_files']) == 8
    for relative, expected in {**figure_seal['files'], **figure_seal['png_files']}.items():
        assert file_sha256(run / 'figures' / relative) == expected, relative
    result = dict(state='passed' if not differences and not changed_main and
        all(not seal['changed'] for seal in seals.values()) else 'failed', completed_at=now(),
        original_protected_files=len(original), original_evaluation_files=len(inventory),
        changed_original_files=differences, changed_original_evaluation_files=changed_main,
        seals=seals, completed_artifacts=completed, figures=figures,
        report_artifact_hashes=report_hashes, required_report_questions=list(REQUIRED_QUESTIONS),
        previous_results_overwritten=False, main_expected_cells=147200)
    atomic_json(run / 'provenance/final_audit.json', result)
    if result['state'] != 'passed':
        raise AssertionError('Preservation audit failed; inspect final_audit.json')
    return result


def package(run_dir):
    run = Path(run_dir)
    final_audit = read_json(run / 'provenance/final_audit.json')
    assert final_audit['state'] == 'passed'
    verify_report_artifacts(run, final_audit)
    destination = run.parent / f'Structured_Evidence_{run.name}_Analysis.zip'
    if destination.exists():
        raise RuntimeError('Existing delivery archive is preserved')
    files = set()
    # Every per-case probability vector is included. Raw observations/checkpoints
    # remain in the separately preserved run, referenced by content address.
    for name in ('reports', 'figures', 'tests', 'manifests', 'provenance', 'commands',
                 'logs', 'dispatch', 'runtime_audit', 'error_audit', 'evidence_export',
                 'validation'):
        files.update(path for path in (run / name).rglob('*') if path.is_file() and
                     path.suffix != '.lock')
    for name in ('REQUEST.txt', 'protocol.json', 'base_config.json', 'source_checks.json',
                 'replay/per_case_predictions.parquet', 'replay/contexts.parquet',
                 'replay/original_reread_differences.parquet', 'replay/inventory.json',
                 'replay/complete.json', 'replay/predictions_complete.json',
                 'invariance/selection.json', 'invariance/transformations_complete.json',
                 'invariance/transformations.csv', 'invariance/shared_clock.csv',
                 'invariance/new_execution_pairs.csv', 'invariance/new_execution_summary.csv',
                 'invariance/new_execution_pairs.parquet',
                 'invariance/complete.json', 'evaluation/main/complete.json',
                 'evaluation/main/per_case_results.parquet',
                 'models/frozen.json', 'models/readers/frozen.json',
                 'models/readers/parameters_frozen.json', 'training/scores_complete.json',
                 'training/score_code_amendment.json'):
        path = run / name
        if path.is_file():
            files.add(path)
    files.update(path for path in (run / 'models').rglob('*.json')
                 if path.name in ('model.json', 'complete.json', 'validation.json', 'selection.json'))
    files.update((run / 'models').glob('cost_beta_seed*.json'))
    files.update((run / 'replay/jobs').glob('*.json.gz'))
    entries = {str(path.relative_to(run)): path for path in sorted(files)}
    entries.update({'code/' + path.name: path for path in
                    (EXPERIMENT / 'structured_evidence_v1').glob('*')
                    if path.is_file() and path.suffix in ('.py', '.md')})
    reference = list((EXPERIMENT / 'src/cgx').rglob('*.py'))
    reference += list((EXPERIMENT / 'schemas').glob('*.json'))
    reference += [EXPERIMENT / name for name in
                  ('configs/protocol.json', 'docs/MAIN_EXPERIMENT_V2.md', 'requirements.txt')
                  if (EXPERIMENT / name).is_file()]
    entries.update({'reference_source/' + str(path.relative_to(EXPERIMENT)): path
                    for path in reference})
    hashes = {name: file_sha256(path) for name, path in entries.items()}
    guide = ('Open reports/REPORT.md first. All per-case probability vectors and aggregate tables are included.\n'
        f'The complete acquired-observation store, training checkpoints, and traces are preserved at: {run}\n'
        'This analysis ZIP does not duplicate those large raw files. Original raw evidence is resolved through '
        'source_checks.json and replay/inventory.json; new evidence through the run public store. '
        'The evidence_export directory contains packed histories, acquisition identity envelopes, '
        'decision traces, and an external raw-blob resolution index. The raw measurement values '
        'are preserved at the recorded locations but are not embedded in this analysis ZIP.\n'
        'Model scope, source-only fits, invariant tests, conditional denominators, timeouts, physical cache '
        'accounting, and limitations are recorded in the accompanying report and manifests.\n')
    temporary = destination.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as archive:
        archive.writestr('START_HERE.txt', guide)
        archive.writestr('SHA256SUMS.json', json.dumps(hashes, sort_keys=True, indent=2) + '\n')
        for name, path in entries.items():
            archive.write(path, name)
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
    temporary.rename(destination)
    result = dict(state='passed', path=str(destination), bytes=destination.stat().st_size,
        sha256=file_sha256(destination), files=len(entries) + 2, created_at=now(),
        raw_observations_preserved_at=str(run / 'public'), raw_observations_in_analysis_zip=False)
    atomic_json(run / 'delivery.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['audit', 'package'])
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    print(audit(args.run_dir, args.workers) if args.command == 'audit' else package(args.run_dir), flush=True)
