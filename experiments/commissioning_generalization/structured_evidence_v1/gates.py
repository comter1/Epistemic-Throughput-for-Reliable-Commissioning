"""Seal source-only readers and enforce the ordered experimental stages."""
import argparse
from pathlib import Path
from cgx.config import EXPERIMENT
from cgx.provenance import atomic_json, file_sha256, read_json, now


def _write_once(path, value):
    if path.exists():
        raise RuntimeError(f'Existing scientific seal is preserved: {path}')
    atomic_json(path, value)


def verify_parameters(run_dir):
    run = Path(run_dir)
    parameters = read_json(run / 'models/readers/parameters_frozen.json')
    assert parameters['state'] == 'parameters_sealed_before_reader_replay'
    for relative, expected in parameters['files'].items():
        assert file_sha256(run / relative) == expected, relative
    assert file_sha256(run / 'tests/reader_fit_audit.json') == parameters['reader_fit_audit_sha256']
    assert file_sha256(run / 'manifests/structured_reader_model_spec.json') == parameters['model_spec_sha256']
    return parameters


def verify_readers(run_dir):
    run = Path(run_dir)
    parameters = verify_parameters(run)
    frozen = read_json(run / 'models/readers/frozen.json')
    assert frozen['state'] == 'sealed_before_reader_replay'
    assert frozen['parameters_frozen_sha256'] == file_sha256(run / 'models/readers/parameters_frozen.json')
    assert frozen['files'] == parameters['files']
    assert frozen['reader_controls_sha256'] == file_sha256(run / 'tests/reader_controls.json')
    assert frozen['reader_cache_equivalence_sha256'] == file_sha256(run / 'tests/reader_cache_equivalence.json')
    for relative, expected in frozen['runtime_files'].items():
        assert file_sha256(EXPERIMENT / relative) == expected, relative
    return frozen


def seal_parameters(run_dir):
    run = Path(run_dir)
    audit_path = run / 'tests/reader_fit_audit.json'
    audit = read_json(audit_path)
    assert audit['state'] == 'passed' and audit['source_only']
    assert audit['crossfit_exclusion_verified'] and not audit['hidden_offsets_or_severities_used']
    files = {}
    for name in ('final', 'crossfit_0', 'crossfit_1', 'crossfit_2'):
        directory = run / 'models/readers' / name
        marker = read_json(directory / 'complete.json')
        model_path = directory / marker['selected_checkpoint']
        assert marker['state'] == 'passed'
        assert file_sha256(model_path) == marker['selected_checkpoint_hash'] == audit['models'][name]['sha256']
        model = read_json(model_path)
        assert model['training_data_only'] and not model['hidden_offsets_or_severities_used']
        for path in (model_path, directory / 'complete.json'):
            files[str(path.relative_to(run))] = file_sha256(path)
    record = dict(state='parameters_sealed_before_reader_replay', created_at=now(), files=files,
        reader_fit_audit_sha256=file_sha256(audit_path),
        model_spec_sha256=file_sha256(run / 'manifests/structured_reader_model_spec.json'),
        final_checkpoint_hash=audit['models']['final']['sha256'])
    _write_once(run / 'models/readers/parameters_frozen.json', record)
    return record


def authorize_replay(run_dir):
    run = Path(run_dir)
    parameters_path = run / 'models/readers/parameters_frozen.json'
    parameters = verify_parameters(run)
    controls = run / 'tests/reader_controls.json'
    assert read_json(controls)['state'] == 'passed'
    cache = run / 'tests/reader_cache_equivalence.json'
    assert read_json(cache)['state'] == 'passed'
    runtime = {str(path.relative_to(EXPERIMENT)): file_sha256(path)
               for path in (EXPERIMENT / 'structured_evidence_v1').glob('*.py')
               if path.name in ('events.py', 'likelihood.py', 'reader.py', 'fit_reader.py')}
    record = dict(state='sealed_before_reader_replay', created_at=now(),
        files=parameters['files'], parameters_frozen_sha256=file_sha256(parameters_path),
        selected_final='models/readers/final/model.json', runtime_files=runtime,
        reader_controls_sha256=file_sha256(controls), reader_cache_equivalence_sha256=file_sha256(cache))
    path = run / 'models/readers/frozen.json'
    _write_once(path, record)
    gate = dict(state='passed', created_at=now(), reader_checkpoint_hash=parameters['final_checkpoint_hash'],
        reader_frozen_sha256=file_sha256(path), scope='fixed-reader old-evidence replay and invariance; no selector optimizer')
    _write_once(run / 'manifests/reader_replay_authorized.json', gate)
    return gate


def authorize_policies(run_dir):
    run = Path(run_dir)
    replay = run / 'replay/complete.json'
    complete = read_json(replay)
    assert complete['state'] == 'passed'
    assert complete['invariance_complete_sha256'] == file_sha256(run / 'invariance/complete.json')
    frozen_path = run / 'models/readers/frozen.json'
    assert complete['reader_frozen_sha256'] == file_sha256(frozen_path)
    verify_readers(run)
    tests = {}
    for name in ('selector_adapters.json', 'policy_initialization_control.json', 'full_immutable_score_lookup.json'):
        path = run / 'tests' / name
        assert read_json(path)['state'] == 'passed', name
        tests['tests/' + name] = file_sha256(path)
    scientific = {str(path.relative_to(EXPERIMENT)): file_sha256(path)
                  for path in (EXPERIMENT / 'structured_evidence_v1').glob('*.py')
                  if path.name in ('policy_training.py', 'selectors.py', 'rollout.py', 'evaluation.py', 'storage.py')}
    score = read_json(run / 'training/scores_complete.json')
    assert score['state'] == 'passed' and score['output_hash'] == file_sha256(run / 'training/scores.sqlite')
    gate = dict(state='authorized_after_reader_replay', created_at=now(),
        reader_replay_complete_path=str(replay), reader_replay_complete_sha256=file_sha256(replay),
        policy_runtime_files=scientific, preflight_tests=tests,
        new_scores_complete_sha256=file_sha256(run / 'training/scores_complete.json'),
        new_scores_sqlite_sha256=score['output_hash'])
    _write_once(run / 'manifests/policy_training_authorized.json', gate)
    return gate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['seal-parameters', 'authorize-replay', 'authorize-policies'])
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    print(globals()[args.stage.replace('-', '_')](args.run_dir), flush=True)
