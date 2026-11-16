"""Portable evidence/provenance indexes; no model calls or acquisitions.

Large observation blobs stay in their immutable public stores. Small packed
histories, existing acquisition identities, graphs/tasks and recorded decisions
are copied into compressed JSONL. Blob locators include compressed-file hashes;
only small metadata is reparsed for canonical content-address verification.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import tempfile
import time

from cgx.provenance import atomic_json, canonical_bytes, file_sha256, hash_json, now, read_json

VERSION = 'structured-evidence-export-v1'
CELL_FIELDS = ('suite', 'method', 'policy_seed', 'execution_cap', 'case_id')
EXPECTED_MAIN = 147200


def read_gzip(path):
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)


class JsonlWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.temporary = self.path.with_suffix(self.path.suffix + '.partial')
        self.rows = 0

    def __enter__(self):
        self.handle = gzip.open(self.temporary, 'wb', compresslevel=1)
        return self

    def write(self, value):
        self.handle.write(canonical_bytes(value) + b'\n')
        self.rows += 1

    def __exit__(self, kind, value, traceback):
        self.handle.close()
        if kind is None:
            self.temporary.replace(self.path)


def _address(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Invalid content address')
    return value


def locate(roots, kind, digest):
    _address(digest)
    for root in roots:
        path = Path(root) / 'public' / kind / (digest + '.json.gz')
        if path.is_file():
            return path
    raise FileNotFoundError(f'{kind}/{digest} absent from {list(map(str, roots))}')


def packed_history(roots, digest):
    path = locate(roots, 'histories', digest)
    value = read_gzip(path)
    if hash_json(value) != digest:
        raise ValueError(f'History canonical address mismatch: {path}')
    return value


def validate_envelope(packed, provenance, expected_hash=None):
    ids = provenance['observation_ids']
    if len(ids) != len(packed['observation_refs']) or not all(isinstance(x, str) and x for x in ids):
        raise ValueError('Acquisition identities do not align with saved observations')
    if 'observation_refs' in provenance and provenance['observation_refs'] != packed['observation_refs']:
        raise ValueError('Provenance observation references changed')
    digest = hash_json(provenance)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError('Acquisition identity envelope hash mismatch')
    return digest


def dependency_keys(roots, packed):
    roots = tuple(map(str, roots))
    return [(roots, 'graphs', packed['graph_ref']), (roots, 'tasks', packed['task_ref'])] + [
        (roots, 'observations', ref) for ref in packed['observation_refs']]


def inspect_blob(item):
    roots, kind, digest = item
    try:
        path = locate(roots, kind, digest)
        before = path.stat()
        value = read_gzip(path) if kind != 'observations' else None
        if value is not None and hash_json(value) != digest:
            raise ValueError('Canonical metadata content address mismatch')
        compressed_sha = file_sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError('Blob changed during export')
        row = dict(state='passed', kind=kind, content_address=digest,
            resolution_roots=list(roots), resolved_path=str(path), compressed_bytes=after.st_size,
            compressed_sha256=compressed_sha, canonical_content_verified=value is not None,
            verification='canonical JSON plus compressed SHA256' if value is not None else
                'existing content-address locator plus compressed SHA256; raw JSON not reparsed')
        return row, value
    except Exception as exc:
        return dict(state='failed', kind=kind, content_address=digest,
            resolution_roots=list(roots), error=f'{type(exc).__name__}:{exc}'), None


def main_prefixes(packed, provenance, trace):
    """Project saved chains; copy identity strings, never assign measurement IDs.

    Prefix content addresses are new *storage* addresses. Intermediate Plan-Once
    trace predictions can be stale by design; diagnosis_updated is not altered.
    """
    validate_envelope(packed, provenance)
    if not packed['observation_refs']:
        raise ValueError('Normal rollout has no initial observation')
    count = 1
    remaining = dict(packed['initial_budget'])
    actions = []

    def prefix(index, source, recorded=None):
        history = dict(packed, observation_refs=packed['observation_refs'][:count],
            remaining=dict(remaining), action_keys=list(actions))
        envelope = dict(provenance, observation_ids=provenance['observation_ids'][:count])
        if 'observation_refs' in envelope:
            envelope['observation_refs'] = packed['observation_refs'][:count]
        return dict(prefix_ordinal=index, provenance_origin=source,
            history_hash=hash_json(history), packed=history,
            provenance_hash=hash_json(envelope), provenance=envelope,
            recorded_prediction=None if recorded is None else recorded.get('prediction'),
            diagnosis_updated=None if recorded is None else recorded.get('diagnosis_updated', False),
            initial_full_prediction_saved=False if index == 0 else None)

    yield prefix(0, 'slice_of_saved_final_envelope_and_initial_budget')
    for index, event in enumerate(trace, 1):
        if 'resources_before' in event and event['resources_before'] != remaining:
            raise ValueError('Recorded decision resources do not form a chain')
        if 'evidence_hash' in event:
            if count >= len(packed['observation_refs']) or event['evidence_hash'] != packed['observation_refs'][count]:
                raise ValueError('Trace observation is not the next saved acquisition')
            if event.get('measurement_acquisition_id') != provenance['observation_ids'][count]:
                raise ValueError('Trace acquisition identity differs from saved provenance')
            count += 1
            actions.append(event['action']['action_key'])
        if 'resources_after' in event:
            remaining = dict(event['resources_after'])
        yield prefix(index, 'slice_of_saved_final_envelope_and_recorded_decision', event)
    if count != len(packed['observation_refs']) or actions != packed['action_keys'] or remaining != packed['remaining']:
        raise ValueError('Trace does not resolve to the exact final public history')


def _old_shard(arguments):
    run, shard = arguments
    path = Path(shard['path'])
    if file_sha256(path) != shard['sha256']:
        raise ValueError('Saved replay job shard changed')
    jobs = read_gzip(path)
    prediction_path = run / 'replay/predictions_structured' / path.name
    if not prediction_path.is_file():
        prediction_path = run / 'replay/predictions_all' / path.name
    predictions = read_gzip(prediction_path)
    if predictions['state'] != 'passed' or predictions['input_sha256'] != shard['sha256']:
        raise ValueError('Saved structured replay prediction provenance changed')
    values = {value['job_id']: value for value in predictions['predictions']}
    if len(values) != len(jobs) or set(values) != {job['job_id'] for job in jobs}:
        raise ValueError('Replay job/prediction coverage differs')
    records, dependencies, failures = [], set(), []
    for job in jobs:
        try:
            roots = (job['store_root'],)
            packed = job['packed'] if job['packed'] is not None else packed_history(roots, job['history_hash'])
            if hash_json(packed) != job['history_hash']:
                raise ValueError('Inline prefix history canonical address mismatch')
            value = values[job['job_id']]
            envelope = {'observation_ids': value['observation_ids']}
            digest = validate_envelope(packed, envelope, value['provenance_hash'])
            if value['observation_refs'] != packed['observation_refs']:
                raise ValueError('Replay input observation references changed')
            records.append(dict(job_id=job['job_id'], case_id=job['case_id'],
                history_hash=job['history_hash'], packed=packed, provenance=envelope,
                provenance_hash=digest, resolution_roots=list(roots),
                inline_diagnostic_prefix=job['packed'] is not None,
                identity_source=dict(path=str(prediction_path), job_id=job['job_id'])))
            dependencies.update(dependency_keys(roots, packed))
        except Exception as exc:
            failures.append(dict(job_id=job['job_id'], history_hash=job['history_hash'],
                error=f'{type(exc).__name__}:{exc}'))
    return records, dependencies, failures, dict(path=str(prediction_path), sha256=file_sha256(prediction_path))


def _main_cell(arguments):
    source, run, path, frozen_hash = arguments
    item = read_json(path)
    row, trace = item['row'], item['trace']
    if item['state'] != 'passed' or item['frozen_hash'] != frozen_hash:
        raise ValueError('Main result does not match frozen models')
    if item['result_hash'] != hash_json(dict(row=row, trace=trace)):
        raise ValueError('Main row/trace hash mismatch')
    cell = {key: row[key] for key in CELL_FIELDS}
    cell['cell_id'] = path.stem
    if hash_json([row['method'], row['policy_seed'], row['execution_cap'], row['byte_fraction'], row['case_id']]) != path.stem:
        raise ValueError('Main result cell key mismatch')
    trace_record = dict(cell, source_result_path=str(path), result_hash=item['result_hash'],
        row=row, trace=trace, final_history_hash=row.get('evidence_trace_hash'),
        final_provenance_hash=row.get('event_provenance_hash'))
    if not row.get('evidence_trace_hash'):
        if not row.get('worker_failure') or not row.get('resource_accounting_unavailable') or trace or row.get('event_provenance_hash'):
            raise ValueError('Missing history is not a recorded outer failure')
        trace_record['history_status'] = 'recorded_outer_failure_no_saved_history'
        return [], set(), trace_record
    roots = (str(run), str(source))
    packed = packed_history(roots, row['evidence_trace_hash'])
    key = _address(row['event_provenance_hash'])
    provenance_path = run / 'public/event_provenance' / row['evidence_trace_hash'] / (key + '.json')
    provenance = read_json(provenance_path)
    validate_envelope(packed, provenance, key)
    records = [dict(cell, prefix_ordinal=-1, is_official_final=True,
        history_hash=row['evidence_trace_hash'], packed=packed, provenance_hash=key,
        provenance=provenance, identity_source=str(provenance_path),
        resolution_roots=list(roots), recorded_prediction=row['prediction'])]
    records.extend(dict(cell, is_official_final=False, resolution_roots=list(roots), **part)
        for part in main_prefixes(packed, provenance, trace))
    trace_record['history_status'] = 'saved_and_resolved'
    return records, set(dependency_keys(roots, packed)), trace_record


def _bounded_map(pool, function, items, batch_size):
    iterator = iter(items)
    while True:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(iterator))
            except StopIteration:
                break
        if not batch:
            return
        yield from pool.map(function, batch)


def _safe_main_cell(arguments):
    try:
        return _main_cell(arguments), None
    except Exception as exc:
        return None, dict(cell_id=arguments[2].stem, path=str(arguments[2]), error=f'{type(exc).__name__}:{exc}')


def export(source_root, run_dir, cohort, workers=12):
    source, run = Path(source_root), Path(run_dir)
    out = run / 'evidence_export' / cohort
    out.mkdir(parents=True, exist_ok=True)
    complete_path = out / 'complete.json'
    implementation = file_sha256(__file__)
    if complete_path.exists():
        previous = read_json(complete_path)
        if previous['state'] != 'passed' or previous['implementation_sha256'] != implementation:
            raise ValueError('Preserving previous export; failed or different implementation needs a separate version')
        for name, digest in previous['output_hashes'].items():
            if file_sha256(out / name) != digest:
                raise ValueError('Completed evidence export changed')
        return previous
    started = time.monotonic()
    counts = Counter()
    dependencies = set()
    input_files = {}
    identities = set()
    if cohort == 'old_histories':
        inventory = read_json(run / 'replay/inventory.json')
        complete = read_json(run / 'replay/complete.json')
        if complete['state'] != 'passed':
            raise ValueError('Reader replay must be complete')
        input_files.update({str(run / 'replay' / name): file_sha256(run / 'replay' / name)
            for name in ('inventory.json', 'complete.json', 'contexts.parquet')})
        if input_files[str(run / 'replay/contexts.parquet')] != inventory['contexts_sha256']:
            raise ValueError('Replay contexts changed')
        expected = inventory['all_prediction_jobs']
        planned_contexts = inventory['contexts']
    elif cohort == 'new_main':
        complete = read_json(run / 'evaluation/main/complete.json')
        frozen = read_json(run / 'models/frozen.json')
        if complete['state'] != 'passed' or complete['rows'] != EXPECTED_MAIN or complete['frozen_hash'] != hash_json(frozen):
            raise ValueError('Full frozen main matrix must be complete')
        paths = sorted((run / 'evaluation/main/cases').glob('*.json'))
        if len(paths) != EXPECTED_MAIN:
            raise ValueError('Main cell file coverage changed')
        expected = planned_contexts = EXPECTED_MAIN
        input_files.update({str(run / name): file_sha256(run / name) for name in
            ('evaluation/main/complete.json', 'evaluation/main/per_case_results.parquet', 'models/frozen.json')})
    else:
        raise ValueError(cohort)
    with ExitStack() as stack:
        writers = {name: stack.enter_context(JsonlWriter(out / (name + '.jsonl.gz')))
            for name in ('histories', 'traces', 'blobs', 'metadata', 'failures')}
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=workers))
        if cohort == 'old_histories':
            tasks = ((run, shard) for shard in inventory['shards'])
            for records, refs, failures, prediction_file in _bounded_map(pool, _old_shard, tasks, workers * 2):
                dependencies.update(refs)
                for record in records:
                    if record['job_id'] in identities:
                        raise ValueError('Duplicate exported replay job')
                    identities.add(record['job_id'])
                    counts['inline_diagnostic_prefixes'] += int(record['inline_diagnostic_prefix'])
                    writers['histories'].write(record)
                for failure in failures:
                    writers['failures'].write(failure)
                counts['failed_contexts'] += len(failures)
                counts['contexts_processed'] += len(records) + len(failures)
                input_files[prediction_file['path']] = prediction_file['sha256']
                if counts['contexts_processed'] % 4096 < 128:
                    progress(out, counts, started, 'histories', expected)
        else:
            tasks = ((source, run, path, complete['frozen_hash']) for path in paths)
            for value, failure in _bounded_map(pool, _safe_main_cell, tasks, workers * 4):
                counts['contexts_processed'] += 1
                if failure is not None:
                    writers['failures'].write(failure)
                    counts['failed_contexts'] += 1
                    continue
                records, refs, trace = value
                dependencies.update(refs)
                if trace['cell_id'] in identities:
                    raise ValueError('Duplicate main cell identity')
                identities.add(trace['cell_id'])
                writers['traces'].write(trace)
                counts['recorded_decisions'] += len(trace['trace'])
                counts['recorded_outer_failures_no_history'] += int(not records)
                for record in records:
                    writers['histories'].write(record)
                    counts['official_final_histories'] += int(record['is_official_final'])
                    counts['projected_prefix_histories'] += int(not record['is_official_final'])
                if counts['contexts_processed'] % 4096 == 0:
                    progress(out, counts, started, 'histories', expected)
        if counts['contexts_processed'] != expected:
            raise ValueError('Evidence export denominator changed')
        for row, value in _bounded_map(pool, inspect_blob, sorted(dependencies), workers * 4):
            writers['blobs'].write(row)
            counts['blob_locators'] += 1
            if row['state'] == 'failed':
                writers['failures'].write(row)
                counts['failed_blob_locators'] += 1
            else:
                counts['verified_compressed_bytes'] += row['compressed_bytes']
                counts['canonical_metadata_blobs'] += int(row['canonical_content_verified'])
                counts['external_observation_blobs'] += int(row['kind'] == 'observations')
                if value is not None:
                    writers['metadata'].write(dict(kind=row['kind'], content_address=row['content_address'], value=value))
            if counts['blob_locators'] % 4096 == 0:
                progress(out, counts, started, 'blob_integrity', len(dependencies))
    result = dict(state='failed' if counts['failed_contexts'] or counts['failed_blob_locators'] else 'passed',
        version=VERSION, cohort=cohort, created_at=now(), implementation_sha256=implementation,
        expected_contexts=expected, planned_prediction_contexts=planned_contexts,
        all_contexts_retained=counts['contexts_processed'] == expected,
        all_dependency_locators_verified=not counts['failed_blob_locators'],
        no_new_measurement_identities=True, no_reader_calls=True, physical_acquisitions=0,
        raw_observations_in_export=False,
        raw_integrity_scope='Every resolved observation has existence/size/compressed SHA256; canonical JSON validated for packed histories, provenance, graphs and tasks. Raw observation JSON is not reparsed.',
        prefix_scope='Public history projected from saved final observation order, saved IDs and recorded decision resources; stored predictions are retained, never recomputed. Initial new-main full prediction was not persisted by the original rollout and is explicitly unavailable.',
        counts=dict(counts), output_rows={name: writer.rows for name, writer in writers.items()},
        input_hashes=input_files, output_hashes={name + '.jsonl.gz': file_sha256(out / (name + '.jsonl.gz')) for name in writers},
        elapsed_seconds=time.monotonic()-started, workers=workers)
    atomic_json(complete_path, result)
    if result['state'] != 'passed':
        raise ValueError('Evidence export incomplete; every failure is retained in failures.jsonl.gz')
    return result


def progress(out, counts, started, stage, expected):
    value = dict(state='running', stage=stage, expected=expected, counts=dict(counts),
        elapsed_seconds=time.monotonic()-started, updated_at=now())
    atomic_json(out / 'progress.json', value)
    print(json.dumps(value), flush=True)


def engineering_controls(run_dir):
    checks = []
    packed = dict(graph_ref='a'*64, task_ref='b'*64, observation_refs=['c'*64, 'd'*64],
        initial_budget=dict(executions=4, bytes=100, calibrations=2, decisions=6),
        remaining=dict(executions=3, bytes=90, calibrations=2, decisions=5),
        action_keys=['inspect'], requirements={})
    provenance = dict(observation_ids=['same-numeric-values-event-1', 'same-numeric-values-event-2'],
        observation_refs=packed['observation_refs'], identity_origin='existing_fixture')
    trace = [dict(action=dict(action_key='inspect'), evidence_hash='d'*64,
        measurement_acquisition_id=provenance['observation_ids'][1], resources_before=packed['initial_budget'],
        resources_after=packed['remaining'], prediction={'recorded': True}, diagnosis_updated=False),
        dict(action=dict(action_key='stop'), prediction={'recorded': True}, stop_reason='policy_stop')]
    prefixes = list(main_prefixes(packed, provenance, trace))
    assert prefixes[-1]['history_hash'] == hash_json(packed)
    assert prefixes[-1]['provenance_hash'] == hash_json(provenance)
    assert prefixes[1]['diagnosis_updated'] is False and prefixes[1]['recorded_prediction'] == {'recorded': True}
    assert prefixes[0]['recorded_prediction'] is None
    assert prefixes[0]['provenance']['observation_ids'] == provenance['observation_ids'][:1]
    checks.extend(['exact_final_chain', 'identity_strings_preserved', 'stale_plan_prediction_labeled', 'initial_prediction_unavailable_explicit'])
    broken = json.loads(json.dumps(trace));broken[0]['measurement_acquisition_id'] = 'invented'
    try:
        list(main_prefixes(packed, provenance, broken))
    except ValueError:
        checks.append('identity_mismatch_rejected')
    else:
        raise AssertionError('Changed acquisition ID accepted')
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); value = {'metadata': [1, 2]}; digest = hash_json(value)
        path = root / 'public/graphs' / (digest + '.json.gz');path.parent.mkdir(parents=True)
        with gzip.open(path, 'wt') as handle:
            json.dump(value, handle)
        row, copied = inspect_blob(((str(root),), 'graphs', digest))
        assert row['state'] == 'passed' and copied == value and row['canonical_content_verified']
        assert inspect_blob(((str(root),), 'graphs', '0'*64))[0]['state'] == 'failed'
        with gzip.open(path, 'wt') as handle:
            json.dump({'changed': True}, handle)
        assert inspect_blob(((str(root),), 'graphs', digest))[0]['state'] == 'failed'
        checks.extend(['canonical_metadata_verified', 'missing_dependency_retained', 'changed_metadata_rejected'])
    result = dict(state='passed', checks=checks, count=len(checks), engineering_fixtures_only=True,
        no_reader_calls=True, physical_acquisitions=0, implementation_sha256=file_sha256(__file__), completed_at=now())
    atomic_json(Path(run_dir) / 'tests/evidence_export_engineering.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['old_histories', 'new_main', 'controls'])
    parser.add_argument('--source-root')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    result = engineering_controls(args.run_dir) if args.command == 'controls' else export(
        args.source_root, args.run_dir, args.command, args.workers)
    print(json.dumps(result, sort_keys=True), flush=True)
