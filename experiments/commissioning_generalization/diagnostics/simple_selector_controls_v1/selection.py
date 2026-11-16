"""Select only from the complete registered source-validation matrix, then seal.

Completed-record predictions are paired reporting outputs, never a selection
criterion. This module opens no new test outcomes and performs no acquisition.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from cgx.config import load_config
from cgx.data.store import Store
from cgx.provenance import atomic_json, file_sha256, hash_json, now, read_json

CAPS = (1, 2, 4)
TEMPLATES = tuple(f'F{i}' for i in range(8))
STOPS = {'rule_none': None, 'rule_0.99': .99, 'rule_0.999': .999}
CANDIDATES = TEMPLATES + tuple(STOPS)
FAMILIES = ('healthy', 'cache_stale', 'queue_hold', 'adapter_scale')
EXPECTED_ROWS = 512 * 3 * 11
RUNTIME_FILES = ('policies.py', 'rollout.py', 'evaluation.py', 'storage.py', 'cache.py')
SCIENCE_FILES = (*RUNTIME_FILES, 'selection.py')
CRITERION = ('configuration_balanced_mean_deployed_final_nll_bits', 'fewer_bytes',
             'fewer_executions', 'fewer_calibrations', 'fixed_candidate_order')
RANK_COLUMNS = ('mean_deployed_final_nll_bits', 'mean_bytes_used',
                'mean_executions_used', 'mean_calibrations_used')
SOURCE_OUTPUTS = ('validation_per_case_online.parquet',
                  'validation_per_case_completed_read.parquet',
                  'validation_acquired_history_index.parquet',
                  'validation_action_traces.jsonl.gz')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_nonnegative(frame, columns):
    for column in columns:
        values = pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float)
        require(bool(np.isfinite(values).all() and (values >= 0).all()),
                f'{column}: missing/nonfinite/negative values cannot be dropped or counted as zero')


def _nullable(value):
    return None if pd.isna(value) else value


def validate_matrix(online, completed, source_cases, split_ids, reader_hash, config_hash):
    """Check all denominators and pairs without consulting a test outcome."""
    case_map = {c['case_id']: c for c in source_cases}
    require(len(source_cases) == len(case_map) == 512, 'Source validation must contain exactly512 distinct cases')
    configs = Counter(c['configuration_id'] for c in source_cases)
    require(len(configs) == 8 and set(configs.values()) == {64}, 'Expected8 validation configurations of64 cases')
    for configuration in configs:
        families = Counter(c['fault_family'] for c in source_cases if c['configuration_id'] == configuration)
        require(set(families) == set(FAMILIES) and set(families.values()) == {16},
                f'Validation family balance differs: {configuration}')
    validation_ids = set(case_map)
    require(validation_ids == set(split_ids['validation']), 'Validation IDs differ from the predeclared split manifest')
    for split, ids in split_ids.items():
        if split != 'validation':
            require(not validation_ids.intersection(ids), f'Validation overlaps{split}')
    expected = {(case, cap, candidate) for case in case_map for cap in CAPS for candidate in CANDIDATES}
    columns = {'case_id', 'configuration_id', 'suite', 'fault_family', 'execution_cap', 'candidate_id',
               'method', 'template_id', 'stop_threshold', 'policy_seed', 'cell_key', 'reader_hash',
               'protocol_hash', 'final_nll_bits', 'bytes_used', 'executions_used', 'calibrations_used',
               'correct_joint', 'healthy_false_alarm', 'fault_miss', 'timeout', 'abstained',
               'worker_failure', 'resource_accounting_unavailable', 'result_mode', 'rule_hash',
               'evidence_trace_hash', 'event_provenance_hash', 'execution_source_key'}
    for mode, frame in (('online', online), ('completed', completed)):
        require(columns.issubset(frame.columns), f'{mode}: missing columns{sorted(columns - set(frame.columns))}')
        require(len(frame) == EXPECTED_ROWS, f'{mode}: expected{EXPECTED_ROWS} logical rows; got{len(frame)}')
        require(not frame.duplicated(['case_id', 'execution_cap', 'candidate_id']).any(), f'{mode}: duplicate logical cell')
        require(not frame.cell_key.duplicated().any(), f'{mode}: duplicate cell key')
        require(set(frame[['case_id', 'execution_cap', 'candidate_id']].itertuples(index=False, name=None)) == expected,
                f'{mode}: missing, extra, or foreign validation cell')
        require(set(frame.suite) == {'validation'} and set(frame.result_mode) == {mode}, f'{mode}: unexpected split/mode')
        require(frame.policy_seed.isna().all(), f'{mode}: deterministic candidates must have null seeds')
        require(set(frame.reader_hash) == {reader_hash}, f'{mode}: reader hash mismatch')
        require(set(frame.protocol_hash) == {config_hash}, f'{mode}: executable config hash mismatch')
        require(frame.configuration_id.eq(frame.case_id.map(lambda key: case_map[key]['configuration_id'])).all(),
                f'{mode}: source configuration assignment changed')
        require(frame.fault_family.eq(frame.case_id.map(lambda key: case_map[key]['fault_family'])).all(),
                f'{mode}: source family assignment changed')
        _finite_nonnegative(frame, ('final_nll_bits',))
        for column in ('bytes_used', 'executions_used', 'calibrations_used'):
            missing = frame[column].isna()
            require((~missing | frame.resource_accounting_unavailable).all(), f'{mode}: undeclared missing resources')
            _finite_nonnegative(frame.loc[~missing], (column,))
        require((frame.executions_used.isna() | frame.executions_used.le(frame.execution_cap)).all()
                and (frame.calibrations_used.isna() | frame.calibrations_used.le(2)).all(),
                f'{mode}: declared resource cap exceeded')
        for candidate, group in frame.groupby('candidate_id', sort=False):
            is_template = candidate in TEMPLATES
            method = 'fixed_template_validation' if is_template else 'interface_rule_se'
            require(set(group.method) == {method}, f'{mode}: candidate method mismatch{candidate}')
            require(set(map(_nullable, group.template_id)) == ({candidate} if is_template else {None}),
                    f'{mode}: template label mismatch{candidate}')
            require(set(map(_nullable, group.stop_threshold)) == ({None} if is_template else {STOPS[candidate]}),
                    f'{mode}: stopping option mismatch{candidate}')
    left = online.set_index('cell_key').sort_index()
    right = completed.set_index('cell_key').sort_index()
    for column in ('case_id', 'configuration_id', 'suite', 'fault_family', 'execution_cap', 'candidate_id',
                   'method', 'template_id', 'stop_threshold', 'reader_hash', 'protocol_hash', 'rule_hash',
                   'evidence_trace_hash', 'event_provenance_hash', 'execution_source_key',
                   'bytes_used', 'executions_used', 'calibrations_used'):
        require(left[column].equals(right[column]), f'Completed-read pairing changed{column}')
    require('completion_status' in completed, 'Missing completed-read outcome status')
    require(set(completed.completion_status).issubset({'reused_verified_online', 'completed_same_history', 'unresolved_kept_online'}),
            'Unknown completed-read status')
    return dict(state='passed', rows_per_mode=EXPECTED_ROWS, source_cases=512, source_configurations=8,
                cases_per_configuration=64, families_per_configuration={family: 16 for family in FAMILIES},
                execution_caps=list(CAPS), candidates=list(CANDIDATES), deterministic_seed=None,
                all_failed_timeout_abstained_rows_retained=True, split_disjoint=True,
                completed_read_paired_same_history=True, new_test_outcomes_opened=False)


def configuration_balanced_mean(frame, column):
    """Equal configurations; all cases including errors remain in each mean."""
    values = [math.fsum(group[column].astype(float)) / len(group)
              for _, group in frame.groupby('configuration_id', sort=True)]
    require(bool(values), 'Cannot average an empty denominator')
    return math.fsum(values) / len(values)


def summarize_candidates(online, completed):
    merged = online.merge(completed[['cell_key', 'final_nll_bits', 'correct_joint', 'healthy_false_alarm',
                                    'fault_miss', 'completion_status']], on='cell_key', validate='one_to_one',
                          suffixes=('', '_completed'))
    merged['completed_unresolved'] = merged.completion_status.eq('unresolved_kept_online')
    def metrics(group, by_configuration):
        average = configuration_balanced_mean if by_configuration else lambda frame, column: math.fsum(frame[column].astype(float)) / len(frame)
        result = dict(rows=len(group), configurations=group.configuration_id.nunique(),
                      mean_deployed_final_nll_bits=average(group, 'final_nll_bits'),
                      mean_completed_final_nll_bits=average(group, 'final_nll_bits_completed'),
                      mean_bytes_used=average(group, 'bytes_used'),
                      mean_executions_used=average(group, 'executions_used'),
                      mean_calibrations_used=average(group, 'calibrations_used'),
                      deployed_accuracy=average(group, 'correct_joint'),
                      completed_accuracy=average(group, 'correct_joint_completed'),
                      timeout_fraction=average(group, 'timeout'), abstention_fraction=average(group, 'abstained'),
                      worker_failure_rows=int(group.worker_failure.sum()), completed_unresolved_rows=int(group.completed_unresolved.sum()),
                      unknown_resource_rows=int(group.resource_accounting_unavailable.sum()),
                      distinct_execution_sources=group.execution_source_key.nunique())
        for name, subset in (('healthy_false_alarm', group[group.fault_family == 'healthy']),
                             ('fault_miss', group[group.fault_family != 'healthy'])):
            result[f'deployed_{name}_rate'] = average(subset, name)
            result[f'completed_{name}_rate'] = average(subset, name + '_completed')
        return result
    summaries = []; config_summaries = []
    for cap in CAPS:
        for candidate in CANDIDATES:
            group = merged[(merged.execution_cap == cap) & (merged.candidate_id == candidate)]
            common = dict(execution_cap=cap, candidate_id=candidate,
                          method='fixed_template_validation' if candidate in TEMPLATES else 'interface_rule_se')
            summaries.append({**common, **metrics(group, True)})
            for configuration, values in group.groupby('configuration_id', sort=True):
                config_summaries.append({**common, 'configuration_id': configuration, **metrics(values, False)})
    return pd.DataFrame(summaries), pd.DataFrame(config_summaries)


def choose_settings(summary):
    """Exact lexicographic order; no after-the-fact tolerance or completed NLL."""
    selected_templates = {}; selected_stops = {}; winners = []; ranked = summary.copy()
    ranked['candidate_rank'] = 0; ranked['selected'] = False
    for cap in CAPS:
        for candidates, destination in ((TEMPLATES, selected_templates), (tuple(STOPS), selected_stops)):
            group = ranked[(ranked.execution_cap == cap) & ranked.candidate_id.isin(candidates)]
            require(set(group.candidate_id) == set(candidates) and len(group) == len(candidates), 'Incomplete selection candidates')
            _finite_nonnegative(group, RANK_COLUMNS[:1])
            for column in RANK_COLUMNS[1:]:
                _finite_nonnegative(group.loc[group[column].notna()], (column,))
            order = {name: i for i, name in enumerate(candidates)}
            records = sorted(group.to_dict('records'), key=lambda row: tuple(
                math.inf if pd.isna(row[name]) else row[name] for name in RANK_COLUMNS) + (order[row['candidate_id']],))
            winner = records[0]
            destination[str(cap)] = winner['candidate_id'] if candidates == TEMPLATES else STOPS[winner['candidate_id']]
            for rank, row in enumerate(records, 1):
                mask = (ranked.execution_cap == cap) & (ranked.candidate_id == row['candidate_id'])
                ranked.loc[mask, 'candidate_rank'] = rank; ranked.loc[mask, 'selected'] = rank == 1
            winners.append({name: _nullable(winner[name]) for name in ('execution_cap', 'candidate_id', 'method', *RANK_COLUMNS,
                                                          'mean_completed_final_nll_bits')})
    settings = dict(state='sealed', selected_template_by_cap=selected_templates,
                    selected_stop_threshold_by_cap=selected_stops, criterion=list(CRITERION),
                    template_tie_order=list(TEMPLATES), stop_tie_order=list(STOPS),
                    numerical_tie_rule='exact equality; no post-outcome tolerance',
                    missing_resource_rule='all rows retained; any unknown cost makes that candidate mean unknown and ranks as positive infinity only at that resource tie-break',
                    aggregation='equal weight per configuration after averaging all64 cases in each configuration',
                    completed_record_nll_used_for_selection=False, selected_candidate_summaries=winners,
                    candidate_summary='validation/candidate_summary.csv',
                    candidate_configuration_summary='validation/candidate_configuration_summary.csv',
                    traceability='validation/selection_traceability.json')
    return settings, ranked


def _prerequisites(run):
    namespace = Path(__file__).parent
    protocol = read_json(run / 'protocol.json')
    inventory = read_json(run / 'source_inventory.json')
    design = read_json(run / 'provenance/validation_design_seal.json')
    marker = read_json(run / 'validation/phase_validation_complete.json')
    config_hash = hash_json(load_config())
    require(protocol['config_hash'] == inventory['executable_config_hash'] == config_hash, 'Executable config changed')
    require(protocol['reader_hash'] == inventory['reader_hash'] == marker['reader_hash'], 'Reader hash changed')
    require(protocol['source_validation_cases'] == 512 and protocol['source_validation_logical_rows'] == EXPECTED_ROWS,
            'Registered validation denominator changed')
    require(protocol['execution_budgets'] == list(CAPS) and tuple(protocol['fixed_templates']) == TEMPLATES,
            'Registered candidates/budgets changed')
    require(protocol['heuristic_stop_options'] == list(STOPS.values()), 'Registered stop order changed')
    require(protocol['selection_rule'] == list(CRITERION), 'Registered selection criterion changed')
    require(design['state'] == 'sealed_before_validation', 'No prior design seal')
    require(design['protocol_sha256'] == file_sha256(run / 'protocol.json'), 'Protocol changed since design seal')
    require(design['source_inventory_sha256'] == file_sha256(run / 'source_inventory.json'), 'Inventory changed since design seal')
    require(set(RUNTIME_FILES).issubset(design['code_hashes']), 'Missing runtime scientific design hashes')
    for name, digest in design['code_hashes'].items():
        require(file_sha256(namespace / name) == digest, f'Design scientific code changed: {name}')
    require(marker['state'] == 'passed' and marker['rows'] == marker['expected'] == EXPECTED_ROWS,
            'Full source validation completion required before any selection')
    require(marker['physical_cases'] == 512 and marker['all_planned_denominators_retained'], 'Incomplete validation denominator')
    for name in SOURCE_OUTPUTS:
        require(name in marker['outputs_sha256'], f'Missing validation output marker: {name}')
        require(file_sha256(run / 'validation' / name) == marker['outputs_sha256'][name], f'Validation output changed: {name}')
    # Test paths are inspected only for existence, never opened. Any new test
    # outcome before this seal is a protocol error, even if nobody has read it.
    require(not any((run / 'evaluation/cases').glob('*.json')), 'New test cells exist before selection seal')
    require(not any((run / 'evaluation').glob('*per_case*.parquet')), 'New test outcomes exist before selection seal')
    return protocol, inventory, design, marker


def seal_selection(run_dir):
    run = Path(run_dir).resolve(); directory = run / 'validation'
    output_names = ('candidate_summary.csv', 'candidate_configuration_summary.csv',
                    'selection_traceability.json', 'selected_settings.json')
    require(not (run / 'selection_manifest.json').exists(), 'Existing selection seal is immutable')
    require(not any((directory / name).exists() for name in output_names), 'Existing selection outputs are preserved; inspect before retry')
    protocol, inventory, design, marker = _prerequisites(run)
    online = pd.read_parquet(directory / SOURCE_OUTPUTS[0])
    completed = pd.read_parquet(directory / SOURCE_OUTPUTS[1])
    history_index = pd.read_parquet(directory / SOURCE_OUTPUTS[2])
    source_cases = list(Store(protocol['source_root']).cases('validation'))
    splits = read_json(run / 'provenance/split_case_ids.json')
    checks = validate_matrix(online, completed, source_cases, splits, protocol['reader_hash'], protocol['config_hash'])
    require(len(history_index) == EXPECTED_ROWS and history_index.cell_key.is_unique,
            'History index does not retain every logical validation cell')
    require(set(history_index.cell_key) == set(online.cell_key), 'History index differs from validation matrix')
    require(history_index.cell_file_sha256.str.fullmatch('[0-9a-f]{64}').all(), 'Missing per-cell file hashes')
    require(history_index.cell_path.map(lambda path: Path(path).resolve().parent == (directory / 'cases').resolve()).all(),
            'Validation traceability points outside its case directory')
    summary, per_config = summarize_candidates(online, completed)
    settings, summary = choose_settings(summary)
    summary.to_csv(directory / output_names[0], index=False)
    per_config.to_csv(directory / output_names[1], index=False)
    candidate_refs = []
    for (cap, candidate), group in online.groupby(['execution_cap', 'candidate_id'], sort=True):
        candidate_refs.append(dict(execution_cap=int(cap), candidate_id=candidate, rows=len(group),
            logical_cell_keys_sha256=hash_json(sorted(group.cell_key)),
            rule_hashes=sorted(set(group.rule_hash)),
            distinct_execution_sources=group.execution_source_key.nunique(),
            failed_rows=int(group.worker_failure.sum()), timeout_rows=int(group.timeout.sum()),
            abstained_rows=int(group.abstained.sum())))
    traceability = dict(state='passed', created_at=now(), checks=checks,
        source_validation_completion_sha256=file_sha256(directory / 'phase_validation_complete.json'),
        validation_design_seal_sha256=file_sha256(run / 'provenance/validation_design_seal.json'),
        split_case_ids_sha256=file_sha256(run / 'provenance/split_case_ids.json'),
        validation_case_ids_sha256=hash_json(sorted(c['case_id'] for c in source_cases)),
        runtime_identity_hash=marker['runtime_identity_hash'], inputs_sha256=marker['outputs_sha256'],
        candidate_results=candidate_refs,
        per_case_trace_resolution=dict(index='validation/' + SOURCE_OUTPUTS[2],
            fields=['cell_key', 'cell_path', 'cell_file_sha256', 'history_hash', 'provenance_hash',
                    'source_logical_cell_key', 'execution_source_key'],
            action_traces='validation/' + SOURCE_OUTPUTS[3]),
        original_rule_candidate='rule_none', original_rule_outcomes_preserved=True,
        inherited_interface_rule_catalog_tie='original max returns first catalog item on equal scores; node/catalog reordering can change tied choices',
        duplicated_plans_are_identity_aliases_not_independent_support=True,
        no_new_acquisition_or_training=True, new_test_outcomes_opened=False)
    atomic_json(directory / output_names[2], traceability)
    settings.update(source_validation_rows=EXPECTED_ROWS,
                    source_validation_completion_sha256=traceability['source_validation_completion_sha256'])
    atomic_json(directory / output_names[3], settings)
    manifest = dict(state='sealed', sealed_at=now(),
        selected_template_by_cap=settings['selected_template_by_cap'],
        selected_stop_threshold_by_cap=settings['selected_stop_threshold_by_cap'],
        code_hashes={name: file_sha256(Path(__file__).parent / name) for name in SCIENCE_FILES},
        config_hash=protocol['config_hash'], reader_hash=protocol['reader_hash'],
        protocol_sha256=file_sha256(run / 'protocol.json'), source_inventory_sha256=file_sha256(run / 'source_inventory.json'),
        validation_design_seal_sha256=traceability['validation_design_seal_sha256'],
        source_validation_completion_sha256=traceability['source_validation_completion_sha256'],
        validation_input_hashes={f'validation/{name}': digest for name, digest in marker['outputs_sha256'].items()},
        selection_output_hashes={f'validation/{name}': file_sha256(directory / name) for name in output_names},
        criterion=list(CRITERION), candidate_orders={'templates': list(TEMPLATES), 'stopping': list(STOPS)},
        completed_record_nll_used_for_selection=False, validation_checks=checks,
        new_test_cells_absent_at_seal=True, new_test_outcomes_opened=False)
    atomic_json(run / 'selection_manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(seal_selection(args.run_dir), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
