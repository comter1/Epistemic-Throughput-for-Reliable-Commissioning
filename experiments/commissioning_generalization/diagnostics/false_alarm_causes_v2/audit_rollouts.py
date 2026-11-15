"""Exhaustive CPU-only actual-history, legality and paired-cache audit."""
import argparse
from pathlib import Path
import pickle

import numpy as np

from cgx.data.store import Store
from cgx.models.features import FEATURE_VERSION
from cgx.observation.actions import action_catalog, feasible_actions, initial_budget, charge
from cgx.provenance import atomic_json, hash_json, read_json
from diagnostics.false_alarm_causes.interventions import load_blob
from .confirmation import audit_trajectory
from .rollouts import METHODS, SEEDS, read_cases, feature_code_hash


def audit_live_structure(item, expected_reader):
    row, prefixes, trace = item['row'], item['prefixes'], item['trace']
    assert row['state'] == 'passed' and row['arm'] == 'R-live'
    assert row['selected_reader_variant'] == expected_reader
    assert len(prefixes) == len(trace) + 1
    assert [p['prefix_ordinal'] for p in prefixes] == list(range(len(prefixes)))
    assert prefixes[-1]['history_hash'] == row['evidence_trace_hash']
    for prefix in prefixes:
        assert prefix['prediction_reader'] == prefix['deployed_q_reader'] == expected_reader
        assert prefix['gate_calibration_q_reader'] == 'original_FINAL'
        assert prefix['gate_head_device'] == 'cpu'
        assert all(type(v) is int and v >= 0 for v in prefix['remaining'].values())
        if prefix['actionable']:
            chosen = int(np.argmax(prefix['original_scores']))
            assert chosen == prefix['chosen_action_index']
            assert prefix['candidate_actions'][chosen] == prefix['chosen_action']
            assert len(prefix['chosen_head_input_deployed']) == 281
            assert np.isfinite(prefix['selected_gate_score'])
        q = prefix['prediction']
        assert 'healthy' in q['target_keys'] and abs(np.exp2(q['log_probabilities_base2']).sum() - 1) < 1e-5
    for i, event in enumerate(trace):
        assert event['action'] == prefixes[i]['chosen_action']
        assert event['history_before'] == prefixes[i]['history_hash']
        assert event['history_after'] == prefixes[i + 1]['history_hash']
        assert charge(prefixes[i]['remaining'], event['actual_cost']) == prefixes[i + 1]['remaining']
        assert all(event['actual_cost'][k] <= event['action']['cost_bound'][k] for k in event['actual_cost'])
    return dict(passed=True, prefixes=len(prefixes), acquired_actions=len(trace))


def run(run_dir, phase='source', case_limit=None):
    run = Path(run_dir)
    cases = read_cases(run / 'manifests' / ('source_cases.json' if phase == 'source' else 'fresh_cases.json'))
    if case_limit is not None:
        cases = cases[:case_limit]
    store = Store(run)
    sealed_tasks = read_json(run / 'manifests' / ('source_tasks.json' if phase == 'source' else 'fresh_tasks.json'))
    histories, features, acquisitions, observations = set(), set(), {}, set()
    new_acquisitions = {}
    selected_reader = read_json(run / 'manifests/source_variant_selection.json')['selected_reader_variant'] if phase == 'live' else None
    catalogs, tasks = {}, {}
    counts = dict(trajectories=0, acquired_actions=0, prefixes=0, calibrated_actions=0,
                  source_train_trajectories=0, source_validation_trajectories=0)
    public_fields = {'graph_ref', 'observation_refs', 'task_ref', 'requirements', 'remaining', 'initial_budget', 'action_keys'}
    for case in cases:
        for method in METHODS:
            for seed in SEEDS:
                path = run / 'trajectories' / phase / 'cases' / f"{hash_json([case['case_id'], method, seed])}.json.gz"
                item = load_blob(path)
                assert item['result_hash'] == hash_json({k: v for k, v in item.items() if k != 'result_hash'})
                result = audit_live_structure(item, selected_reader) if phase == 'live' else audit_trajectory(item)
                assert result['passed'], (case['case_id'], method, seed, result)
                counts['trajectories'] += 1
                counts['prefixes'] += result['prefixes']
                counts['acquired_actions'] += result['acquired_actions']
                if phase == 'source':
                    assert item['row']['source_split'] == case['split']
                    assert item['row']['source_fold'] == case['source_fold']
                    assert item['row']['oof_reader_name'] == f"crossfit_{case['source_fold']}"
                    counts['source_' + case['split'] + '_trajectories'] += 1
                if phase == 'live':
                    a0_path = run / 'trajectories/confirmation/cases' / path.name
                    assert item['initial_observation_ref'] == load_blob(a0_path)['initial_observation_ref']
                previous = None
                for index, prefix in enumerate(item['prefixes']):
                    digest = prefix['history_hash']
                    packed = store.history_references(digest)
                    assert set(packed) == public_fields
                    assert packed['remaining'] == prefix['remaining']
                    assert len(packed['observation_refs']) == prefix['observation_count'] == index + 1
                    assert len(packed['action_keys']) == index
                    assert packed['observation_refs'][0] == item['initial_observation_ref']
                    if previous is not None:
                        event = item['trace'][index - 1]
                        assert previous['observation_refs'] == packed['observation_refs'][:-1]
                        assert packed['observation_refs'][-1] == event['observation_ref']
                        assert packed['action_keys'][-1] == event['action']['action_key']
                        executions = prefix['executions_used']
                        calibrations = prefix['calibrations_used']
                        task = tasks.setdefault(packed['task_ref'], store._get('tasks', packed['task_ref']))
                        assert task == sealed_tasks[case['case_id']]
                        expected = hash_json([case, task, event['action'], executions, calibrations, 'A'])
                        assert event['acquisition_identity'] == expected
                        acquisitions[expected] = event['action']
                        if not event['observation_reused']:
                            new_acquisitions[expected] = event['action']
                        counts['calibrated_actions'] += int(bool(event['action']['calibration_pair']))
                    if packed['graph_ref'] not in catalogs:
                        graph = store._get('graphs', packed['graph_ref'])
                        catalogs[packed['graph_ref']] = action_catalog(graph)
                        assert packed['initial_budget'] == initial_budget(graph, 1., 4)
                    assert prefix['candidate_actions'] == feasible_actions(catalogs[packed['graph_ref']], prefix['remaining'])
                    if digest not in features:
                        with Path(prefix['feature_cache_path']).open('rb') as handle:
                            saved = pickle.load(handle)
                        assert saved['history_hash'] == digest
                        assert saved['feature_version'] == FEATURE_VERSION
                        assert saved['scientific_code_hash'] == feature_code_hash()
                        assert np.count_nonzero(saved['features'].nodes[:, 26:29]) == 0
                        features.add(digest)
                    observations.update(packed['observation_refs'])
                    histories.add(digest)
                    previous = packed
    assert counts['trajectories'] == len(cases) * 6
    result = dict(state='passed', phase=phase, profile_subset=case_limit is not None,
                  **counts, distinct_cases=len(cases), unique_histories=len(histories),
                  unique_features=len(features), unique_observation_contents=len(observations),
                  reader_prediction_requests=counts['prefixes'],
                  avoided_reader_forwards_by_exact_history_reuse=counts['prefixes'] - len(histories),
                  unique_physical_acquisition_identities=len(acquisitions),
                  unique_additional_execution_realizations=sum(bool(a['ports']) for a in acquisitions.values()),
                  unique_calibration_realizations=sum(bool(a['calibration_pair']) for a in acquisitions.values()),
                  new_additional_execution_realizations=sum(bool(a['ports']) for a in new_acquisitions.values()),
                  new_calibration_realizations=sum(bool(a['calibration_pair']) for a in new_acquisitions.values()),
                  every_candidate_legal_and_in_original_order=True, every_raw_feature_q_input_empty=True,
                  every_acquired_observation_exact_next_prefix=True, acquisition_identity_includes_case_action_ordinals_and_streams=True,
                  public_history_contains_no_truth=True, all_failures_preserved_or_stage_fails=True,
                  nominal_tasks_match_sealed_manifest=True,
                  live_initial_observations_identical_to_A0=phase == 'live',
                  live_reader_rule=selected_reader,
                  full_model_forwards=0, new_feature_builds=0, new_physical_executions=0)
    suffix = '_profile' if case_limit is not None else ''
    atomic_json(run / 'tests' / f'{phase}_rollout_audit{suffix}.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--phase', choices=('source', 'confirmation', 'live'), default='source')
    parser.add_argument('--case-limit', type=int)
    args = parser.parse_args()
    run(args.run_dir, args.phase, args.case_limit)
