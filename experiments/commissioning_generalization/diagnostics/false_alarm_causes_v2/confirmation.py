"""Exact-prefix fresh comparisons and evaluator-only auditing for V2.

The baseline really acquires every observation. S1 and every gate stop only at
an actually available prefix; final-only reader variants reuse that same prefix.
No truth, outcome-based threshold, or best-prefix oracle enters a decision.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from cgx.provenance import atomic_json, file_sha256, hash_json, read_json
from cgx.observation.actions import charge
from diagnostics.false_alarm_causes.interventions import load_blob, prediction_metrics
from .head_data import plain
from .rollouts import METHODS, SEEDS, read_cases


def identity(row):
    return row['case_id'], row['method_key'], int(row['policy_seed'])


def predicted(q):
    return q['target_keys'][int(np.argmax(q['log_probabilities_base2']))]


def first_execution_prefix(prefixes):
    return next((p for p in prefixes if p['executions_used'] >= 1), prefixes[-1])


def gated_prefix(prefixes, scores):
    """Only original a0, actual prefix and a public predicted scalar are used."""
    for prefix in prefixes:
        action = prefix['chosen_action']
        if not prefix['actionable'] or action is None or action['kind'] == 'stop':
            return prefix
        score = scores[prefix['prefix_ordinal']]
        if not np.isfinite(score):
            raise FloatingPointError('A continuing gate score must be finite')
        if score <= 0:
            return prefix
    return prefixes[-1]


def audit_trajectory(item):
    row, trace, prefixes = item['row'], item['trace'], item['prefixes']
    if row['state'] != 'passed':
        return dict(passed=False, state=row['state'], error=row.get('error'), denominator_retained=True)
    assert len(prefixes) == len(trace) + 1
    assert [p['prefix_ordinal'] for p in prefixes] == list(range(len(prefixes)))
    assert prefixes[-1]['history_hash'] == row['evidence_trace_hash']
    for prefix in prefixes:
        assert prefix['prediction_reader'] == 'final'
        assert all(type(v) is int and v >= 0 for v in prefix['remaining'].values())
        assert all(prefix['remaining'][k] <= prefix['initial_budget'][k] for k in prefix['remaining'])
        if prefix['actionable']:
            i = int(np.argmax(prefix['original_scores']))
            assert i == prefix['chosen_action_index']
            assert prefix['candidate_actions'][i] == prefix['chosen_action']
            assert prefix['candidate_actions'][i]['action_key'] == prefix['chosen_action_key']
            assert len(prefix['chosen_head_input_final']) == 281
        q = prefix['prediction']
        assert 'healthy' in q['target_keys']
        assert abs(np.exp2(q['log_probabilities_base2']).sum() - 1) < 1e-5
    for i, event in enumerate(trace):
        before, after = prefixes[i], prefixes[i + 1]
        assert event['action'] == before['chosen_action']
        assert event['history_before'] == before['history_hash']
        assert event['history_after'] == after['history_hash']
        assert charge(before['remaining'], event['actual_cost']) == after['remaining']
        assert event['action']['kind'] != 'stop'
        assert after['observation_count'] == before['observation_count'] + 1
        assert all(event['actual_cost'][k] <= event['action']['cost_bound'][k] for k in event['actual_cost'])
    return dict(passed=True, prefixes=len(prefixes), acquired_actions=len(trace),
                argmax_proposer_unchanged=True, budgets_legal=True, actual_prefix_chain=True)


def baseline_items(run):
    run = Path(run)
    cases = read_cases(run / 'manifests/fresh_cases.json')
    for case in cases:
        for method in METHODS:
            for seed in SEEDS:
                path = run / 'trajectories/confirmation/cases' / f"{hash_json([case['case_id'], method, seed])}.json.gz"
                if not path.exists():
                    raise FileNotFoundError(f'Missing sealed confirmation trajectory: {path}')
                item = load_blob(path)
                assert item['result_hash'] == hash_json({k: v for k, v in item.items() if k != 'result_hash'})
                yield case, item


def prepare(run_dir):
    run = Path(run_dir)
    assert read_json(run / 'trajectories/confirmation/complete.json')['trajectories_present'] == 768
    assert read_json(run / 'head_replay_fresh_complete.json')['state'] == 'passed'
    choices = read_json(run / 'head_selection.json')['methods']
    reader_choice = read_json(run / 'manifests/source_reader_selection.json')
    variant = reader_choice.get('selected_variant')
    assert variant in (None, 'R-P', 'R-C', 'R-PC')
    g1_frame = pd.read_parquet(run / 'fresh_G1_prefixes.parquet')
    g1 = {identity(row): plain(row) for row in g1_frame.to_dict('records')}
    score_frame = pd.read_parquet(run / 'correction_scores_fresh_replay.parquet')
    scores = defaultdict(dict)
    for row in score_frame.to_dict('records'):
        choice = choices[row['method_key']]
        if row['cell'] == choice['cell'] and row['step'] == choice['step']:
            scores[identity(row)][int(row['decision_index'])] = float(row['score'])
    arms = ['A0', 'S1', 'G1'] + (['R1', 'RG1'] if variant is not None else [])
    availability = dict(selected_reader_variant=variant,
                        arms={arm: dict(available=arm in arms,
                              reason='source-selected reader rule' if arm in ('R1', 'RG1') and variant else
                              'no reader correction passed source validation' if arm not in arms else 'prespecified comparison')
                              for arm in ('A0', 'S1', 'R1', 'G1', 'RG1')})
    atomic_json(run / 'arm_availability.json', availability)
    records, tasks, audits = [], {}, []
    for case, item in baseline_items(run):
        row, prefixes = item['row'], item['prefixes']
        key = identity(row)
        audit = audit_trajectory(item)
        audits.append(dict(case_id=key[0], method_key=key[1], policy_seed=key[2], **audit))
        if row['state'] != 'passed':
            # Failures are explicitly present in every arm's denominator. Such a
            # row is an abstention, never manufactured as a healthy prediction.
            for arm in arms:
                records.append(dict(row, arm=arm, available=True, retained_prefix_ordinal=None,
                                    fault_family=case['fault_family'], final_prediction=None))
            continue
        gate = gated_prefix(prefixes, scores[key])
        assert gate['history_hash'] == g1[key]['history_hash']
        assert gate['prefix_ordinal'] == int(g1[key]['decision_index'])
        chosen = {'A0': prefixes[-1], 'S1': first_execution_prefix(prefixes), 'G1': gate}
        if variant is not None:
            chosen.update(R1=prefixes[-1], RG1=gate)
        for arm, prefix in chosen.items():
            record = {k: row[k] for k in ('configuration_id', 'case_id', 'method_key', 'policy_seed',
                                         'state', 'true_target_key', 'is_healthy')}
            record.update(arm=arm, fault_family=case['fault_family'], history_hash=prefix['history_hash'],
                          retained_prefix_ordinal=prefix['prefix_ordinal'], original_prediction=prefix['prediction'],
                          final_prediction=None if arm in ('R1', 'RG1') else prefix['prediction'],
                          initial_budget=prefix['initial_budget'], remaining=prefix['remaining'],
                          selected_reader_variant=variant if arm in ('R1', 'RG1') else 'original',
                          selected_gate_cell=choices[row['method_key']]['cell'] if arm in ('G1', 'RG1') else None,
                          selected_gate_step=choices[row['method_key']]['step'] if arm in ('G1', 'RG1') else None,
                          gate_selected_as_deployable_improvement=choices[row['method_key']]['guardrails_passed']
                              if arm in ('G1', 'RG1') else None,
                          physical_wall_seconds=row['end_to_end_seconds'] if arm == 'A0' else None,
                          physical_wall_scope='shared actual A0 acquisition window' if arm == 'A0' else
                              'exact retained A0 prefix; logical cost only; never separately timed',
                          termination_reason=prefix['terminal_reason'] or 'prespecified_early_stop')
            records.append(record)
            if arm in ('R1', 'RG1'):
                digest = prefix['history_hash']
                if digest not in tasks:
                    tasks[digest] = dict(history_hash=digest, feature_cache_path=prefix['feature_cache_path'],
                        provenance=[f"initial:{case['case_id']}"] +
                            [t['acquisition_identity'] for t in item['trace'][:prefix['prefix_ordinal']]], contexts=[])
                tasks[digest]['contexts'].append({**{k: row[k] for k in ('case_id', 'method_key', 'policy_seed')}, 'arm': arm})
    pd.DataFrame(records).to_parquet(run / 'confirmation_retained_prefixes.parquet', index=False)
    atomic_json(run / 'confirmation_reader_tasks.json', dict(variant=variant, tasks=list(tasks.values()),
                selected_before_fresh=True, source_reader_selection_sha256=file_sha256(run / 'manifests/source_reader_selection.json')))
    atomic_json(run / 'tests/confirmation_prefix_audit.json', dict(state='passed' if all(a['passed'] for a in audits) else 'incomplete',
                trajectories=len(audits), expected_trajectories=768, audits=audits,
                gate_reads_truth=False, first_execution_rule_reads_truth=False, stopped_prefix_is_actual=True))
    print(json.dumps(dict(stage='confirmation_prepare', rows=len(records), reader_tasks=len(tasks), variant=variant)), flush=True)


def finish(run_dir):
    run = Path(run_dir)
    retained = [plain(row) for row in pd.read_parquet(run / 'confirmation_retained_prefixes.parquet').to_dict('records')]
    tasks = read_json(run / 'confirmation_reader_tasks.json')
    reread = {}
    if tasks['variant'] is not None:
        payload = read_json(run / 'confirmation_reader_predictions.json')
        assert payload['variant'] == tasks['variant']
        reread = payload['predictions']
        assert set(reread) == {t['history_hash'] for t in tasks['tasks']}
    baseline = {identity(item['row']): item for _, item in baseline_items(run)}
    rows = []
    prefix_rows = []
    decisions = []
    choices = read_json(run / 'head_selection.json')['methods']
    gate_scores = {}
    for score in pd.read_parquet(run / 'correction_scores_fresh_replay.parquet').to_dict('records'):
        choice = choices[score['method_key']]
        if score['cell'] == choice['cell'] and score['step'] == choice['step']:
            gate_scores[(*identity(score), int(score['decision_index']))] = float(score['score'])
    retained_gates = {identity(row): row['retained_prefix_ordinal'] for row in retained if row['arm'] == 'G1'}
    for record in retained:
        item = baseline[identity(record)]
        row = {k: v for k, v in record.items() if k not in
               ('initial_budget', 'remaining', 'original_prediction', 'final_prediction')}
        if record['state'] != 'passed':
            row.update(healthy_false_alarm=None, fault_miss=None, correct_joint=False, final_nll_bits=None,
                       watchdog_failed=True, abstained=True)
            rows.append(row)
            continue
        prefixes, truth = item['prefixes'], record['true_target_key']
        q = reread[record['history_hash']] if record['arm'] in ('R1', 'RG1') else record['final_prediction']
        metrics = prediction_metrics(q, truth)
        base_q = prefixes[-1]['prediction']
        fixed = prediction_metrics(base_q)['max_fault_target']
        lp = dict(zip(q['target_keys'], q['log_probabilities_base2']))
        all_correct = [predicted(p['prediction']) == truth for p in prefixes]
        first = first_execution_prefix(prefixes)
        first_reached = first['executions_used'] >= 1 and first['prefix_ordinal'] <= record['retained_prefix_ordinal']
        first_correct = predicted(first['prediction']) == truth if first_reached else None
        final_correct = metrics['correct_joint']
        before_stop = all_correct[:record['retained_prefix_ordinal'] + 1]
        first_wrong = next((i for i, correct in enumerate(before_stop) if not correct), None)
        if first_wrong is None and not final_correct:
            first_wrong = record['retained_prefix_ordinal']
        remaining, initial = record['remaining'], record['initial_budget']
        row.update(metrics, max_fault_log_odds=lp[metrics['max_fault_target']] - lp['healthy'],
                   fixed_fault_target=fixed, fixed_fault_log_odds=lp[fixed] - lp['healthy'],
                   executions_used=initial['executions'] - remaining['executions'],
                   calibrations_used=initial['calibrations'] - remaining['calibrations'],
                   decisions_used=initial['decisions'] - remaining['decisions'], bytes_used=initial['bytes'] - remaining['bytes'],
                   initial_correct=all_correct[0], first_execution_reached=first_reached,
                   first_execution_correct=first_correct,
                   reference_A0_first_execution_correct=predicted(first['prediction']) == truth if first['executions_used'] >= 1 else None,
                   trajectory_prefix_diagnosis_reader='original_FINAL; R* changes only retained final reading',
                   first_correct_final_wrong=bool(first_correct and not final_correct) if first_reached else None,
                   first_wrong_final_correct=bool(not first_correct and final_correct) if first_reached else None,
                   initially_wrong_still_wrong=bool(not all_correct[0] and not final_correct),
                   first_wrong_prefix=first_wrong,
                   later_recovered=bool(first_wrong is not None and (any(before_stop[first_wrong + 1:]) or final_correct)),
                   watchdog_failed=False, abstained=False)
        rows.append(row)
    for key, item in baseline.items():
        if item['row']['state'] != 'passed':
            continue
        truth = item['row']['true_target_key']
        fixed = prediction_metrics(item['prefixes'][-1]['prediction'])['max_fault_target']
        for p in item['prefixes']:
            metrics = prediction_metrics(p['prediction'], truth)
            lp = dict(zip(p['prediction']['target_keys'], p['prediction']['log_probabilities_base2']))
            prefix_rows.append(dict(case_id=key[0], method_key=key[1], policy_seed=key[2],
                configuration_id=item['row']['configuration_id'], prefix_ordinal=p['prefix_ordinal'],
                history_hash=p['history_hash'], executions_used=p['executions_used'], calibrations_used=p['calibrations_used'],
                original_chosen_action=p['chosen_action_key'], original_chosen_score=p['chosen_score'],
                fixed_fault_target=fixed, fixed_fault_log_odds=lp[fixed] - lp['healthy'],
                max_fault_log_odds=lp[metrics['max_fault_target']] - lp['healthy'], **metrics))
            action = p['chosen_action']
            score = gate_scores.get((*key, p['prefix_ordinal']))
            event = item['trace'][p['prefix_ordinal']] if p['prefix_ordinal'] < len(item['trace']) else None
            decisions.append(dict(case_id=key[0], configuration_id=item['row']['configuration_id'],
                method_key=key[1], policy_seed=key[2], prefix_ordinal=p['prefix_ordinal'],
                history_hash=p['history_hash'], proposed_action_key=action['action_key'] if action else None,
                proposed_action_kind=action['kind'] if action else None,
                proposed_ports=json.dumps(action['ports'] if action else []),
                actual_next_action_key=event['action']['action_key'] if event else None,
                actual_next_observation_ref=event['observation_ref'] if event else None,
                actual_next_acquisition_identity=event['acquisition_identity'] if event else None,
                record_counts_by_observation=json.dumps(p['record_counts_by_observation']),
                total_record_count=p['total_record_count'], sequence_count=p['sequence_count'],
                p_healthy=metrics['p_healthy'], predicted_target_key=metrics['predicted_target_key'],
                original_q=p['chosen_score'], selected_gate_prediction=score,
                gate_says_stop=bool(score is not None and score <= 0),
                original_policy_stop=bool(action is not None and action['kind'] == 'stop'),
                prefix_reached_by_G1=p['prefix_ordinal'] <= retained_gates[key],
                G1_stops_here=p['prefix_ordinal'] == retained_gates[key],
                remaining=json.dumps(p['remaining'], sort_keys=True),
                executions_used=p['executions_used'], calibrations_used=p['calibrations_used']))
    frame = pd.DataFrame(rows)
    expected = set(baseline)
    assert len(expected) == 768
    for arm, group in frame.groupby('arm'):
        assert len(group) == 768, (arm, len(group))
        assert {identity(row) for row in group.to_dict('records')} == expected
        assert group.case_id.nunique() == 128
        assert group[group.is_healthy].case_id.nunique() == 64
        assert len(group[group.is_healthy]) == 384
    for keys, group in frame.groupby(['case_id', 'method_key', 'policy_seed']):
        if {'G1', 'RG1'}.issubset(set(group.arm)):
            pair = group[group.arm.isin(['G1', 'RG1'])]
            assert pair.history_hash.nunique() == 1
            assert pair.retained_prefix_ordinal.nunique() == 1
    frame.to_csv(run / 'confirmation_per_case.csv', index=False)
    pd.DataFrame(prefix_rows).to_csv(run / 'confirmation_prefix_metrics.csv', index=False)
    pd.DataFrame(decisions).to_csv(run / 'confirmation_decisions.csv', index=False)
    atomic_json(run / 'tests/confirmation_denominators.json', dict(state='passed', rows=len(frame),
        arms=sorted(frame.arm.unique()), trajectories_per_arm=768, physical_cases_per_arm=128,
        healthy_physical_cases=64, faulty_physical_cases=64, healthy_runs_per_arm=384, faulty_runs_per_arm=384,
        healthy_runs_per_method=192, faulty_runs_per_method=192, policy_seeds=list(SEEDS),
        failures_retained=int((frame.state != 'passed').sum()),
        main_method_names=list(METHODS), no_independent_walltime_for_replayed_arms=True))
    print(json.dumps(dict(stage='confirmation_finish', rows=len(frame), arms=sorted(frame.arm.unique()))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'finish'))
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else finish)(args.run_dir)
