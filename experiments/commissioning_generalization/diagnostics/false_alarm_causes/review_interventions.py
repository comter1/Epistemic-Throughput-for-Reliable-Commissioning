"""Independent read-only validation of acquired intervention artifacts; no forwards."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from cgx.data.store import Store
from cgx.observation.actions import action_catalog, feasible_actions, initial_budget
from cgx.provenance import atomic_json, hash_json
from .ledger import TimedStage


def review(run_dir,phase='discovery'):
    run_dir=Path(run_dir);store=Store(run_dir)
    manifest=json.loads((run_dir/'manifests'/f'interventions_{phase}.json').read_text())
    cases=json.loads(Path(manifest['case_manifest']).read_text())
    expected={(c['case_id'],m,s,a) for c in cases[:manifest['cases']]
              for m in manifest['methods'] for s in manifest['seeds'] for a in manifest.get('analysis_arms',manifest['arms'])}
    found=set();failures=[];counts=Counter();states=Counter();graphs={}
    def check(name,ok,key):
        counts[name+'_checked']+=1
        if not ok:
            counts[name+'_failed']+=1
            failures.append(dict(check=name,key=list(key)))
    with TimedStage(run_dir,'intervention_independent_review'):
        for path in sorted((run_dir/'interventions'/phase/'cases').glob('*.json.gz')):
            with gzip.open(path,'rt') as handle:item=json.load(handle)
            row=item['row'];trace=item['trace'];key=(row['case_id'],row['method_key'],row['policy_seed'],row['arm'])
            check('declared_row_identity',key in expected,key)
            check('unique_row_identity',key not in found,key);found.add(key);states[row['state']]+=1
            packed=store.history_references(row['evidence_trace_hash'])
            graph=graphs.setdefault(packed['graph_ref'],store._get('graphs',packed['graph_ref']))
            catalog=action_catalog(graph);budget=initial_budget(graph,1.,4)
            check('same_initial_caps',packed['initial_budget']==budget,key)
            check('trace_matches_acquired_history_refs',packed['observation_refs']==[item['initial_observation_ref']]+
                  [t['observation_ref'] for t in trace],key)
            check('trace_matches_public_action_keys',packed['action_keys']==[t['action']['action_key'] for t in trace],key)
            current=budget.copy();execution=calibration=0;prediction=item['row']['initial_prediction']
            for step in trace:
                action=step['action'];before=step['resources_before'];after=step['resources_after']
                check('resource_chain',before==current,key)
                legal={a['action_key']:a for a in feasible_actions(catalog,before)}
                check('actually_selected_action_legal',action['action_key'] in legal and legal[action['action_key']]==action,key)
                charge={k:before[k]-after[k] for k in before}
                check('charged_within_public_bounds',all(0<=charge[k]<=action['cost_bound'][k] for k in charge),key)
                check('execution_calibration_decision_charges',charge['executions']==int(bool(action['ports']))
                      and charge['calibrations']==int(bool(action['calibration_pair'])) and charge['decisions']==1,key)
                execution+=bool(action['ports']);calibration+=bool(action['calibration_pair'])
                check('separate_execution_and_burst_indices',step['execution_index']==execution and step['burst_index']==calibration,key)
                check('prediction_before_matches_last_valid',step['prediction_before']==prediction,key)
                if step['diagnosis_updated']:prediction=step['prediction']
                current=after
            check('final_resources_match_actual_trace',current==packed['remaining'],key)
            check('last_prediction_is_returned',prediction==row['prediction'],key)
            check('reported_resources_are_actual',all(row[k+'_used']==budget[k]-current[k]
                  for k in ('executions','calibrations','bytes','decisions')),key)
            p=np.exp2(row['prediction']['log_probabilities_base2'])
            check('finite_normalized_probabilities',np.isfinite(p).all() and abs(p.sum()-1)<1e-5,key)
            if row['arm'] in ('C','D'):
                planpath=run_dir/'manifests/calibration_plans'/f'{hash_json([row["case_id"],row["method_key"],row["policy_seed"]])}.json'
                plan=json.loads(planpath.read_text());n=len(plan['actions'])
                check('prepaid_pairs_match_sealed_public_plan',row['calibration_plan_hash']==plan['plan_hash']
                      and [t['action'] for t in trace[:n]]==plan['actions'],key)
            if row['arm']=='S':
                source=run_dir/'interventions'/phase/'cases'/f'{hash_json([row["case_id"],row["method_key"],row["policy_seed"],"A"])}.json.gz'
                with gzip.open(source,'rt') as handle:aitem=json.load(handle)
                arow=aitem['row'];apacked=store.history_references(arow['evidence_trace_hash'])
                first=next((i+1 for i,t in enumerate(aitem['trace']) if t['action']['ports']),None)
                stop=len(aitem['trace']) if first is None else first
                expected_trace=aitem['trace'][:stop]
                valid=[(i+1,t['prediction']) for i,t in enumerate(expected_trace) if t['diagnosis_updated']]
                index,q=valid[-1] if valid else (0,arow['initial_prediction'])
                check('S_exact_first_execution_A_trace_prefix',trace==expected_trace and row['source_A_prefix_decision_index']==stop,key)
                check('S_no_future_observations_or_action_keys',packed['observation_refs']==apacked['observation_refs'][:stop+1]
                      and packed['action_keys']==apacked['action_keys'][:stop],key)
                check('S_exact_completed_prefix_prediction',row['prediction']==q and row['last_valid_prediction_prefix']==index,key)
                check('S_same_initial_public_context',all(packed[k]==apacked[k] for k in ('graph_ref','task_ref','initial_budget','requirements'))
                      and row['initial_prediction']==arow['initial_prediction'],key)
                check('S_first_execution_rule_no_outcome_condition',row['first_robot_execution_exists']==(first is not None)
                      and row['nonanticipatory_public_rule'] and not row['rule_uses_prediction_or_truth'],key)
                check('S_shared_prefix_identity',row['paired_A_evidence_trace_hash']==arow['evidence_trace_hash']
                      and item['shared_real_prefix'] and not item['independently_resampled'],key)
                check('S_no_new_call_or_fabricated_standalone_timing',all(row[k]==0 for k in (
                      'additional_model_forward_calls','additional_feature_build_calls','additional_robot_execution_attempts'))
                      and row['end_to_end_seconds'] is None and not row['standalone_wall_seconds_measured'],key)
                check('S_same_frozen_weights',row['checkpoint_hash']==arow['checkpoint_hash']
                      and row['diagnostic_checkpoint_hash']==arow['diagnostic_checkpoint_hash'],key)
        result=dict(phase=phase,expected_rows=len(expected),available_rows=len(found),missing_rows=len(expected-found),
                    missing_examples=[list(k) for k in sorted(expected-found)[:10]],states=dict(states),
                    checks=dict(counts),failure_examples=failures[:100],
                    input_boundary_review='Source inspected: B/D use dict(initial) with only true remaining/action_keys; explicit initial_q; current feasible action list. All final readers consume acquired complete history.',
                    shared_prefix_review='S stopping index uses first action with robot ports, independent of q/labels/future observations. Output history and completed q must equal that exact A prefix; standalone S wall is unmeasured.',
                    initialization_failure_caveat='run_case_job pre-arm failures have no outputs. Final aggregation must join sealed expected identities and retain missing/failed arms explicitly.',
                    numeric_forwards=0)
        atomic_json(run_dir/f'intervention_independent_review_{phase}.json',result)
        print(json.dumps({k:result[k] for k in ('phase','expected_rows','available_rows','missing_rows','states','failure_examples')}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True)
    parser.add_argument('--phase',default='discovery');args=parser.parse_args()
    review(args.run_dir,args.phase)
