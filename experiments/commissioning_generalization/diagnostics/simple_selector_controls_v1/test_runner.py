"""Source-only engineering qualification; no new test outcome is opened."""
import argparse
from copy import deepcopy
import importlib
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import numpy as np

from cgx.config import load_config
from cgx.observation.actions import action_catalog, initial_budget, can_afford, charge
from cgx.provenance import atomic_json, file_sha256, hash_json, now, read_json
from structured_evidence_v1.events import prepare_history
from structured_evidence_v1.reader import load_reader
from .cache import ExtensionCachedSystem, FRESH_EXECUTION_CAP, FreshExecutionCapExceeded, reserve_attempt
from .evaluation import _initialize, _execute_canonical, acquisition_profile, canonical_job, runtime_files
from .policies import SimplePolicy
from .rollout import complete_final_read, rollout
from .storage import ExtensionStore


def run_controls(source_root,sed_run,run_dir):
    run=Path(run_dir);started=time.monotonic();store=ExtensionStore(source_root,sed_run,run)
    config=load_config();reader=load_reader(sed_run,'final');controls={};records=[]
    validation=list(store.cases('validation'));tests=[c for split in ('test_seen','test_connections','test_size') for c in store.cases(split)]
    assert len(validation)==512 and len(tests)==3200
    assert not {c['case_id'] for c in validation}&{c['case_id'] for c in tests}
    controls['source_test_ids_disjoint']=True
    selected=read_json(run/'validation/pilot_cases.json')['case_ids'][0]
    case=next(c for c in validation if c['case_id']==selected)
    identity=dict(runtime_files=runtime_files(),reader_hash=reader.checkpoint_hash,qualification=True)
    _initialize(source_root,sed_run,run,identity)
    with (patch('torch.load',side_effect=AssertionError('A simple rule must never load weights')),
          patch('cgx.models.PolicyModel',side_effect=AssertionError('No neural selector construction'))):
        for spec in (dict(method='full_once_100'),dict(method='full_once_100_cal2'),
                dict(method='random_legal',choice_seed=11,case_id=case['case_id']),
                dict(method='interface_rule_se',stop_threshold=None)):
            job=dict(case=case,execution_cap=1,spec=spec)
            key=hash_json(['source-engineering-qualification',identity,case['case_id'],spec])
            payload=_execute_canonical(job,spec,1,key)
            assert not payload['online']['worker_failure'],payload['online'].get('traceback')
            assert payload['online']['feasible']
            assert payload['completed']['completion_status']!='unresolved_kept_online'
            atomic_json(run/'tests/qualification_records'/(spec['method']+'.json'),payload)
            records.append(payload)
    controls['simple_rules_do_not_load_neural_selectors']=True
    full,calibrated,random,rule=records
    assert full['online']['executions_used']==1 and full['online']['calibrations_used']==0
    assert calibrated['online']['executions_used']==1 and 0<=calibrated['online']['calibrations_used']<=2
    assert calibrated['online']['bytes_used']>=full['online']['bytes_used']
    controls['real_full_once_one_execution_no_calibration']=True
    controls['real_full_once_calibration_charged']=True
    for payload in records:
        history=store.load_history(payload['online']['evidence_trace_hash'])
        catalog=action_catalog(history['graph'],config);lookup={a['action_key']:a for a in catalog}
        remaining=deepcopy(history['initial_budget'])
        for entry in payload['trace']:
            action=entry.get('action')
            if not action or action['kind']=='stop':continue
            assert lookup[action['action_key']]==action and can_afford(remaining,action['cost_bound'])
            assert entry['resources_before']==remaining
            actual={k:entry['resources_before'][k]-entry['resources_after'][k] for k in remaining}
            remaining=charge(remaining,actual)
        assert remaining==history['remaining']
        for field in ('executions_used','bytes_used','calibrations_used','decisions_used','evidence_trace_hash','event_provenance_hash'):
            assert payload['online'][field]==payload['completed'][field]
    controls['actual_legal_actions_charges_and_completed_history_unchanged']=True
    history=store.load_history(full['online']['evidence_trace_hash']);provenance=full['event_provenance']
    original=reader.predict(history,provenance=provenance)
    reference=np.asarray(original['log_probabilities_base2'])
    for cap in (1,2,4):
        for method in ('full_once_100','full_once_100_cal2'):
            public={'graph':history['graph'],'remaining':initial_budget(history['graph'],1.,cap)}
            plan=SimplePolicy({'method':method}).plan(public,action_catalog(history['graph'],config),None)
            baseline=SimplePolicy({'method':method}).plan({**public,'remaining':initial_budget(history['graph'],1.,1)},action_catalog(history['graph'],config),None)
            assert plan==baseline
        changed=deepcopy(history);changed['initial_budget']=initial_budget(history['graph'],1.,cap)
        used={k:history['initial_budget'][k]-history['remaining'][k] for k in history['remaining']}
        changed['remaining']=charge(changed['initial_budget'],used)
        actual=reader.predict(changed,provenance=provenance)
        assert actual['target_keys']==original['target_keys']
        assert np.array_equal(reference,np.asarray(actual['log_probabilities_base2']))
    cap_marker=dict(state='passed',created_at=now(),source_case_id=case['case_id'],
        history_hash=full['online']['evidence_trace_hash'],provenance_hash=full['online']['event_provenance_hash'],
        reader_hash=reader.checkpoint_hash,budgets=[1,2,4],identical_full_once_plans=True,
        identical_completed_posterior_under_budget_only_views=True,physical_noise_identities_unchanged=True)
    atomic_json(run/'tests/one_pass_cap_views.json',cap_marker)
    controls['one_pass_cap_views_verified']=True
    for method in ('full_once_100','random_legal','interface_rule_se'):
        row=deepcopy(full['online']);row['method']=method;row['final_online_prediction_complete']=False
        reread=complete_final_read(row,history,provenance,reader,120)
        assert np.array_equal(reference,np.asarray(reread['prediction']['log_probabilities_base2']))
    controls['same_history_same_posterior_across_method_labels']=True
    duplicate={**history,'observations':history['observations']+[history['observations'][-1]]}
    duplicate_provenance={**provenance,'observation_ids':provenance['observation_ids']+[provenance['observation_ids'][-1]],
        'observation_refs':provenance['observation_refs']+[provenance['observation_refs'][-1]]}
    predicted=reader.predict(duplicate,provenance=duplicate_provenance)
    assert np.array_equal(reference,np.asarray(predicted['log_probabilities_base2']))
    # A small identity ingress fixture uses real stored values, without claiming
    # that copying them creates a fresh physical observation in the experiment.
    observation={**history['observations'][-1],'records':history['observations'][-1]['records'][:20], 'calibration_bursts':[]}
    tiny={'graph':history['graph'],'observations':[observation,observation]}
    same=prepare_history(tiny,{'observation_ids':['same','same']})
    distinct=prepare_history(tiny,{'observation_ids':['first','second']})
    assert len(distinct['events'])==2*len(same['events'])>0
    controls['same_ids_deduplicate_equal_values_distinct_ids_retained']=True
    cold=ExtensionCachedSystem(store.graph(case['configuration_id']),case,store.task(case['case_id']),config=config,
        implementation=next(c for c in store.configurations() if c['configuration_id']==case['configuration_id'])['implementation'],
        asset_path=Path(source_root)/'assets/mujoco_menagerie/franka_emika_panda',
        cache_root=run/'cache/physical',readonly_cache_roots=[Path(sed_run)/'cache/physical'],attempt_root=run)
    first=full['trace'][0]['action'];before=cold.robot_attempts
    value=cold.acquire(first,1,0)
    assert cold.robot_attempts==before and cold.cache_hits==1
    assert hash_json(value.observation)==full['trace'][0]['evidence_hash']
    assert cold.last_acquisition_id==full['trace'][0]['measurement_acquisition_id']
    controls['exact_cache_preserves_event_identity_and_logical_costs']=True
    with tempfile.TemporaryDirectory(prefix='simple-selector-cap-control-') as temporary:
        reserved=reserve_attempt(temporary,'unit-fixture',1,1)
        path=Path(temporary)/'resources/fresh_attempt_budget.json';state=read_json(path)
        assert state['robot_execution_attempts']==state['calibration_attempts']==1
        state['robot_execution_attempts']=FRESH_EXECUTION_CAP;atomic_json(path,state)
        try:reserve_attempt(temporary,'unit-over-cap',1,0)
        except FreshExecutionCapExceeded:pass
        else:raise AssertionError('Fresh attempt cap failed open')
        assert read_json(path)['robot_execution_attempts']==FRESH_EXECUTION_CAP
    controls['durable_cap_reserves_before_call_and_rejects_overrun']=True
    assert reader.checkpoint_hash==read_json(run/'protocol.json')['reader_hash']
    assert reader.parameter_fingerprint()==load_reader(sed_run,'final').parameter_fingerprint()
    controls['same_frozen_reader_parameters_and_numerical_settings']=True
    # A failed action is attempted, not successfully acquired coverage.
    failed=acquisition_profile([{'action':first,'stop_reason':'acquisition_error'}],None)
    assert not failed['full_trace_used'] and failed['sample_rates']==[] and failed['attempted_action_family_counts']
    controls['failed_acquisition_not_reported_as_recorded_coverage']=True
    result=dict(state='passed',created_at=now(),source_only=True,source_case_id=case['case_id'],
        qualification_not_formal_validation=True,controls=controls,executed_real_rollouts=len(records),
        no_new_test_outcomes_opened=True,all_real_attempts_charged_to_shared_cap=True,
        reader_hash=reader.checkpoint_hash,runtime_files=runtime_files(),elapsed_seconds=time.monotonic()-started)
    atomic_json(run/'tests/runner_controls.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--sed-run',required=True);p.add_argument('--run-dir',required=True)
    a=p.parse_args();print(run_controls(a.source_root,a.sed_run,a.run_dir),flush=True)
