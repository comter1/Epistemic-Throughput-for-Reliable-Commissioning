"""Execute fixed registered controls; preserve every logical row and exact evidence."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import itertools
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from cgx.config import load_config
from cgx.graphs import legal_targets
from cgx.provenance import atomic_json, file_sha256, hash_json, now, read_json
from structured_evidence_v1.reader import load_reader
from .cache import FRESH_EXECUTION_CAP, file_lock
from .policies import SimplePolicy
from .rollout import rollout, complete_final_read, target_key
from .storage import ExtensionStore

_WORKER=None
RUNTIME_FILES=('policies.py','rollout.py','cache.py','storage.py','evaluation.py')
PHASE_SPLITS={'validation':('validation',),'primary':('test_connections',),
    'transfer':('test_seen','test_size'),'all':('test_connections','test_seen','test_size')}
EXPECTED_ROWS={'validation':16896,'primary':32000,'transfer':32000,'all':64000}


def runtime_files():
    return {name:file_sha256(Path(__file__).parent/name) for name in RUNTIME_FILES}


def _require(source_root,sed_run,run,phase):
    from structured_evidence_v1.evaluation import require_frozen
    frozen=require_frozen(sed_run)
    protocol=read_json(run/'protocol.json')
    if protocol.get('fresh_robot_execution_attempt_cap',190000)!=FRESH_EXECUTION_CAP:
        raise ValueError('Fresh-attempt cap differs from the registered190000')
    if protocol.get('completed_read_watchdog_seconds',120)!=120:
        raise ValueError('Completed-reader watchdog differs from registered120 seconds')
    design=read_json(run/'provenance/validation_design_seal.json')
    if design['state']!='sealed_before_validation':raise ValueError('Validation design must be sealed')
    if design['protocol_sha256']!=file_sha256(run/'protocol.json'):raise ValueError('Sealed protocol changed')
    if design['source_inventory_sha256']!=file_sha256(run/'source_inventory.json'):raise ValueError('Sealed source inventory changed')
    for name,digest in design['code_hashes'].items():
        if file_sha256(Path(__file__).parent/name)!=digest:raise ValueError(f'Validation design code changed: {name}')
    selected=None
    if phase!='validation':
        selected=read_json(run/'selection_manifest.json')
        if selected['state']!='sealed':raise ValueError('Source validation settings must be sealed before test')
        if selected.get('reader_hash',frozen['reader_hash'])!=frozen['reader_hash']:
            raise ValueError('Selected settings use a different reader')
        experiment=Path(__file__).parents[2]
        for name,digest in selected['code_hashes'].items():
            path=Path(name) if Path(name).is_absolute() else (experiment/name if '/' in name else Path(__file__).parent/name)
            if file_sha256(path)!=digest:raise ValueError(f'Sealed extension code changed: {name}')
        if selected.get('config_hash',hash_json(load_config()))!=hash_json(load_config()):
            raise ValueError('Sealed executable configuration changed')
    identity=dict(version='cgx-simple-selector-evaluation-v1',runtime_files=runtime_files(),
        config_hash=hash_json(load_config()),reader_hash=frozen['reader_hash'],
        old_frozen_sha256=file_sha256(Path(sed_run)/'models/frozen.json'),
        protocol_sha256=file_sha256(run/'protocol.json'))
    # Selection does not change the execution of already fixed validation recipes.
    return identity,selected


def logical_key(job):
    case=job['case'];spec=job['spec']
    return hash_json([case['split'],case['configuration_id'],case['case_id'],spec['method'],
        job['execution_cap'],spec.get('choice_seed'),job.get('candidate_id')])


def canonical_job(job):
    """Declared identical plans share one rollout, not independent timing trials."""
    spec=deepcopy(job['spec']);cap=job['execution_cap']
    if spec['method']=='fixed_template_validation':
        alias={'F0':'full_once_100','F1':'full_once_100_cal2','F2':'coverage_forward'}.get(spec['template_id'])
        if alias:spec={'method':alias}
    if spec['method'] in ('full_once_100','full_once_100_cal2'):cap=1
    if spec['method']=='random_legal':spec['case_id']=job['case']['case_id']
    return spec,cap


def planned_jobs(store,phase,selected=None,case_subset=None):
    cases=[c for split in PHASE_SPLITS[phase] for c in store.cases(split)]
    if case_subset is not None:
        wanted=set(case_subset)
        if not wanted.issubset({c['case_id'] for c in cases}):raise ValueError('Subset crosses the declared phase split')
        cases=[c for c in cases if c['case_id'] in wanted]
    for case in sorted(cases,key=lambda c:hash_json(['simple-selector-case-order-v1',c['case_id']])):
        if phase=='validation':
            for cap in (1,2,4):
                for template in ('F0','F1','F2','F3','F4','F5','F6','F7'):
                    yield dict(case=case,execution_cap=cap,candidate_id=template,
                        spec=dict(method='fixed_template_validation',template_id=template))
                for threshold,name in ((None,'rule_none'),(.99,'rule_0.99'),(.999,'rule_0.999')):
                    yield dict(case=case,execution_cap=cap,candidate_id=name,
                        spec=dict(method='interface_rule_se',stop_threshold=threshold))
        else:
            for method in ('full_once_100','full_once_100_cal2'):
                yield dict(case=case,execution_cap=1,spec=dict(method=method))
            for cap in (1,2,4):
                for seed in (11,23,37):
                    yield dict(case=case,execution_cap=cap,spec=dict(method='random_legal',choice_seed=seed,case_id=case['case_id']))
                yield dict(case=case,execution_cap=cap,spec=dict(method='coverage_forward'))
                yield dict(case=case,execution_cap=cap,spec=dict(method='fixed_template_validation',
                    template_id=selected['selected_template_by_cap'][str(cap)]))
                yield dict(case=case,execution_cap=cap,spec=dict(method='interface_rule_se',
                    stop_threshold=selected['selected_stop_threshold_by_cap'][str(cap)]))


def _initialize(source_root,sed_run,run_root,identity):
    global _WORKER
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    store=ExtensionStore(source_root,sed_run,run_root)
    reader=load_reader(sed_run,'final',device='cpu')
    completed_reader=load_reader(sed_run,'final',device='cpu')
    assert reader.checkpoint_hash==completed_reader.checkpoint_hash==identity['reader_hash']
    assert reader.parameter_fingerprint()==completed_reader.parameter_fingerprint()
    marker=Path(run_root)/'tests/one_pass_cap_views.json'
    cap_views=marker.exists() and read_json(marker)['state']=='passed'
    _WORKER=dict(source_root=Path(source_root),sed_run=Path(sed_run),run=Path(run_root),
        identity=identity,identity_hash=hash_json(identity),config=load_config(),store=store,
        reader=reader,completed_reader=completed_reader,reader_fingerprint=reader.parameter_fingerprint(),
        configurations={c['configuration_id']:c for c in store.configurations()},
        warmed_policies=set(),one_pass_cap_views_verified=cap_views)


def _outer_failure(case,graph,cap,error,seconds):
    keys,prior=legal_targets(graph);truth=target_key(case);predicted=keys[int(np.argmax(prior))]
    loss=-math.log2(prior[keys.index(truth)])
    row=dict(case_id=case['case_id'],configuration_id=case['configuration_id'],suite=case['split'],
        fault_family=case['fault_family'],byte_fraction=1.,execution_cap=cap,
        initial_nll_bits=loss,final_nll_bits=loss,prior_nll_bits=loss,score_improvement_bits=0.,
        target_cardinality=len(keys),predicted_target_key=predicted,true_target_key=truth,
        correct_joint=predicted==truth,correct_mechanism=predicted.rsplit(':',1)[-1]==truth.rsplit(':',1)[-1],
        healthy_false_alarm=truth=='healthy' and predicted!='healthy',fault_miss=truth!='healthy' and predicted=='healthy',
        prediction=dict(target_keys=list(keys),log_probabilities_base2=np.log2(prior).tolist()),
        timeout=isinstance(error,TimeoutError),abstained=True,
        errors=[f'worker_error:{type(error).__name__}:{error}'],worker_failure=True,
        traceback=traceback.format_exc(),resource_accounting_unavailable=True,
        evidence_trace_hash=None,event_provenance_hash=None,final_online_prediction_complete=False,
        end_to_end_seconds=seconds,diagnostic_family='structured_evidence_v1',feasible=None)
    for name in ('bytes_used','executions_used','calibrations_used','decisions_used','initial_bytes',
            'selection_seconds','inference_seconds','simulation_seconds','recorder_seconds',
            'warmup_seconds','selection_warmup_seconds','physical_cache_hits','new_robot_execution_attempts',
            'new_calibration_attempts','actual_new_physical_seconds','imported_physical_cache_hits',
            'extension_physical_cache_hits','reader_calls','reader_warmup_calls','selector_calls',
            'selector_warmup_calls','planning_calls'):row[name]=None
    return row


def acquisition_profile(trace,history):
    counts={};attempted={};rates={};rate_sequence=[];chosen=[];acquired=[];full=0
    for entry in trace:
        action=entry.get('action')
        if not action or action['kind']=='stop':continue
        kind=action['kind'];attempted[kind]=attempted.get(kind,0)+1
        chosen.append(action['action_key'])
        if not entry.get('evidence_hash'):continue
        counts[kind]=counts.get(kind,0)+1;acquired.append(action['action_key'])
        full+=int(kind=='full_trace')
        if action['ports']:
            rate=str(action['sample_hz']);rates[rate]=rates.get(rate,0)+1;rate_sequence.append(action['sample_hz'])
    ports=set()
    if history is not None:
        for observation in history['observations'][1:]:
            for record in observation.get('records',[]):
                port=record.get('port_id',record.get('port'))
                if port and port!='external:out':ports.add(port)
    return dict(action_family_counts=counts,attempted_action_family_counts=attempted,
        sample_rate_counts=rates,sample_rates=rate_sequence,unique_sample_rates=sorted(map(int,rates)),
        full_trace_used=full>0,full_trace_action_count=full,chosen_action_keys=chosen,acquired_action_keys=acquired,
        distinct_port_coverage=len(ports),acquired_distinct_ports=sorted(ports))


def _execute_canonical(job,spec,cap,key):
    w=_WORKER;store=w['store'];case=job['case'];graph=store.graph(case['configuration_id'])
    policy=SimplePolicy(spec);policy._cgx_eval_warmed=policy.rule_hash in w['warmed_policies']
    started=time.monotonic();history=None;trace=[];plan_metadata={}
    try:
        row,trace,history,_=rollout(graph,case,store.task(case['case_id']),w['config'],w['reader'],policy,1.,cap,
            implementation=w['configurations'][case['configuration_id']]['implementation'],
            asset_path=w['source_root']/'assets/mujoco_menagerie/franka_emika_panda',store=store,
            source_root=w['source_root'],run_root=w['run'],sed_run_root=w['sed_run'])
        if getattr(policy,'_cgx_eval_warmed',False):w['warmed_policies'].add(policy.rule_hash)
        plan_metadata=deepcopy(policy.plan_metadata)
        row.update(fault_family=case['fault_family'],healthy_false_alarm=case['fault_family']=='healthy' and row['predicted_target_key']!='healthy',
            fault_miss=case['fault_family']!='healthy' and row['predicted_target_key']=='healthy',worker_failure=False)
        provenance=store.load_provenance(row['evidence_trace_hash'],row['event_provenance_hash'])
        completed=complete_final_read(row,history,provenance,w['completed_reader'],120)
    except Exception as exc:
        row=_outer_failure(case,graph,cap,exc,time.monotonic()-started)
        completed={**deepcopy(row),**dict(completion_status='unresolved_kept_online',
            completed_read_error='No complete saved final history after outer worker failure',
            completed_read_calls=0,completed_read_seconds=0.,completion_reused_probability=False,
            completion_watchdog_seconds=120,actual_history_unchanged=True,
            online_timeout=row['timeout'],online_abstained=row['abstained'],abandoned_checks_not_recovered=True)}
    for reader in (w['reader'],w['completed_reader']):
        if reader.parameter_fingerprint()!=w['reader_fingerprint']:raise RuntimeError('Frozen SED parameter mutation')
    common=dict(inference_device='cpu',cpu_threads_per_worker=1,rollout_execution='independent_process',
        neural_selector_loaded=False,reader_hash=w['reader'].checkpoint_hash,
        reader_parameter_fingerprint=w['reader_fingerprint'],checkpoint_hash=None,
        execution_rule_hash=policy.rule_hash,execution_source_key=key,execution_source_cap=cap,
        **acquisition_profile(trace,history))
    row.update(common);completed.update(common)
    stopped_early=row.get('earlier_termination_affected_acquisition',bool(row.get('worker_failure')))
    for record in (row,completed):record.update(earlier_termination_affected_acquisition=stopped_early,
        online_termination_errors=list(row['errors']))
    packed=store.history_references(row['evidence_trace_hash']) if row['evidence_trace_hash'] else None
    envelope=store.load_provenance(row['evidence_trace_hash'],row['event_provenance_hash']) if row['evidence_trace_hash'] else None
    return dict(state='passed',online=row,completed=completed,trace=trace,plan_metadata=plan_metadata,
        packed_history=packed,event_provenance=envelope,canonical_spec=spec,source_cell_key=logical_key(job),
        runtime_identity_hash=w['identity_hash'],created_at=now())


def _job(job):
    w=_WORKER;spec,cap=canonical_job(job)
    if cap!=job['execution_cap'] and not w['one_pass_cap_views_verified']:
        raise RuntimeError('One-pass cap-view equivalence control must pass before reuse')
    source_key=hash_json([w['identity_hash'],job['case']['case_id'],spec,cap])
    cache=w['run']/'rollouts'/source_key[:2]/(source_key+'.json')
    with file_lock(cache.with_suffix('.lock')):
        reused=cache.exists()
        if reused:
            payload=read_json(cache)
            expected=payload.pop('result_hash')
            if hash_json(payload)!=expected:raise ValueError('Canonical rollout content changed')
            if payload['runtime_identity_hash']!=w['identity_hash']:raise ValueError('Rollout runtime identity changed')
        else:
            payload=_execute_canonical(job,spec,cap,source_key)
            atomic_json(cache,{**payload,'result_hash':hash_json(payload)})
    requested=SimplePolicy(job['spec']);cell=logical_key(job)
    common=dict(method=job['spec']['method'],policy_seed=job['spec'].get('choice_seed'),
        candidate_id=job.get('candidate_id'),template_id=requested.template_id,
        stop_threshold=job['spec'].get('stop_threshold'),execution_cap=job['execution_cap'],
        cell_key=cell,rule_hash=requested.rule_hash,protocol_hash=hash_json(w['config']),
        runtime_identity_hash=w['identity_hash'],reused_rollout=reused,
        source_logical_cell_key=payload['source_cell_key'],canonical_method=spec['method'],
        cap_view_reuse=cap!=job['execution_cap'],comparison_identity_alias=spec['method']!=job['spec']['method'] or cap!=job['execution_cap'],
        one_pass_cap_views_verified=w['one_pass_cap_views_verified'])
    result=deepcopy(payload)
    for mode in ('online','completed'):
        result[mode].update(common,result_mode=mode)
        if reused:result[mode].update(wall_time_scope='reused_identical_rollout_reference_not_new_timing')
    result['logical_spec']=job['spec'];result['logical_cell_key']=cell
    result['result_hash']=hash_json(result)
    return result


def phase_directory(run,phase):
    return run/('validation' if phase=='validation' else 'evaluation')


def _case_path(run,phase,key):
    return phase_directory(run,phase)/'cases'/(key+'.json')


def evaluate(source_root,sed_run,run_root,phase,workers=32,shard=0,shards=1,case_subset=None):
    run=Path(run_root);identity,selected=_require(source_root,sed_run,run,phase)
    if not 0<=shard<shards:raise ValueError('Invalid shard')
    store=ExtensionStore(source_root,sed_run,run);subset=None
    if case_subset:
        value=read_json(case_subset);subset=value['case_ids'] if isinstance(value,dict) else value
    jobs=[j for j in planned_jobs(store,phase,selected,subset) if int(j['case']['case_id'],16)%shards==shard]
    directory=phase_directory(run,phase);directory.mkdir(parents=True,exist_ok=True)
    qualifier='subset_'+hash_json(sorted(subset))[:12] if subset is not None else 'all'
    task=f'{phase}_{qualifier}_shard{shard}of{shards}';started=time.monotonic()
    with file_lock(directory/(task+'.lock')):
        pending=[];existing=0
        for job in jobs:
            path=_case_path(run,phase,logical_key(job))
            if path.exists():
                value=read_json(path);expected=value.pop('result_hash')
                if hash_json(value)!=expected or value['runtime_identity_hash']!=hash_json(identity):
                    raise ValueError('Saved logical cell changed or belongs to another runtime; preserve and inspect')
                existing+=1
            else:pending.append(job)
        count=existing
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_initialize,
                initargs=(str(source_root),str(sed_run),str(run),identity)) as pool:
            iterator=iter(pending)
            while batch:=list(itertools.islice(iterator,max(128,workers*4))):
                for job,value in zip(batch,pool.map(_job,batch,chunksize=1),strict=True):
                    atomic_json(_case_path(run,phase,logical_key(job)),value);count+=1
                    if count%100==0:
                        progress=dict(state='running',task=task,completed=count,expected=len(jobs),
                            resumed_cells=existing,elapsed_seconds=time.monotonic()-started)
                        atomic_json(directory/(task+'_progress.json'),progress);print(json.dumps(progress),flush=True)
        result=dict(state='passed',task=task,cells=len(jobs),resumed_cells=existing,
            elapsed_seconds=time.monotonic()-started,workers=workers,shard=shard,shards=shards,
            phase=phase,subset_case_ids=subset,runtime_identity_hash=hash_json(identity))
        atomic_json(directory/(task+'_complete.json'),result)
        return result


def finish(source_root,sed_run,run_root,phase):
    run=Path(run_root);identity,selected=_require(source_root,sed_run,run,phase)
    store=ExtensionStore(source_root,sed_run,run);jobs=list(planned_jobs(store,phase,selected))
    assert len(jobs)==EXPECTED_ROWS[phase]
    directory=phase_directory(run,phase);online=[];completed=[];histories=[];traces=[];seen=set()
    for job in jobs:
        key=logical_key(job);assert key not in seen;seen.add(key)
        path=_case_path(run,phase,key);value=read_json(path);expected=value.pop('result_hash')
        assert hash_json(value)==expected and value['runtime_identity_hash']==hash_json(identity)
        a,b=value['online'],value['completed'];assert a['cell_key']==b['cell_key']==key
        assert a['evidence_trace_hash']==b['evidence_trace_hash']
        assert a['event_provenance_hash']==b['event_provenance_hash']
        online.append(a);completed.append(b)
        histories.append(dict(cell_key=key,suite=a['suite'],configuration_id=a['configuration_id'],case_id=a['case_id'],
            method=a['method'],execution_cap=a['execution_cap'],policy_seed=a['policy_seed'],
            history_hash=a['evidence_trace_hash'],provenance_hash=a['event_provenance_hash'],
            source_logical_cell_key=a['source_logical_cell_key'],execution_source_key=a['execution_source_key'],
            history_path=store.resolve_blob('histories',a['evidence_trace_hash']) if a['evidence_trace_hash'] else None,
            provenance_path=store.resolve_provenance(a['evidence_trace_hash'],a['event_provenance_hash']) if a['evidence_trace_hash'] else None,
            packed_history_json=json.dumps(value['packed_history'],sort_keys=True),
            event_provenance_json=json.dumps(value['event_provenance'],sort_keys=True),
            cell_path=str(path),cell_file_sha256=file_sha256(path),outer_failure_no_history=a.get('worker_failure',False)))
        traces.append(dict(cell_key=key,source_logical_cell_key=a['source_logical_cell_key'],
            execution_source_key=a['execution_source_key'],trace=value['trace'],plan_metadata=value['plan_metadata']))
    outputs={}
    for suffix,rows in [('per_case_online',online),('per_case_completed_read',completed),('acquired_history_index',histories)]:
        path=directory/f'{phase}_{suffix}.parquet';pd.DataFrame(rows).to_parquet(path,index=False);outputs[path.name]=file_sha256(path)
        if phase=='all':
            alias=directory/(suffix+'.parquet');alias.write_bytes(path.read_bytes());outputs[alias.name]=file_sha256(alias)
    trace_path=directory/f'{phase}_action_traces.jsonl.gz'
    with gzip.open(trace_path,'wt') as stream:
        for row in traces:stream.write(json.dumps(row,sort_keys=True)+'\n')
    outputs[trace_path.name]=file_sha256(trace_path)
    if phase=='all':
        alias=directory/'action_traces.jsonl.gz';alias.write_bytes(trace_path.read_bytes());outputs[alias.name]=file_sha256(alias)
    result=dict(state='passed',phase=phase,rows=len(online),expected=EXPECTED_ROWS[phase],
        physical_cases=len({r['case_id'] for r in online}),reader_hash=identity['reader_hash'],
        runtime_identity_hash=hash_json(identity),all_planned_denominators_retained=True,
        official_rows_unchanged_by_completed_read=True,no_new_acquisitions_for_completed_read=True,
        new_network_training=False,one_pass_cap_views_verified=all(r['one_pass_cap_views_verified'] for r in online),
        unresolved_completed_reads=sum(r['completion_status']=='unresolved_kept_online' for r in completed),
        outputs_sha256=outputs,completed_at=now())
    atomic_json(directory/f'phase_{phase}_complete.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['evaluate','finish'])
    parser.add_argument('--source-root',required=True);parser.add_argument('--sed-run',required=True)
    parser.add_argument('--run-dir',required=True);parser.add_argument('--phase',required=True,choices=PHASE_SPLITS)
    parser.add_argument('--workers',type=int,default=32);parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1);parser.add_argument('--case-subset')
    a=parser.parse_args()
    if a.command=='evaluate':result=evaluate(a.source_root,a.sed_run,a.run_dir,a.phase,a.workers,a.shard,a.shards,a.case_subset)
    else:result=finish(a.source_root,a.sed_run,a.run_dir,a.phase)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':main()
