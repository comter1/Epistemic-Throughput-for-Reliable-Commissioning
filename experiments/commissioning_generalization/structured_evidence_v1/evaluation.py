"""Full original benchmark/validation matrix with a frozen structured reader.

Independent process workers retain the original per-call wall deadlines. Each
matrix cell is atomically written and uniquely owned; methods, budgets, seeds,
case manifests and normalization rules are imported from the old benchmark.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import fcntl
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch

from cgx.baselines import RuntimePolicy
from cgx.config import load_config
from cgx.evaluation.design import main_methods, conditions, trained_method
from cgx.evaluation.rollout import target_key
from cgx.graphs import legal_targets
from cgx.provenance import atomic_json, read_json, hash_json, file_sha256, now, verify_hashes
from cgx.training.engine import load_policy
from cgx.training.stopping import validate_completion
from .rollout import rollout
from .selectors import model_spec, construct_policy
from .storage import OverlayStore

_WORKER=None


def _initialize(source_root,run_root,config,spec,method,beta):
    global _WORKER
    from .reader import load_reader
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    store=OverlayStore(source_root,run_root)
    configurations={r['configuration_id']:r for r in store.configurations()}
    _WORKER=dict(source_root=Path(source_root),run_root=Path(run_root),config=config,
        reader=load_reader(run_root,'final',device='cpu'),policy=RuntimePolicy(method,construct_policy(spec),beta),
        store=store,configurations=configurations)


def _job(job):
    case,cap,fraction=job
    w=_WORKER;store=w['store'];graph=store.graph(case['configuration_id']);started=time.monotonic()
    try:
        row,traces,_,_=rollout(graph,case,store.task(case['case_id']),w['config'],w['reader'],w['policy'],fraction,cap,
            implementation=w['configurations'][case['configuration_id']]['implementation'],
            asset_path=w['source_root']/'assets/mujoco_menagerie/franka_emika_panda',store=store,
            source_root=w['source_root'],run_root=w['run_root'])
    except Exception as exc:
        # Preserve the planned denominator and explicit prior fallback even if
        # construction fails before the original rollout's own error handler.
        keys,prior=legal_targets(graph);truth=target_key(case);index=keys.index(truth)
        prediction={'target_keys':list(keys),'log_probabilities_base2':np.log2(prior).tolist()}
        predicted=keys[int(np.argmax(prior))]
        row=dict(case_id=case['case_id'],configuration_id=case['configuration_id'],suite=case['split'],
            byte_fraction=fraction,execution_cap=cap,initial_nll_bits=float(-np.log2(prior[index])),
            final_nll_bits=float(-np.log2(prior[index])),prior_nll_bits=float(-np.log2(prior[index])),
            score_improvement_bits=0.,target_cardinality=len(keys),predicted_target_key=predicted,true_target_key=truth,
            correct_joint=predicted==truth,correct_mechanism=predicted.rsplit(':',1)[-1]==truth.rsplit(':',1)[-1],
            bytes_used=None,executions_used=None,calibrations_used=None,decisions_used=None,initial_bytes=None,
            selection_seconds=None,inference_seconds=None,simulation_seconds=None,recorder_seconds=None,
            warmup_seconds=None,selection_warmup_seconds=None,end_to_end_seconds=time.monotonic()-started,
            timeout=isinstance(exc,TimeoutError),abstained=True,errors=[f'worker_error:{type(exc).__name__}:{exc}'],
            visible_truncation=None,evidence_trace_hash=None,event_provenance_hash=None,feasible=None,
            unrestricted_reference=False,prediction=prediction,worker_failure=True,traceback=traceback.format_exc(),
            physical_cache_hits=None,new_robot_execution_attempts=None,new_calibration_attempts=None,
            physical_acquisition_measured_seconds=None,diagnostic_family='structured_evidence_v1',
            logical_acquisition_reference_seconds=None,actual_new_physical_seconds=None,provenance_seconds=None,
            resource_accounting_unavailable=True,wall_time_scope='failed_before_or_outside_normal_rollout')
        traces=[]
    row.update(inference_device='cpu',rollout_execution='independent_process',cpu_threads_per_worker=1,
        healthy_false_alarm=case['fault_family']=='healthy' and row['predicted_target_key']!='healthy',
        fault_miss=case['fault_family']!='healthy' and row['predicted_target_key']=='healthy',fault_family=case['fault_family'])
    return row,traces


def map_rollouts(source_root,run_root,config,model,method,jobs,*,workers=32,beta=.5):
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_initialize,
            initargs=(str(source_root),str(run_root),config,model_spec(model),method,beta)) as pool:
        iterator=iter(jobs)
        while batch:=list(itertools.islice(iterator,max(128,workers*4))):
            yield from pool.map(_job,batch,chunksize=1)


def validation_callback(source_root,run_root,config,workers=32,artifact_directory=None):
    store=OverlayStore(source_root,run_root);cases=list(store.cases('validation'));checks=0
    if artifact_directory:
        existing=list(Path(artifact_directory).glob('check_*.json'))
        checks=max((int(p.stem.split('_')[-1]) for p in existing),default=0)
    assert len(cases)==config['splits']['validation']['configurations']*config['splits']['validation']['cases_per_configuration']
    def evaluate(model,reader,method):
        nonlocal checks
        checks+=1;started=time.monotonic();was_training=model.training
        deployed='cost_aware_adaptation' if method=='gnn_myopic' else method
        by_config=defaultdict(list)
        try:
            jobs=[(c,config['evaluation']['main_execution_cap'],config['evaluation']['primary_byte_fraction']) for c in cases]
            for row,trace in map_rollouts(source_root,run_root,config,model,deployed,jobs,workers=workers):
                by_config[row['configuration_id']].append(row)
            assert sum(map(len,by_config.values()))==len(cases)
            def mean(field):
                if any(r.get(field) is None for group in by_config.values() for r in group):return None
                return float(np.mean([np.mean([r[field] for r in group]) for group in by_config.values()]))
            metric=dict(mean_final_nll_bits=mean('final_nll_bits'),mean_bytes_used=mean('bytes_used'),
                case_count=len(cases),configurations=len(by_config),byte_fraction=config['evaluation']['primary_byte_fraction'],
                validation_only=True,timeouts=sum(r['timeout'] for group in by_config.values() for r in group),
                abstentions=sum(r['abstained'] for group in by_config.values() for r in group),
                resource_accounting_unavailable=sum(r.get('resource_accounting_unavailable',False) for group in by_config.values() for r in group),
                elapsed_seconds=time.monotonic()-started,rollout_workers=workers,inference_device='cpu')
            if artifact_directory:
                atomic_json(Path(artifact_directory)/f'check_{checks:04d}.json',dict(metric=metric,
                    rows=[r for group in by_config.values() for r in group],reader_hash=reader.checkpoint_hash))
            if metric['resource_accounting_unavailable']:
                raise RuntimeError('Validation retained a worker failure with unknown resource usage; actual-byte checkpoint tie-break cannot be evaluated')
            return metric
        finally:
            model.train(was_training)
    return evaluate


def select_beta(source_root,run_root,seed,workers=32):
    run=Path(run_root);config=load_config();store=OverlayStore(source_root,run_root)
    cases=list(store.cases('validation'));model=load_policy(run,'gnn_myopic',seed,device='cpu')
    output=run/'models'/f'cost_beta_seed{seed}.json';curves=[]
    for beta in config['methods']['cost_aware_adaptation']['beta_grid']:
        rows=[r for r,_ in map_rollouts(source_root,run,config,model,'cost_aware_adaptation',
            [(c,4,1.) for c in cases],workers=workers,beta=beta)]
        groups=defaultdict(list)
        for row in rows:groups[row['configuration_id']].append(row)
        unavailable=sum(r.get('resource_accounting_unavailable',False) for r in rows)
        if unavailable:
            atomic_json(run/'validation'/f'cost_beta_seed{seed}_{beta}.json',dict(rows=rows,
                state='failed',resource_accounting_unavailable=unavailable))
            raise RuntimeError('Beta validation retained unknown resource usage; actual-byte tie-break cannot be evaluated')
        mean=lambda field:float(np.mean([np.mean([r[field] for r in g]) for g in groups.values()]))
        curves.append(dict(beta=beta,loss=mean('final_nll_bits'),bytes=mean('bytes_used'),
            cases=len(rows),abstentions=sum(r['abstained'] for r in rows)))
        atomic_json(run/'validation'/f'cost_beta_seed{seed}_{beta}.json',dict(rows=rows,metric=curves[-1]))
    selected=min(curves,key=lambda x:(x['loss'],x['bytes'],x['beta']))['beta']
    result=dict(state='passed',validation_only=True,seed=seed,selected_beta=selected,curves=curves,
        checkpoint_hash=model.checkpoint_hash,cases=len(cases))
    atomic_json(output,result);return result


def freeze_models(run_root):
    from .reader import load_reader
    run=Path(run_root);config=load_config();files={};inventory={}
    for method in config['methods']['learned']:
        for seed in config['seeds']['training']:
            directory=run/'models/policies'/f'{method}_seed{seed}'
            marker=read_json(directory/'complete.json');history=read_json(directory/'validation.json')
            validate_completion(marker,history,config['training'],config['training']['policy_update_cap'])
            best=min(history,key=lambda r:(r['mean_final_nll_bits'],r.get('mean_bytes_used',0),r['iteration']))
            filename=f"checkpoint_{best['iteration']:07d}.pt"
            assert marker['selected_checkpoint']==filename
            assert file_sha256(directory/filename)==marker['selected_checkpoint_hash']
            for p in (directory/filename,directory/'complete.json',directory/'validation.json'):
                files[str(p.relative_to(run))]=file_sha256(p)
            inventory[f'{method}_seed{seed}']=dict(checkpoint=filename,selected_iteration=best['iteration'],actual_updates=marker['actual_updates'])
    beta={}
    for seed in config['seeds']['training']:
        path=run/'models'/f'cost_beta_seed{seed}.json';value=read_json(path)
        assert value['state']=='passed' and value['validation_only']
        files[str(path.relative_to(run))]=file_sha256(path);beta[str(seed)]=value['selected_beta']
    reader=load_reader(run,'final',device='cpu')
    scientific_code={p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')
        if p.name in ('reader.py','events.py','likelihood.py','selectors.py','rollout.py','evaluation.py','policy_training.py','storage.py')}
    record=dict(state='passed',frozen_at=now(),validation_only=True,files=files,
        inventory=inventory,cost_aware_beta=beta,reader_hash=reader.checkpoint_hash,
        protocol_hash=hash_json(config),scientific_code=scientific_code,online_learning=False)
    path=run/'models/frozen.json'
    if path.exists():
        old=read_json(path)
        for key in ('files','reader_hash','protocol_hash','scientific_code'):assert old[key]==record[key]
        return old
    atomic_json(path,record);return record


def require_frozen(run):
    from .reader import load_reader
    run=Path(run);frozen=read_json(run/'models/frozen.json')
    assert frozen['state']=='passed' and frozen['protocol_hash']==hash_json(load_config())
    verify_hashes(frozen['files'],run)
    assert load_reader(run,'final',device='cpu').checkpoint_hash==frozen['reader_hash']
    for name,sha in frozen['scientific_code'].items():assert file_sha256(Path(__file__).parent/name)==sha,name
    return frozen


def evaluate(source_root,run_root,method,seed,workers=32,suite='all',shard=0,shards=1):
    run=Path(run_root);config=load_config();frozen=require_frozen(run);frozen_hash=hash_json(frozen)
    assert (method,seed) in main_methods(config)
    assert 0<=shard<shards
    store=OverlayStore(source_root,run);suites=[s for s in config['splits'] if s.startswith('test_')] if suite=='all' else [suite]
    cases=[c for split in suites for c in store.cases(split)]
    for split in suites:
        assert sum(c['split']==split for c in cases)==config['splits'][split]['configurations']*config['splits'][split]['cases_per_configuration']
    cases=[c for c in cases if int(c['case_id'],16)%shards==shard]
    model=load_policy(run,trained_method(method),seed,device='cpu') if seed is not None else None
    beta=frozen['cost_aware_beta'].get(str(seed),.5) if method=='cost_aware_adaptation' else .5
    directory=run/'evaluation/main';(directory/'cases').mkdir(parents=True,exist_ok=True)
    task_name=f'{method}_seed{seed}_{suite}_shard{shard}of{shards}'
    lock_path=directory/(task_name+'.lock')
    with lock_path.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=[];expected=[];started=time.monotonic()
        for cap,fraction in conditions(config,method):
            for case in cases:
                key=hash_json([method,seed,cap,fraction,case['case_id']]);path=directory/'cases'/(key+'.json')
                expected.append(key)
                if path.exists():
                    saved=read_json(path);assert saved['frozen_hash']==frozen_hash
                    assert saved['result_hash']==hash_json({'row':saved['row'],'trace':saved['trace']})
                else:pending.append((key,path,(case,cap,fraction)))
        complete=len(expected)-len(pending)
        for (key,path,_),(row,trace) in zip(pending,map_rollouts(source_root,run,config,model,method,
                [p[2] for p in pending],workers=workers,beta=beta),strict=True):
            row.update(method=method,policy_seed=seed,protocol_hash=frozen['protocol_hash'],reader_hash=frozen['reader_hash'],
                checkpoint_hash=model.checkpoint_hash if model is not None else hash_json({'method':method,'rule':'original_full_recording'}),
                technical_smoke=False)
            payload=dict(row=row,trace=trace)
            atomic_json(path,dict(state='passed',frozen_hash=frozen_hash,result_hash=hash_json(payload),**payload))
            complete+=1
            if complete%100==0:
                progress=dict(state='running',task=task_name,completed=complete,expected=len(expected),elapsed_seconds=time.monotonic()-started)
                atomic_json(directory/(task_name+'_progress.json'),progress);print(json.dumps(progress),flush=True)
        result=dict(state='passed',task=task_name,cells=len(expected),frozen_hash=frozen_hash,
            elapsed_seconds=time.monotonic()-started,workers=workers,shard=shard,shards=shards)
        atomic_json(directory/(task_name+'_complete.json'),result);return result


def finish(source_root,run_root):
    run=Path(run_root);config=load_config();frozen=require_frozen(run);fh=hash_json(frozen)
    store=OverlayStore(source_root,run);cases=[c for s in config['splits'] if s.startswith('test_') for c in store.cases(s)]
    rows=[];expected=set()
    for method,seed in main_methods(config):
        for cap,fraction in conditions(config,method):
            for case in cases:
                key=hash_json([method,seed,cap,fraction,case['case_id']]);expected.add(key)
                p=read_json(run/'evaluation/main/cases'/(key+'.json'))
                assert p['state']=='passed' and p['frozen_hash']==fh
                assert p['result_hash']==hash_json({'row':p['row'],'trace':p['trace']})
                rows.append(p['row'])
    actual={p.stem for p in (run/'evaluation/main/cases').glob('*.json')}
    assert actual==expected
    import pandas as pd
    frame=pd.DataFrame(rows);frame.to_parquet(run/'evaluation/main/per_case_results.parquet',index=False)
    result=dict(state='passed',expected=len(expected),rows=len(rows),physical_cases=len(cases),
        frozen_hash=fh,abstentions=int(frame.abstained.sum()),timeouts=int(frame.timeout.sum()),
        method_seed_budget_cells=len(expected)//len(cases),all_planned_denominators_retained=True)
    atomic_json(run/'evaluation/main/complete.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['evaluate','select-beta','freeze','finish'])
    p.add_argument('--source-root',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--method');p.add_argument('--seed',type=int);p.add_argument('--workers',type=int,default=32)
    p.add_argument('--suite',default='all');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args()
    if a.command=='evaluate':result=evaluate(a.source_root,a.run_dir,a.method,a.seed,a.workers,a.suite,a.shard,a.shards)
    elif a.command=='select-beta':result=select_beta(a.source_root,a.run_dir,a.seed,a.workers)
    elif a.command=='freeze':result=freeze_models(a.run_dir)
    else:result=finish(a.source_root,a.run_dir)
    print(json.dumps(result,sort_keys=True),flush=True)
