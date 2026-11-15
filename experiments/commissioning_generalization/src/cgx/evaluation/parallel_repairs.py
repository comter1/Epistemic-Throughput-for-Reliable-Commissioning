"""Parallel real post-repair verification/audit on the frozen acquired decisions."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing as mp
import itertools
import time
import numpy as np
from cgx.provenance import read_json,atomic_json,hash_json,verify_hashes
from cgx.graphs.manifests import load_cases
from .matrix import require_frozen,manifest_maps,verify_result_inventory,validate_result_artifact,result_identity
from .design import seeded_method
from .parallel import evaluation_workers

_CONTEXT=None


def _initialize(root,config):
    global _CONTEXT
    from cgx.data.store import Store
    _CONTEXT=(Path(root),config,Store(root))


def _one(job):
    from cgx.env.system import System
    from .repairs import verification
    root,config,store=_CONTEXT
    case,graph,task,implementation,acquired,threshold=job
    confidence=float(np.exp2(max(acquired['prediction']['log_probabilities_base2'])))
    system=System(graph,case,task,config,root/'assets/mujoco_menagerie/franka_emika_panda',implementation)
    result=verification(system,case,config,acquired['predicted_target_key'],confidence,threshold,acquired['abstained'],store=store)
    result.update(case_id=case['case_id'],configuration_id=case['configuration_id'],suite=case['split'],method=acquired['method'],policy_seed=acquired['policy_seed'],
        confidence=confidence,threshold=threshold,evidence_trace_hash=acquired['evidence_trace_hash'],diagnostic_bytes=acquired['bytes_used'],
        diagnostic_executions=acquired['executions_used'],total_deployed_executions=1+acquired['executions_used']+result['public_verification_count'],
        total_deployed_bytes=acquired['initial_bytes']+acquired['bytes_used']+result['public_cost']['bytes'],
        total_deployed_seconds=acquired['end_to_end_seconds']+result['repair_seconds']+result['public_cost']['wall_seconds'],technical_smoke=False)
    return result


def evaluate_repairs_parallel(root,config,resume=False):
    root=Path(root);frozen=require_frozen(root,config);frozen_hash=hash_json(frozen)
    if read_json(root/'evaluation/main/complete.json')['state']!='passed':raise ValueError('Complete main acquisition required')
    thresholds=read_json(root/'models/thresholds.json');subset=read_json(root/'manifests/repair_subset.json')
    cases={c['case_id']:c for split in config['splits'] if split.startswith('test_') for c in load_cases(root,split)}
    rows,graphs,tasks=manifest_maps(root);inputs={}
    for path in (root/'evaluation/main/cases').glob('*.json'):
        artifact=read_json(path);row=artifact['row']
        if row['byte_fraction']!=config['evaluation']['repair_byte_fraction'] or row['execution_cap']!=config['evaluation'].get('repair_execution_cap',4) or row['method'] not in config['evaluation']['repair_methods']:continue
        validate_result_artifact(artifact,result_identity(row),frozen_hash)
        key=(row['case_id'],row['method'],row['policy_seed'])
        if key in inputs:raise ValueError('Ambiguous repair acquisition input')
        inputs[key]=row
    expected=set();pending=[];jobs=[];output=root/'evaluation/repairs';started=time.monotonic()
    for cid in [cid for item in subset for cid in item['case_ids']]:
        case=cases[cid];configuration=case['configuration_id']
        for method in config['evaluation']['repair_methods']:
            seeds=config['evaluation']['repair_training_seeds'] if seeded_method(method,config) else [None]
            for seed in seeds:
                key=(cid,method,seed);digest=hash_json(key);path=output/'cases'/f'{digest}.json'
                if digest in expected:raise ValueError('Duplicate repair cell')
                expected.add(digest)
                if resume and path.exists():
                    old=read_json(path)
                    if old.get('state')!='passed' or old.get('frozen_hash')!=frozen_hash or old.get('result_hash')!=hash_json(old['row']):raise ValueError('Repair artifact changed')
                    continue
                if key not in inputs:raise ValueError('Missing required frozen acquisition for repair')
                threshold_key=f'{method}:{seed}'
                if threshold_key not in thresholds['repair_confidence']:raise ValueError('Missing frozen repair threshold')
                jobs.append((case,graphs[configuration],tasks[cid],rows[configuration]['implementation'],inputs[key],thresholds['repair_confidence'][threshold_key]));pending.append(path)
    iterator=iter(zip(pending,jobs));count=0
    with ProcessPoolExecutor(max_workers=evaluation_workers(),mp_context=mp.get_context('spawn'),initializer=_initialize,initargs=(str(root),config)) as pool:
        while batch:=list(itertools.islice(iterator,256)):
            for (path,_),result in zip(batch,pool.map(_one,[job for _,job in batch],chunksize=4),strict=True):
                atomic_json(path,dict(state='passed',frozen_hash=frozen_hash,result_hash=hash_json(result),row=result));count+=1
            atomic_json(output/'progress.json',dict(state='running',completed_this_invocation=count,expected=len(expected),elapsed_seconds=time.monotonic()-started))
    verify_hashes(frozen['files'],root);complete=verify_result_inventory(output/'cases',expected)
    report=dict(state='passed',rows=complete,expected_rows=len(expected),completed_this_invocation=count,workers=evaluation_workers(),elapsed_seconds=time.monotonic()-started)
    atomic_json(output/'complete.json',report);return report
