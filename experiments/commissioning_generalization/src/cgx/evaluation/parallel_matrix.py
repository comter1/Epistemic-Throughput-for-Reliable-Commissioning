"""Complete, resumable matrix execution using independent CPU case workers."""
from pathlib import Path
import json
import time
import numpy as np
from cgx.config import EXPERIMENT
from cgx.provenance import atomic_json,read_json,hash_json,file_sha256,verify_hashes,append_jsonl,now
from cgx.graphs.manifests import load_cases,write_jsonl
from cgx.training import load_reader,load_policy
from .parallel import rollout_pool,map_rollouts,evaluation_workers
from .matrix import require_frozen,manifest_maps,validate_result_artifact,verify_result_inventory
from .design import main_methods,conditions


def evaluate_parallel(root,config,suite="all",resume=False,sensitivity=False):
    root=Path(root);frozen=require_frozen(root,config);frozen_hash=hash_json(frozen)
    rows,graphs,tasks=manifest_maps(root)
    suites=[s for s in config['splits'] if s.startswith('test_')] if suite=='all' else [suite]
    if any(s not in config['splits'] or not s.startswith('test_') for s in suites):raise ValueError('Unknown test suite')
    append_jsonl(root/'test_access.jsonl',dict(time=now(),purpose='parallel_frozen_test_splits',suites=suites,frozen_model_hash=frozen_hash))
    cases=[c for split in suites for c in load_cases(root,split)]
    for split in suites:
        selected=[c for c in cases if c['split']==split]
        expected=config['splits'][split]['configurations']*config['splits'][split]['cases_per_configuration']
        if len(selected)!=expected or len({c['case_id'] for c in selected})!=expected:raise ValueError('Missing/duplicate sealed test cases')
    if sensitivity:
        ids={cid for r in read_json(root/'manifests/round_sensitivity_subset.json') for cid in r['case_ids']}
        cases=[c for c in cases if c['case_id'] in ids]
    reader=load_reader(root,'final',device='cpu')
    methods=main_methods(config)
    output=root/'evaluation'/('round_sensitivity' if sensitivity else 'main');(output/'cases').mkdir(parents=True,exist_ok=True)
    environment_hash=file_sha256(root/'environment.json');manifest_hash=file_sha256(root/'manifests/hashes.json')
    expected_keys=set();completed=0;started=time.monotonic()
    for method,seed in methods:
        model=load_policy(root,'gnn_myopic' if method=='cost_aware_adaptation' else method,seed,device='cpu') if seed is not None else None
        beta=read_json(root/'models/thresholds.json')['cost_aware_beta'].get(str(seed),.5) if method=='cost_aware_adaptation' else .5
        checkpoint_hash=model.checkpoint_hash if model is not None else hash_json({'method':method,'rule_source':file_sha256(EXPERIMENT/'src/cgx/baselines/policies.py')})
        pending=[];jobs=[]
        for cap,fraction in conditions(config,method,sensitivity):
            if True:
                for case in cases:
                    key=hash_json([method,seed,cap,fraction,case['case_id']]);path=output/'cases'/f'{key}.json'
                    if key in expected_keys:raise ValueError('Duplicate matrix cell')
                    expected_keys.add(key)
                    if resume and path.exists():
                        validate_result_artifact(read_json(path),key,frozen_hash);completed+=1;continue
                    cid=case['configuration_id'];pending.append((key,path))
                    jobs.append((graphs[cid],case,tasks[case['case_id']],fraction,cap,rows[cid]['implementation'],method=='full_recording_unrestricted_reference'))
        if not pending:continue
        with rollout_pool(reader,model,method,config,root/'assets/mujoco_menagerie/franka_emika_panda',beta=beta,store_root=root) as pool:
            for (key,path),(result,traces) in zip(pending,map_rollouts(pool,jobs),strict=True):
                result.update(method=method,policy_seed=seed,protocol_hash=frozen['protocol_hash'],environment_hash=environment_hash,
                    manifest_hash=manifest_hash,checkpoint_hash=checkpoint_hash,reader_hash=reader.checkpoint_hash,technical_smoke=False)
                payload={'row':result,'trace':traces}
                atomic_json(path,dict(state='passed',frozen_hash=frozen_hash,result_hash=hash_json(payload),**payload))
                validate_result_artifact(read_json(path),key,frozen_hash)
                completed+=1
                if completed%100==0:
                    progress=dict(state='running',completed=completed,method=method,seed=seed,workers=evaluation_workers(),elapsed_seconds=time.monotonic()-started)
                    atomic_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    verify_hashes(frozen['files'],root)
    count=verify_result_inventory(output/'cases',expected_keys,allow_other_suites=suite!='all')
    report=dict(state='passed',frozen_hash=frozen_hash,suites=suites,case_files=count,expected_for_requested_suites=len(expected_keys),
                rollout_workers=evaluation_workers(),inference_device='cpu',elapsed_seconds=time.monotonic()-started)
    atomic_json(output/('complete.json' if suite=='all' else f'complete-{suite}.json'),report);return report


_REREAD=None


def _reread_init(root,config):
    global _REREAD
    from cgx.data.store import Store
    from functools import lru_cache
    reader=load_reader(root,'independent',device='cpu');store=Store(root)
    @lru_cache(maxsize=64)
    def predict(digest):
        from .rollout import call_deadline,normalized_prediction
        from cgx.graphs import legal_targets
        history=store.load_history(digest);keys,prior=legal_targets(history['graph'])
        prediction={'target_keys':keys,'log_probabilities_base2':np.log2(prior).tolist()}
        error=None;stamp=time.monotonic()
        try:
            with call_deadline(config['resources']['diagnosis_timeout_seconds_per_call']):
                prediction=normalized_prediction(reader.predict(history),keys,prior)
        except Exception as exc:error=f'{type(exc).__name__}:{exc}'
        return prediction,error,time.monotonic()-stamp
    _REREAD=(predict,reader.checkpoint_hash)


def _reread_one(path):
    predict,reader_hash=_REREAD;row=read_json(path)['row'];digest=row['evidence_trace_hash']
    p,error,seconds=predict(digest);index=p['target_keys'].index(row['true_target_key'])
    return dict(source_result_hash=file_sha256(path),evidence_trace_hash=digest,final_nll_bits=-p['log_probabilities_base2'][index],
        method=row['method'],policy_seed=row['policy_seed'],suite=row['suite'],configuration_id=row['configuration_id'],
        case_id=row['case_id'],byte_fraction=row['byte_fraction'],execution_cap=row['execution_cap'],reacquired_executions=0,reader_hash=reader_hash,
        abstained=error is not None,error=error,inference_seconds=seconds,inference_device='cpu',warmup_seconds=0.)


def reread_parallel(root,config):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp,itertools
    root=Path(root);frozen=require_frozen(root,config)
    complete=read_json(root/'evaluation/main/complete.json')
    if complete['state']!='passed':raise ValueError('Complete acquisition required')
    inputs=sorted((root/'evaluation/main/cases').glob('*.json'))
    if len(inputs)!=complete['expected_for_requested_suites']:raise ValueError('Incomplete acquisition matrix')
    output=[];iterator=iter(inputs);started=time.monotonic()
    with ProcessPoolExecutor(max_workers=evaluation_workers(),mp_context=mp.get_context('spawn'),initializer=_reread_init,initargs=(str(root),config)) as pool:
        while batch:=list(itertools.islice(iterator,512)):
            output.extend(pool.map(_reread_one,batch,chunksize=8))
            atomic_json(root/'evaluation/reread_progress.json',dict(state='running',completed=len(output),total=len(inputs),elapsed_seconds=time.monotonic()-started))
    write_jsonl(root/'evaluation/independent_reader.jsonl',output);verify_hashes(frozen['files'],root)
    report=dict(state='passed',rows=len(output),unique_transcripts=len({r['evidence_trace_hash'] for r in output}),reacquired_executions=0,
        output_hash=file_sha256(root/'evaluation/independent_reader.jsonl'),reader_hash=output[0]['reader_hash'],frozen_hash=hash_json(frozen),
        workers=evaluation_workers(),inference_device='cpu',elapsed_seconds=time.monotonic()-started)
    atomic_json(root/'evaluation/independent_reader_complete.json',report);return report
