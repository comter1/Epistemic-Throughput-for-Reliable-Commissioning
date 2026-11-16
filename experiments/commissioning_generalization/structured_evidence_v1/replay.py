"""Read the exact saved Panda histories with frozen diagnostic readers.

No environment or simulator is imported. Evaluator labels never enter a reader.
The original deployed result is retained alongside an untimed native GNN reread,
R-C, and the Structured Evidence Diagnoser. Parallel workers own disjoint shards.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import pandas as pd

from cgx.data.store import Store
from cgx.provenance import atomic_json, file_sha256, hash_json, read_json, verify_hashes

VERSION = 'structured-evidence-replay-v1'
READERS = ('original_as_deployed', 'original_GNN_reread', 'R-C', 'structured')
EXPECTED_MAIN_ROWS = 147200


def _json(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def write_gzip(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with gzip.open(tmp, 'wt') as handle: json.dump(value, handle, default=_json)
    tmp.replace(path)


def read_gzip(path):
    with gzip.open(path, 'rt') as handle: return json.load(handle)


def prediction_metrics(prediction, truth):
    keys = list(prediction['target_keys'])
    lp = np.asarray(prediction['log_probabilities_base2'], dtype=np.float64)
    if len(keys) != len(set(keys)) or 'healthy' not in keys or truth not in keys:
        raise ValueError('Invalid prediction target space')
    if lp.shape != (len(keys),) or not np.isfinite(lp).all():
        raise ValueError('Invalid prediction values')
    probability = np.exp2(lp)
    if abs(probability.sum()-1.) > 1e-5: raise ValueError('Unnormalized prediction')
    probability /= probability.sum()
    best = int(probability.argmax()); healthy = keys.index('healthy'); target = keys.index(truth)
    predicted = keys[best]; is_healthy = truth == 'healthy'
    one_hot = np.zeros(len(keys)); one_hot[target] = 1.
    return dict(predicted_target_key=predicted, p_healthy=float(probability[healthy]),
        max_fault_probability=float(np.max(np.delete(probability, healthy))),
        confidence=float(probability[best]), correct=int(predicted == truth),
        is_healthy=is_healthy, healthy_false_alarm=int(is_healthy and predicted != 'healthy'),
        fault_miss=int(not is_healthy and predicted == 'healthy'),
        joint_fault_correct=int(not is_healthy and predicted == truth),
        nll_bits=float(-lp[target]), brier=float(np.square(probability-one_hot).sum()),
        target_keys=json.dumps(keys), full_log_probabilities=json.dumps(lp.tolist()))


def provenance_for_history(history, case_id):
    """Opaque identities from actual acquisition positions, not numerical values.

    Packaging transforms are applied only AFTER these identities are attached.
    This adapts the legacy fixed one-block-per-acquisition storage contract. It
    must not be used to assign new IDs to split or duplicated storage blocks.
    """
    execution = burst = 0; ids = []
    for index, obs in enumerate(history['observations']):
        if index == 0:
            ids.append(hash_json([case_id, 'initial_acquisition']))
        else:
            cost = obs['actual_cost']
            execution += int(cost['executions']); burst += int(cost['calibrations'])
            ids.append(hash_json([case_id, 'acquisition', execution, burst, obs['action_key']]))
    if len(ids) != len(set(ids)):
        raise ValueError('Legacy acquisition metadata does not uniquely identify its blocks')
    return {'observation_ids': ids}


def materialize_packed(store, packed):
    return dict(graph=store._get('graphs', packed['graph_ref']),
        observations=[store.load_observation(ref) for ref in packed['observation_refs']],
        nominal_task=store._get('tasks', packed['task_ref']),
        requirements=deepcopy(packed['requirements']), remaining=dict(packed['remaining']),
        initial_budget=dict(packed['initial_budget']), action_keys=list(packed['action_keys']))


def _problem_contexts(store_root, paths, cohort, v1=False):
    store = Store(store_root)
    for path in paths:
        item = read_gzip(path); row = item['row']
        if v1 and row['arm'] != 'A': continue
        trace = item['trace']; packed = store.history_references(row['evidence_trace_hash'])
        if len(packed['observation_refs']) != len(trace)+1:
            raise AssertionError('Legacy prefix acquisition chain is incomplete')
        first_execution = next((i+1 for i,t in enumerate(trace)
            if t['resources_before']['executions'] > t['resources_after']['executions']), None)
        for i in range(len(trace)+1):
            pp = dict(packed, observation_refs=packed['observation_refs'][:i+1],
                remaining=dict(packed['initial_budget'] if i == 0 else trace[i-1]['resources_after']),
                action_keys=[t['action']['action_key'] for t in trace[:i]])
            pred = row['initial_prediction'] if v1 and i == 0 else (
                trace[i-1]['prediction'] if v1 else item['prefixes'][i]['prediction'])
            metadata = dict(cohort=cohort, suite='diagnostic_confirmation',
                case_id=row['case_id'], configuration_id=row['configuration_id'],
                method=row['method_key'], policy_seed=int(row['policy_seed']), execution_cap=4,
                true_target_key=row['true_target_key'],
                fault_family='healthy' if row['true_target_key']=='healthy' else row['true_target_key'].split(':',1)[1],
                prefix_ordinal=i, is_terminal=i==len(trace), first_execution_ordinal=first_execution,
                source_trajectory=str(path), history_hash=hash_json(pp),
                executions_used=packed['initial_budget']['executions']-pp['remaining']['executions'],
                calibrations_used=packed['initial_budget']['calibrations']-pp['remaining']['calibrations'],
                bytes_used=packed['initial_budget']['bytes']-pp['remaining']['bytes'],
                decisions_used=packed['initial_budget']['decisions']-pp['remaining']['decisions'],
                original_timeout=False, original_abstained=False,
                action_kind='initial' if i==0 else trace[i-1]['action']['kind'],
                ports=json.dumps([] if i==0 else trace[i-1]['action']['ports']))
            yield metadata, pred, pp


def inventory(source_root, run_dir, v1_run, v2_run, chunk_size=128):
    source_root, run = Path(source_root), Path(run_dir)
    output = run/'replay'; output.mkdir(parents=True, exist_ok=True)
    if (output/'inventory.json').exists():
        existing = read_json(output/'inventory.json')
        if existing['source_table_sha256'] != file_sha256(source_root/'results/per_case_results.parquet'):
            raise ValueError('Original result table changed')
        return existing
    frame = pd.read_parquet(source_root/'results/per_case_results.parquet')
    if len(frame) != EXPECTED_MAIN_ROWS or frame.case_id.nunique() != 3200:
        raise ValueError('Incomplete original main matrix')
    jobs = {}; contexts = []
    def register(meta, pred, store_root, packed=None):
        key = hash_json([str(store_root), meta['history_hash'], meta['case_id']])
        job = dict(job_id=key, store_root=str(store_root), history_hash=meta['history_hash'],
            case_id=meta['case_id'], packed=packed)
        if key in jobs and jobs[key]['packed'] is None and packed is not None: jobs[key]['packed']=packed
        else: jobs.setdefault(key, job)
        contexts.append(dict(meta, job_id=key, stored_prediction=json.dumps(pred, default=_json)))
    resource_fields=('executions_used','calibrations_used','bytes_used','decisions_used',
        'selection_seconds','inference_seconds','end_to_end_seconds','simulation_seconds','recorder_seconds')
    for row in frame.to_dict('records'):
        meta={k:row[k] for k in ('suite','case_id','configuration_id','method','policy_seed','execution_cap',
                                 'true_target_key','fault_family')}
        meta.update(cohort='main', history_hash=row['evidence_trace_hash'], prefix_ordinal=-1,
            is_terminal=True, first_execution_ordinal=None, original_timeout=bool(row['timeout']),
            original_abstained=bool(row['abstained']), source_trajectory=None)
        meta.update({k:row[k] for k in resource_fields})
        register(meta,row['prediction'],source_root)
    main_jobs=len(jobs)
    for root,cohort,v1,subdir in [(Path(v1_run),'fa_v1',True,'interventions/confirmation/cases'),
                                (Path(v2_run),'fa_v2',False,'trajectories/confirmation/cases')]:
        for meta,pred,packed in _problem_contexts(root,sorted((root/subdir).glob('*.json.gz')),cohort,v1):
            register(meta,pred,root,packed)
    context=pd.DataFrame(contexts)
    context.to_parquet(output/'contexts.parquet',index=False)
    ordered=[jobs[k] for k in sorted(jobs)]
    shards=[]
    for off in range(0,len(ordered),chunk_size):
        part=ordered[off:off+chunk_size]; index=off//chunk_size
        path=output/'jobs'/f'{index:05d}.json.gz';write_gzip(path,part)
        shards.append(dict(index=index,path=str(path),jobs=len(part),sha256=file_sha256(path)))
    result=dict(state='passed',version=VERSION,main_rows=len(frame),main_physical_cases=3200,
        main_unique_final_histories=main_jobs,all_prediction_jobs=len(jobs),contexts=len(context),
        contexts_by_cohort=context.groupby('cohort').size().to_dict(),
        source_table_sha256=file_sha256(source_root/'results/per_case_results.parquet'),
        contexts_sha256=file_sha256(output/'contexts.parquet'),chunk_size=chunk_size,shards=shards,
        source_root=str(source_root),v1_run=str(v1_run),v2_run=str(v2_run),physical_acquisitions=0)
    atomic_json(output/'inventory.json',result)
    return result


_STORES = {}; _STRUCTURED = None; _MODE = 'all'


def _worker_init(run_dir, mode):
    import torch
    torch.set_num_threads(1)
    global _STRUCTURED, _MODE
    _MODE=mode
    if mode!='baselines':
        from .reader import load_reader
        _STRUCTURED=load_reader(run_dir,name='final',device='cpu')


def _prepare(job):
    from cgx.models.features import tensorize_history
    stamp=time.perf_counter()
    store=_STORES.setdefault(job['store_root'],Store(job['store_root']))
    packed=job['packed'] or store.history_references(job['history_hash'])
    if hash_json(packed) != job['history_hash']: raise ValueError('Packed public history hash mismatch')
    history=materialize_packed(store,packed)
    provenance=provenance_for_history(history,job['case_id'])
    structured_start=time.perf_counter()
    structured=_STRUCTURED.predict(history,provenance=dict(provenance,observation_refs=packed['observation_refs'])) if _MODE!='baselines' else None
    common=dict(job_id=job['job_id'],structured=structured,
        structured_seconds=time.perf_counter()-structured_start if structured is not None else None,
        observation_ids=provenance['observation_ids'],observation_refs=packed['observation_refs'],
        provenance_hash=hash_json(provenance),
        raw_record_count=sum(len(o.get('records',())) for o in history['observations']))
    if _MODE=='structured':
        return dict(common,preparation_seconds=time.perf_counter()-stamp)
    features=tensorize_history(history)
    ports=defaultdict(list)
    for obs in history['observations']:
        grouped=defaultdict(list)
        for record in obs.get('records',()): grouped[record['port']].append(record)
        for port,records in grouped.items():
            owner=features.node_lookup.get(port.split(':')[0],-1)
            if owner>=0:ports[(owner,port)].append((len(records),any(r['missing'] for r in records)))
    raw=np.zeros((len(features.nodes),3),dtype=np.float32)
    for (owner,_),rows in ports.items():
        raw[owner,0]+=1
        raw[owner,1]+=np.mean([r[0] for r in rows])/1000
        raw[owner,2]+=np.mean([r[1] for r in rows])
    corrected=features.nodes.copy();corrected[:,23:26]=np.log1p(raw)
    corrected[:features.physical_count,31]=1-np.minimum(corrected[:features.physical_count,23],1)
    return dict(common,features=features,corrected_nodes=corrected,preparation_seconds=time.perf_counter()-stamp)


def _bounded_map(pool, jobs, ahead):
    iterator=iter(jobs);pending=deque()
    for _ in range(ahead):
        try:pending.append(pool.submit(_prepare,next(iterator)))
        except StopIteration:break
    while pending:
        result=pending.popleft().result();yield result
        try:pending.append(pool.submit(_prepare,next(iterator)))
        except StopIteration:pass


def _gnn_batch(model, prepared, check_noop=False):
    import torch
    from cgx.models.features import collate_features
    device=next(model.parameters()).device
    features=[p['features'] for p in prepared];batch=collate_features(features,device)
    captured={}
    def hook(module,args,output): captured['pools']=output
    handle=model.backbone.evidence.register_forward_hook(hook)
    with torch.inference_mode():native=model(batch).cpu().numpy()
    handle.remove();old_forward=model.backbone.evidence.forward
    try:
        model.backbone.evidence.forward=lambda b:captured['pools']
        if check_noop:
            with torch.inference_mode():noop=model(batch).cpu().numpy()
            error=float(np.max(np.abs(np.exp2(native)-np.exp2(noop))))
            if error>1e-6:raise AssertionError(f'Batched pool no-op mismatch: {error}')
        else:error=None
        batch['nodes']=torch.as_tensor(np.concatenate([p['corrected_nodes'] for p in prepared]),device=device)
        with torch.inference_mode():corrected=model(batch).cpu().numpy()
    finally:model.backbone.evidence.forward=old_forward
    if device.type=='cuda':torch.cuda.synchronize(device)
    output=[]
    for i,part in enumerate(prepared):
        keys=list(part['features'].targets)
        predictions=dict(original_GNN_reread=dict(target_keys=keys,log_probabilities_base2=native[i,:len(keys)].tolist()),
            **{'R-C':dict(target_keys=keys,log_probabilities_base2=corrected[i,:len(keys)].tolist())})
        if part['structured'] is not None:predictions['structured']=part['structured']
        for pred in predictions.values():
            if list(pred['target_keys'])!=keys:raise ValueError('Readers disagree on legal target ordering')
        output.append(dict(job_id=part['job_id'],predictions=predictions,
            structured_seconds=part['structured_seconds'],preparation_seconds=part['preparation_seconds'],
            observation_ids=part['observation_ids'],observation_refs=part['observation_refs'],
            provenance_hash=part['provenance_hash'],
            raw_record_count=part['raw_record_count']))
    return output,error


def worker(source_root,run_dir,worker_index,worker_count,device='cuda:0',workers=16,batch_size=8,limit_shards=None,mode='all',prefetch=None,output_tag=''):
    import torch
    from diagnostics.false_alarm_causes.core import inference_settings,load_reader
    run=Path(run_dir);manifest=read_json(run/'replay/inventory.json')
    if output_tag and not output_tag.replace('_','').isalnum():raise ValueError('Unsafe output tag')
    namespace=mode+('_'+output_tag if output_tag else '')
    ahead=max(workers,batch_size*2) if prefetch is None else int(prefetch)
    if ahead<workers:raise ValueError('Prefetch must cover allocated workers')
    if mode=='baselines':
        gate=read_json(run/'manifests/reader_baseline_replay_authorized.json')
        if gate['state']!='passed':raise ValueError('Baseline-only replay has not been authorized')
        reader_hash=None
    else:
        from .reader import load_reader as load_structured
        gate=read_json(run/'manifests/reader_replay_authorized.json')
        if gate['state']!='passed':raise ValueError('Reader selection has not been sealed for replay')
        frozen=read_json(run/'models/readers/frozen.json')
        if frozen['state']!='sealed_before_reader_replay':raise ValueError('Reader fit manifest not sealed')
        if gate['reader_frozen_sha256']!=file_sha256(run/'models/readers/frozen.json'):
            raise ValueError('Reader freeze manifest changed after authorization')
        verify_hashes(frozen['files'],run)
        verify_hashes(frozen['runtime_files'],Path(__file__).parents[1])
        structured=load_structured(run,name='final',device='cpu')
        if gate['reader_checkpoint_hash']!=structured.checkpoint_hash:raise ValueError('Structured reader changed after seal')
        reader_hash=structured.checkpoint_hash;del structured
    inference_settings();model=load_reader(source_root,'final',device) if mode!='structured' else None
    selected=[s for s in manifest['shards'] if s['index']%worker_count==worker_index]
    if limit_shards is not None:selected=selected[:limit_shards]
    started=time.monotonic();completed=0;forward_items=0;noop_errors=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=_worker_init,initargs=(str(run),mode)) as pool:
        for shard in selected:
            dest=run/f'replay/predictions_{namespace}'/f"{shard['index']:05d}.json.gz"
            if dest.exists():
                previous=read_gzip(dest)
                if previous['input_sha256']!=shard['sha256'] or previous['reader_checkpoint_hash']!=reader_hash:
                    raise ValueError('Replay resume inputs changed')
                completed+=len(previous['predictions']);continue
            if file_sha256(shard['path'])!=shard['sha256']:raise ValueError('Replay job manifest changed')
            rows=[];pending=[];shard_start=time.monotonic()
            for prepared in _bounded_map(pool,read_gzip(shard['path']),ahead):
                if mode=='structured':
                    rows.append(dict(job_id=prepared['job_id'],predictions={'structured':prepared['structured']},
                        **{k:prepared[k] for k in ('structured_seconds','preparation_seconds','observation_ids','observation_refs','provenance_hash','raw_record_count')}))
                    continue
                pending.append(prepared)
                if len(pending)>=batch_size:
                    values,error=_gnn_batch(model,pending,check_noop=not noop_errors)
                    rows.extend(values);forward_items+=2*len(pending);pending=[]
                    if error is not None:noop_errors.append(error)
            if pending:
                values,error=_gnn_batch(model,pending,check_noop=not noop_errors)
                rows.extend(values);forward_items+=2*len(pending)
                if error is not None:noop_errors.append(error)
            write_gzip(dest,dict(state='passed',input_sha256=shard['sha256'],reader_checkpoint_hash=reader_hash,
                original_reader_checkpoint_hash=model.checkpoint_hash if model else None,predictions=rows,
                device=device,worker_index=worker_index,cpu_workers=workers,batch_size=batch_size,prefetch=ahead,
                wall_seconds=time.monotonic()-shard_start,model_forward_items=2*len(rows) if model else 0,mode=mode))
            completed+=len(rows)
            print(json.dumps(dict(stage='old_history_replay',worker_index=worker_index,
                shard=shard['index'],completed_jobs=completed,elapsed_seconds=time.monotonic()-started)),flush=True)
    result=dict(state='passed',worker_index=worker_index,worker_count=worker_count,device=device,mode=mode,
        completed_jobs=completed,new_gnn_forward_items=forward_items,physical_acquisitions=0,
        noop_max_probability_error=max(noop_errors,default=0.),wall_seconds=time.monotonic()-started)
    atomic_json(run/'replay/workers'/f'{namespace}_{worker_index:03d}.json',result)
    return result


def finish(run_dir):
    run=Path(run_dir);manifest=read_json(run/'replay/inventory.json');cache={}
    gate=read_json(run/'manifests/reader_replay_authorized.json')
    if gate['state']!='passed':raise ValueError('Reader replay gate missing')
    frozen=read_json(run/'models/readers/frozen.json');verify_hashes(frozen['files'],run)
    if gate['reader_frozen_sha256']!=file_sha256(run/'models/readers/frozen.json'):
        raise ValueError('Reader freeze manifest changed after authorization')
    verify_hashes(frozen['runtime_files'],Path(__file__).parents[1])
    if file_sha256(run/'replay/contexts.parquet')!=manifest['contexts_sha256']:
        raise ValueError('Sealed replay contexts changed')
    ready=run/'replay/predictions_complete.json'
    if ready.exists():
        result=read_json(ready)
        if result['reader_checkpoint_hash']!=gate['reader_checkpoint_hash']:
            raise ValueError('Completed predictions used another reader')
        if result['per_case_predictions_sha256']!=file_sha256(run/'replay/per_case_predictions.parquet'):
            raise ValueError('Completed per-case predictions changed')
        return _seal_complete(run,result)
    original_hash=read_json(Path(manifest['source_root'])/'models/readers/final/complete.json')['selected_checkpoint_hash']
    for shard in manifest['shards']:
        expected={j['job_id'] for j in read_gzip(shard['path'])}
        combined=run/'replay/predictions_all'/f"{shard['index']:05d}.json.gz"
        modes=('all',) if combined.exists() else ('baselines','structured')
        for mode in modes:
            part=read_gzip(run/f'replay/predictions_{mode}'/f"{shard['index']:05d}.json.gz")
            if part['state']!='passed' or part['input_sha256']!=shard['sha256']:raise ValueError('Invalid prediction shard')
            if {r['job_id'] for r in part['predictions']}!=expected or len(part['predictions'])!=len(expected):
                raise ValueError('Prediction shard job support changed')
            if mode!='baselines' and part['reader_checkpoint_hash']!=gate['reader_checkpoint_hash']:
                raise ValueError('Prediction used an unsealed structured reader')
            if mode!='structured' and part['original_reader_checkpoint_hash']!=original_hash:
                raise ValueError('Prediction used another original GNN')
            for row in part['predictions']:
                if row['job_id'] in cache:
                    if set(row['predictions'])&set(cache[row['job_id']]['predictions']):raise ValueError('Duplicate replay prediction job')
                    if row['observation_ids']!=cache[row['job_id']]['observation_ids']:raise ValueError('Replay provenance changed between readers')
                    cache[row['job_id']]['predictions'].update(row['predictions'])
                    if mode=='structured':cache[row['job_id']]['structured_seconds']=row['structured_seconds']
                else:cache[row['job_id']]=row
    if len(cache)!=manifest['all_prediction_jobs']:raise ValueError('Reader replay coverage incomplete')
    contexts=pd.read_parquet(run/'replay/contexts.parquet');outputs=[];numerics=[]
    for context in contexts.to_dict('records'):
        saved=json.loads(context.pop('stored_prediction'));values=cache[context['job_id']]
        predictions=dict(values['predictions'],original_as_deployed=saved)
        if set(predictions)!=set(READERS):raise ValueError('Incomplete reader comparison')
        old=values['predictions']['original_GNN_reread']
        if saved['target_keys']!=old['target_keys']:raise ValueError('Stored target remapping changed')
        numerics.append(dict(job_id=context['job_id'],cohort=context['cohort'],case_id=context['case_id'],
            original_timeout=context['original_timeout'],original_abstained=context['original_abstained'],
            probability_error=float(np.max(np.abs(np.exp2(saved['log_probabilities_base2'])-np.exp2(old['log_probabilities_base2'])))),
            argmax_changed=int(np.argmax(saved['log_probabilities_base2'])!=np.argmax(old['log_probabilities_base2']))))
        for name,pred in predictions.items():
            outputs.append(dict(context,reader=name,**prediction_metrics(pred,context['true_target_key']),
                structured_processing_seconds=values['structured_seconds'],
                replay_preparation_seconds=values['preparation_seconds'],
                provenance_hash=hash_json({'observation_ids':values['observation_ids']}),
                observation_references=json.dumps(values['observation_refs']),
                physical_acquisitions=0,wall_time_scope='offline_rereading_shared_saved_evidence'))
    frame=pd.DataFrame(outputs);frame.to_parquet(run/'replay/per_case_predictions.parquet',index=False)
    frame.drop(columns=['full_log_probabilities','target_keys']).to_csv(run/'replay/per_case_metrics.csv',index=False)
    pd.DataFrame(numerics).to_parquet(run/'replay/original_reread_differences.parquet',index=False)
    main=frame[frame.cohort=='main']
    if len(main)!=EXPECTED_MAIN_ROWS*len(READERS):raise ValueError('Main replay denominator changed')
    result=dict(state='passed',rows=len(frame),main_rows=len(main),main_physical_cases=main.case_id.nunique(),
        unique_histories=len(cache),readers=list(READERS),physical_acquisitions=0,
        reader_checkpoint_hash=gate['reader_checkpoint_hash'],
        reader_frozen_sha256=file_sha256(run/'models/readers/frozen.json'),
        original_deadline_results_retained_separately=True,
        per_case_predictions_sha256=file_sha256(run/'replay/per_case_predictions.parquet'),
        completed_at=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    atomic_json(run/'replay/predictions_complete.json',result)
    return _seal_complete(run,result)


def _seal_complete(run,result):
    invariant_path=run/'invariance/complete.json'
    if not invariant_path.exists():
        return dict(result,state='pending_invariance',policy_training_authorized=False)
    invariant=read_json(invariant_path)
    if invariant['state']!='passed':raise ValueError('Invariance checks have not passed')
    if invariant['predictions_complete_sha256']!=file_sha256(run/'replay/predictions_complete.json'):
        raise ValueError('Invariance used another prediction table')
    result.update(invariance_complete_sha256=file_sha256(invariant_path),
                  reader_frozen_sha256=file_sha256(run/'models/readers/frozen.json'))
    atomic_json(run/'replay/complete.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('inventory');p.add_argument('--source-root',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--v1-run',required=True);p.add_argument('--v2-run',required=True);p.add_argument('--chunk-size',type=int,default=128)
    p=sub.add_parser('worker');p.add_argument('--source-root',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--worker-index',type=int,required=True);p.add_argument('--worker-count',type=int,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=16);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--limit-shards',type=int)
    p.add_argument('--mode',choices=('baselines','structured','all'),default='all')
    p.add_argument('--prefetch',type=int);p.add_argument('--output-tag',default='')
    p=sub.add_parser('finish');p.add_argument('--run-dir',required=True)
    args=vars(parser.parse_args());command=args.pop('command');print(globals()[command](**args),flush=True)
