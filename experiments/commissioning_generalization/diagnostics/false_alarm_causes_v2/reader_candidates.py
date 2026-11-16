"""Prespecified public-only reader rules and SOURCE validation selection.

Every real reading stays in the GRU and edge summaries. R-P changes local
measurement averaging only; R-C changes only four declared count fields.
"""
from collections import defaultdict
import argparse
import gzip
import json
import math
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import torch

from cgx.data.store import Store
from cgx.models.features import FEATURE_VERSION,collate_features
from cgx.provenance import atomic_json,hash_json,read_json,file_sha256
from diagnostics.false_alarm_causes.core import inference_settings,load_reader
from .hooks import FrozenReaderHooks,prediction_error
from .ledger import Ledger,TimedStage

VARIANTS = ('original', 'R-P', 'R-C', 'R-PC')


def candidate_tensors(state, variant):
    if variant not in VARIANTS:
        raise ValueError(f'Undeclared reader variant: {variant}')
    local, nodes = state.local.clone(), state.batch['nodes'].clone()
    ports = defaultdict(list)
    for index, metadata in enumerate(state.sequence_metadata):
        port = metadata['port']
        owner = state.node_lookup.get(port.split(':')[0], -1)
        if owner >= 0:
            ports[(owner,port)].append(index)
    if variant in ('R-P', 'R-PC'):
        local.zero_()
        by_node = defaultdict(list)
        for (owner,port), indices in ports.items():
            by_node[owner].append(state.sequence_vectors[indices].mean(0))
        for owner, vectors in by_node.items():
            local[owner] = torch.stack(vectors).mean(0)
    if variant in ('R-C', 'R-PC'):
        raw = np.zeros((len(nodes),3), dtype=np.float32)
        for (owner,port), indices in ports.items():
            rows = [state.sequence_metadata[i] for i in indices]
            raw[owner,0] += 1
            raw[owner,1] += np.mean([r['record_count'] for r in rows])/1000
            raw[owner,2] += np.mean([r['missing'] for r in rows])
        nodes[:,23:26] = torch.from_numpy(np.log1p(raw))
        nn = len(state.node_lookup)
        nodes[:nn,31] = 1-torch.minimum(nodes[:nn,23], torch.ones(nn))
    return local, nodes


def predict_variant_state(hooks, state, variant='original'):
    if variant == 'original':
        return state.prediction
    local, nodes = candidate_tensors(state,variant)
    return hooks.predict_state(state, local=local, nodes=nodes, detail='reader_candidate_'+variant)


def evaluate_variants(hooks, history, metadata, provenance=None):
    """Metadata is evaluator-only; prepare/prediction receive history alone."""
    state = hooks.prepare(history,provenance)
    fixed_fault=metrics(state.prediction,metadata['true_target_key'])['max_fault_target']
    rows=[]
    for variant in VARIANTS:
        stamp=time.perf_counter()
        prediction=predict_variant_state(hooks,state,variant)
        rows.append(dict(metadata,variant=variant,**metrics(prediction,metadata['true_target_key'],fixed_fault),
                         processing_seconds=time.perf_counter()-stamp))
    return rows


def metrics(prediction, truth, fixed_fault=None):
    keys=prediction['target_keys'];lp=np.asarray(prediction['log_probabilities_base2'],np.float64)
    healthy=keys.index('healthy'); faults=[i for i,k in enumerate(keys) if k!='healthy']
    fault=max(faults,key=lambda i:lp[i]);best=int(lp.argmax());predicted=keys[best]
    fixed=fault if fixed_fault is None else keys.index(fixed_fault)
    return dict(predicted_target_key=predicted,p_healthy=float(np.exp2(lp[healthy])),
                max_fault_probability=float(np.exp2(lp[fault])),max_fault_target=keys[fault],
                max_fault_log_odds=float(lp[fault]-lp[healthy]),fixed_fault_target=keys[fixed],
                fixed_fault_log_odds=float(lp[fixed]-lp[healthy]),
                is_healthy=truth=='healthy',false_alarm=int(truth=='healthy' and predicted!='healthy'),
                missed_fault=int(truth!='healthy' and predicted=='healthy'),
                joint_fault_correct=int(truth!='healthy' and predicted==truth),
                correct=int(predicted==truth),true_target_nll_bits=float(-lp[keys.index(truth)]))


def select_source_candidate(frame):
    """Input must contain ONLY new source-validation cases, asserted explicitly.

    Do not accept V1 confirmation/discovery rows just because a caller labels
    their split as validation. The orchestrator supplies sealed source IDs.
    """
    if 'source_purpose' not in frame or set(frame.source_purpose)!={'source_validation'}:
        raise ValueError('Candidate selection is restricted to new source validation')
    if not set(frame.variant).issubset(VARIANTS):
        raise ValueError('Unexpected candidate')
    grouped=[]
    for variant,part in frame.groupby('variant'):
        cfg=[]
        for _,p in part.groupby('configuration_id'):
            healthy=p[p.is_healthy];faulty=p[~p.is_healthy]
            cfg.append(dict(nll=p.true_target_nll_bits.mean(),fa=healthy.false_alarm.mean(),
                            miss=faulty.missed_fault.mean(),joint=faulty.joint_fault_correct.mean()))
        row=dict(variant=variant,**pd.DataFrame(cfg).mean().to_dict(),
                 processing_seconds=part.processing_seconds.mean(),runs=len(part),
                 healthy_runs=int(part.is_healthy.sum()),physical_cases=part.case_id.nunique())
        grouped.append(row)
    table=pd.DataFrame(grouped);base=table[table.variant=='original'].iloc[0]
    table['passes_guardrails']=(table.fa<=base.fa+1e-12)&(table['miss']<=base['miss']+.02+1e-12)&(table.joint>=base.joint-.02-1e-12)
    # A correction must improve log loss; an equal/worse candidate is not an improvement.
    table['improves_nll']=table.nll<base.nll
    allowed=table[(table.variant!='original')&table.passes_guardrails&table.improves_nll]
    selected=None
    if len(allowed):
        tied=allowed[allowed.nll<=allowed.nll.min()+.005].copy()
        tied['complexity']=tied.variant.map({'R-P':1,'R-C':1,'R-PC':2})
        selected=tied.sort_values(['complexity','processing_seconds','variant']).iloc[0].variant
    return selected,table


def prepare_stored(hooks,run,history_hash,provenance=None,feature_cache_path=None):
    """Reuse the exact public raw-feature cache made by the rollout CPU worker."""
    run=Path(run);history=Store(run).load_history(history_hash)
    path=Path(feature_cache_path) if feature_cache_path else run/'cache/features'/(history_hash+'.pkl')
    with path.open('rb') as handle:payload=pickle.load(handle)
    assert payload['history_hash']==history_hash and payload['feature_version']==FEATURE_VERSION
    import cgx.models.features as feature_module
    assert payload['scientific_code_hash']==file_sha256(feature_module.__file__)
    hooks.ledger.reserve('cache_hit',detail='reader_candidate_features')
    return hooks.prepare(history,provenance=provenance,features=payload['features'],identity=history_hash)


class SourceBatchAudit:
    """Reconstruct actual original source batches for any >1e-6 shape rounding.

    Original history ordering, initial profile/resume boundary and active-prefix
    support determine the batch. No labels enter this numerical check.
    """
    def __init__(self,hooks,run):
        self.hooks,self.run=hooks,Path(run);self.prefixes=None;self.cache={}

    def original_batch(self,case_id,method,seed,ordinal,history_hash):
        if self.prefixes is None:
            self.plan=read_json(self.run/'manifests/source_rollouts.json')
            cases=[c['case_id'] for c in read_json(self.run/'manifests/source_cases.json')]
            prof=set(read_json(self.run/'source_workload_forecast.json')['profile_case_ids'])
            self.jobs={profile:[(cid,m,s) for cid in cases if (cid in prof)==profile
                              for m in self.plan['methods'] for s in self.plan['seeds']] for profile in (False,True)}
            self.profile_cases=prof
            cols=['case_id','method_key','policy_seed','prefix_ordinal','history_hash','feature_cache_path']
            table=pd.read_parquet(self.run/'source_prefixes.parquet',columns=cols)
            self.prefixes={(r.case_id,r.method_key,int(r.policy_seed),int(r.prefix_ordinal)):
                           (r.history_hash,r.feature_cache_path) for r in table.itertuples()}
        jobs=self.jobs[case_id in self.profile_cases];position=jobs.index((case_id,method,int(seed)))
        start=position//self.plan['window']*self.plan['window'];unique={}
        for job in jobs[start:start+self.plan['window']]:
            item=self.prefixes.get((*job,int(ordinal)))
            if item is not None:unique.setdefault(item[0],item[1])
        group=list(unique.items());target=next(i for i,(digest,path) in enumerate(group) if digest==history_hash)
        size=self.plan['gpu_batch_size'];chunk=group[target//size*size:target//size*size+size]
        key=tuple(digest for digest,path in chunk)
        if key not in self.cache:
            features=[]
            for digest,path in chunk:
                with open(path,'rb') as handle:payload=pickle.load(handle)
                assert payload['history_hash']==digest and payload['feature_version']==FEATURE_VERSION
                features.append(payload['features'])
            self.hooks.ledger.reserve('model_forward',len(features),detail='reader_source_batch_reconstruction')
            with torch.inference_mode():values=self.hooks.model(collate_features(features,self.hooks.device)).cpu().numpy()
            self.cache[key]={digest:dict(target_keys=list(f.targets),log_probabilities_base2=values[i,:len(f.targets)].tolist())
                             for i,((digest,path),f) in enumerate(zip(chunk,features))}
        else:self.hooks.ledger.reserve('cache_hit',detail='reader_source_batch_reconstruction')
        return self.cache[key][history_hash],dict(original_batch_size=len(chunk),original_batch_history_hashes=list(key),
                                                original_window_start_job=start,original_prefix_ordinal=ordinal)

    def compare(self,state,stored,case_id,method,seed,ordinal,history_hash):
        assert state.prediction['target_keys']==stored['target_keys']
        native=np.asarray(state.prediction['log_probabilities_base2'],np.float64)
        saved=np.asarray(stored['log_probabilities_base2'],np.float64)
        error=float(np.max(np.abs(np.exp2(native)-np.exp2(saved))))
        same=int(native.argmax())==int(saved.argmax())
        row=dict(max_abs_probability_error=error,argmax_identical=same,
                 native_argmax_margin=float(np.diff(np.sort(np.exp2(native))[-2:])[0]),
                 stored_argmax_margin=float(np.diff(np.sort(np.exp2(saved))[-2:])[0]),
                 max_abs_log2_probability_difference=float(np.max(np.abs(native-saved))),
                 original_batch_reconstruction_required=error>1e-6)
        if not same or error>2e-6:
            atomic_json(self.run/'tests/reader_source_numerical_failure.json',dict(case_id=case_id,method_key=method,
                        policy_seed=seed,history_hash=history_hash,**row))
            raise AssertionError(f'Cross-batch numeric bound/argmax failed: {row}')
        if error>1e-6:
            replay,metadata=self.original_batch(case_id,method,seed,ordinal,history_hash)
            row.update(metadata,original_batch_reconstruction=prediction_error(replay,stored,tolerance=1e-6))
        return row


def validate_source(root,run,device='cuda:0'):
    run=Path(run);status=read_json(run/'trajectories/source/complete.json')
    if status['state']!='passed' or status['trajectories_present']!=1920:
        raise ValueError('All bounded source rollouts must finish before reader selection')
    cases=[c for c in read_json(run/'manifests/source_cases.json') if c['purpose']=='source_validation']
    assert len(cases)==64 and len({c['configuration_id'] for c in cases})==8
    inference_settings();hooks=FrozenReaderHooks(load_reader(root,'final',device),Ledger(run),run/'cache/reader_states',detail='reader_source_validation')
    batch_audit=SourceBatchAudit(hooks,run)
    rows=[];checks=[];started=time.monotonic()
    for case in cases:
        for method in ('gnn_adaptive','gnn_depth2'):
            for seed in (11,23,37):
                path=run/'trajectories/source/cases'/(hash_json([case['case_id'],method,seed])+'.json.gz')
                with gzip.open(path,'rt') as handle:item=json.load(handle)
                row=item['row'];assert row['case_id']==case['case_id'] and row['state']=='passed'
                if row['source_split']!='validation':raise ValueError('Non-validation source reached candidate selection')
                provenance=['initial:'+case['case_id']]+[t['acquisition_identity'] for t in item['trace']]
                state=prepare_stored(hooks,run,row['evidence_trace_hash'],provenance)
                numeric=batch_audit.compare(state,row['prediction'],case['case_id'],method,seed,len(item['trace']),
                                            row['evidence_trace_hash'])
                ti=state.targets.index(row['true_target_key'])
                checks.append(dict(case_id=case['case_id'],method_key=method,policy_seed=seed,**numeric,
                    native_minus_stored_true_target_nll_bits=float(row['prediction']['log_probabilities_base2'][ti]-
                        state.prediction['log_probabilities_base2'][ti])))
                fixed_fault=metrics(state.prediction,row['true_target_key'])['max_fault_target']
                for variant in VARIANTS:
                    stamp=time.perf_counter();pred=predict_variant_state(hooks,state,variant)
                    if hooks.device.type=='cuda':torch.cuda.synchronize(hooks.device)
                    rows.append(dict(case_id=case['case_id'],configuration_id=case['configuration_id'],method_key=method,
                        policy_seed=seed,source_purpose='source_validation',true_target_key=row['true_target_key'],
                        history_hash=row['evidence_trace_hash'],variant=variant,**metrics(pred,row['true_target_key'],fixed_fault),
                        fault_family=case['fault_family'],processing_seconds=time.perf_counter()-stamp,
                        target_keys=json.dumps(pred['target_keys']),full_log_probabilities=json.dumps(pred['log_probabilities_base2'])))
        print(json.dumps(dict(stage='reader_source_validation',completed_cases=len(rows)//24,total_cases=64,
                              elapsed_seconds=time.monotonic()-started)),flush=True)
    frame=pd.DataFrame(rows)
    if len(frame)!=1536 or len(frame.drop_duplicates(['case_id','method_key','policy_seed','variant']))!=1536:
        raise AssertionError('Incomplete candidate validation support')
    selected,summary=select_source_candidate(frame)
    frame.to_csv(run/'reader_candidate_validation.csv',index=False)
    summary.to_csv(run/'reader_candidate_validation_summary.csv',index=False)
    frame.groupby(['variant','method_key','policy_seed','is_healthy']).agg(
        runs=('case_id','size'),false_alarm=('false_alarm','sum'),missed_fault=('missed_fault','sum'),
        joint_fault_correct=('joint_fault_correct','sum'),nll_bits=('true_target_nll_bits','mean')).reset_index().to_csv(
        run/'reader_candidate_validation_per_seed.csv',index=False)
    selection=dict(state='passed',selected_variant=selected,available_as_improvement=selected is not None,
        checkpoint_hash=hooks.checkpoint_hash,source_case_manifest_sha256=file_sha256(run/'manifests/source_cases.json'),
        validation_cases=64,configurations=8,trajectories=384,variant_rows=1536,
        selection_rule='one shared variant across methods and seeds; equal-configuration mean final joint NLL; stated error guardrails; require NLL improvement',
        original_role='strict NLL-improvement reference and guardrail baseline; not a zero-complexity candidate in the 0.005-bit tie',
        minimum_nll_improvement_bits=0.,minimum_improvement_comparison='strictly greater than zero; no 0.005-bit minimum',
        tie_rule='among qualified non-original candidates within0.005bits of best, prefer single modification then measured processing cost then name',
        original_tie_interpretation_confirmed_before_candidate_results=True,
        max_native_vs_batched_stored_probability_error=max(x['max_abs_probability_error'] for x in checks),
        native_vs_stored_over_1e_6_count=sum(x['max_abs_probability_error']>1e-6 for x in checks),
        native_vs_stored_engineering_bound=2e-6,PCE_reconstruction_tolerance_unchanged=1e-6,
        all_over_1e_6_reconstructed_in_original_batch=True,
        max_abs_native_vs_stored_true_target_nll_difference_bits=max(abs(x['native_minus_stored_true_target_nll_bits']) for x in checks),
        numerical_comparison='four candidates share native singleton/partial forward; original stored batched baseline audited separately; no case removed',
        all_native_vs_stored_argmax_identical=True,summary=summary.to_dict('records'),
        reads_fresh_outcomes=False,reads_V1_outcomes=False,source_validation_only=True)
    atomic_json(run/'manifests/source_reader_selection.json',selection)
    atomic_json(run/'tests/reader_candidate_validation_reconstruction.json',dict(state='passed',checks=checks))
    return selection


def confirmation_reread(root,run,tasks,device='cuda:0'):
    run=Path(run);selection=read_json(run/'manifests/source_reader_selection.json');variant=selection['selected_variant']
    if variant is None:raise ValueError('No source-validated reader improvement; R1/RG1 unavailable')
    requests=read_json(tasks)
    if isinstance(requests,dict):requests=requests.get('tasks',requests.get('histories'))
    if not isinstance(requests,list):raise ValueError('Expected exact-history task list')
    inference_settings();hooks=FrozenReaderHooks(load_reader(root,'final',device),Ledger(run),run/'cache/reader_states',detail='reader_confirmation')
    predictions={}
    for index,request in enumerate(requests):
        digest=request['history_hash']
        state=prepare_stored(hooks,run,digest,request.get('provenance'),request.get('feature_cache_path'))
        predictions[digest]=predict_variant_state(hooks,state,variant)
        if (index+1)%32==0:print(json.dumps(dict(stage='confirmation_reader_reread',completed=index+1,total=len(requests))),flush=True)
    result=dict(variant=variant,predictions=predictions,checkpoint_hash=hooks.checkpoint_hash,
                source_selection_sha256=file_sha256(run/'manifests/source_reader_selection.json'),
                tasks_sha256=file_sha256(tasks),unique_histories=len(predictions))
    atomic_json(run/'confirmation_reader_predictions.json',result)
    return dict(variant=variant,unique_histories=len(predictions))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--confirmation-tasks');args=parser.parse_args()
    with TimedStage(args.run,'reader_confirmation_reread' if args.confirmation_tasks else 'reader_candidate_validation'):
        result=(confirmation_reread(args.root,args.run,args.confirmation_tasks,args.device) if args.confirmation_tasks
                else validate_source(args.root,args.run,args.device))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
