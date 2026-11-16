"""Frozen source-only OOF labels and matched action-head inputs.

No head is fitted here. Original FINAL-q head inputs and base raw features are
reused from actual source rollouts; only OOF reader/backbone calls are new.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import torch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

from cgx.models.features import FEATURE_VERSION, collate_features, collate_actions
import cgx.models.features as feature_module
from cgx.provenance import atomic_json, file_sha256, hash_json, read_json
from cgx.training.engine import load_reader, load_policy
from .ledger import Ledger

FEATURE_CODE_HASH=file_sha256(Path(feature_module.__file__))


def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return value.item()
    return value


def feature(row,run):
    path=Path(row.get('feature_cache_path') or run/'cache/features'/f"{row['history_hash']}.pkl")
    with path.open('rb') as handle:cached=pickle.load(handle)
    if isinstance(cached,dict):
        assert cached.get('history_hash',row['history_hash'])==row['history_hash']
        assert cached.get('feature_version',FEATURE_VERSION)==FEATURE_VERSION
        assert cached.get('scientific_code_hash',FEATURE_CODE_HASH)==FEATURE_CODE_HASH
        return cached.get('features',cached.get('feature'))
    return cached


def policy_feature(base,row,q,method):
    """Same transformation as original engine; does not reconstruct raw records."""
    nodes=base.nodes.copy();probabilities=dict(zip(q['target_keys'],np.exp2(q['log_probabilities_base2'])))
    for key,index in zip(base.targets[1:],base.target_nodes[1:]):nodes[index,27]=probabilities.get(key,0.)
    nodes[:,28]=probabilities.get('healthy',0.)
    resources=base.resources.copy()
    if method=='gnn_depth2':
        initial=plain(row['initial_budget']);remaining=plain(row['remaining'])
        if isinstance(initial,str):initial=json.loads(initial)
        if isinstance(remaining,str):remaining=json.loads(remaining)
        decisions=min(2,remaining['decisions'])
        resources[3]=decisions/max(1,initial['decisions']);resources[6]=decisions/10
    return replace(base,nodes=nodes,resources=resources)


def chosen_action(row):
    if row.get('chosen_action') is not None:return plain(row['chosen_action'])
    actions=plain(row['candidate_actions'])
    if isinstance(actions,str):actions=json.loads(actions)
    return next(a for a in actions if a['action_key']==row['chosen_action_key'])


def prediction(row):
    p=plain(row.get('prediction',row.get('prediction_final')))
    if isinstance(p,str):p=json.loads(p)
    return p


def _reader_name(row,case_lookup,assignment):
    case=case_lookup[row['case_id']]
    for candidate in (row,case,assignment.get(row['configuration_id'],{})):
        for key in ('oof_reader_name','oof_reader'):
            if candidate.get(key) is not None:return str(candidate[key])
        for key in ('source_fold','oof_fold'):
            if candidate.get(key) is not None:return f"crossfit_{int(candidate[key])}"
    raise KeyError(f"Missing sealed OOF assignment for {row['configuration_id']}")


def run(root,run_dir,device='cuda:1',batch_size=8):
    root,run=Path(root),Path(run_dir);ledger=Ledger(run)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.perf_counter();cpu=time.process_time()
    completion=read_json(run/'trajectories/source/complete.json')
    assert completion['state']=='passed' and completion['trajectories_present']==1920 and completion['failed_trajectories']==0
    data=run/'source_prefixes.parquet';frame=pd.read_parquet(data)
    cases=read_json(run/'manifests/source_cases.json')
    if isinstance(cases,dict):cases=cases.get('cases',cases.get('source_cases'))
    case_lookup={c['case_id']:c for c in cases}
    assert len(case_lookup)==320
    assert len(frame[['case_id','method_key','policy_seed']].drop_duplicates())==1920
    assert not frame[['case_id','method_key','policy_seed','decision_index']].duplicated().any()
    assignments={}
    for name in ('oof_assignments.json','source_oof_assignments.json'):
        p=run/'manifests'/name
        if p.exists():assignments=read_json(p)
    rows=[plain(x) for x in frame.to_dict('records')]
    for row in rows:
        assert row['case_id'] in case_lookup,'Non-source case reached head preparation'
        case=case_lookup[row['case_id']]
        row['source_split']=row.get('source_split') or case.get('source_split') or case['split']
        assert row['source_split'] in ('train','validation')
        row['oof_reader_name']=_reader_name(row,case_lookup,assignments)
    plan={'source_cases_sha256':file_sha256(run/'manifests/source_cases.json'),
          'source_prefixes_sha256':file_sha256(data),'source_rows':len(rows),
          'feature_version':FEATURE_VERSION,'feature_code_sha256':FEATURE_CODE_HASH,
          'head_data_code_sha256':file_sha256(Path(__file__)),
          'full_forward_upper_bound':len({(r['history_hash'],r['oof_reader_name']) for r in rows})+sum(bool(r.get('actionable')) for r in rows),
          'raw_feature_constructions':0,'support_rule':'all actual source prefixes; fit all non-STOP proposed continuations, irrespective of return sign',
          'held_out_test_or_v1_outcomes_read':False,'feature_transform':'copy original node q fields; original depth2 decision cap; no raw feature rebuild'}
    atomic_json(run/'manifests/head_data_workload.json',plan)
    cache=run/'cache/head_data';cache.mkdir(parents=True,exist_ok=True)
    readers={name:load_reader(root,name,device=device) for name in sorted({r['oof_reader_name'] for r in rows})}
    unique={}
    for row in rows:
        model=readers[row['oof_reader_name']]
        assert row['configuration_id'] not in model.training_configurations,'Original OOF topology exclusion failed'
        unique.setdefault((row['oof_reader_name'],row['history_hash']),row)
    qcache={};new_reader=0;new_policy=0
    for name,model in readers.items():
        tasks=[(key,row) for key,row in unique.items() if key[0]==name]
        pending=[]
        for key,row in tasks:
            path=cache/'oof_predictions'/f"{hash_json([model.checkpoint_hash,FEATURE_VERSION,FEATURE_CODE_HASH,row['history_hash']])}.json"
            if path.exists():qcache[key]=read_json(path)['prediction']
            else:pending.append((key,row,path))
        for offset in range(0,len(pending),batch_size):
            chunk=pending[offset:offset+batch_size];features=[feature(row,run) for _,row,_ in chunk]
            assert all(np.count_nonzero(f.nodes[:,27:29])==0 for f in features),'Base reader features contain q'
            ledger.reserve('model_forward',len(chunk),detail='head_data:oof_reader')
            with torch.inference_mode():lp=model(collate_features(features,device)).cpu().numpy()
            for i,((key,row,path),f) in enumerate(zip(chunk,features)):
                q={'target_keys':list(f.targets),'log_probabilities_base2':lp[i,:len(f.targets)].tolist()}
                qcache[key]=q;atomic_json(path,{'prediction':q,'checkpoint_hash':model.checkpoint_hash,
                    'history_hash':row['history_hash'],'configuration_excluded':True})
            new_reader+=len(chunk)
        print(json.dumps({'stage':'head_data_oof','reader':name,'unique_histories':len(tasks),'new_forwards':len(pending)}),flush=True)
    all_output=[]
    for (method,seed),subset in frame.groupby(['method_key','policy_seed'],sort=True):
        selected=[r for r in rows if r['method_key']==method and int(r['policy_seed'])==int(seed)]
        model=load_policy(root,method,int(seed),device=device)
        pending=[];x_oof={};score_oof={}
        for row in selected:
            if not row.get('actionable') or chosen_action(row)['kind']=='stop':continue
            key=hash_json([model.checkpoint_hash,readers[row['oof_reader_name']].checkpoint_hash,FEATURE_CODE_HASH,
                          method,int(seed),row['history_hash'],row['chosen_action_key']])
            path=cache/'oof_head_inputs'/f'{key}.npz'
            if path.exists():
                with np.load(path) as saved:x_oof[key]=saved['x'];score_oof[key]=float(saved['score'])
            else:pending.append((key,row,path))
        for offset in range(0,len(pending),batch_size):
            chunk=pending[offset:offset+batch_size]
            features=[policy_feature(feature(row,run),row,qcache[(row['oof_reader_name'],row['history_hash'])],method) for _,row,_ in chunk]
            batch=collate_features(features,device);actions=collate_actions(batch,[[chosen_action(row)] for _,row,_ in chunk],device)
            captured=[]
            hook=model.action_head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu().numpy().copy()))
            try:
                ledger.reserve('model_forward',len(chunk),detail='head_data:oof_backbone')
                with torch.inference_mode():scores=model(batch,actions).cpu().numpy()
            finally:hook.remove()
            assert len(captured)==1 and captured[0].shape==(len(chunk),281)
            for i,(key,row,path) in enumerate(chunk):
                path.parent.mkdir(parents=True,exist_ok=True)
                x_oof[key]=captured[0][i]
                score_oof[key]=float(scores[i])
                np.savez_compressed(path,x=x_oof[key],score=scores[i],checkpoint_hash=model.checkpoint_hash)
            new_policy+=len(chunk)
        trajectories=defaultdict(list)
        for row in selected:trajectories[row['case_id']].append(row)
        fit_rows=[];xo=[];xf=[]
        for case_id,trajectory in sorted(trajectories.items()):
            trajectory.sort(key=lambda r:int(r['decision_index']))
            by_index={int(r['decision_index']):r for r in trajectory};last=max(by_index)
            for row in trajectory:
                i=int(row['decision_index']);remaining=row['remaining']
                if isinstance(remaining,str):remaining=json.loads(remaining)
                represented=min(2,remaining['decisions']) if method=='gnn_depth2' else remaining['decisions']
                action=chosen_action(row) if row.get('actionable') else None
                eligible=action is not None and action['kind']!='stop'
                endpoint=min(last,i+represented) if eligible else i
                end=by_index[endpoint]
                q=qcache[(row['oof_reader_name'],row['history_hash'])]
                qe=qcache[(end['oof_reader_name'],end['history_hash'])]
                final=prediction(row);final_end=prediction(end)
                truth=row.get('true_target_key')
                if truth is None:
                    case=case_lookup[case_id];truth='healthy' if case['fault_family']=='healthy' else f"{case['fault_node']}:{case['fault_family']}"
                target=lambda p:float(p['log_probabilities_base2'][p['target_keys'].index(truth)])
                out={k:row[k] for k in ('configuration_id','case_id','method_key','policy_seed','decision_index','history_hash','source_split','oof_reader_name')}
                out.update(eligible_for_fit=eligible,actionable=bool(row.get('actionable')),stop_reference=0.,
                    chosen_stop=action is not None and action['kind']=='stop',resource_or_error_terminal=not bool(row.get('actionable')),
                    chosen_action_key=action['action_key'] if action else None,
                    chosen_action_kind=action['kind'] if action else None,original_q=float(row.get('chosen_score',row.get('original_chosen_score',0.)) or 0.),
                    endpoint_decision_index=endpoint,endpoint_history_hash=end['history_hash'],represented_horizon=represented,
                    observed_continuation_decisions=endpoint-i,return_oof=target(qe)-target(q),return_final=target(final_end)-target(final),
                    true_target_key=truth,is_healthy=truth=='healthy',fault_family=case_lookup[case_id]['fault_family'],
                    executions_used=int(row.get('executions_used',0)),calibrations_used=int(row.get('calibrations_used',0)),
                    terminal_reason=row.get('terminal_reason') or trajectory[-1].get('terminal_reason'),
                    oof_checkpoint_hash=readers[row['oof_reader_name']].checkpoint_hash,policy_checkpoint_hash=model.checkpoint_hash,
                    oof_prediction=q,final_prediction=final)
                all_output.append(out)
                if eligible:
                    key=hash_json([model.checkpoint_hash,readers[row['oof_reader_name']].checkpoint_hash,FEATURE_CODE_HASH,
                                  method,int(seed),row['history_hash'],row['chosen_action_key']])
                    out['original_q_with_oof_input']=score_oof[key]
                    x=np.asarray(row['chosen_head_input_final'],dtype=np.float32)
                    assert x.shape==(281,) and np.isfinite(x).all()
                    fit_rows.append(out);xo.append(x_oof[key]);xf.append(x)
        directory=run/'head_data';directory.mkdir(exist_ok=True)
        name=f'{method}_seed{int(seed)}'
        pd.DataFrame(fit_rows).to_parquet(directory/(name+'.parquet'),index=False)
        np.savez_compressed(directory/(name+'.npz'),x_oof=np.asarray(xo,np.float32),x_final=np.asarray(xf,np.float32),
            y_oof=np.asarray([r['return_oof'] for r in fit_rows],np.float32),y_final=np.asarray([r['return_final'] for r in fit_rows],np.float32))
        print(json.dumps({'stage':'head_data_policy','method':method,'seed':int(seed),'eligible_rows':len(fit_rows)}),flush=True)
    result=pd.DataFrame(all_output)
    result.to_parquet(run/'continuing_value_source_per_prefix.parquet',index=False)
    result.drop(columns=['oof_prediction','final_prediction']).to_csv(run/'continuing_value_source_per_prefix.csv',index=False)
    calibration_tables(result,run)
    atomic_json(run/'head_data_complete.json',{'state':'passed','rows':len(result),'fit_rows':int(result.eligible_for_fit.sum()),
        'new_oof_reader_forwards':new_reader,'new_oof_policy_forwards':new_policy,'new_raw_feature_builds':0,
        'source_manifest_hash':plan['source_cases_sha256'],'source_prefixes_hash':plan['source_prefixes_sha256'],
        'feature_code_sha256':FEATURE_CODE_HASH,'head_data_code_sha256':plan['head_data_code_sha256'],
        'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'device':device,
        'test_and_v1_labels_read':False,'all_oof_configuration_exclusions_passed':True})


def calibration_tables(frame,run):
    """Conditional source-validation diagnostics, never a fitting filter."""
    val=frame[frame.source_split.eq('validation')].copy()
    val['q_bin']=pd.cut(val.original_q,[-np.inf,0,.1,.25,.5,1,2,np.inf],right=True).astype(str)
    val.groupby(['method_key','policy_seed','actionable','chosen_stop','resource_or_error_terminal','terminal_reason'],dropna=False).size().reset_index(name='prefixes').to_csv(
        run/'continuing_value_terminal_and_stop_counts.csv',index=False)
    val=val[val.eligible_for_fit]
    output=[]
    groupings=(['method_key'],['method_key','fault_family'],['method_key','executions_used'],
               ['method_key','q_bin'],['method_key','fault_family','executions_used','q_bin'])
    for group,positive_only in [(g,False) for g in groupings]+[(['method_key'],True),(['method_key','fault_family'],True)]:
        subset=val[val.original_q.gt(0)] if positive_only else val
        for identity,part in subset.groupby(group,dropna=False,sort=True):
            if not isinstance(identity,tuple):identity=(identity,)
            for teacher in ('oof','final'):
                p=part.copy();p['realized_return']=p['return_'+teacher]
                p['positive_q_negative_return']=(p.original_q.gt(0)&p.realized_return.lt(0)).astype(float)
                p['label_difference']=p.return_final-p.return_oof
                p['q_input_difference']=p.original_q-p.original_q_with_oof_input
                measures=['original_q','realized_return','positive_q_negative_return','label_difference','q_input_difference']
                # Prefixes within seed and case, then seeds within physical case,
                # then equal cases within configuration, then configurations.
                per_seed=p.groupby(['configuration_id','case_id','is_healthy','policy_seed'],sort=True)[measures].mean().reset_index()
                cases=per_seed.groupby(['configuration_id','case_id','is_healthy'],sort=True)[measures].mean().reset_index()
                configs=[c for _,c in cases.groupby('configuration_id',sort=True)]
                estimate=np.mean([c[measures].mean().to_numpy() for c in configs],axis=0)
                rng=np.random.default_rng(671932);selected=rng.integers(len(configs),size=(2000,len(configs)))
                draws=np.zeros((2000,len(configs),len(measures)))
                for ci,config in enumerate(configs):
                    rr,cc=np.where(selected==ci)
                    result=np.zeros((len(rr),len(measures)))
                    for _,stratum in config.groupby('is_healthy',sort=True):
                        values=stratum[measures].to_numpy()
                        sampled=rng.integers(len(values),size=(len(rr),len(values)))
                        result+=values[sampled].sum(1)/len(config)
                    draws[rr,cc]=result
                low,high=np.quantile(draws.mean(1),[.025,.975],axis=0)
                row=dict(zip(group,identity));row.update(grouping='/'.join(group),teacher=teacher,positive_q_only=positive_only,prefixes=len(p),
                    physical_cases=p.case_id.nunique(),configurations=p.configuration_id.nunique(),
                    positive_q_prefixes=int(p.original_q.gt(0).sum()),
                    positive_q_negative_return_count=int(p.positive_q_negative_return.sum()),
                    interpretation='sampled original-continuation return, not an oracle stopping label',
                    weighting='prefixes within seed; seeds within case; cases within configuration; equal configurations',
                    bootstrap_strata='healthy/faulty case counts retained within sampled configuration',
                    bootstrap_repetitions=2000)
                for i,name in enumerate(measures):row.update({name:float(estimate[i]),name+'_ci_low':float(low[i]),name+'_ci_high':float(high[i])})
                positive=p[p.original_q.gt(0)]
                if len(positive):
                    positive_seeds=positive.groupby(['configuration_id','case_id','policy_seed']).realized_return.mean()
                    row['mean_return_when_q_positive']=float(positive_seeds.groupby(level=[0,1]).mean().groupby(level=0).mean().mean())
                else:row['mean_return_when_q_positive']=None
                output.append(row)
    pd.DataFrame(output).to_csv(run/'continuing_value_calibration.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--device',default='cuda:1');p.add_argument('--batch-size',type=int,default=8);a=p.parse_args()
    run(a.root,a.run_dir,a.device,a.batch_size)
