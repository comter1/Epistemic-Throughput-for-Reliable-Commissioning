"""One precommitted actual prefix per discovery trajectory; all-prefix targeting.

The limited prefix sample preserves all configurations, cases, methods and
three seeds while protecting the shared forward budget for fresh confirmation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
from cgx.observation.actions import action_catalog,feasible_actions
from cgx.provenance import atomic_json,read_json,hash_json,file_sha256
from . import interventions as iv
from .branches import targeted,truth_for


def arm_path(run,case,method,seed):
    return Path(run)/'interventions/discovery/cases'/f"{hash_json([case['case_id'],method,seed,'A'])}.json.gz"


def seal(run):
    run=Path(run);path=run/'manifests/q_input_prefix_subset.json'
    if path.exists():return read_json(path)
    cases=read_json(run/'manifests/discovery_cases.json');rows=[]
    for case in cases:
        for method in iv.METHODS:
            for seed in iv.SEEDS:
                item=iv.load_blob(arm_path(run,case,method,seed))
                if item['row']['state']!='passed':raise ValueError('Discovery A is incomplete; preserve this denominator')
                candidates=[];refs=[item['initial_observation_ref']]
                for index,tr in enumerate([None]+item['trace']):
                    if tr is not None:refs.append(tr['observation_ref'])
                    # Every remaining calibration or execution can be bought
                    # by at least one catalog action under the full byte cap;
                    # worker still verifies exact legal actions on materialize.
                    remaining=tr['resources_after'] if tr else {'decisions':6,'executions':4,'calibrations':2}
                    if remaining['decisions']<=0 or remaining['executions']+remaining['calibrations']<=0:continue
                    digest=hash_json([refs,remaining])
                    candidates.append({'decision_index':index,'prefix_reference_hash':digest,
                                       'hash_rank':hash_json(['CGX-FA-CAUSE-V1:q-input-prefix',case['case_id'],method,seed,digest])})
                noninitial=[c for c in candidates if c['decision_index']>0]
                selected=min(noninitial or candidates,key=lambda c:c['hash_rank'])
                rows.append({'case':case,'method_key':method,'policy_seed':seed,**selected,
                             'source_A_evidence_hash':item['row']['evidence_trace_hash']})
    result={'selection_rule':'minimum fixed hash among noninitial acquired A prefixes with resources for action; otherwise initial',
            'sample_count':len(rows),'prefixes_per_trajectory':1,'all_methods_seeds_cases_retained':True,
            'all_prefixes_enumerated_for_action_targeting_without_forward':True,
            'maximum_model_and_feature_calls':len(rows)*4,'source_file_sha256':file_sha256(Path(__file__)),'prefixes':rows}
    result['manifest_hash']=hash_json(result);atomic_json(path,result);return result


def job(selected):
    runner=iv._RUNNER;case=selected['case'];method=selected['method_key'];seed=selected['policy_seed']
    identity=hash_json([case['case_id'],method,seed]);path=runner.run/'q_input_subset'/f'{identity}.json.gz'
    if path.exists():return iv.load_blob(path)
    started=time.perf_counter();cpu=time.process_time()
    base={'configuration_id':case['configuration_id'],'case_id':case['case_id'],'method_key':method,'policy_seed':seed}
    try:
        item=iv.load_blob(arm_path(runner.run,case,method,seed))
        initial,_=runner.initial(case,method,seed);history=deepcopy(initial);catalog=action_catalog(history['graph'],runner.config)
        targeting=[];prefix=None
        for index,trace in enumerate(item['trace']):
            legal=feasible_actions(catalog,history['remaining']);q=trace['prediction_before'];suspect=iv.prediction_metrics(q)['max_fault_target']
            targeting.append({**base,'decision_index':index,'source':'discovery_A_acquired_action','action_key':trace['action']['action_key'],
                'action_kind':trace['action']['kind'],'suspect_target':suspect,
                'selected_targets_suspect':targeted(trace['action'],history['graph'],suspect),
                'legal_fraction_targeting_suspect':float(np.mean([targeted(a,history['graph'],suspect) for a in legal])),
                'legal_actions':len(legal),'full_trace_excluded_from_targeted':True,
                'q_healthy_before':iv.prediction_metrics(q)['p_healthy'],
                'predicted_target_before':iv.prediction_metrics(q)['predicted_target_key'],
                'true_target_key':truth_for(case),'is_healthy':case['fault_family']=='healthy'})
            if index==selected['decision_index']:prefix=deepcopy(history)
            history['observations'].append(runner.output.load_observation(trace['observation_ref']))
            history['remaining']=dict(trace['resources_after']);history['action_keys'].append(trace['action']['action_key'])
        if selected['decision_index']==len(item['trace']):prefix=history
        if prefix is None:raise ValueError('Selected prefix absent from actual trace')
        index=selected['decision_index'];q0=item['row']['initial_prediction']
        latest=q0 if index==0 else item['trace'][index-1]['prediction']
        legal=feasible_actions(catalog,prefix['remaining'])
        if len(legal)<=1:raise ValueError('Preselected prefix has no continuing legal action')
        model=runner.policy(method,seed)
        a=runner.score(model,prefix,legal,latest,'fixed_q:latest')
        b=runner.score(model,prefix,legal,q0,'fixed_q:initial')
        left=legal[int(a.argmax())];right=legal[int(b.argmax())];suspect=iv.prediction_metrics(latest)['max_fault_target']
        row={**base,'state':'passed','decision_index':index,'evidence_hash':hash_json(prefix),
             'prefix_reference_hash':selected['prefix_reference_hash'],'checkpoint_hash':model.checkpoint_hash,
             'same_raw_records_and_legal_actions':True,'legal_action_count':len(legal),'prefix_sampling':'one fixed-hash actual prefix per A trajectory',
             'action_changed':left['action_key']!=right['action_key'],'latest_action_key':left['action_key'],'initial_q_action_key':right['action_key'],
             'latest_action_kind':left['kind'],'initial_q_action_kind':right['kind'],
             'latest_targets_suspect':targeted(left,prefix['graph'],suspect),'frozen_q_targets_suspect':targeted(right,prefix['graph'],suspect),
             'legal_fraction_targeting_suspect':float(np.mean([targeted(x,prefix['graph'],suspect) for x in legal])),
             'max_score_difference':float(np.max(np.abs(a-b))),'true_target_key':truth_for(case),'is_healthy':case['fault_family']=='healthy',
             'latest_q':latest,'initial_q':q0,
             'scores':[{'action_key':action['action_key'],'kind':action['kind'],'ports':action['ports'],
                        'latest_q_score':float(x),'initial_q_score':float(y)} for action,x,y in zip(legal,a,b)]}
        result={'state':'passed','row':row,'targeting':targeting,'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu}
    except Exception as exc:
        result={'state':'failed','row':{**base,'state':'failed','error':type(exc).__name__+':'+str(exc)},'targeting':[],
                'traceback':traceback.format_exc()}
    iv.dump_blob(path,result);return result


def run(root,run_dir,workers=16):
    directory=Path(run_dir);plan=seal(directory);results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=iv.initialize,
                             initargs=(root,run_dir,['cuda:2','cuda:3'])) as pool:
        for i,result in enumerate(pool.map(job,plan['prefixes'],chunksize=1),1):
            results.append(result)
            if i%32==0:print(json.dumps({'stage':'fixed_q_subset','completed':i,'total':len(plan['prefixes']),'failed':sum(r['state']!='passed' for r in results)}),flush=True)
    frame=pd.DataFrame([r['row'] for r in results]);frame.to_parquet(directory/'q_input_action_checks.parquet',index=False)
    frame.drop(columns=['scores','latest_q','initial_q'],errors='ignore').to_csv(directory/'q_input_action_checks.csv',index=False)
    pd.DataFrame([t for r in results for t in r['targeting']]).to_csv(directory/'action_suspect_targeting.csv',index=False)
    atomic_json(directory/'q_input_action_checks_complete.json',{'state':'passed' if all(r['state']=='passed' for r in results) else 'incomplete',
                'jobs':len(results),'failed':[r for r in results if r['state']!='passed']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--workers',type=int,default=16);args=parser.parse_args();run(args.root,args.run_dir,args.workers)
