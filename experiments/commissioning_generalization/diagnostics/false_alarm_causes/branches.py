"""Fixed-record calibration, explicit q-input checks, and matched-horizon branches.

All actions come from public evidence and frozen models. Private labels enter
only the outcome columns, after acquisitions and predictions have finished.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
from cgx.provenance import atomic_json,read_json,hash_json,file_sha256
from cgx.observation.actions import action_catalog,feasible_actions,charge
from .core import prefix_descriptors,materialize_prefix,reconstruct_prefixes
from . import interventions as iv


def truth_for(case):
    return 'healthy' if case['fault_family']=='healthy' else case['fault_node']+':'+case['fault_family']


def original_prediction(runner,history,meta):
    path=runner.run/'cache/predictions/final'/f"{meta['evidence_prefix_hash']}.json"
    if path.exists():
        saved=read_json(path)
        return {'target_keys':saved['target_keys'],'log_probabilities_base2':saved['full_log_probabilities']}
    return runner.predict(history,'same_records:original_reader')[0]


def coverage(history):
    domains={n['id']+':'+p['name']:n['clock_domain'] for n in history['graph']['nodes'] for p in n['ports']}
    counts=Counter()
    for obs in history['observations']:
        matches=defaultdict(list)
        for record in obs['records']:
            if record['port'] in domains:matches[record['message_token']].append(record)
        for records in matches.values():
            for i,record in enumerate(records):
                for peer in records[i+1:]:
                    a,b=domains[record['port']],domains[peer['port']]
                    if a!=b:counts[tuple(sorted((a,b)))]+=1
    return counts


def calibration_job(job):
    case,method,seed=job;runner=iv._RUNNER
    key=hash_json([case['case_id'],method,seed]);path=runner.run/'same_records'/f'{key}.json.gz'
    if path.exists():return iv.load_blob(path)
    stamp=time.perf_counter();cpu=time.process_time()
    base={'configuration_id':case['configuration_id'],'case_id':case['case_id'],'method_key':method,'policy_seed':seed}
    try:
        item=runner.artifact(case,method,seed)
        packed,meta=prefix_descriptors(runner.root,item)[-1]
        history=materialize_prefix(runner.root,packed);q=original_prediction(runner,history,meta)
        measured=[b for o in history['observations'] for b in o['calibration_bursts']]
        measured_pairs={tuple(sorted(b['pair'])) for b in measured}
        counts=coverage(history)
        legal={tuple(a['calibration_pair']):a for a in action_catalog(history['graph'],runner.config) if a['kind']=='calibration_only'}
        ranked=sorted((p for p in legal if p not in measured_pairs),key=lambda p:(-counts[p],hash_json(['same-record-calibration',case['case_id'],method,seed,p])))
        pairs=ranked[:max(0,2-len(measured_pairs))]
        plan={**base,'ordinary_records_hash':hash_json([o['records'] for o in history['observations'] if o['records']]),
              'original_prefix_hash':meta['evidence_prefix_hash'],'existing_pairs':[list(p) for p in sorted(measured_pairs)],
              'pairs':[list(p) for p in pairs],'coverage':[counts[p] for p in pairs],
              'cost_bound':{k:sum(legal[p]['cost_bound'][k] for p in pairs) for k in history['remaining']},
              'rule':'acquired cross-clock message equality pairs, count descending, fixed hash tie; fill to two distinct pairs',
              'uses_private_truth_or_offsets':False}
        atomic_json(runner.run/'manifests/same_record_calibration_plans'/f'{key}.json',plan)
        variants=[('retain',history,q,{'executions':0,'bytes':0,'calibrations':0,'decisions':0})]
        removed=deepcopy(history)
        for obs in removed['observations']:obs['calibration_bursts']=[]
        remove_q=runner.predict(removed,'same_records:remove_calibration')[0] if measured else q
        variants.append(('remove',removed,remove_q,{'executions':0,'bytes':0,'calibrations':0,'decisions':0}))
        supplemented=deepcopy(history);extra={k:0 for k in history['remaining']}
        if pairs:
            system=runner.system(case,history)
            executions=history['initial_budget']['executions']-history['remaining']['executions']
            bursts=history['initial_budget']['calibrations']-history['remaining']['calibrations']
            for pair in pairs:
                bursts+=1;obs,reused=runner.acquire(system,legal[pair],executions,bursts,'same_records:calibration')
                supplemented['observations'].append(obs)
                supplemented['action_keys'].append(legal[pair]['action_key'])
                for k in extra:extra[k]+=obs['actual_cost'][k]
            # Supplemental reader-only evidence may exceed original resources.
            # Increase initial accounting by exactly its explicit extra cost;
            # the original remaining resources then stay nonnegative and the
            # history's actual-cost identity still holds. Reader ignores these
            # resource fields, as independently checked in the input audit.
            supplemented['initial_budget']={k:history['initial_budget'][k]+extra[k] for k in extra}
            added_q=runner.predict(supplemented,'same_records:extra_calibration')[0]
        else:added_q=q
        variants.append(('supplement_to_two_pairs',supplemented,added_q,extra))
        rows=[]
        truth=truth_for(case)
        for name,h,p,cost in variants:
            assert hash_json([o['records'] for o in h['observations'] if o['records']])==plan['ordinary_records_hash']
            rows.append({**base,'variant':name,'state':'passed','source':'original_final_acquired_records',
                         'checkpoint_hash':runner.reader.checkpoint_hash,'original_evidence_hash':meta['evidence_prefix_hash'],
                         'evidence_hash':runner.output.put_history(h),'ordinary_records_hash':plan['ordinary_records_hash'],
                         'original_calibrations':len(measured),'added_calibrations':cost['calibrations'],
                         'extra_bytes':cost['bytes'],'extra_decisions':cost['decisions'],'extra_executions':cost['executions'],
                         'exceeds_original_budget':any(cost[k]>history['remaining'][k] for k in cost),
                         'supplementary_evidence':bool(cost['calibrations']),
                         'added_pairs':json.dumps(plan['pairs']) if name=='supplement_to_two_pairs' else '[]',
                         'added_pair_coverage':json.dumps(plan['coverage']) if name=='supplement_to_two_pairs' else '[]',
                         'prediction':p,**iv.prediction_metrics(p,truth)})
        result={'state':'passed','rows':rows,'wall_seconds':time.perf_counter()-stamp,'cpu_seconds':time.process_time()-cpu}
    except Exception as exc:
        result={'state':'failed','rows':[{**base,'state':'failed','error':type(exc).__name__+':'+str(exc)}],
                'traceback':traceback.format_exc()}
    iv.dump_blob(path,result);return result


def targeted(action,graph,target):
    if action['kind'] in ('stop','full_trace') or not action['ports']:return False
    node=target.split(':')[0]
    neighborhood={node}
    for edge in graph['edges']:
        if edge['source']==node:neighborhood.add(edge['target'])
        if edge['target']==node:neighborhood.add(edge['source'])
    return bool({port.split(':')[0] for port in action['ports']} & neighborhood)


def q_input_job(job):
    case,method,seed=job;runner=iv._RUNNER
    key=hash_json([case['case_id'],method,seed]);outpath=runner.run/'q_input_checks'/f'{key}.json.gz'
    if outpath.exists():return iv.load_blob(outpath)
    base={'configuration_id':case['configuration_id'],'case_id':case['case_id'],'method_key':method,'policy_seed':seed}
    try:
        path=runner.run/'interventions/discovery/cases'/f"{hash_json([case['case_id'],method,seed,'A'])}.json.gz"
        item=iv.load_blob(path)
        if item['row']['state']!='passed':raise ValueError('A arm incomplete')
        initial,_=runner.initial(case,method,seed);history=deepcopy(initial)
        q0=item['row']['initial_prediction'];current=q0;rows=[];model=runner.policy(method,seed)
        catalog=action_catalog(history['graph'],runner.config)
        traces=[None]+item['trace']
        for index,trace in enumerate(traces):
            if trace is not None:
                history['observations'].append(runner.output.load_observation(trace['observation_ref']))
                history['remaining']=trace['resources_after'];history['action_keys'].append(trace['action']['action_key'])
                if not trace['diagnosis_updated']:break
                current=trace['prediction']
            actions=feasible_actions(catalog,history['remaining'])
            if len(actions)<=1:continue
            latest=runner.score(model,history,actions,current,'fixed_q:latest')
            frozen=runner.score(model,history,actions,q0,'fixed_q:initial')
            left=actions[int(latest.argmax())];right=actions[int(frozen.argmax())]
            suspect=iv.prediction_metrics(current)['max_fault_target']
            row={**base,'state':'passed','decision_index':index,'evidence_hash':hash_json(history),
                 'same_raw_records_and_legal_actions':True,'legal_action_count':len(actions),
                 'action_changed':left['action_key']!=right['action_key'],'latest_action_key':left['action_key'],
                 'initial_q_action_key':right['action_key'],'latest_action_kind':left['kind'],'initial_q_action_kind':right['kind'],
                 'max_score_difference':float(np.max(np.abs(latest-frozen))),
                 'latest_suspect':suspect,'latest_targets_suspect':targeted(left,history['graph'],suspect),
                 'frozen_q_targets_suspect':targeted(right,history['graph'],suspect),
                 'legal_fraction_targeting_suspect':float(np.mean([targeted(a,history['graph'],suspect) for a in actions])),
                 'latest_q':current,'initial_q':q0,
                 'scores':[{'action_key':a['action_key'],'action_kind':a['kind'],'ports':a['ports'],
                            'calibration_pair':a['calibration_pair'],'latest_q_score':float(x),'initial_q_score':float(y)}
                           for a,x,y in zip(actions,latest,frozen)]}
            rows.append(row)
        result={'state':'passed','rows':rows}
    except Exception as exc:
        result={'state':'failed','rows':[{**base,'state':'failed','error':type(exc).__name__+':'+str(exc)}],'traceback':traceback.format_exc()}
    iv.dump_blob(outpath,result);return result


@contextmanager
def measurement_replicate(replicate):
    """Separate observer noise without changing plant disturbances or task.

    The original implementation derives all substreams from one root. This
    task-local wrapper adds a suffix only to external measurement, acquisition
    phase, timestamp and calibration seeds; plant and token streams stay exact.
    """
    import cgx.env.system as system_module
    import cgx.observation.clocks as clocks_module
    old_system=system_module.stable_seed;old_clocks=clocks_module.stable_seed
    def seed(*parts):
        return old_clocks(*parts,'CGX-FA-CAUSE-V1:measurement_replicate',replicate) if len(parts)>1 and parts[1] in ('external','arm','timestamp','calibration') else old_clocks(*parts)
    system_module.stable_seed=seed;clocks_module.stable_seed=seed
    try:yield
    finally:system_module.stable_seed=old_system;clocks_module.stable_seed=old_clocks


def choose_branches(actions,scores):
    chosen=actions[int(scores.argmax())];selected=[chosen]
    ordinary=[a for a in actions if a['ports'] and not a['calibration_pair']]
    order=lambda a:hash_json(['branch-action',a['action_key']])
    def add(action):
        if action is not None and action['action_key'] not in {a['action_key'] for a in selected}:selected.append(action)
    near=sorted(ordinary,key=lambda a:(abs(a['cost_bound']['bytes']-chosen['cost_bound']['bytes']),order(a)))
    add(next((a for a in near if a['action_key']!=chosen['action_key']),None))
    add(next(iter(sorted((a for a in actions if a['kind']=='calibration_only'),key=order)),None))
    chosen_nodes={p.split(':')[0] for p in chosen['ports']}
    other=sorted(ordinary,key=lambda a:(len(chosen_nodes & {p.split(':')[0] for p in a['ports']}),abs(a['cost_bound']['bytes']-chosen['cost_bound']['bytes']),order(a)))
    add(next((a for a in other if a['action_key'] not in {x['action_key'] for x in selected}),None))
    return selected[:4]


def branch_plan(root,run_dir,replicates=3):
    run_dir=Path(run_dir);path=run_dir/'manifests/action_branch_plan.json'
    if path.exists():return read_json(path)
    cases=read_json(run_dir/'manifests/discovery_cases.json');configs=sorted({c['configuration_id'] for c in cases})
    rows=[]
    from .core import load_artifact
    for method_index,method in enumerate(iv.METHODS):
        for config_index,config in enumerate(configs):
            healthy=(config_index+method_index)%2==0
            eligible=[c for c in cases if c['configuration_id']==config and (c['fault_family']=='healthy')==healthy]
            case=min(eligible,key=lambda c:hash_json(['branch-case',method,c['case_id']]))
            seed=iv.SEEDS[config_index%len(iv.SEEDS)]
            descriptors=prefix_descriptors(root,load_artifact(root,case['case_id'],method,seed))
            candidates=[(p,m) for p,m in descriptors if p['remaining']['executions']>0 and p['remaining']['decisions']>0]
            packed,meta=min(candidates,key=lambda pm:hash_json(['branch-prefix',method,pm[1]['evidence_prefix_hash']]))
            rows.append({'case':case,'method_key':method,'policy_seed':seed,'packed':packed,'meta':meta,
                         'horizon':min(2,packed['remaining']['decisions']) if method=='gnn_depth2' else packed['remaining']['decisions']})
    plan={'prefixes':rows,'prefix_count':len(rows),'noise_replicates':replicates,'maximum_actions':4,
          'selection':'one fixed-hash case/prefix per method and configuration; alternating healthy/fault and policy seed',
          'horizon_rule':'gnn_depth2 at most two decisions; gnn_adaptive full remaining decisions; same frozen continuation after alternatives',
          'noise_rule':'retain task, private system, plant and token RNG; new external/arm/timestamp/calibration observer substreams',
          'STOP':'analytic zero reference; no simulator attempt',
          'source_file_sha256':file_sha256(Path(__file__)),
          'maximum_action_steps':sum(x['horizon'] for x in rows)*4*replicates,
          'maximum_robot_executions':sum(min(x['horizon'],x['packed']['remaining']['executions']) for x in rows)*4*replicates}
    plan['plan_hash']=hash_json(plan);atomic_json(path,plan);return plan


def branch_job(job):
    selected,replicates=job;runner=iv._RUNNER
    case=selected['case'];method=selected['method_key'];seed=selected['policy_seed'];meta=selected['meta']
    key=hash_json([case['case_id'],method,seed,meta['evidence_prefix_hash']]);path=runner.run/'action_branches'/f'{key}.json.gz'
    if path.exists():return iv.load_blob(path)
    base={'configuration_id':case['configuration_id'],'case_id':case['case_id'],'method_key':method,'policy_seed':seed,
          'decision_index':meta['decision_index'],'evidence_prefix_hash':meta['evidence_prefix_hash'],'horizon':selected['horizon']}
    try:
        history=materialize_prefix(runner.root,selected['packed']);prediction=original_prediction(runner,history,meta)
        model=runner.policy(method,seed);catalog=action_catalog(history['graph'],runner.config)
        # Model's represented horizon and continuation use this same truncated
        # remaining decision count. Other acquisition resources remain actual.
        state=deepcopy(history);state['remaining']['decisions']=selected['horizon']
        actions=feasible_actions(catalog,state['remaining']);scores=runner.score(model,state,actions,prediction,'branches:initial_score')
        alternatives=choose_branches(actions,scores);scoremap={a['action_key']:float(s) for a,s in zip(actions,scores)}
        chosen=actions[int(scores.argmax())];rows=[];traces=[]
        atomic_json(runner.run/'manifests/action_branch_actions'/f'{key}.json',{
            **base,'actions':alternatives,'scores':scoremap,'chosen_action':chosen,'selection_uses_truth':False})
        for replicate in range(replicates):
            for alternative in alternatives:
                h=deepcopy(state);q=deepcopy(prediction);first_q=q;steps=[]
                exec_index=meta['executions_used'];burst_index=meta['calibrations_used'];action=alternative
                noisy_case=deepcopy(case);noisy_case['measurement_noise_replicate']=replicate
                system=runner.system(noisy_case,h)
                failed=None
                try:
                    with measurement_replicate(replicate):
                        for depth in range(selected['horizon']):
                            if action['kind']=='stop':break
                            exec_index+=int(bool(action['ports']));burst_index+=int(bool(action['calibration_pair']))
                            obs,reused=runner.acquire(system,action,exec_index,burst_index,'branches:acquisition')
                            h['remaining']=charge(h['remaining'],obs['actual_cost']);h['observations'].append(obs);h['action_keys'].append(action['action_key'])
                            q,_=runner.predict(h,'branches:reader')
                            if depth==0:first_q=q
                            steps.append({'action':action,'observation_ref':runner.output.put_observation(obs),'prediction':q,
                                          'execution_index':exec_index,'burst_index':burst_index,'resources_after':dict(h['remaining'])})
                            if depth+1>=selected['horizon'] or h['remaining']['decisions']<=0:break
                            legal=feasible_actions(catalog,h['remaining'])
                            values=runner.score(model,h,legal,q,'branches:continuation');action=legal[int(values.argmax())]
                except Exception as exc:failed=type(exc).__name__+':'+str(exc)
                # Labels enter only here, after the branch and all actions.
                truth=truth_for(case);target=prediction['target_keys'].index(truth)
                immediate=first_q['log_probabilities_base2'][target]-prediction['log_probabilities_base2'][target]
                cumulative=q['log_probabilities_base2'][target]-prediction['log_probabilities_base2'][target]
                rows.append({**base,'state':'failed' if failed else 'passed','error':failed,
                             'noise_replicate':replicate,'action_key':alternative['action_key'],'action_kind':alternative['kind'],
                             'action_ports':json.dumps(alternative['ports']),'action_cost_bound_bytes':alternative['cost_bound']['bytes'],
                             'predicted_q':scoremap[alternative['action_key']],'max_predicted_q':float(max(scores)),'STOP_score':0.,
                             'policy_selected':alternative['action_key']==chosen['action_key'],
                             'immediate_sample_gain_bits':immediate,'matched_horizon_sample_gain_bits':cumulative,
                             'predicted_minus_matched_horizon_gain':scoremap[alternative['action_key']]-cumulative,
                             'decisions_completed':len(steps),'executions_used':exec_index-meta['executions_used'],
                             'calibrations_used':burst_index-meta['calibrations_used'],
                             'bytes_used':state['remaining']['bytes']-h['remaining']['bytes'],
                             'true_target_key':truth,'is_healthy':truth=='healthy',
                             'comparison_is_exact_expected_Q':False,'continuation_policy':'same frozen checkpoint, greedy feasible actions',
                             'diagnostic_reader':'final_GNN (training Q targets use OOF readers; this is deployment sample gain)'})
                traces.append({'action_key':alternative['action_key'],'noise_replicate':replicate,'steps':steps})
        result={'state':'passed' if all(r['state']=='passed' for r in rows) else 'incomplete','rows':rows,'traces':traces}
    except Exception as exc:
        result={'state':'failed','rows':[{**base,'state':'failed','error':type(exc).__name__+':'+str(exc)}],'traceback':traceback.format_exc()}
    iv.dump_blob(path,result);return result


def run(root,run_dir,stage,workers=4,devices=('cuda:2','cuda:3'),replicates=3):
    run_dir=Path(run_dir);cases=read_json(run_dir/'manifests/discovery_cases.json')
    function={'calibration':calibration_job,'q_input':q_input_job,'branches':branch_job}[stage]
    if stage=='branches':
        plan=branch_plan(root,run_dir,replicates)
        jobs=[(p,plan['noise_replicates']) for p in plan['prefixes']]
    else:jobs=[(c,m,s) for c in cases for m in iv.METHODS for s in iv.SEEDS]
    atomic_json(run_dir/'manifests'/f'{stage}_workload.json',{'stage':stage,'jobs':len(jobs),'workers':workers,
                'devices':list(devices),'source_hash':file_sha256(Path(__file__))})
    results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=iv.initialize,
                             initargs=(str(root),str(run_dir),list(devices))) as pool:
        for i,result in enumerate(pool.map(function,jobs,chunksize=1),1):
            results.append(result)
            if i%8==0:print(json.dumps({'stage':stage,'jobs_completed':i,'total_jobs':len(jobs),'failed':sum(r['state']!='passed' for r in results)}),flush=True)
    rows=[r for result in results for r in result['rows']]
    filename={'calibration':'calibration_same_records','q_input':'q_input_action_checks','branches':'action_value_checks'}[stage]
    frame=pd.DataFrame(rows);frame.to_parquet(run_dir/(filename+'.parquet'),index=False)
    flat=frame.drop(columns=['scores','prediction','latest_q','initial_q'],errors='ignore');flat.to_csv(run_dir/(filename+'.csv'),index=False)
    atomic_json(run_dir/(filename+'_complete.json'),{'state':'passed' if all(r['state']=='passed' for r in results) else 'incomplete',
                'jobs':len(jobs),'rows':len(rows),'failed':[r for r in results if r['state']!='passed']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--stage',required=True,choices=('calibration','q_input','branches'))
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--devices',nargs='+',default=['cuda:2','cuda:3'])
    parser.add_argument('--replicates',type=int,default=3)
    args=parser.parse_args();run(args.root,args.run_dir,args.stage,args.workers,args.devices,args.replicates)
