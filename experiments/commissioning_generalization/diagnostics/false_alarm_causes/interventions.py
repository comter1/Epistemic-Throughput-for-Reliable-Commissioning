"""Frozen-checkpoint interventions for CGX-FA-CAUSE-V1, never main-run writes.

A: acquired evidence + current q; B: initial evidence + initial q;
C/D: A/B with two public-rule calibrations prepaid at the beginning.
The same final reader always reads all actually acquired observations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch

from cgx.config import load_config
from cgx.data.store import Store
from cgx.env.system import System, generate_task
from cgx.evaluation.rollout import call_deadline
from cgx.models.features import FEATURE_VERSION, tensorize_history, collate_features, collate_actions
from cgx.observation.actions import initial_budget, action_catalog, feasible_actions, charge
from cgx.provenance import atomic_json, read_json, hash_json, file_sha256
from cgx.seeds import seed_for
from cgx.training.engine import load_reader, load_policy
from .ledger import Ledger

METHODS = ('gnn_depth2', 'gnn_adaptive')
SEEDS = (11, 23, 37)
WATCHDOG_SECONDS = 240.0


def dump_blob(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with gzip.open(temporary,'wt',compresslevel=1) as handle:json.dump(obj,handle,separators=(',',':'))
    temporary.replace(path)


def load_blob(path):
    with gzip.open(path,'rt') as handle:return json.load(handle)


def stage_add(timing, name, seconds):
    timing[name]=timing.get(name,0.)+float(seconds)


def prediction_metrics(prediction, truth=None):
    keys=prediction['target_keys']; logp=np.asarray(prediction['log_probabilities_base2'])
    p=np.exp2(logp); i=int(np.argmax(p)); healthy=keys.index('healthy')
    faults=[j for j,key in enumerate(keys) if key!='healthy'];j=max(faults,key=lambda j:p[j])
    out={'predicted_target_key':keys[i],'p_healthy':float(p[healthy]),
         'max_fault_probability':float(p[j]),'max_fault_target':keys[j],
         'fault_minus_healthy':float(p[j]-p[healthy])}
    if truth is not None:
        out.update(true_target_key=truth,is_healthy=truth=='healthy',
                   healthy_false_alarm=bool(truth=='healthy' and keys[i]!='healthy'),
                   fault_miss=bool(truth!='healthy' and keys[i]=='healthy'),
                   correct_joint=keys[i]==truth,final_nll_bits=float(-logp[keys.index(truth)]))
    return out


class Runner:
    def __init__(self, root, run, device='cpu'):
        self.root=Path(root);self.run=Path(run);self.device=device
        self.config=load_config();self.ledger=Ledger(self.run)
        self.source=Store(root);self.output=Store(run)
        self.reader=load_reader(root,'final',device=device)
        self.policies={};self.predictions={};self.scores={};self.timings=defaultdict(float)
        self.assets=self.root/'assets/mujoco_menagerie/franka_emika_panda'
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False

    def policy(self, method, seed):
        key=(method,seed)
        if key not in self.policies:self.policies[key]=load_policy(self.root,method,seed,device=self.device)
        return self.policies[key]

    def sync(self):
        if str(self.device).startswith('cuda'):torch.cuda.synchronize(self.device)

    def predict(self, history, detail):
        # Resource state and action keys remain in the key even though the main
        # diagnostic backbone does not consume the resource feature vector.
        key=hash_json([self.reader.checkpoint_hash,FEATURE_VERSION,history])
        if key in self.predictions:return deepcopy(self.predictions[key]),{'cache_hit':True}
        started=time.perf_counter()
        from .core import predict
        with call_deadline(WATCHDOG_SECONDS):
            prediction,timing=predict(self.reader,history,self.ledger,detail=detail,
                                      cache_dir=self.run/'cache'/'reader_predictions')
        self.predictions[key]=deepcopy(prediction)
        stage_add(self.timings,'diagnosis_total',time.perf_counter()-started)
        return prediction,timing

    def score(self, model, history, actions, prediction, detail):
        stamp=time.perf_counter()
        state=dict(history)
        if model.method=='gnn_depth2':state['remaining']={**history['remaining'],'decisions':min(2,history['remaining']['decisions'])}
        stage_add(self.timings,'history_copy',time.perf_counter()-stamp)
        key=hash_json([model.checkpoint_hash,FEATURE_VERSION,state,prediction,[a['action_key'] for a in actions]])
        if key in self.scores:return self.scores[key].copy()
        with call_deadline(WATCHDOG_SECONDS):
            self.ledger.reserve('feature_build',detail=detail+':selector')
            stamp=time.perf_counter()
            features=tensorize_history(state,prediction=prediction)
            batch=collate_features([features],self.device)
            candidates=collate_actions(batch,[actions],self.device)
            stage_add(self.timings,'selector_feature_build',time.perf_counter()-stamp)
            self.ledger.reserve('model_forward',detail=detail+':selector')
            stamp=time.perf_counter()
            with torch.inference_mode():scores=model(batch,candidates)
            self.sync();stage_add(self.timings,'selector_model',time.perf_counter()-stamp)
            values=scores.detach().cpu().numpy()
        if not np.isfinite(values).all():raise FloatingPointError('Nonfinite selector scores')
        self.scores[key]=values.copy()
        return values

    def artifact(self, case, method, seed, cap=4):
        digest=hash_json([method,seed,cap,1.0,case['case_id']])
        path=self.root/'evaluation/main/cases'/f'{digest}.json'
        item=read_json(path)
        if item['result_hash']!=hash_json({'row':item['row'],'trace':item['trace']}):raise ValueError('Original result hash mismatch')
        return item

    def initial(self, case, method, seed, fresh=False):
        if not fresh:
            item=self.artifact(case,method,seed)
            stamp=time.perf_counter()
            packed=self.source.history_references(item['row']['evidence_trace_hash'])
            history={'graph':self.source._get('graphs',packed['graph_ref']),
                     'observations':[self.source.load_observation(packed['observation_refs'][0])],
                     'nominal_task':self.source._get('tasks',packed['task_ref']),
                     'requirements':packed['requirements'],'initial_budget':packed['initial_budget'],
                     'remaining':dict(packed['initial_budget']),'action_keys':[]}
            stage_add(self.timings,'original_record_read',time.perf_counter()-stamp)
            return history,item
        cached=self.run/'cache'/'fresh_initial'/f"{case['case_id']}.json.gz"
        cached.parent.mkdir(parents=True,exist_ok=True)
        with cached.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            return self._fresh_initial(case,cached)

    def _fresh_initial(self, case, cached):
        if cached.exists():return load_blob(cached),None
        graph=deepcopy(self.source.graph(case['configuration_id']))
        task=generate_task(case['streams']['task'])
        system=System(graph,case,task,self.config,self.assets)
        self.ledger.reserve('robot_execution',detail='fresh:initial')
        stamp=time.perf_counter()
        with call_deadline(WATCHDOG_SECONDS):initial=system.run(ports=(),sample_hz=10,execution_index=0)
        stage_add(self.timings,'actual_simulation',time.perf_counter()-stamp)
        rng=np.random.default_rng(seed_for(self.config['seeds']['manifest'],'evaluation_node_order',case['case_id']))
        rng.shuffle(graph['nodes'])
        budget=initial_budget(graph,1.0,4)
        history={'graph':graph,'observations':[initial.observation],'remaining':dict(budget),
                 'initial_budget':budget,'action_keys':[],'nominal_task':task,
                 'requirements':{k:self.config['system'][k] for k in ('position_tolerance_m','hold_seconds','joint_limit_tolerance_rad')}}
        dump_blob(cached,history)
        return history,None

    def system(self, case, history):
        # Physics uses the manifest graph order, as the original rollout does;
        # only model-side graph nodes are independently permuted.
        return System(self.source.graph(case['configuration_id']),case,history['nominal_task'],self.config,self.assets)

    def acquire(self, system, action, execution_index, burst_index, detail):
        key=hash_json([system.case,system.task,action,execution_index,burst_index,system.implementation])
        path=self.run/'cache'/'acquired'/f'{key}.json.gz'
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            return self._acquire_locked(system,action,execution_index,burst_index,detail,path)

    def _acquire_locked(self,system,action,execution_index,burst_index,detail,path):
        # This cache is consulted only AFTER the arm chooses the action. It never
        # reveals a future observation to the selector. Logical resource cost is
        # charged on every arm even if a deterministic realization is reused.
        if path.exists():
            stamp=time.perf_counter();observation=load_blob(path)
            stage_add(self.timings,'acquisition_cache_read',time.perf_counter()-stamp)
            return observation,True
        if action['ports']:self.ledger.reserve('robot_execution',detail=detail)
        if action['calibration_pair']:self.ledger.reserve('calibration',detail=detail)
        stamp=time.perf_counter()
        with call_deadline(WATCHDOG_SECONDS):result=system.acquire(action,execution_index,burst_index)
        stage_add(self.timings,'actual_simulation' if action['ports'] else 'actual_calibration',time.perf_counter()-stamp)
        dump_blob(path,result.observation)
        return result.observation,False

    def calibration_plan(self, case, method, seed, history, prediction):
        path=self.run/'manifests'/'calibration_plans'/f'{hash_json([case["case_id"],method,seed])}.json'
        if path.exists():return read_json(path)
        model=self.policy(method,seed);catalog=action_catalog(history['graph'],self.config)
        ordinary=[a for a in feasible_actions(catalog,history['remaining']) if a['ports'] and not a['calibration_pair']]
        values=self.score(model,history,ordinary,prediction,'calibration_plan')
        top=ordinary[int(np.argmax(values))]
        graph=history['graph'];nodes={n['id']:n for n in graph['nodes']};ports=set(top['ports'])
        declared=Counter();covered=Counter()
        for e in graph['edges']:
            pair=tuple(sorted((nodes[e['source']]['clock_domain'],nodes[e['target']]['clock_domain'])))
            if pair[0]==pair[1]:continue
            declared[pair]+=1
            if f"{e['source']}:{e['source_port']}" in ports and f"{e['target']}:{e['target_port']}" in ports:covered[pair]+=1
        legal={tuple(a['calibration_pair']):a for a in catalog if a['kind']=='calibration_only'}
        tie=lambda p:hash_json(['CGX-FA-CAUSE-V1:calibration-pair',case['case_id'],method,seed,p])
        ranking=sorted((p for p in legal if covered[p]>0),key=lambda p:(-covered[p],tie(p)))
        ranking+=sorted((p for p in legal if covered[p]==0),key=lambda p:(-declared[p],tie(p)))
        pairs=ranking[:2]
        plan={'case_id':case['case_id'],'configuration_id':case['configuration_id'],'method_key':method,'policy_seed':seed,
              'checkpoint_hash':model.checkpoint_hash,'initial_evidence_hash':hash_json(history),
              'top_ordinary_action':top,'top_ordinary_score':float(max(values)),
              'pairs':[list(p) for p in pairs],'actions':[legal[p] for p in pairs],
              'cost_bound':{k:sum(legal[p]['cost_bound'][k] for p in pairs) for k in history['remaining']},
              'ranking':[{'pair':list(p),'covered_connections':covered[p],'declared_connections':declared[p],'tie_hash':tie(p)} for p in ranking],
              'selection_uses_private_truth':False,'rule':'covered declared cross-clock connections, then declared connections, then fixed hash',
              'created_before_any_arm_for_key':True}
        plan['plan_hash']=hash_json(plan);atomic_json(path,plan)
        return plan

    def run_arm(self, case, method, seed, arm, initial, plan, phase='discovery'):
        model=self.policy(method,seed);started=time.perf_counter();history=deepcopy(initial)
        system=self.system(case,history);prediction,_=self.predict(history,phase+':initial_reader');initial_q=deepcopy(prediction)
        execution_index=burst_index=0;trace=[];errors=[];termination='resources_exhausted'
        queued=list(plan['actions']) if arm in ('C','D') else []
        catalog=action_catalog(history['graph'],self.config)
        frozen_evidence=arm in ('B','D');frozen_q=arm in ('B','D','E')
        try:
            while history['remaining']['decisions']>0:
                actions=feasible_actions(catalog,history['remaining'])
                if all(a['kind']=='stop' for a in actions):break
                if queued:
                    action=queued.pop(0);score=None;selection='precommitted_calibration'
                    if action not in actions:raise ValueError('Precommitted calibration infeasible')
                else:
                    selector=dict(initial) if frozen_evidence else dict(history)
                    selector['remaining']=dict(history['remaining']);selector['action_keys']=list(history['action_keys'])
                    q=initial_q if frozen_q else prediction
                    values=self.score(model,selector,actions,q,phase+':'+arm)
                    index=int(np.argmax(values));action=actions[index];score=float(values[index]);selection='frozen_checkpoint'
                if action['kind']=='stop':termination='policy_stop';break
                before=dict(history['remaining']);prior_prediction=deepcopy(prediction)
                execution_index+=int(bool(action['ports']));burst_index+=int(bool(action['calibration_pair']))
                observation,reused=self.acquire(system,action,execution_index,burst_index,phase+':'+arm)
                history['remaining']=charge(history['remaining'],observation['actual_cost'])
                history['observations'].append(observation);history['action_keys'].append(action['action_key'])
                trace.append({'decision_index':len(trace)+1,'action':action,'selected_score':score,'selection':selection,
                              'observation_ref':self.output.put_observation(observation),'observation_reused':reused,
                              'prediction_before':prior_prediction,'diagnosis_updated':False,
                              'resources_before':before,'resources_after':dict(history['remaining']),
                              'execution_index':execution_index,'burst_index':burst_index})
                prediction,timing=self.predict(history,phase+':'+arm+':reader')
                trace[-1].update(prediction=prediction,reader_timing=timing,diagnosis_updated=True)
        except Exception as exc:
            errors.append(type(exc).__name__+':'+str(exc));termination='wide_watchdog_or_error'
        truth='healthy' if case['fault_family']=='healthy' else case['fault_node']+':'+case['fault_family']
        used={k:history['initial_budget'][k]-history['remaining'][k] for k in history['remaining']}
        row={'phase':phase,'configuration_id':case['configuration_id'],'case_id':case['case_id'],'method_key':method,'policy_seed':seed,
             'arm':arm,'intervention_name':{'A':'Current-evidence/current-q; free calibration','B':'Selector-keeps-initial-evidence',
             'C':'Current-evidence/current-q; prepaid calibration','D':'Selector-keeps-initial-evidence; prepaid calibration',
             'E':'Current evidence; selector keeps initial q'}[arm],
             'checkpoint_hash':model.checkpoint_hash,'diagnostic_checkpoint_hash':self.reader.checkpoint_hash,
             'execution_cap':4,'watchdog_seconds_per_call':WATCHDOG_SECONDS,'original_two_second_deadlines_disabled':True,
             'device':self.device,'feature_version':FEATURE_VERSION,'calibration_plan_hash':plan['plan_hash'],
             'executions_used':used['executions'],'calibrations_used':used['calibrations'],'bytes_used':used['bytes'],
             'decisions_used':used['decisions'],'end_to_end_seconds':time.perf_counter()-started,
             'execution_attempts_in_arm':execution_index,'calibration_attempts_in_arm':burst_index,
             'last_valid_prediction_prefix':max([0]+[t['decision_index'] for t in trace if t['diagnosis_updated']]),
             'evidence_trace_hash':self.output.put_history(history),'state':'passed' if not errors else 'failed',
             'errors':errors,'termination_reason':termination,'prediction':prediction,'initial_prediction':initial_q,
             **prediction_metrics(prediction,truth)}
        return {'row':row,'trace':trace,'initial_observation_ref':self.output.put_observation(initial['observations'][0])}


_RUNNER=None
def initialize(root, run, devices):
    global _RUNNER
    identity=mp.current_process()._identity
    index=(identity[0]-1 if identity else 0)%len(devices)
    _RUNNER=Runner(root,run,devices[index])


def prepare_case_job(job):
    case,method,seed,phase,arms=job;runner=_RUNNER
    initial,_=runner.initial(case,method,seed,phase=='confirmation')
    prediction,_=runner.predict(initial,phase+':initial_plan')
    plan=runner.calibration_plan(case,method,seed,initial,prediction)
    return {'case_id':case['case_id'],'method_key':method,'policy_seed':seed,'plan_hash':plan['plan_hash']}


def run_case_job(job):
    case,method,seed,phase,arms=job;runner=_RUNNER
    identity=hash_json([phase,case['case_id'],method,seed])
    directory=runner.run/'interventions'/phase/'jobs';directory.mkdir(parents=True,exist_ok=True)
    marker=directory/f'{identity}.json'
    if marker.exists() and read_json(marker).get('state')=='passed':return read_json(marker)
    stamp=time.perf_counter();cpu=time.process_time();before=dict(runner.timings)
    try:
        initial,original=runner.initial(case,method,seed,phase=='confirmation')
        prediction,_=runner.predict(initial,phase+':initial_plan')
        plan=runner.calibration_plan(case,method,seed,initial,prediction)
        outputs=[]
        for arm in arms:
            path=runner.run/'interventions'/phase/'cases'/f'{hash_json([case["case_id"],method,seed,arm])}.json.gz'
            if path.exists() and load_blob(path)['row']['state']=='passed':outputs.append(str(path));continue
            item=runner.run_arm(case,method,seed,arm,initial,plan,phase)
            if original is not None:
                item['original_row']={k:original['row'][k] for k in ('prediction','timeout','abstained','final_nll_bits','predicted_target_key','evidence_trace_hash')}
            dump_blob(path,item);outputs.append(str(path))
        result={'state':'passed' if all(load_blob(p)['row']['state']=='passed' for p in outputs) else 'incomplete',
                'case_id':case['case_id'],'method_key':method,'policy_seed':seed,'phase':phase,'arms':list(arms),
                'outputs':outputs,'wall_seconds':time.perf_counter()-stamp,'cpu_seconds':time.process_time()-cpu,
                'device':runner.device,'timing':{k:v-before.get(k,0.) for k,v in runner.timings.items()}}
    except Exception as exc:
        result={'state':'failed','case_id':case['case_id'],'method_key':method,'policy_seed':seed,'phase':phase,
                'error':type(exc).__name__+':'+str(exc),'traceback':traceback.format_exc(),'wall_seconds':time.perf_counter()-stamp}
    atomic_json(marker,result)
    return result


def run(root, run_dir, phase='discovery', workers=8, devices=('cuda:2','cuda:3'), arms=('A','B','C','D'), limit=None):
    run_dir=Path(run_dir)
    manifest=run_dir/'manifests'/('fresh_cases.json' if phase=='confirmation' else 'discovery_cases.json')
    cases=read_json(manifest)
    if isinstance(cases,dict):cases=cases.get('cases',cases.get('selected_cases'))
    if limit is not None:cases=cases[:limit]
    jobs=[(c,m,s,phase,tuple(arms)) for c in cases for m in METHODS for s in SEEDS]
    plan={'phase':phase,'case_manifest':str(manifest),'case_manifest_sha256':file_sha256(manifest),'cases':len(cases),
          'jobs':len(jobs),'arms':list(arms),'methods':list(METHODS),'seeds':list(SEEDS),
          'devices':list(devices),'workers':workers,'watchdog_seconds_per_call':WATCHDOG_SECONDS,
          'initial_reused':phase!='confirmation','maximum_additional_logical_executions':len(jobs)*len(arms)*4,
          'cache_boundary':'Only previously acquired identical action realization after action selection; no policy peeking',
          'source_file_sha256':file_sha256(Path(__file__))}
    path=run_dir/'manifests'/f'interventions_{phase}.json'
    if path.exists() and read_json(path)!=plan:raise ValueError('Intervention workload manifest differs on resume')
    atomic_json(path,plan)
    results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize,
                             initargs=(str(root),str(run_dir),list(devices))) as pool:
        plans=list(pool.map(prepare_case_job,jobs,chunksize=1))
        atomic_json(run_dir/'manifests'/f'calibration_plans_{phase}_sealed.json',
                    {'plans':plans,'plans_hash':hash_json(plans),'sealed_before_all_arms':True})
        print(json.dumps({'phase':phase,'calibration_plans_sealed':len(plans)}),flush=True)
        for i,result in enumerate(pool.map(run_case_job,jobs,chunksize=1),1):
            results.append(result)
            if i%8==0:print(json.dumps({'phase':phase,'completed_jobs':i,'total_jobs':len(jobs),'failed_jobs':sum(r['state']!='passed' for r in results)}),flush=True)
    rows=[]
    for item in results:
        for path in item.get('outputs',[]):rows.append(load_blob(path)['row'])
    if rows:
        output=run_dir/('fresh_intervention_outcomes.parquet' if phase=='confirmation' else 'intervention_outcomes.parquet')
        pd.DataFrame(rows).to_parquet(output,index=False)
    atomic_json(run_dir/'interventions'/phase/'complete.json',{'state':'passed' if all(r['state']=='passed' for r in results) else 'incomplete',
                'jobs':len(results),'rows':len(rows),'failed_jobs':[r for r in results if r['state']!='passed']})
    print(json.dumps({'phase':phase,'rows':len(rows),'jobs':len(results)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--phase',default='discovery',choices=('discovery','confirmation'))
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--devices',nargs='+',default=['cuda:2','cuda:3'])
    parser.add_argument('--arms',nargs='+',default=['A','B','C','D']);parser.add_argument('--limit',type=int)
    args=parser.parse_args();run(args.root,args.run_dir,args.phase,args.workers,args.devices,args.arms,args.limit)
