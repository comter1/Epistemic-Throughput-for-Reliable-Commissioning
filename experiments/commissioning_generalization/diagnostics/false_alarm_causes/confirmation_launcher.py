"""Fresh A with a precommitted STOP after its first extra robot execution.

S shares A's physically acquired prefix and the prediction actually computed at
that prefix. The universal stopping rule is nonanticipatory and never inspects
labels, confidence, correctness, or later observations. A retains the complete
original online policy. No independent S wall time is fabricated.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import pandas as pd
from cgx.provenance import atomic_json,read_json,hash_json,file_sha256
from . import interventions as iv
from .branches import truth_for


def derive_stop_first(runner,case,method,seed):
    path=runner.run/'interventions/confirmation/cases'/f"{hash_json([case['case_id'],method,seed,'S'])}.json.gz"
    if path.exists():return str(path)
    source=runner.run/'interventions/confirmation/cases'/f"{hash_json([case['case_id'],method,seed,'A'])}.json.gz"
    item=iv.load_blob(source);started=time.perf_counter()
    first=next((i+1 for i,t in enumerate(item['trace']) if t['action']['ports']),None)
    stop=len(item['trace']) if first is None else first
    trace=deepcopy(item['trace'][:stop])
    history=runner.output.load_history(item['row']['evidence_trace_hash'])
    history['observations']=history['observations'][:stop+1]
    history['action_keys']=history['action_keys'][:stop]
    history['remaining']=dict(trace[-1]['resources_after'] if trace else history['initial_budget'])
    valid=[(i+1,t['prediction']) for i,t in enumerate(trace) if t['diagnosis_updated']]
    last_valid,q=valid[-1] if valid else (0,item['row']['initial_prediction'])
    used={k:history['initial_budget'][k]-history['remaining'][k] for k in history['remaining']}
    completed=(first is not None and trace[-1]['diagnosis_updated']) or (first is None and item['row']['state']=='passed')
    row={**item['row'],'arm':'S','intervention_name':'STOP after first additional robot execution and completed diagnosis',
         'state':'passed' if completed else 'failed','errors':[] if completed else item['row']['errors'],
         'termination_reason':'precommitted_stop_after_first_robot_execution' if first is not None else item['row']['termination_reason'],
         'executions_used':used['executions'],'calibrations_used':used['calibrations'],'bytes_used':used['bytes'],
         'decisions_used':used['decisions'],'execution_attempts_in_arm':used['executions'],
         'calibration_attempts_in_arm':used['calibrations'],'last_valid_prediction_prefix':last_valid,
         'prediction':q,'evidence_trace_hash':runner.output.put_history(history),
         'end_to_end_seconds':None,'standalone_wall_seconds_measured':False,
         'timing_source':'A shared physical prefix; standalone S wall time was not measured',
         'additional_model_forward_calls':0,'additional_feature_build_calls':0,'additional_robot_execution_attempts':0,
         'paired_A_evidence_trace_hash':item['row']['evidence_trace_hash'],
         'source_A_prefix_decision_index':stop,'first_robot_execution_exists':first is not None,
         'nonanticipatory_public_rule':True,'rule_uses_prediction_or_truth':False,
         **iv.prediction_metrics(q,truth_for(case))}
    row['derivation_wall_seconds']=time.perf_counter()-started
    out={'row':row,'trace':trace,'initial_observation_ref':item['initial_observation_ref'],
         'shared_real_prefix':True,'independently_resampled':False,'source_A_file':str(source)}
    iv.dump_blob(path,out);return str(path)


def confirmation_job(job):
    result=iv.run_case_job(job);runner=iv._RUNNER;case,method,seed,phase,arms=job
    if result['state']=='failed':return result
    try:
        path=derive_stop_first(runner,case,method,seed)
        result=dict(result,outputs=result['outputs']+[path],analysis_arms=['A','S'])
        result['state']='passed' if all(iv.load_blob(p)['row']['state']=='passed' for p in result['outputs']) else 'incomplete'
    except Exception as exc:
        result=dict(result,state='failed',derivation_error=type(exc).__name__+':'+str(exc),traceback=traceback.format_exc())
    atomic_json(runner.run/'interventions/confirmation/jobs_with_shared_prefix'/f"{hash_json([case['case_id'],method,seed])}.json",result)
    return result


def run(root,run_dir,workers=32,devices=('cuda:0','cuda:1','cuda:2','cuda:3')):
    run_dir=Path(run_dir)
    selection_path=run_dir/'manifests/confirmation_selection.json'
    if not selection_path.exists():raise ValueError('Parent must seal the scientific selection before fresh simulation')
    selection=read_json(selection_path)
    if selection.get('analysis_arms')!=['A','S']:raise ValueError('Launcher only implements the sealed A/S public stop rule')
    manifest=run_dir/'manifests/fresh_cases.json';cases=read_json(manifest)
    # The sealed tasks, generated from independent streams, are authoritative.
    tasks=read_json(run_dir/'manifests/fresh_tasks.json')
    from cgx.env.system import generate_task
    for case in cases:
        if tasks[case['case_id']]!=generate_task(case['streams']['task']):raise ValueError('Fresh task differs from sealed task manifest')
    jobs=[(case,method,seed,'confirmation',('A',)) for case in cases for method in iv.METHODS for seed in iv.SEEDS]
    plan={'phase':'confirmation','case_manifest':str(manifest),'case_manifest_sha256':file_sha256(manifest),
          'cases':len(cases),'jobs':len(jobs),'arms':['A'],'analysis_arms':['A','S'],
          'methods':list(iv.METHODS),'seeds':list(iv.SEEDS),'devices':list(devices),'workers':workers,
          'watchdog_seconds_per_call':iv.WATCHDOG_SECONDS,'maximum_new_robot_executions':len(cases)+4*len(jobs),
          'maximum_model_and_feature_calls':30*len(jobs),
          'S_rule':'Follow A until first additional robot execution has a completed reader prediction, then STOP; if A already stopped, retain its endpoint',
          'S_additional_physical_or_model_calls':0,'S_standalone_wall_time_measured':False,
          'selection_manifest_sha256':file_sha256(selection_path),
          'runner_source_sha256':file_sha256(Path(iv.__file__)),'launcher_source_sha256':file_sha256(Path(__file__))}
    path=run_dir/'manifests/interventions_confirmation.json'
    if path.exists() and read_json(path)!=plan:raise ValueError('Fresh workload differs from previously sealed run')
    atomic_json(path,plan)
    results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=iv.initialize,
                             initargs=(str(root),str(run_dir),list(devices))) as pool:
        plans=list(pool.map(iv.prepare_case_job,jobs,chunksize=1))
        atomic_json(run_dir/'manifests/calibration_plans_confirmation_sealed.json',
                    {'plans':plans,'plans_hash':hash_json(plans),'sealed_before_all_arms':True})
        print(json.dumps({'phase':'confirmation','calibration_plans_sealed':len(plans),'physical_arms':['A'],'analysis_arms':['A','S']}),flush=True)
        for i,result in enumerate(pool.map(confirmation_job,jobs,chunksize=1),1):
            results.append(result)
            if i%32==0:print(json.dumps({'phase':'confirmation','jobs_completed':i,'total_jobs':len(jobs),'failed':sum(r['state']!='passed' for r in results)}),flush=True)
    rows=[iv.load_blob(path)['row'] for result in results for path in result.get('outputs',[])]
    pd.DataFrame(rows).to_parquet(run_dir/'fresh_intervention_outcomes.parquet',index=False)
    pd.DataFrame(rows).drop(columns=['prediction','initial_prediction'],errors='ignore').to_csv(run_dir/'fresh_case_confirmation.csv',index=False)
    atomic_json(run_dir/'interventions/confirmation/complete.json',{
        'state':'passed' if all(r['state']=='passed' for r in results) else 'incomplete','jobs':len(jobs),'rows':len(rows),
        'physical_arms':['A'],'analysis_arms':['A','S'],'failed_jobs':[r for r in results if r['state']!='passed']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--workers',type=int,default=32);parser.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1','cuda:2','cuda:3'])
    args=parser.parse_args();run(args.root,args.run_dir,args.workers,args.devices)
