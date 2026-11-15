"""Complete V1-confirmation reader probes, no new physical measurements.

CPU processes build features; exactly one process runs GPU inference. Restarted
runs retain completed trajectory shards and cached exact native states.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
import gzip
import itertools
import json
import math
import multiprocessing
import os
import pickle
import time

import numpy as np
import pandas as pd

from cgx.data.store import Store
from cgx.models.features import tensorize_history
from cgx.provenance import atomic_json, hash_json, read_json
from diagnostics.false_alarm_causes.core import inference_settings, load_reader
from .hooks import FrozenReaderHooks, sequence_metadata, prediction_error
from .reader_candidates import metrics
from .ledger import Ledger, TimedStage


def _read(path):
    with gzip.open(path,'rt') as handle:return json.load(handle)


def collect_items(v1):
    rows=[]
    for path in sorted((Path(v1)/'interventions/confirmation/cases').glob('*.json.gz')):
        item=_read(path)
        if item['row']['arm']=='A':rows.append((str(path),item['row']))
    keys={(r['case_id'],r['method_key'],r['policy_seed']) for _,r in rows}
    if len(rows)!=768 or len(keys)!=768 or len({r['case_id'] for _,r in rows})!=128:
        raise AssertionError('Expected complete 128 x 2 x 3 V1 actual trajectories')
    if {r['method_key'] for _,r in rows}!={'gnn_adaptive','gnn_depth2'} or {r['policy_seed'] for _,r in rows}!={11,23,37}:
        raise AssertionError('V1 method/seed support differs from sealed primary population')
    return rows


def _prepare_job(args):
    path,v1,run=args
    item=_read(path);row=item['row'];trace=item['trace'];store=Store(v1)
    packed=store.history_references(row['evidence_trace_hash'])
    history=store.load_history(row['evidence_trace_hash'])
    if len(history['observations'])!=len(trace)+1:
        raise AssertionError('Actual trace and observations disagree')
    acquisition_ids=[hash_json([row['case_id'],'initial',item['initial_observation_ref']])]
    acquisition_ids += [hash_json([row['case_id'],t['action'],t['execution_index'],t['burst_index'],t['observation_ref']]) for t in trace]
    execution_indices=[i+1 for i,t in enumerate(trace) if t['action']['ports'] and t['resources_before']['executions']>t['resources_after']['executions']]
    specs=[]
    for i in range(len(history['observations'])):
        h=dict(history,observations=history['observations'][:i+1], action_keys=history['action_keys'][:i],
               remaining=history['initial_budget'] if i==0 else trace[i-1]['resources_after'])
        specs.append((f'prefix_{i}',h,acquisition_ids[:i+1],
                      row['initial_prediction'] if i==0 else trace[i-1]['prediction']))
    for name,index in [('copy_first',execution_indices[0] if execution_indices else None),
                       ('copy_last',execution_indices[-1] if execution_indices else None)]:
        if index is not None:
            h=dict(history,observations=history['observations']+[history['observations'][index]])
            specs.append((name,h,acquisition_ids+[acquisition_ids[index]],None))
    ledger=Ledger(run);prepared=[]
    for name,h,provenance,stored in specs:
        ident=hash_json(dict(packed,observation_refs=[hash_json(o) for o in h['observations']],
                            remaining=h['remaining'],action_keys=h['action_keys'],provenance=provenance))
        ledger.reserve('feature_build',detail='reader_A_features')
        features=tensorize_history(h)
        metadata=sequence_metadata(h,provenance)
        # Raw records stay in immutable Store. Send only exact features and metadata to GPU.
        thin=dict(h,observations=[{} for _ in h['observations']])
        prepared.append(dict(name=name,history=thin,features=features,provenance=provenance,
                             identity=ident,prepared_metadata=metadata,stored=stored))
    job=dict(row=row,trace=trace,execution_indices=execution_indices,specs=prepared,
             initial_ports=sorted({r['port'] for r in history['observations'][0].get('records',[])}))
    out=Path(run)/'cache/reader_feature_jobs'/ (Path(path).name+'.pkl')
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix('.tmp')
    with tmp.open('wb') as handle:pickle.dump(job,handle,protocol=5)
    tmp.replace(out)
    return str(out)


def shapley(values):
    result={}
    for dim,label in enumerate('PCE'):
        total=0.
        others=[i for i in range(3) if i!=dim]
        for mask in itertools.product((0,1),repeat=2):
            key=[0,0,0]
            for i,v in zip(others,mask):key[i]=v
            before=''.join(map(str,key));key[dim]=1;after=''.join(map(str,key))
            size=sum(mask);weight=math.factorial(size)*math.factorial(2-size)/6
            total+=weight*(values[after]-values[before])
        result['shapley_'+label]=total
    if abs(sum(result.values())-(values['111']-values['000']))>1e-8:
        raise AssertionError('Shapley efficiency failed')
    return result


def _coverage(job,index,anchor):
    trace=job['trace'];seen=set(job['initial_ports'])
    for t in trace[:index-1]:seen.update(t['action']['ports'])
    ports=set(trace[index-1]['action']['ports'])
    current=trace[index-1]['action']
    previous_pairs={tuple(t['action']['calibration_pair']) for t in trace[:index-1] if t['action'].get('calibration_pair')}
    pair=current.get('calibration_pair')
    counts=Counter(port for t in trace[:index] for port in t['action']['ports'])
    sides={}
    for port,count in counts.items():
        owner,side=port.split(':',1);sides.setdefault(owner,{'in':0,'out':0})
        sides[owner]['in' if side.startswith('in') else 'out']+=count
    return dict(new_port_count=len(ports-seen),repeated_port_count=len(ports & seen),
                repeats_only=bool(ports) and not bool(ports-seen),
                new_calibration_pair=bool(pair and tuple(pair) not in previous_pairs),
                action_kind=current['kind'],ports=json.dumps(sorted(ports)),
                component_side_counts=json.dumps(sides,sort_keys=True),
                repeated_same_side_components=sum(max(s.values())>1 and min(s.values())==0 for s in sides.values()),
                balanced_io_components=sum(min(s.values())>0 for s in sides.values()))


def analyze_job(path,hooks,run):
    with open(path,'rb') as handle:job=pickle.load(handle)
    row=job['row'];meta={k:row[k] for k in ('case_id','configuration_id','method_key','policy_seed','true_target_key')}
    states={};checks=[]
    for spec in job['specs']:
        name=spec.pop('name');stored=spec.pop('stored')
        states[name]=hooks.prepare(**spec)
        if stored:checks.append(dict(history=name,**prediction_error(states[name].prediction,stored)))
    finalindex=len(job['trace']);final=states[f'prefix_{finalindex}'];executions=job['execution_indices']
    baseline=metrics(final.prediction,row['true_target_key'])
    population=dict(meta,**baseline,actual_prefixes=finalindex+1,additional_executions=len(executions),
                    termination_reason=row['termination_reason'],has_execution_anchor=bool(executions))
    pairs=[]
    if executions:
        anchor=executions[0];first=states[f'prefix_{anchor}']
        firstmetrics=metrics(first.prediction,row['true_target_key'])
        actual_metrics={i:metrics(states[f'prefix_{i}'].prediction,row['true_target_key']) for i in range(finalindex+1)}
        wrong_steps=[i for i in range(anchor+1,finalindex+1) if not actual_metrics[i]['correct']]
        first_wrong=wrong_steps[0] if wrong_steps else None
        recovered=bool(first_wrong is not None and any(actual_metrics[i]['correct'] for i in range(first_wrong+1,finalindex+1)))
        population.update(first_predicted_target=firstmetrics['predicted_target_key'],
                          first_correct=firstmetrics['correct'],first_false_alarm=firstmetrics['false_alarm'],
                          first_to_final_wrong=int(firstmetrics['correct'] and not baseline['correct']),
                          first_wrong_after_anchor=first_wrong,later_recovered=recovered,
                          transient_mistake=bool(wrong_steps and baseline['correct']),
                          first_wrong_final_correct=int(not firstmetrics['correct'] and baseline['correct']),
                          initially_wrong_still_wrong=int(not firstmetrics['correct'] and not baseline['correct']))
        for name,index in [('copy_first',executions[0]),('copy_last',executions[-1])]:
            pairs.append((name,final,states[name],dict(prefix_index=finalindex,copy_acquisition_index=index,
                         copied_block_has_calibration=bool(job['trace'][index-1]['action'].get('calibration_pair')),
                         is_final=True,first_correct=firstmetrics['correct'],
                         initially_correct_healthy=bool(firstmetrics['correct'] and row['true_target_key']=='healthy'))))
        for index in range(anchor+1,finalindex+1):
            pairs.append(('real_prefix',first,states[f'prefix_{index}'],dict(prefix_index=index,
                         anchor_index=anchor,is_final=index==finalindex,first_correct=firstmetrics['correct'],
                         initially_correct_healthy=bool(firstmetrics['correct'] and row['true_target_key']=='healthy'),
                         first_wrong_after_anchor=first_wrong,later_recovered=recovered,
                         mistake_persists_to_endpoint=bool(not actual_metrics[index]['correct'] and not baseline['correct']),
                         **_coverage(job,index,anchor))))
    else:
        population['status']='no additional execution; retained endpoint; copy/anchor unavailable'
    rows=[];audits=[];attributions=[]
    for kind,base,changed,extra in pairs:
        preds,audit=hooks.factorial(base,changed)
        pairid=hash_json([meta,kind,extra['prefix_index']])
        common=dict(meta,comparison=kind,pair_id=pairid,**extra)
        fixed=metrics(base.prediction,row['true_target_key'])['max_fault_target']
        pairrows=[]
        for mask,pred in preds.items():
            record=dict(common,cell=mask,P=int(mask[0]),C=int(mask[1]),E=int(mask[2]),
                        **metrics(pred,row['true_target_key'],fixed),
                        target_keys=json.dumps(pred['target_keys']),
                        full_log_probabilities=json.dumps(pred['log_probabilities_base2']))
            pairrows.append(record)
        for outcome in ('fixed_fault_log_odds','max_fault_log_odds','p_healthy','true_target_nll_bits'):
            values={r['cell']:r[outcome] for r in pairrows}
            attributions.append(dict(common,outcome=outcome,total_change=values['111']-values['000'],**shapley(values)))
        audits.append(dict(common,**audit));rows.extend(pairrows)
    result=dict(population=population,rows=rows,audits=audits,attributions=attributions,stored_checks=checks)
    out=Path(run)/'reader/shards'/(hash_json([meta['case_id'],meta['method_key'],meta['policy_seed']])+'.json.gz')
    out.parent.mkdir(parents=True,exist_ok=True)
    temp=out.with_suffix('.tmp')
    with gzip.open(temp,'wt') as handle:json.dump(result,handle)
    temp.replace(out)
    Path(path).unlink() # exact native states remain cached; large temporary raw features are unnecessary
    return dict(identity=out.stem,pairs=len(pairs),states=len(states),population=population)


def aggregate(run):
    run=Path(run);parts=[_read(p) for p in sorted((run/'reader/shards').glob('*.json.gz'))]
    population=pd.DataFrame([p['population'] for p in parts]);population.to_csv(run/'reader_population.csv',index=False)
    frame=pd.DataFrame([r for p in parts for r in p['rows']]);frame.to_csv(run/'reader_factorial_per_case.csv',index=False)
    frame[frame.comparison=='real_prefix'].to_csv(run/'reader_real_prefix_effects.csv',index=False)
    attrib=pd.DataFrame([r for p in parts for r in p['attributions']]);attrib.to_csv(run/'reader_factorial_shapley.csv',index=False)
    identity=['case_id','method_key','policy_seed']
    endpoint=frame[(frame.comparison=='real_prefix') & frame.is_final].copy()
    endpoint['comparison']='real_endpoint';endpoint['actual_later_measurement_exists']=True
    present=set(map(tuple,endpoint[identity].drop_duplicates().values))
    zero_rows=[]
    for _,p in population[population.has_execution_anchor].iterrows():
        ident=tuple(p[k] for k in identity)
        if ident in present:continue
        base=frame[(frame.case_id==p.case_id)&(frame.method_key==p.method_key)&(frame.policy_seed==p.policy_seed)&
                   (frame.comparison=='copy_first')&(frame.cell=='000')].iloc[0].to_dict()
        for mask in itertools.product((0,1),repeat=3):
            zero_rows.append(dict(base,comparison='real_endpoint',cell=''.join(map(str,mask)),
                P=mask[0],C=mask[1],E=mask[2],actual_later_measurement_exists=False,
                status='first-execution endpoint is final; identical state, all effects exactly zero'))
    if zero_rows:endpoint=pd.concat([endpoint,pd.DataFrame(zero_rows)],ignore_index=True)
    assert len(endpoint)==int(population.has_execution_anchor.sum())*8
    endpoint.to_csv(run/'reader_real_endpoint_effects.csv',index=False)
    endpoint_attr=attrib[(attrib.comparison=='real_prefix') & attrib.is_final].copy()
    endpoint_attr['comparison']='real_endpoint';endpoint_attr['actual_later_measurement_exists']=True
    zeros=[]
    for _,r in endpoint[~endpoint.actual_later_measurement_exists].drop_duplicates(identity).iterrows():
        for outcome in ('fixed_fault_log_odds','max_fault_log_odds','p_healthy','true_target_nll_bits'):
            zeros.append({**{k:r[k] for k in identity+['configuration_id','true_target_key','initially_correct_healthy']},
                'comparison':'real_endpoint','outcome':outcome,'total_change':0.,'shapley_P':0.,'shapley_C':0.,
                'shapley_E':0.,'actual_later_measurement_exists':False})
    if zeros:endpoint_attr=pd.concat([endpoint_attr,pd.DataFrame(zeros)],ignore_index=True)
    endpoint_attr.to_csv(run/'reader_real_endpoint_shapley.csv',index=False)
    endpoint.groupby(['method_key','cell','is_healthy']).agg(
        trajectory_denominator=('case_id','size'),physical_cases=('case_id','nunique'),
        false_alarm=('false_alarm','sum'),missed_fault=('missed_fault','sum'),
        joint_fault_correct=('joint_fault_correct','sum'),p_healthy=('p_healthy','mean'),
        fixed_fault_log_odds=('fixed_fault_log_odds','mean')).reset_index().to_csv(run/'reader_real_endpoint_counts.csv',index=False)
    retained=[]
    for _,p in population[~population.has_execution_anchor].iterrows():
        for mask in itertools.product((0,1),repeat=3):
            retained.append(dict(p,cell=''.join(map(str,mask)),has_execution_anchor=False,
                                 status='earlier original STOP; original endpoint retained, no P/C/E contrast defined'))
    full_endpoint=pd.concat([endpoint.assign(has_execution_anchor=True),pd.DataFrame(retained)],ignore_index=True)
    full_endpoint.groupby(['method_key','cell','is_healthy']).agg(
        trajectory_denominator=('case_id','size'),physical_cases=('case_id','nunique'),
        defined_PCE_contrasts=('has_execution_anchor','sum'),false_alarm=('false_alarm','sum'),
        missed_fault=('missed_fault','sum'),joint_fault_correct=('joint_fault_correct','sum')).reset_index().to_csv(
        run/'reader_real_endpoint_full_population_counts.csv',index=False)
    configs=frame.groupby(['comparison','cell','method_key','configuration_id','is_healthy'],dropna=False).agg(
        pair_rows=('case_id','size'),physical_cases=('case_id','nunique'),false_alarm=('false_alarm','mean'),
        missed_fault=('missed_fault','mean'),joint_fault_correct=('joint_fault_correct','mean'),
        p_healthy=('p_healthy','mean'),fixed_fault_log_odds=('fixed_fault_log_odds','mean'),
        true_target_nll_bits=('true_target_nll_bits','mean')).reset_index()
    configs.to_csv(run/'reader_factorial_per_configuration.csv',index=False)
    configs.groupby(['comparison','cell','method_key','is_healthy']).mean(numeric_only=True).reset_index().to_csv(run/'reader_factorial_pair_summary.csv',index=False)
    outcomes=['false_alarm','missed_fault','joint_fault_correct','p_healthy','fixed_fault_log_odds','max_fault_log_odds','true_target_nll_bits']
    combined=pd.concat([frame,endpoint],ignore_index=True)
    primary=[]
    for label,part in [('all_eligible_trajectories',combined),('initially_correct_healthy',combined[combined.initially_correct_healthy])]:
        # Each case/seed trajectory has equal weight before seed and configuration averages.
        trajectory=part.groupby(['comparison','cell','method_key','configuration_id','case_id','policy_seed','is_healthy'])[outcomes].mean().reset_index()
        case=trajectory.groupby(['comparison','cell','method_key','configuration_id','case_id','is_healthy'])[outcomes].mean().reset_index()
        config=case.groupby(['comparison','cell','method_key','configuration_id','is_healthy'])[outcomes].mean().reset_index()
        result=config.groupby(['comparison','cell','method_key','is_healthy'])[outcomes].mean().reset_index()
        result['subgroup']=label;result['averaging']='prefixes within case/seed; seeds within case; cases within configuration; configurations equally'
        primary.append(result)
    pd.concat(primary,ignore_index=True).to_csv(run/'reader_factorial_summary.csv',index=False)
    frame.groupby(['comparison','cell','method_key','policy_seed','is_healthy']).agg(
        rows=('case_id','size'),false_alarm=('false_alarm','sum'),missed_fault=('missed_fault','sum'),
        joint_fault_correct=('joint_fault_correct','sum')).reset_index().to_csv(run/'reader_factorial_per_seed.csv',index=False)
    subgroups=[('all_eligible_pairs',frame),('initially_correct_healthy',frame[frame.initially_correct_healthy]),
               ('real_endpoint',endpoint),
               ('real_endpoint_initially_correct_healthy',endpoint[endpoint.initially_correct_healthy]),
               ('copy_without_calibration',frame[frame.copied_block_has_calibration==False]),
               ('copy_with_calibration',frame[frame.copied_block_has_calibration==True]),
               ('real_repeated_ports_only',frame[frame.repeats_only==True]),
               ('real_new_ports',frame[frame.new_port_count>0]),
               ('real_new_calibration_pair',frame[frame.new_calibration_pair==True])]
    subgroup_rows=[]
    for name,part in subgroups:
        for keys,partgroup in part.groupby(['comparison','cell','method_key','is_healthy']):
            subgroup_rows.append(dict(subgroup=name,comparison=keys[0],cell=keys[1],method_key=keys[2],is_healthy=keys[3],
                pair_rows=len(partgroup),physical_cases=partgroup.case_id.nunique(),false_alarm=int(partgroup.false_alarm.sum()),
                missed_fault=int(partgroup.missed_fault.sum()),joint_fault_correct=int(partgroup.joint_fault_correct.sum()),
                p_healthy=partgroup.p_healthy.mean(),fixed_fault_log_odds=partgroup.fixed_fault_log_odds.mean()))
    pd.DataFrame(subgroup_rows).to_csv(run/'reader_factorial_subgroup_summary.csv',index=False)
    checks=[r for p in parts for r in p['stored_checks']];audits=[r for p in parts for r in p['audits']]
    atomic_json(run/'reader_reconstruction_audit.json',dict(trajectories=len(parts),physical_cases=population.case_id.nunique(),
        no_execution_anchor=int((~population.has_execution_anchor).sum()),pairs=len(audits),cells=len(frame),
        stored_distributions_checked=len(checks),max_stored_probability_error=max(r['max_abs_probability_error'] for r in checks),
        max_noop_error=max(a['base_reconstruction']['max_abs_probability_error'] for a in audits),
        max_changed_error=max(a['changed_reconstruction']['max_abs_probability_error'] for a in audits),
        edge_presence_comparisons=sum(bool(a['edge_added'] or a['edge_removed']) for a in audits),
        all_unassigned_tensors_identical=all(a['unassigned_tensors_identical'] for a in audits)))
    with gzip.open(run/'reader_tensor_audits.json.gz','wt') as handle:json.dump(audits,handle)
    return dict(trajectories=len(parts),pairs=len(audits),cells=len(frame))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--v1',required=True)
    parser.add_argument('--run',required=True);parser.add_argument('--workers',type=int,default=24)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--profile',type=int,default=0)
    parser.add_argument('--aggregate-only',action='store_true');args=parser.parse_args()
    if args.aggregate_only:print(aggregate(args.run));return
    inference_settings();items=collect_items(args.v1);run=Path(args.run)
    pending=[]
    for path,row in items:
        shard=run/'reader/shards'/(hash_json([row['case_id'],row['method_key'],row['policy_seed']])+'.json.gz')
        if not shard.exists():pending.append((path,args.v1,args.run))
    if args.profile:pending=pending[:args.profile]
    hooks=FrozenReaderHooks(load_reader(args.root,'final',args.device),Ledger(run),run/'cache/reader_states')
    begin=time.monotonic();done=0
    with TimedStage(run,'reader_factorial_profile' if args.profile else 'reader_factorial'),ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        # Keep at most 2*workers large jobs on disk and in the work queue.
        iterator=iter(pending);queue=[]
        for _ in range(min(len(pending),2*args.workers)):
            queue.append(pool.submit(_prepare_job,next(iterator)))
        while queue:
            future=queue.pop(0);path=future.result();summary=analyze_job(path,hooks,run);done+=1
            print(json.dumps(dict(completed=done,total=len(pending),elapsed_seconds=time.monotonic()-begin,**{k:v for k,v in summary.items() if k!='population'})),flush=True)
            try:queue.append(pool.submit(_prepare_job,next(iterator)))
            except StopIteration:pass
    if not args.profile:print(aggregate(run),flush=True)


if __name__=='__main__':main()
