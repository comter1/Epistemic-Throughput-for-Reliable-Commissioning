"""Auditable conditional denominators and paired topology/case statistics.

Policy seeds are repeated measurements of a physical case, not independent
robots. Paired uncertainty averages seeds first, resamples configurations, then
cases within their original fault-family strata. All timeouts remain included.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cgx.provenance import atomic_json,file_sha256,read_json
from .replay import EXPECTED_MAIN_ROWS,prediction_metrics

METHODS={'gnn_fixed_plan':'GNN Plan-Once','gnn_adaptive':'GNN Replan-Each-Round',
    'gnn_depth2':'GNN 2-Step Planner','transformer_adaptive':'Transformer Replan-Each-Round',
    'cost_aware_adaptation':'GNN Cost-Aware One-Step','full_recording_feasible':'Full Recording'}
METRICS=('accuracy','healthy_false_alarm_rate','fault_miss_rate','fault_joint_accuracy','nll_bits','brier')
BOOTSTRAPS=2000
BOOTSTRAP_SEED=671917
ECE_BINS=10
TIMEOUT_STAGES=('selection','diagnostic','acquisition','infrastructure','unclassified')
TIMEOUT_EXCEPTIONS={'TimeoutError','CallDeadlineExceeded'}
CELL_KEY=['suite','method','policy_seed','execution_cap','case_id']
QUOTED={'gnn_fixed_plan':(65.25,68.38,71.63),'gnn_adaptive':(71.58,68.,62.56),
    'gnn_depth2':(72.75,71.54,65.48),'transformer_adaptive':(71.98,70.,66.10)}


def main_cell_keys(frame):
    """Full Recording has no policy seed; None/NaN serialize identically here."""
    return {(r.suite,r.method,None if pd.isna(r.policy_seed) else int(r.policy_seed),
             int(r.execution_cap),r.case_id) for r in frame[CELL_KEY].itertuples(index=False)}


def timeout_annotations(errors,timeout_flag=False):
    """Classify explicit error provenance, without inferring from bad accuracy.

    A case can have more than one stage, so stage counts need not sum to the
    global timeout flag. Warmup and fixed-plan deadlines count as selection or
    diagnosis, with their more specific prefix retained in the per-case record.
    Generic worker failures without a timeout exception are not relabeled.
    """
    if errors is None or (isinstance(errors,float) and np.isnan(errors)):errors=[]
    elif isinstance(errors,str):errors=[errors]
    stages=set();substages=set()
    mapping={'selection':'selection','selection_warmup':'selection','fixed_plan':'selection',
        'diagnostic':'diagnostic','diagnostic_warmup':'diagnostic',
        'acquisition':'acquisition','initial_execution':'acquisition',
        'worker':'infrastructure','infrastructure':'infrastructure'}
    for error in errors:
        pieces=str(error).split(':',2);prefix=pieces[0]
        explicit=prefix.endswith('_timeout')
        exception=prefix.endswith('_error') and len(pieces)>1 and pieces[1] in TIMEOUT_EXCEPTIONS
        if not (explicit or exception):continue
        substage=prefix.removesuffix('_timeout').removesuffix('_error')
        substages.add(substage);stages.add(mapping.get(substage,'unclassified'))
    from_errors=bool(stages)
    flag=bool(timeout_flag) if timeout_flag is not None and not pd.isna(timeout_flag) else False
    if flag and not stages:stages.add('unclassified')
    return {**{'timeout_'+s:int(s in stages) for s in TIMEOUT_STAGES},
        'timeout_any_from_errors':int(from_errors),
        'timeout_flag_without_error_stage':int(flag and not from_errors),
        'timeout_error_without_flag':int(from_errors and not flag),
        'timeout_substages':json.dumps(sorted(substages))}


def ece(confidence,correct,bins=ECE_BINS):
    confidence=np.asarray(confidence,dtype=float);correct=np.asarray(correct,dtype=float)
    if not len(confidence):return float('nan')
    index=np.minimum((confidence*bins).astype(int),bins-1);result=0.
    for b in range(bins):
        mask=index==b
        if mask.any():result+=mask.mean()*abs(confidence[mask].mean()-correct[mask].mean())
    return float(result)


def aggregate(part):
    healthy=part[part.is_healthy];faulty=part[~part.is_healthy]
    result=dict(rows=len(part),physical_cases=part.case_id.nunique(),configurations=part.configuration_id.nunique(),
        policy_seeds=part.policy_seed.nunique(),healthy_rows=len(healthy),fault_rows=len(faulty),
        healthy_physical_cases=healthy.case_id.nunique(),fault_physical_cases=faulty.case_id.nunique(),
        correct_count=int(part.correct.sum()),false_alarm_count=int(healthy.healthy_false_alarm.sum()),
        miss_count=int(faulty.fault_miss.sum()),fault_joint_correct_count=int(faulty.joint_fault_correct.sum()),
        accuracy=part.correct.mean(),healthy_false_alarm_rate=healthy.healthy_false_alarm.mean(),
        fault_miss_rate=faulty.fault_miss.mean(),fault_joint_accuracy=faulty.joint_fault_correct.mean(),
        nll_bits=part.nll_bits.mean(),brier=part.brier.mean(),ece=ece(part.confidence,part.correct),ece_bins=ECE_BINS)
    for column in ('executions_used','bytes_used','initial_bytes','calibrations_used','decisions_used','selection_seconds',
        'inference_seconds','simulation_seconds','recorder_seconds','end_to_end_seconds',
        'warmup_seconds','selection_warmup_seconds',
        'physical_cache_hits','new_robot_execution_attempts','physical_acquisition_measured_seconds',
        'new_calibration_attempts','actual_new_physical_seconds','logical_acquisition_reference_seconds','provenance_seconds',
        'structured_processing_seconds'):
        if column in part:
            result['mean_'+column]=part[column].mean()
            result[column+'_known_rows']=int(part[column].notna().sum())
            result[column+'_unknown_rows']=int(part[column].isna().sum())
            if column in ('physical_cache_hits','new_robot_execution_attempts','new_calibration_attempts','actual_new_physical_seconds'):
                result['sum_known_'+column]=part[column].sum(min_count=1)
    for column in ('timeout','abstained','original_timeout','original_abstained','worker_failure','resource_accounting_unavailable'):
        if column in part:result[column+'_count']=int(part[column].fillna(False).sum())
    stage_columns=['timeout_'+s for s in TIMEOUT_STAGES]+[
        'timeout_any_from_errors','timeout_flag_without_error_stage','timeout_error_without_flag']
    if 'errors' in part:
        annotated=part if all(c in part for c in stage_columns) else pd.DataFrame([
            timeout_annotations(row.get('errors'),row.get('timeout',False)) for row in part.to_dict('records')])
        for column in stage_columns:result[column+'_count']=int(annotated[column].sum())
    return result


def summary_table(frame,keys):
    return pd.DataFrame([dict(zip(keys,key if isinstance(key,tuple) else (key,)),**aggregate(part))
        for key,part in frame.groupby(keys,dropna=False,sort=True)])


def _metric_vectors(part):
    # Conditional rates are masked BEFORE averaging policy seeds or bootstraps.
    return part.assign(accuracy=part.correct.astype(float),
        healthy_false_alarm_rate=part.healthy_false_alarm.where(part.is_healthy),
        fault_miss_rate=part.fault_miss.where(~part.is_healthy),
        fault_joint_accuracy=part.joint_fault_correct.where(~part.is_healthy))


def paired_bootstrap(left,right,repetitions=BOOTSTRAPS,seed=BOOTSTRAP_SEED):
    """Return right-minus-left paired effects; family strata stay balanced."""
    keys=['configuration_id','case_id','fault_family']
    a=_metric_vectors(left).groupby(keys)[list(METRICS)].mean()
    b=_metric_vectors(right).groupby(keys)[list(METRICS)].mean()
    if not a.index.equals(b.index):raise ValueError('Paired contrast changed physical case/family support')
    delta=b-a;configurations=sorted(delta.index.get_level_values(0).unique());rng=np.random.default_rng(seed)
    choice=rng.integers(0,len(configurations),(repetitions,len(configurations)))
    draws=np.full((repetitions,len(configurations),len(METRICS)),np.nan)
    points=[]
    for ci,configuration in enumerate(configurations):
        group=delta.xs(configuration);points.append(group.mean().to_numpy())
        rr,slots=np.where(choice==ci);total=np.zeros((len(rr),len(METRICS)));count=np.zeros_like(total)
        for _,family in group.groupby(level='fault_family'):
            values=family.to_numpy();indices=rng.integers(0,len(values),(len(rr),len(values)))
            samples=values[indices];total+=np.nansum(samples,axis=1);count+=np.isfinite(samples).sum(axis=1)
        draws[rr,slots]=np.divide(total,count,out=np.full_like(total,np.nan),where=count>0)
    sample=np.nanmean(draws,axis=1);point=np.nanmean(points,axis=0)
    rows=[]
    for i,metric in enumerate(METRICS):
        values=sample[:,i]
        rows.append(dict(metric=metric,difference=float(point[i]),ci_low=float(np.quantile(values,.025)),
            ci_high=float(np.quantile(values,.975)),physical_cases=len(delta),configurations=len(configurations),
            bootstrap_replicates=repetitions,bootstrap_seed=seed,policy_seed_replicates_averaged=True,
            interval_interpretation='Descriptive paired 95% percentile CI; no multiplicity-adjusted significance claim'))
    return rows


def normalize_main(frame,version):
    rows=[]
    for row in frame.to_dict('records'):
        prediction=row.pop('prediction')
        row.update(prediction_metrics(prediction,row['true_target_key']))
        row.update(timeout_annotations(row.get('errors'),row.get('timeout',False)))
        row.update(version=version,history_hash=row.get('evidence_trace_hash'),
            fault_family='healthy' if row['true_target_key']=='healthy' else row['true_target_key'].split(':',1)[1])
        rows.append(row)
    result=pd.DataFrame(rows)
    if len(result)!=EXPECTED_MAIN_ROWS or result.case_id.nunique()!=3200 or result.duplicated(CELL_KEY).any():
        raise ValueError('Full main evaluation matrix is incomplete or duplicated')
    return result


def _contrast(left,right,metadata):
    return [dict(metadata,**r) for r in paired_bootstrap(left,right)]


def timeout_stage_table(frame,keys):
    """Every stage uses all planned rows in the cell as its denominator."""
    rows=[]
    for key,part in frame.groupby(keys,dropna=False,sort=True):
        metadata=dict(zip(keys,key if isinstance(key,tuple) else (key,)))
        for stage in TIMEOUT_STAGES:
            selected=part[part['timeout_'+stage].astype(bool)]
            rows.append(dict(metadata,timeout_stage=stage,planned_rows=len(part),
                physical_cases=part.case_id.nunique(),timeout_stage_rows=len(selected),
                timeout_stage_physical_cases=selected.case_id.nunique(),
                timeout_stage_fraction=len(selected)/len(part),
                correct_among_stage=int(selected.correct.sum()),
                incorrect_among_stage=int((1-selected.correct).sum()),
                mean_nll_among_stage=selected.nll_bits.mean(),
                stage_counts_can_overlap=True,classification_denominator_retained=True))
    return pd.DataFrame(rows)


def error_channel_interpretation(run_dir,cohorts=('old_histories','new_main')):
    """Describe observed model channels; never turn a footprint into blame."""
    run=Path(run_dir);lines=['## Available modeled evidence in wrong diagnoses']
    categories=['structural_absence_of_true_fault_channel','process_variance_only_channel_present_but_wrong',
        'direct_modeled_channel_present_but_wrong','healthy_false_alarm','unreplayable']
    labels={'old_histories':'Old acquired histories with the structured reader','new_main':'New main pipeline'}
    for cohort in cohorts:
        label=labels[cohort]
        directory=run/'error_audit'/cohort;marker=read_json(directory/'complete.json')
        if marker['state']!='passed' or not marker['correct_and_wrong_full_denominators_preserved'] or not marker['structural_categories_not_causal_error_labels']:
            raise ValueError('Required descriptive model-channel audit incomplete')
        path=directory/'primary_b4_summary.csv'
        if file_sha256(path)!=marker['output_hashes'][path.name]:raise ValueError('Error audit summary changed')
        table=pd.read_csv(path);table=table[table.fault_family=='all']
        totals=pd.read_csv(directory/'summary.csv')
        scope='frozen-reader reread' if cohort=='old_histories' else 'official new-main prediction'
        lines.extend([f"**{label}:** {marker['wrong_contexts']}/{marker['all_contexts']} planned cells have a wrong {scope}. The audit retains every wrong context and uses all correct and wrong cells as the denominator.",
            'Across all suites/budgets, wrong-context categories are '+', '.join(
                f"{category}={int(totals[category+'_rows'].sum())}" for category in categories)+'.',
            '| Primary-suite B4 policy | All N | Wrong | No modeled true-fault channel | Variance-only channel | Direct channel present | Healthy false alarm | Unreplayable |',
            '|---|---:|---:|---:|---:|---:|---:|---:|'])
        for row in table.itertuples():
            counts=[int(getattr(row,category+'_rows')) for category in categories]
            if sum(counts)!=int(row.all_wrong_rows):raise ValueError('Error footprint categories changed wrong denominator')
            lines.append(f"| {METHODS[row.method]} | {int(row.all_rows)} | {int(row.all_wrong_rows)} | "+' | '.join(map(str,counts))+' |')
        details=pd.read_parquet(directory/'per_error_context.parquet')
        direct=details[(details.footprint_category=='direct_modeled_channel_present_but_wrong')&
            details.fault_family.isin(['cache_stale','queue_hold'])]
        if len(direct):
            known=direct.true_fault_within_clock_mixed_groups.notna()
            absent=int((direct.loc[known,'true_fault_within_clock_mixed_groups']==0).sum())
            median=direct.true_vs_healthy_log_likelihood_bits.median()
            lines.append(f'Among {len(direct)} timing-fault error contexts with a direct modeled channel, {absent}/{int(known.sum())} assessed contexts have no within-clock mixed packet×clock stratum. Their median true-target versus healthy log-likelihood ratio is {median:+.5f} bits. This describes the available cross-clock evidence and the fixed model’s scores; it does not prove that clock uncertainty caused those errors. Counter definitions are recorded in `error_audit/COUNTER_SEMANTICS.json`.')
        if cohort=='new_main':
            corrected=int(totals.untimed_correct_wrong_rows.sum())
            if corrected!=marker['official_wrong_contexts_correct_if_reread']:
                raise ValueError('Untimed reread correction denominator changed')
            lines.append(f'{corrected} officially wrong new-main contexts become correct when the same saved final evidence is read without the diagnostic deadline. Footprint categories index official error contexts, which need not be errors by the uncensored frozen likelihood model. Other runtime failures and abandoned actions are not repaired by rereading.')
    lines.append('“No modeled true-fault channel” is not “no information in the raw records.” This SED ignores external motion as fault evidence, and another model could extract useful information from unpaired adapter records or other signals. The old policies were trained with another reader. Conversely, a present mean/gain or variance channel can be weak, noisy, clock-confounded or misspecified; presence does not prove identifiability. These post hoc categories quantify model-channel availability and cannot uniquely assign an error to diagnosis versus acquisition selection.')
    return lines


def selected_validation_interpretation(run_dir):
    """Keep completed SOURCE checkpoint checks separate from early snapshots."""
    path=Path(run_dir)/'reports/validation_timeout_audit_refreshed.json'
    if not path.exists():return []
    audit=read_json(path);completed=audit['completed_models'];selected=audit['final_selected_by_method_seed']
    if audit['state']!='passed' or not audit['source_validation'] or len(completed)!=15 or audit['models_still_training']!=0:
        raise ValueError('Refresh the SOURCE validation audit after all 15 models complete before final reporting')
    completion_keys={(r['method'],int(r['seed'])):int(r['selected_iteration']) for r in completed}
    selected_keys={(r['method'],int(r['seed'])):int(r['selected_iteration']) for r in selected}
    if len(selected)!=15 or len(completion_keys)!=15 or selected_keys!=completion_keys:
        raise ValueError('Selected SOURCE validation checks do not match all completed checkpoints')
    totals=audit['final_selected_checkpoint_totals']
    if any(sum(int(r[key]) for r in selected)!=int(totals[key]) for key in ('rows','timeouts','abstentions')):
        raise ValueError('Selected SOURCE validation aggregate changed its denominator')
    frozen=read_json(Path(run_dir)/'models/frozen.json');inventory=frozen['inventory']
    if frozen['state']!='passed' or not frozen['validation_only']:
        raise ValueError('Final reporting requires the completed SOURCE-selected model freeze')
    if set(inventory)!={f'{method}_seed{seed}' for method,seed in completion_keys}:
        raise ValueError('Frozen inventory changed the completed model set')
    for (method,seed),iteration in completion_keys.items():
        model=inventory[f'{method}_seed{seed}']
        if int(model['selected_iteration'])!=iteration or int(model['actual_updates'])<iteration:
            raise ValueError('SOURCE validation checkpoint or training length disagrees with the frozen inventory')
    betas=frozen['cost_aware_beta']
    if set(betas)!={str(seed) for _,seed in completion_keys} or len(betas)!=3:
        raise ValueError('All three frozen Cost-Aware beta values are required')
    if not all(np.isfinite(float(beta)) and 0<=float(beta)<=1 for beta in betas.values()):
        raise ValueError('Frozen Cost-Aware beta lies outside the original interpolation range')
    labels={**METHODS,'gnn_myopic':'GNN One-Step (Cost-Aware backbone)'}
    lines=['## SOURCE validation at the final selected checkpoints',
        f"The refreshed SOURCE audit ({audit['created_at']}) includes all {len(completed)} completed models and zero models still training. Their final selected checkpoint checks contain {int(totals['timeouts'])}/{int(totals['rows'])} recorded timeouts across {int(totals['physical_cases'])} shared validation cases. These are checkpoint-selection validation runs, not held-out main outcomes; the repeated model/seed checks are not independent physical cases. The earlier first-check snapshot above remains a distinct intermediate record.",
        '| Training model | Seed | Completed updates (final iteration) | Selected iteration | SOURCE timeout/N | Error stages |',
        '|---|---:|---:|---:|---:|---|']
    for row in sorted(selected,key=lambda r:(r['method'],r['seed'])):
        stages=', '.join(f'{key}={value}' for key,value in sorted(row['timeout_stages'].items())) or 'none'
        updates=inventory[f"{row['method']}_seed{row['seed']}"]['actual_updates']
        lines.append(f"| {labels[row['method']]} | {row['seed']} | {updates} | {row['selected_iteration']} | {row['timeouts']}/{row['rows']} | {stages} |")
    lines.append('Selected checkpoints follow the frozen source-validation selection rule. The GNN One-Step backbone rows use the original cost-aware checkpoint-validation setting at β=0.5; the later β-grid search and final β-tuned main policy are separate evaluations. This table does not establish a causal effect of training, concurrency or model structure on latency, and does not replace the official main timeout-stage table. `validation_timeout_audit_refreshed.json` retains all intermediate checks, selected-checkpoint labels and input hashes; stage counts can overlap within a run.')
    lines.extend(['Completed updates (the final iteration) and selected iterations above are read from `models/frozen.json`; training can continue beyond the checkpoint ultimately selected. Allocation-only resumption preserved completed iterations; an uninstrumented amount of unsaved work may have been discarded or replayed, so this is not a literal count of all optimizer calls. `reports/uniform16_transition.json` records the allocation transition, and `reports/policy_execution_resources.json` records cumulative attempt wall time separately. The same frozen inventory records the SOURCE-selected Cost-Aware interpolation values:',
        '| Cost-Aware policy seed | Frozen β |','|---|---:|'])
    for seed,beta in sorted(betas.items(),key=lambda pair:int(pair[0])):
        lines.append(f'| {seed} | {float(beta):g} |')
    lines.append('The unchanged RuntimePolicy score is `−[β × norm(predicted_next_loss) + (1−β) × norm(resource_cost)]`, with min–max normalization over feasible actions (`reference_source/src/cgx/baselines/policies.py`). STOP uses current posterior entropy for its loss term. At β=1 the explicit normalized-cost term has zero weight, while hard execution, byte and calibration budgets still constrain actions; at β=0 the score uses normalized resource cost alone. These boundary values describe the selected policy behavior without changing its original formula or training.')
    return lines


def replay_reports(run_dir):
    run=Path(run_dir);out=run/'reports';out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_parquet(run/'replay/per_case_predictions.parquet')
    main=frame[frame.cohort=='main'];diagnostic=frame[frame.cohort!='main']
    keys=['cohort','suite','reader','method','execution_cap']
    summary_table(main,keys).to_csv(out/'reader_replay_summary.csv',index=False)
    summary_table(main,keys+['fault_family']).to_csv(out/'reader_replay_family_summary.csv',index=False)
    summary_table(main,keys+['policy_seed']).to_csv(out/'reader_replay_seed_summary.csv',index=False)
    summary_table(diagnostic[diagnostic.is_terminal],keys).to_csv(out/'problematic_terminal_summary.csv',index=False)
    summary_table(diagnostic,keys+['prefix_ordinal']).to_csv(out/'problematic_prefix_summary.csv',index=False)
    contrasts=[]
    for (suite,method,cap),part in main.groupby(['suite','method','execution_cap']):
        for baseline in ('original_as_deployed','original_GNN_reread','R-C'):
            contrasts.extend(_contrast(part[part.reader==baseline],part[part.reader=='structured'],
                dict(contrast='reader_only',suite=suite,method=method,execution_cap=cap,
                    left=baseline,right='structured')))
    pd.DataFrame(contrasts).to_csv(out/'reader_replay_paired_contrasts.csv',index=False)
    # Continuation harm compares the exact recorded prefix after the first real
    # execution with the terminal prefix. Calibrations before it do not count.
    pairs=[];ineligible=[]
    for (cohort,path,reader),part in diagnostic.groupby(['cohort','source_trajectory','reader']):
        final=part[part.is_terminal].iloc[0];first=part[part.prefix_ordinal==part.first_execution_ordinal]
        if len(first)==0 and pd.isna(final.first_execution_ordinal) and final.executions_used==0:
            ineligible.append(dict(cohort=cohort,source_trajectory=path,reader=reader,case_id=final.case_id,
                method=final.method,policy_seed=final.policy_seed,fault_family=final.fault_family,
                reason='Recorded policy stopped without an additional physical execution',
                terminal_correct=int(final.correct),terminal_nll_bits=float(final.nll_bits)))
            continue
        if len(first)!=1:raise ValueError('Acquired first real execution prefix missing or duplicated')
        first=first.iloc[0]
        pairs.append(dict(cohort=cohort,source_trajectory=path,reader=reader,case_id=final.case_id,
            configuration_id=final.configuration_id,fault_family=final.fault_family,
            method=final.method,policy_seed=final.policy_seed,is_healthy=final.is_healthy,
            first_history_hash=first.history_hash,terminal_history_hash=final.history_hash,
            first_p_healthy=first.p_healthy,terminal_p_healthy=final.p_healthy,
            first_correct=int(first.correct),terminal_correct=int(final.correct),
            first_false_alarm=int(first.healthy_false_alarm),terminal_false_alarm=int(final.healthy_false_alarm),
            correct_to_wrong=int(first.correct and not final.correct),wrong_to_correct=int(not first.correct and final.correct),
            nll_change_bits=float(final.nll_bits-first.nll_bits)))
    pairs=pd.DataFrame(pairs);pairs.to_csv(out/'continuation_pairs.csv',index=False)
    ineligible=pd.DataFrame(ineligible);ineligible.to_csv(out/'continuation_ineligible.csv',index=False)
    rows=[]
    for key,part in pairs.groupby(['cohort','reader','method','fault_family']):
        excluded=0 if ineligible.empty else len(ineligible[(ineligible.cohort==key[0])&
            (ineligible.reader==key[1])&(ineligible.method==key[2])&(ineligible.fault_family==key[3])])
        rows.append(dict(zip(['cohort','reader','method','fault_family'],key),rows=len(part),
            no_additional_execution_trajectories=excluded,all_terminal_trajectories=len(part)+excluded,
            physical_cases=part.case_id.nunique(),first_correct_count=int(part.first_correct.sum()),
            terminal_correct_count=int(part.terminal_correct.sum()),first_false_alarm_count=int(part.first_false_alarm.sum()),
            terminal_false_alarm_count=int(part.terminal_false_alarm.sum()),
            correct_to_wrong_count=int(part.correct_to_wrong.sum()),wrong_to_correct_count=int(part.wrong_to_correct.sum()),
            mean_nll_change_bits=part.nll_change_bits.mean()))
    pd.DataFrame(rows).to_csv(out/'continuation_summary.csv',index=False)
    differences=pd.read_parquet(run/'replay/original_reread_differences.parquet')
    numerical=[]
    for key,part in differences.groupby(['cohort','original_timeout','original_abstained']):
        numerical.append(dict(zip(['cohort','original_timeout','original_abstained'],key),rows=len(part),
            argmax_changed_count=int(part.argmax_changed.sum()),max_probability_error=float(part.probability_error.max()),
            mean_probability_error=float(part.probability_error.mean())))
    pd.DataFrame(numerical).to_csv(out/'original_deployed_vs_native_numerics.csv',index=False)
    result=dict(state='passed',reader_replay_rows=len(frame),contrasts=len(contrasts),continuation_pairs=len(pairs),
        source_sha256=file_sha256(run/'replay/per_case_predictions.parquet'))
    atomic_json(out/'reader_replay_complete.json',result);return result


def replay_policy_contrasts(run_dir):
    """Old acquisition policies compared under each common frozen reader."""
    run=Path(run_dir);frame=pd.read_parquet(run/'replay/per_case_predictions.parquet')
    main=frame[frame.cohort=='main'];contrasts=[]
    for (suite,reader,cap),part in main.groupby(['suite','reader','execution_cap']):
        contrasts.extend(_contrast(part[part.method=='gnn_fixed_plan'],part[part.method=='gnn_adaptive'],
            dict(contrast='old_actions_replan_minus_plan_once',suite=suite,reader=reader,execution_cap=cap,
                left='gnn_fixed_plan',right='gnn_adaptive')))
    for (suite,reader,method),part in main[main.method!='full_recording_feasible'].groupby(['suite','reader','method']):
        contrasts.extend(_contrast(part[part.execution_cap==1],part[part.execution_cap==4],
            dict(contrast='old_actions_B4_minus_B1',suite=suite,reader=reader,method=method,
                left='B1',right='B4',execution_cap=4)))
    pd.DataFrame(contrasts).to_csv(run/'reports/reader_old_policy_contrasts.csv',index=False)
    return dict(state='passed',contrasts=len(contrasts),selection_policies_unchanged=True)


def main_reports(source_root,run_dir):
    run=Path(run_dir);out=run/'reports';out.mkdir(parents=True,exist_ok=True)
    complete=read_json(run/'evaluation/main/complete.json')
    if complete['state']!='passed' or complete['rows']!=EXPECTED_MAIN_ROWS:raise ValueError('Full new main experiment required')
    old=normalize_main(pd.read_parquet(Path(source_root)/'results/per_case_results.parquet'),'original')
    new=normalize_main(pd.read_parquet(run/'evaluation/main/per_case_results.parquet'),'structured')
    if main_cell_keys(old)!=main_cell_keys(new):
        raise ValueError('New main changed the original method/seed/budget/case matrix')
    frame=pd.concat([old,new],ignore_index=True);frame.to_parquet(out/'main_per_case_predictions.parquet',index=False)
    frame.drop(columns=['full_log_probabilities','target_keys'],errors='ignore').to_csv(out/'main_per_case_metrics.csv',index=False)
    keys=['version','suite','method','execution_cap'];summary=summary_table(frame,keys)
    summary.to_csv(out/'main_summary.csv',index=False)
    summary_table(frame,keys+['fault_family']).to_csv(out/'main_family_summary.csv',index=False)
    summary_table(frame,keys+['policy_seed']).to_csv(out/'main_seed_summary.csv',index=False)
    timeout_stage_table(frame,keys).to_csv(out/'main_timeout_stages.csv',index=False)
    timeout_stage_table(frame,keys+['policy_seed']).to_csv(out/'main_timeout_stages_by_seed.csv',index=False)
    quoted=[]
    for method,values in QUOTED.items():
        for cap,quote in zip((1,2,4),values):
            match=summary[(summary.version=='original')&(summary.suite=='test_connections')&
                (summary.method==method)&(summary.execution_cap==cap)].iloc[0]
            if match.rows!=4800 or abs(match.accuracy*100-quote)>.006:
                raise ValueError('Original quoted accuracy or denominator no longer matches')
            newer=summary[(summary.version=='structured')&(summary.suite=='test_connections')&
                (summary.method==method)&(summary.execution_cap==cap)].iloc[0]
            quoted.append(dict(method=method,public_method=METHODS[method],execution_cap=cap,
                user_quoted_original_accuracy_percent=quote,original_exact_accuracy_percent=match.accuracy*100,
                original_correct_count=int(match.correct_count),original_denominator=int(match.rows),
                new_accuracy_percent=newer.accuracy*100,new_correct_count=int(newer.correct_count),
                new_denominator=int(newer.rows),physical_cases=int(match.physical_cases),
                new_minus_old_percentage_points=(newer.accuracy-match.accuracy)*100))
    pd.DataFrame(quoted).to_csv(out/'critical_original_new_accuracy.csv',index=False)
    contrasts=[]
    for (suite,method,cap),part in frame.groupby(['suite','method','execution_cap']):
        contrasts.extend(_contrast(part[part.version=='original'],part[part.version=='structured'],
            dict(contrast='new_minus_old',suite=suite,method=method,execution_cap=cap,left='original',right='structured')))
    for (version,suite,method),part in frame[frame.method!='full_recording_feasible'].groupby(['version','suite','method']):
        for cap in (2,4):
            contrasts.extend(_contrast(part[part.execution_cap==1],part[part.execution_cap==cap],
                dict(contrast='budget_change',version=version,suite=suite,method=method,execution_cap=cap,left='B1',right=f'B{cap}')))
    for (version,suite,cap),part in frame.groupby(['version','suite','execution_cap']):
        a=part[part.method=='gnn_fixed_plan'];b=part[part.method=='gnn_adaptive']
        contrasts.extend(_contrast(a,b,dict(contrast='replan_minus_plan_once',version=version,suite=suite,
            method='gnn_adaptive_vs_gnn_fixed_plan',execution_cap=cap,left='gnn_fixed_plan',right='gnn_adaptive')))
    pd.DataFrame(contrasts).to_csv(out/'main_paired_contrasts.csv',index=False)
    replay=pd.read_parquet(run/'replay/per_case_predictions.parquet')
    replay=replay[(replay.cohort=='main')&(replay.reader=='structured')]
    decomposition=[]
    for (suite,method,cap),part in replay.groupby(['suite','method','execution_cap']):
        oldpart=old[(old.suite==suite)&(old.method==method)&(old.execution_cap==cap)]
        newpart=new[(new.suite==suite)&(new.method==method)&(new.execution_cap==cap)]
        for label,left,right in [('reader_mapping_at_fixed_old_histories',oldpart,part),
            ('policy_retraining_acquisition_and_runtime',part,newpart)]:
            decomposition.extend(_contrast(left,right,dict(suite=suite,method=method,
                execution_cap=cap,component=label,left='old_histories',right='structured_reader_or_new_pipeline')))
    pd.DataFrame(decomposition).to_csv(out/'diagnosis_acquisition_decomposition.csv',index=False)
    result=dict(state='passed',original_rows=len(old),new_rows=len(new),physical_cases=3200,
        old_and_new_keys_identical=True,conditional_denominators_explicit=True,confidence_bins=ECE_BINS,
        uncertainty='2000 paired hierarchical bootstrap replicates; policy seeds averaged per physical case; configurations then fault-stratified cases',
        original_sha256=file_sha256(Path(source_root)/'results/per_case_results.parquet'),
        new_sha256=file_sha256(run/'evaluation/main/per_case_results.parquet'))
    atomic_json(out/'main_complete.json',result);return result


def interpret(run_dir):
    """Generate observations from completed tables, without a success premise."""
    run=Path(run_dir);out=run/'reports';summary=pd.read_csv(out/'main_summary.csv')
    audit=read_json(run/'runtime_audit/complete.json')
    if audit['state']!='passed' or not audit['full_matrix_denominator_unchanged'] or not audit['official_results_unchanged']:
        raise ValueError('Predeclared post-main reader deadline audit is incomplete')
    contrasts=pd.read_csv(out/'main_paired_contrasts.csv');inv=read_json(run/'invariance/complete.json')
    continuation=pd.read_csv(out/'continuation_summary.csv');table=pd.read_csv(out/'critical_original_new_accuracy.csv')
    shared=pd.read_csv(run/'invariance/shared_clock.csv');control=pd.read_csv(run/'invariance/transformations.csv')
    fit=read_json(run/'models/readers/final/complete.json');fit_audit=read_json(run/'tests/reader_fit_audit.json')
    scores=read_json(run/'training/scores_complete.json');support=read_json(run/'tests/selector_adapters.json')
    if scores['state']!='passed' or scores['old_reward_labels_read'] or scores['transitions']!=support['source_transitions']:
        raise ValueError('New diagnostic rewards must cover the unchanged source transitions')
    new_execution=pd.read_parquet(run/'invariance/new_execution_pairs.parquet')
    new_execution=new_execution[new_execution.reader=='structured']
    case_seed=new_execution.groupby(['cohort','configuration_id','case_id','method','policy_seed']).nll_change_bits.mean()
    new_execution_mean=case_seed.groupby(['cohort','configuration_id','case_id']).mean().groupby(
        ['cohort','configuration_id']).mean().mean()
    def cell(version,suite,method,cap):
        return summary[(summary.version==version)&(summary.suite==suite)&(summary.method==method)&(summary.execution_cap==cap)].iloc[0]
    def rate(r,field,numerator,denominator):return f"{r[field]*100:.2f}% ({int(r[numerator])}/{int(r[denominator])})"
    lines=['# Structured Evidence Diagnoser: completed Panda benchmark',
        'All original main rows are retained, and the new matrix uses the same 3,200 physical cases, three test suites, policy seeds and resource caps. The 147,200 rows per experiment include 4800 runs per learned-method/budget primary-suite cell (1600 physical cases × three policy seeds). Healthy and faulty conditional denominators there are 1200 and 3600, respectively.',
        'The reader-only study uses the exact saved old evidence. Original deployed deadline/fallback outputs and untimed native GNN rereads are separate labels. R-C changes only the previously selected count features. The new-policy experiment changes both the reader and the newly trained policy, so its total gain cannot by itself identify an acquisition mechanism.',
        'Confidence intervals are descriptive paired 95% hierarchical bootstrap intervals (2000 replicates, seeds averaged within case, configurations then fault-family-stratified cases). No unadjusted significance claim is made.',
        '## Model scope and training support',
        f"The final likelihood fit uses {fit['training_cases']} training cases across {len(fit['training_configurations'])} training configurations. Three topology-cross-fit models exclude their assigned configurations; the exclusion audit is {fit_audit['crossfit_exclusion_verified']}. Gaussian-mixture component counts are fixed before evaluation, and distribution parameters are learned from source observations without true offsets or injected numeric severities.",
        f"All {scores['transitions']:,} source transitions were rescored using the structured topology-exclusion readers ({scores['unique_predictions']:,} case/history predictions); no old diagnostic-reward labels were read. Source behavior support is unchanged, including {support['stop_transitions']}/{support['source_transitions']} STOP transitions ({100*support['stop_transitions']/support['source_transitions']:.4f}%). This experiment does not rebalance STOP examples. The OOF and deployed readers share the structured model family but have separately fitted source parameters; it is not a final-reader-only reward experiment. All {support['matched_original_initializations']} policy initializations match the original controls, with architectures, optimization and stopping rules retained.",
        '; '.join(f"{family}: {counts['cases_with_local_pair']}/{counts['total_family_cases']} source family cases have an acquired local input/output pair" for family,counts in sorted(fit['support'].items()))+'. These are fitting-support counts, not held-out successes.',
        'External motion observations currently have a fault-independent likelihood and cancel in fault normalization; an external-only initial history therefore returns the graph prior. Missing and timeout indicators do not distinguish faults in this model. Timing residuals are conditionally independent after the shared packet epoch, clock offsets and fault delay; additional downstream age correlations are not modeled. A passed identity/covariance test does not remove these modeling limitations.',
        'Reader-only results re-read saved evidence offline without the old call deadline; the original deployed outputs are retained separately. The new main pipeline applies the preserved live deadlines and includes failed-call prior fallbacks. The acquisition/retraining component below therefore also includes deployment runtime effects; it is not a pure action-quality estimate.',
        'Wall times were measured on shared nodes, with co-tenancy, CPU quotas, concurrency and allocation changes recorded in `reports/main_resources_node*.json` and `reports/main_allocation_epochs.json`. This is not an exclusive-hardware benchmark: contention and throttling can affect both wall time and the preserved two-second selection/diagnosis deadlines. Every timeout row remains in the official results and classification denominators.',
        '## Critical primary-suite comparison',
        '| Method | B | Original quoted % | Original correct/N | Structured % | Structured correct/N |',
        '|---|---:|---:|---:|---:|---:|']
    reader_summary=pd.read_csv(out/'reader_replay_summary.csv')
    reader_summary=reader_summary[(reader_summary.suite=='test_connections')&(reader_summary.execution_cap==4)]
    reader_section=['## Reader-only comparison on exact old acquired evidence',
        'The acquisition policies and every stored record are fixed in this comparison. Values below are primary-suite B4 results; the main comparison after retraining follows separately. Original deployed and native untimed GNN outputs both remain in the audit tables.',
        '| Old acquisition policy | N | Original deployed correct | Native GNN correct | R-C correct | Structured correct | Original → structured healthy false alarms | Original → structured fault misses |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        part=reader_summary[reader_summary.method==method]
        old=part[part.reader=='original_as_deployed'].iloc[0];native=part[part.reader=='original_GNN_reread'].iloc[0]
        corrected=part[part.reader=='R-C'].iloc[0];structured=part[part.reader=='structured'].iloc[0]
        reader_section.append(f"| {METHODS[method]} | {int(old.rows)} | {int(old.correct_count)} | {int(native.correct_count)} | {int(corrected.correct_count)} | {int(structured.correct_count)} | {int(old.false_alarm_count)} → {int(structured.false_alarm_count)} / {int(old.healthy_rows)} | {int(old.miss_count)} → {int(structured.miss_count)} / {int(old.fault_rows)} |")
    reader_section.append('The structured likelihood changes more than acquisition-count inputs, so its full improvement cannot be assigned to count removal alone. R-C is the narrower count-feature intervention. The old policies were trained with the original diagnostic model, and their STOP/acquisition decisions can be poorly matched to the structured reader. In particular, external-only evidence is uninformative to this new reader. All such stopped cases remain in the denominator. `reader_old_policy_contrasts.csv` reports paired budget and Plan/Replan comparisons with each reader held fixed; a confidence interval spanning zero does not establish equivalence.')
    position=lines.index('## Critical primary-suite comparison')
    lines[position:position]=reader_section
    for r in table.to_dict('records'):
        lines.append(f"| {r['public_method']} | {r['execution_cap']} | {r['user_quoted_original_accuracy_percent']:.2f} | {r['original_correct_count']}/{r['original_denominator']} | {r['new_accuracy_percent']:.2f} | {r['new_correct_count']}/{r['new_denominator']} |")
    current=summary[summary.version=='structured']
    stage_totals={s:int(current['timeout_'+s+'_count'].sum()) for s in TIMEOUT_STAGES}
    if any(stage_totals.values()):
        lines.extend(['## Runtime deadlines by stage',
            'All timed-out and abstained cells remain in the official classification denominators. Error strings distinguish where a call stopped; a selection timeout can retain an already correct diagnostic posterior, so timeout counts and wrong-answer counts are separate. Stage counts can overlap when one planned cell records multiple failure stages.',
            'Across all 147,200 new main cells: '+', '.join(f'{stage}={count}' for stage,count in stage_totals.items())+'.',
            '| Primary-suite method | B | Planned N | Selection deadline cells | Diagnostic deadline cells | Acquisition deadline cells | Infrastructure deadline cells | Unclassified deadline cells |',
            '|---|---:|---:|---:|---:|---:|---:|---:|'])
        for r in current[current.suite=='test_connections'].itertuples():
            counts=[int(getattr(r,'timeout_'+s+'_count')) for s in TIMEOUT_STAGES]
            if not any(counts):continue
            lines.append(f"| {METHODS[r.method]} | {r.execution_cap} | {int(r.rows)} | "+' | '.join(map(str,counts))+' |')
        lines.append('Per-seed and all-suite counts, physical-case counts and correctness among stage-affected cells are saved in `main_timeout_stages.csv` and `main_timeout_stages_by_seed.csv`. The separate untimed-reader audit addresses diagnostic deadlines only. It does not rerun selection or recover acquisitions abandoned after a selector deadline; therefore a budget effect with many selector timeouts is not a clean test of the diagnostic representation alone.')
        flag_mismatch=int(current.timeout_error_without_flag_count.sum()+current.timeout_flag_without_error_stage_count.sum())
        if flag_mismatch:
            lines.append(f'{flag_mismatch} cells have a disagreement between the recorded timeout flag and recognized timeout error provenance. Both fields are retained; unclassified flags are not assigned to the reader or selector without evidence.')
    validation_audit=out/'validation_timeout_audit.json'
    if validation_audit.exists():
        source=read_json(validation_audit)
        checks=[r for r in source['by_method_seed_check'] if r['method']=='gnn_depth2' and r['seed']==23]
        if checks:
            first=sorted(checks,key=lambda r:r['check'])[0]
            if first['timeouts']:
                iterations=', '.join(map(str,first.get('checkpoint_iterations',[]))) or 'unavailable'
                stages=', '.join(f'{key}={value}' for key,value in first['timeout_stages'].items())
                lines.append(f"A separate SOURCE validation snapshot ({source['created_at']}) recorded {first['timeouts']}/{first['rows']} timeouts for GNN 2-Step seed 23, {first['check']}, checkpoint iteration(s) {iterations}: {stages}. The recorded selection status was `{first['selection_status']}`. This describes an intermediate training-validation record, not the final held-out main result. The comparison with the original validation aggregate is not matched for chosen evidence or worker concurrency and cannot identify a causal hardware/model contribution. Full input hashes, workloads and per-family/configuration counts remain in `validation_timeout_audit.json`.")
    lines.extend(selected_validation_interpretation(run))
    lines.extend(['## Required interpretation',
        f"1. **Does the structured diagnoser eliminate sensitivity to duplicated evidence?** The declared real-history tests cover {inv['test_A_B_cases']} physical cases and duplicate counts 0–32. Maximum absolute posterior-probability change is {inv['test_A_B_max_probability_error']:.3g} (tolerance 1e-6). This establishes the tested storage-identity invariance, not general diagnostic accuracy.",
        f"2. **Does it correctly distinguish new measurements from repeated storage?** Preserved IDs deduplicate; {int((control.test=='identity_equal_values_distinct_id').sum())} equal-value/distinct-ID ingress checks retain the added event. Test C evaluates {inv['test_C_structured_pairs']} genuinely acquired execution increments from the saved diagnostic trajectories. Mean true-label NLL change after the new execution is {new_execution_mean:+.6f} bits (configuration/case balanced, policy seeds averaged; negative is improvement). Per-family changes are in `invariance/new_execution_summary.csv`; posterior change or improved NLL is not forced.",
        f"3. **Does it correctly preserve shared-error uncertainty?** Test D covers {inv['test_D_cases']} acquired cross-clock histories. For the explicitly uncalibrated learned healthy timing control, the average-transit variance is 2τ² + 2σ²/N, whereas the independent-clock debug model gives (2τ² + 2σ²)/N. The nonzero shared term is retained; its observed learned value is {shared.shared_clock_variance.iloc[0]:.6g} s². Actual marginalized relative-clock posterior covariance is also reported. Informative calibration or identifiable hypotheses may legitimately reduce that conditional covariance."])
    lines.append('4. **Does it reduce the healthy false alarms previously caused by continued investigation?** Exact first-execution to terminal comparisons follow; each denominator counts fixed saved trajectories, with physical cases reported separately.')
    for (cohort,method),part in continuation[continuation.fault_family=='healthy'].groupby(['cohort','method']):
        text=[]
        for reader in ('original_as_deployed','original_GNN_reread','R-C','structured'):
            r=part[part.reader==reader].iloc[0]
            text.append(f"{reader}: {int(r.first_false_alarm_count)}/{int(r.rows)} → {int(r.terminal_false_alarm_count)}/{int(r.rows)}")
        lines.append(f"   - {cohort}, {METHODS[method]}: {'; '.join(text)} ({int(r.physical_cases)} physical cases).")
    lines.append('5. **Does adaptive investigation still degrade as B increases?** Primary-suite paired B4−B1 differences below show accuracy in percentage points and NLL in bits (lower NLL is better), with descriptive 95% CIs. B is an additional-execution cap: more allowed budget need not produce more acquired evidence when the policy stops earlier. Mean recorded additional execution and byte usage is shown for B1 → B4; classification denominators remain complete when resource values are unavailable.')
    for method in ('gnn_adaptive','gnn_depth2','transformer_adaptive','cost_aware_adaptation'):
        paired=contrasts[(contrasts.contrast=='budget_change')&(contrasts.version=='structured')&
            (contrasts.suite=='test_connections')&(contrasts.method==method)&(contrasts.execution_cap==4)]
        r=paired[paired.metric=='accuracy'].iloc[0];nll=paired[paired.metric=='nll_bits'].iloc[0]
        direction='decreases' if r.difference<0 else 'increases' if r.difference>0 else 'is unchanged'
        first=cell('structured','test_connections',method,1);last=cell('structured','test_connections',method,4)
        resources=[]
        for column,label,format_spec in [('executions_used','additional executions','.3f'),('bytes_used','additional bytes',',.0f')]:
            means=[format(v,format_spec) if pd.notna(v) else 'unavailable' for v in (first['mean_'+column],last['mean_'+column])]
            entry=f"mean {label} {' → '.join(means)}"
            if int(first[column+'_unknown_rows']) or int(last[column+'_unknown_rows']):
                entry+=f" (known/unknown rows {int(first[column+'_known_rows'])}/{int(first[column+'_unknown_rows'])} → {int(last[column+'_known_rows'])}/{int(last[column+'_unknown_rows'])})"
            resources.append(entry)
        lines.append(f"   - {METHODS[method]}: accuracy {direction}, {100*r.difference:+.2f} [{100*r.ci_low:+.2f}, {100*r.ci_high:+.2f}]; NLL {nll.difference:+.4f} [{nll.ci_low:+.4f}, {nll.ci_high:+.4f}] bits; {'; '.join(resources)}.")
    lines.append('6. **Does Replan now beat, match, or still lose to Plan-Once?** Primary-suite paired structured Replan−Plan-Once differences show accuracy in percentage points and NLL in bits (lower NLL is better), with descriptive 95% CIs. An interval spanning zero is not evidence of equivalence:')
    for cap in (1,2,4):
        paired=contrasts[(contrasts.contrast=='replan_minus_plan_once')&(contrasts.version=='structured')&
            (contrasts.suite=='test_connections')&(contrasts.execution_cap==cap)]
        r=paired[paired.metric=='accuracy'].iloc[0];nll=paired[paired.metric=='nll_bits'].iloc[0]
        direction='higher' if r.difference>0 else 'lower' if r.difference<0 else 'equal'
        lines.append(f"   - B={cap}: accuracy {direction}, {100*r.difference:+.2f} [{100*r.ci_low:+.2f}, {100*r.ci_high:+.2f}]; NLL {nll.difference:+.4f} [{nll.ci_low:+.4f}, {nll.ci_high:+.4f}] bits.")
    lines.append('7. **Does topology generalization remain?** Frozen structured Replan B4 exact-target accuracy is:')
    for suite in ('test_seen','test_connections','test_size'):
        r=cell('structured',suite,'gnn_adaptive',4)
        lines.append(f"   - {suite}: {rate(r,'accuracy','correct_count','rows')}; NLL {r.nll_bits:.4f} bits; healthy false alarms {rate(r,'healthy_false_alarm_rate','false_alarm_count','healthy_rows')}; recorded timeouts {int(r.timeout_count)}/{int(r.rows)}, abstentions {int(r.abstained_count)}/{int(r.rows)}.")
    lines.extend(['   These are separate transfer settings; the size shift is not pooled with the primary unseen-connections result. Generalization is measured by these held-out outcomes, not inferred from a passed identity test.',
        '8. **Which remaining errors are caused by diagnosis, and which by acquisition selection?** `diagnosis_acquisition_decomposition.csv` contains paired, identical-cell contrasts for original deployed predictions → structured reader on old histories (reader mapping), and structured reader on old histories → new policy plus structured reader (policy retraining/acquisition/runtime). The components add to the total old-to-new difference. Continued-execution prefix changes identify sensitivity to the acquired continuation under the fixed old policy. A remaining wrong posterior may reflect model misspecification, ambiguous/noisy observed evidence, or an uninformative chosen action; this experiment does not uniquely assign each error to diagnosis versus selection. Full Recording is an acquired-evidence reference, not an oracle. No per-case causal label is invented.',
        '## Audit files',
        '`reports/main_per_case_predictions.parquet`, `replay/per_case_predictions.parquet`, `replay/contexts.parquet`, and `replay/jobs/` retain target vectors and source history references. New raw acquired histories and provenance remain under the run public store. Summary, family, seed, resource, continuation and paired-contrast CSVs accompany the eight figure families. Timeouts, abstentions and failed rollout fallbacks stay in denominators. Offline replay times and physical-cache wall times are labeled separately from original deployed runtime.'])
    decomposition=pd.read_csv(out/'diagnosis_acquisition_decomposition.csv')
    selected=decomposition[(decomposition.suite=='test_connections')&(decomposition.method=='gnn_adaptive')&
        (decomposition.execution_cap==4)&(decomposition.metric=='nll_bits')]
    if len(selected)!=2:raise ValueError('Primary diagnosis/acquisition decomposition missing')
    observations='; '.join(f"{r.component}: {r.difference:+.5f} bits [{r.ci_low:+.5f}, {r.ci_high:+.5f}]" for r in selected.itertuples())
    lines.insert(-2,'For primary-suite Replan B4, the paired NLL decomposition is '+observations+'. Negative is improvement.')
    lines[-2:-2]=error_channel_interpretation(run)
    audit_lines=['## Reader deadline sensitivity (separate from official results)']
    if audit['selected_cells']:
        audit_lines.append(f"The predeclared error-string rule selected {audit['selected_cells']} diagnostic-timeout cells. {audit['replayed_cells']} were reread successfully without the reader deadline; {audit['unreplayable_cells']} could not be reread and retain their official predictions. The full sensitivity matrix still contains {audit['full_matrix_rows']} cells. Selection uses error provenance, not fault labels or favorable probabilities.")
        sensitivity=pd.read_csv(run/'runtime_audit/full_matrix_summary.csv')
        for method in ('gnn_fixed_plan','gnn_adaptive','full_recording_feasible'):
            part=sensitivity[(sensitivity.suite=='test_connections')&(sensitivity.method==method)&(sensitivity.execution_cap==4)]
            official=part[part.version=='official'].iloc[0];untimed=part[part.version=='reader_deadline_sensitivity'].iloc[0]
            audit_lines.append(f"- Primary-suite {METHODS[method]}, B4: official accuracy {official.accuracy*100:.2f}% ({int(official.correct_count)}/{int(official.rows)}), deadline-sensitivity accuracy {untimed.accuracy*100:.2f}% ({int(untimed.correct_count)}/{int(untimed.rows)}); NLL {official.nll_bits:.4f} → {untimed.nll_bits:.4f} bits.")
        audit_lines.append('This measures the effect of rereading the already acquired final evidence. It does not recover abandoned later actions, redo physical executions, change frozen parameters, remove selector deadlines, or replace the official main comparison. Warm-cache adaptive calls and deferred final calls can have different runtime costs, so a remaining Plan/Replan difference is not automatically attributable to diagnostic representation.')
    else:
        audit_lines.append(f"No eligible diagnostic-timeout cells occurred in the {audit['full_matrix_rows']}-cell new main matrix. The sensitivity table therefore leaves all official predictions unchanged.")
    lines[-2:-2]=audit_lines
    if 'resource_accounting_unavailable_count' in summary:
        missing=int(summary[summary.version=='structured'].resource_accounting_unavailable_count.sum())
        if missing:
            lines.append(f'{missing} new planned rollout cells failed before complete resource accounting was available. Their diagnostic prior fallbacks remain in every classification denominator. Resource means report their observed known-row denominators and unknown counts; unavailable work is not recorded as zero. Durable physical-attempt files retain any interrupted or retried work separately.')
    # Keep Markdown table rows adjacent; prose and list blocks remain separated.
    text='\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n'
    path=out/'REPORT.md';path.write_text(text);return dict(state='passed',report=str(path),sha256=file_sha256(path))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['replay_reports','replay_policy_contrasts','main_reports','interpret'])
    parser.add_argument('--run-dir',required=True);parser.add_argument('--source-root')
    args=vars(parser.parse_args());command=args.pop('command')
    if command!='main_reports':args.pop('source_root')
    print(globals()[command](**args),flush=True)
