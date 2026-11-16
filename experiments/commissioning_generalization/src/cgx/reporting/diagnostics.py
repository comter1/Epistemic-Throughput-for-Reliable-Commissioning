"""Post-evaluation stratification and a source-validation quality threshold.

Private family/severity annotations enter result tables here, after acquisition.
No policy, reader, feature builder or data collector imports this enrichment.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from cgx.provenance import read_json,file_sha256,atomic_json

FAMILIES=('healthy','cache_stale','queue_hold','adapter_scale')
GROUP=['suite','method','policy_seed','byte_fraction','execution_cap']


def validation_quality_target(root):
    root=Path(root);directory=root/'models/readers/final'
    marker=read_json(directory/'complete.json');history=read_json(directory/'validation.json')
    if marker.get('state')!='passed' or marker.get('technical_smoke'):
        raise ValueError('Quality target requires the completed formal final reader')
    filename=marker['selected_checkpoint'];path=directory/filename
    if file_sha256(path)!=marker['selected_checkpoint_hash']:raise ValueError('Quality-target reader checkpoint hash mismatch')
    selected=[row for row in history if f"checkpoint_{row['iteration']:07d}.pt"==filename]
    if len(selected)!=1:raise ValueError('Selected reader validation metric missing/ambiguous')
    target=float(selected[0]['mean_final_nll_bits'])
    if not np.isfinite(target):raise ValueError('Quality-target validation NLL is nonfinite')
    return dict(target_nll_bits=target,source='validation',rule='selected_final_reader_mean_validation_nll',
        reader_checkpoint_hash=marker['selected_checkpoint_hash'],selected_iteration=selected[0]['iteration'],
        validation_history_hash=file_sha256(directory/'validation.json'),
        metric_boundary='equal case/configuration mean across all acquired validation histories',
        crossing_rule='first registered resource-cap point meeting target; no interpolation or extrapolation')


def enrich_evaluator_results(frame,cases,graphs):
    """Strict evaluator joins, never a public-history modification."""
    metadata=[];seen=set()
    for case in cases:
        if case['case_id'] in seen:raise ValueError('Duplicate private case identity during reporting')
        seen.add(case['case_id']);graph=graphs[case['configuration_id']]
        family=case['fault_family'];target='healthy' if family=='healthy' else case['fault_node']+':'+family
        metadata.append(dict(case_id=case['case_id'],_expected_configuration=case['configuration_id'],_expected_target=target,
            fault_family=family,severity=case['severity'],component_count=len(graph['nodes']),
            splitter_count=sum(n['type']=='splitter' for n in graph['nodes']),merger_count=sum(n['type']=='merger' for n in graph['nodes'])))
    descriptors=pd.DataFrame(metadata)
    # A supplied evaluator count must agree with the sealed graph; never overwrite an error.
    base=frame.rename(columns={'component_count':'_recorded_component_count'}) if 'component_count' in frame else frame
    enriched=base.merge(descriptors,on='case_id',how='left',validate='many_to_one')
    if enriched.fault_family.isna().any():raise ValueError('Result references an unregistered private case')
    if not enriched.configuration_id.eq(enriched._expected_configuration).all() or not enriched.true_target_key.eq(enriched._expected_target).all():
        raise ValueError('Private case/result configuration or target mismatch')
    if '_recorded_component_count' in enriched and not enriched._recorded_component_count.eq(enriched.component_count).all():
        raise ValueError('Recorded topology size differs from sealed graph')
    enriched=enriched.drop(columns=[x for x in ('_recorded_component_count','_expected_configuration','_expected_target') if x in enriched])
    enriched['severity']=pd.to_numeric(enriched.severity,errors='raise')
    enriched['severity_magnitude']=enriched.severity.abs()
    enriched['predicted_mechanism']=enriched.predicted_target_key.map(lambda key:'healthy' if key=='healthy' else key.rsplit(':',1)[-1])
    if not enriched.predicted_mechanism.isin(FAMILIES).all():raise ValueError('Prediction mechanism outside registered target space')
    enriched['healthy_false_alarm']=np.where(enriched.fault_family.eq('healthy'),enriched.predicted_target_key.ne('healthy').astype(float),np.nan)
    if 'initial_task_pass' not in enriched:enriched['initial_task_pass']=None
    if 'initial_hold_error_m' not in enriched:enriched['initial_hold_error_m']=np.nan
    enriched['initial_task_impact']=enriched.initial_task_pass.map(lambda value:'unknown' if pd.isna(value) else ('passed' if value else 'failed'))
    return enriched


def _confusion(frame,grouping):
    counts=frame.groupby(grouping+['predicted_mechanism'],dropna=False).size().unstack('predicted_mechanism',fill_value=0).reindex(columns=FAMILIES,fill_value=0)
    denominator=counts.sum(axis=1)
    fractions=counts.div(denominator,axis=0).rename(columns=lambda key:'fraction_'+key)
    counts=counts.rename(columns=lambda key:'count_'+key)
    result=pd.concat([counts,fractions],axis=1);result['cases']=denominator
    return result.reset_index()


def cost_to_quality(points,quality_target):
    if quality_target.get('source')!='validation' or quality_target.get('rule')!='selected_final_reader_mean_validation_nll':
        raise ValueError('Cost-to-quality target must be sealed from source validation')
    target=float(quality_target['target_nll_bits'])
    if not np.isfinite(target):raise ValueError('Nonfinite quality target')
    rows=[]
    axis=quality_target.get('budget_axis','byte_fraction')
    keys=['suite','method','policy_seed']+(['execution_cap'] if axis=='byte_fraction' else ['byte_fraction'])
    for values,group in points.groupby(keys,dropna=False):
        group=group.sort_values(axis);qualifying=group[group.final_nll_bits.le(target)]
        row=dict(zip(keys,values));row.update(target_nll_bits=target,status='not_reached',byte_fraction=np.nan,
            actual_bytes=np.nan,actual_executions=np.nan,actual_calibrations=np.nan,observed_nll_bits=np.nan,
            selected_point_feasible=None,reference_only=row['method']=='full_recording_unrestricted_reference',
            registered_points_observed=len(group),interpolated=False,extrapolated=False)
        if len(qualifying):
            selected=qualifying.iloc[0]
            row.update(status='reached',byte_fraction=float(selected.byte_fraction),actual_bytes=float(selected.bytes_used),
                actual_executions=float(selected.executions_used),actual_calibrations=float(selected.calibrations_used),
                observed_nll_bits=float(selected.final_nll_bits),selected_point_feasible=bool(selected.get('all_cases_feasible',True)))
            row['execution_cap']=int(selected.execution_cap)
        rows.append(row)
    return pd.DataFrame(rows)


def write_diagnostic_tables(root,frame,quality_target):
    root=Path(root);output=root/'results';output.mkdir(parents=True,exist_ok=True)
    grouping=GROUP+['configuration_id']
    healthy=frame[frame.fault_family.eq('healthy')]
    per_config=healthy.groupby(grouping,dropna=False).agg(healthy_cases=('case_id','size'),false_alarms=('healthy_false_alarm','sum'),
        healthy_false_alarm_rate=('healthy_false_alarm','mean'),healthy_abstention_rate=('abstained','mean')).reset_index()
    per_config.to_csv(output/'healthy_false_alarm_per_configuration.csv',index=False)
    per_seed=per_config.groupby(GROUP,dropna=False).agg(healthy_cases=('healthy_cases','sum'),false_alarms=('false_alarms','sum'),
        configurations=('configuration_id','size'),healthy_false_alarm_rate=('healthy_false_alarm_rate','mean'),
        healthy_abstention_rate=('healthy_abstention_rate','mean')).reset_index()
    per_seed.to_csv(output/'healthy_false_alarm_per_seed.csv',index=False)
    for name,extra in [('mechanism',[]),('topology_size',['component_count']),('severity',['severity']),('joint_size_severity',['component_count','severity'])]:
        _confusion(frame,grouping+['fault_family']+extra).to_csv(output/f'confusion_by_{name}.csv',index=False)
    impact=frame.groupby(grouping+['fault_family','component_count','severity','initial_task_impact'],dropna=False).agg(
        cases=('case_id','size'),final_nll_bits=('final_nll_bits','mean'),joint_accuracy=('correct_joint','mean'),
        mean_initial_hold_error_m=('initial_hold_error_m','mean'),timeout_rate=('timeout','mean'),abstention_rate=('abstained','mean')).reset_index()
    impact.to_csv(output/'fault_impact_strata.csv',index=False)
    # Cases -> systems -> seeds: do not let graph size or case counts change statistical weight.
    metrics=['final_nll_bits','bytes_used','executions_used','calibrations_used']
    system_points=frame.groupby(grouping,dropna=False)[metrics].mean().reset_index()
    feasible=frame.groupby(grouping,dropna=False).feasible.all().reset_index(name='all_cases_feasible') if 'feasible' in frame else None
    if feasible is not None:system_points=system_points.merge(feasible,on=grouping,validate='one_to_one')
    else:system_points['all_cases_feasible']=True
    points=system_points.groupby(GROUP,dropna=False)[metrics].mean().reset_index()
    feasibility=system_points.groupby(GROUP,dropna=False).all_cases_feasible.all().reset_index()
    points=points.merge(feasibility,on=GROUP,validate='one_to_one')
    points.to_csv(output/'cost_quality_recorded_points.csv',index=False)
    crossings=cost_to_quality(points,quality_target);crossings.to_csv(output/'cost_to_quality_per_seed.csv',index=False)
    atomic_json(output/'cost_to_quality_target.json',quality_target)
    atomic_json(output/'diagnostic_table_definitions.json',{
        'annotations':'Private fault family/severity joined only after acquisition into evaluator outputs; never model input.',
        'healthy_false_alarm':'Nonhealthy top-1 prediction among all truly healthy cases, including timeout/abstention rows; operational abstention is reported separately.',
        'severity':'Signed injection parameter within its own fault family; healthy severity is undefined (blank/NaN), not zero.',
        'confusion':'Counts and conditional prediction fractions for every observed true-family/configuration stratum; all four predicted families shown, including zero cells.',
        'impact':'Initial common execution requirement outcome; unknown outcomes remain a separate stratum. No case is filtered by task impact.',
        'quality':'Validation-selected final reader NLL target. First actually recorded resource-cap point meeting it; not_reached has missing costs, no extrapolation.',
        'weighting':'Average cases within configuration, then configurations equally; training seeds remain separate.'})
    return dict(healthy_rows=len(healthy),impact_strata=len(impact),cost_quality_groups=len(crossings))
