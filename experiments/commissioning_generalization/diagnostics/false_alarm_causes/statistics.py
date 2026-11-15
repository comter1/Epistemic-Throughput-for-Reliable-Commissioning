"""Paired physical-case statistics with seed averaging and hierarchical intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

PAIR = ['configuration_id','case_id','method_key','policy_seed']
CASE = PAIR[:3]
METRICS = {'healthy_false_alarm':'healthy', 'fault_miss':'faulty',
           'correct_joint':'faulty','final_nll_bits':'all',
           'executions_used':'all','calibrations_used':'all','bytes_used':'all'}
METRIC_GROUPS = list(METRICS.items()) + [('final_nll_bits','faulty'),('final_nll_bits','healthy'),('correct_joint','all')]


def estimate(frame, metric, contrast, phase, baseline, intervention, group='all'):
    """All paired rows must exist; evaluate seeds, then cases, then configurations."""
    rows=[]
    for method,part in frame.groupby('method_key',sort=True):
        if group=='healthy':part=part[part.is_healthy]
        if group=='faulty':part=part[~part.is_healthy]
        if part.empty:continue
        assert not part[PAIR].duplicated().any()
        assert part[['baseline','intervention']].notna().all().all(), 'Missing pairs must be resolved, not dropped'
        cases=part.groupby(CASE,sort=True)[['baseline','intervention']].mean().reset_index()
        configs=[g[['baseline','intervention']].to_numpy(float) for _,g in cases.groupby('configuration_id',sort=True)]
        point=np.mean([g.mean(0) for g in configs],axis=0)
        rng=np.random.default_rng(671922)
        sampled_configs=np.zeros((2000,len(configs),2))
        # Configurations and cases are resampled together for both paired arms.
        selected=rng.integers(len(configs),size=(2000,len(configs)))
        for ci,values in enumerate(configs):
            repetitions,slots=np.where(selected==ci)
            indices=rng.integers(len(values),size=(len(repetitions),len(values)))
            sampled_configs[repetitions,slots]=values[indices].mean(1)
        sampled=sampled_configs.mean(1)
        delta=sampled[:,1]-sampled[:,0]
        row=dict(phase=phase,method_key=method,contrast=contrast,baseline_name=baseline,
                 intervention_name=intervention,metric=metric,subgroup=group,
                 baseline=float(point[0]),intervention=float(point[1]),difference=float(point[1]-point[0]),
                 paired_ci_low=float(np.quantile(delta,.025)),paired_ci_high=float(np.quantile(delta,.975)),
                 baseline_ci_low=float(np.quantile(sampled[:,0],.025)),baseline_ci_high=float(np.quantile(sampled[:,0],.975)),
                 intervention_ci_low=float(np.quantile(sampled[:,1],.025)),intervention_ci_high=float(np.quantile(sampled[:,1],.975)),
                 case_weighted_baseline=float(cases.baseline.mean()),case_weighted_intervention=float(cases.intervention.mean()),
                 configurations=len(configs),physical_cases=len(cases),paired_seed_rows=len(part),bootstrap_repetitions=2000)
        if metric in ('healthy_false_alarm','fault_miss') and not contrast.startswith('('):
            row.update(baseline_positive_seed_rows=int(part.baseline.sum()),
                       remaining_positive_seed_rows=int(part.intervention.sum()),
                       cases_with_any_remaining_positive=int((part.groupby('case_id').intervention.max()>0).sum()))
        rows.append(row)
    return rows


def compare(left,right,metric,contrast,phase,baseline,intervention,group='all'):
    a=left[PAIR+['is_healthy',metric]].rename(columns={metric:'baseline'})
    b=right[PAIR+[metric]].rename(columns={metric:'intervention'})
    paired=a.merge(b,on=PAIR,how='outer',validate='one_to_one',indicator=True)
    assert (paired._merge=='both').all(), f'Unpaired rows for {phase}/{contrast}/{metric}'
    return estimate(paired,metric,contrast,phase,baseline,intervention,group)


def interventions(run, fresh=False):
    path=run/('fresh_intervention_outcomes.parquet' if fresh else 'intervention_outcomes.parquet')
    if not path.exists():return []
    data=pd.read_parquet(path);phase='confirmation' if fresh else 'discovery'
    selection=run/'manifests/confirmation_selection.json'
    expected_arms=(set(json.loads(selection.read_text())['analysis_arms']) if selection.exists() else
                   set(json.loads((run/'manifests/interventions_confirmation.json').read_text())['arms'])) if fresh else set('ABCD')
    assert set(data.arm)==expected_arms, 'A complete intervention arm is missing'
    assert not data[PAIR+['arm']].duplicated().any()
    assert len(data)==(128 if fresh else 64)*6*data.arm.nunique(), 'Do not omit failed/missing preselected cases'
    contrasts=[('A',arm) for arm in sorted(data.arm.unique()) if arm!='A']
    if not fresh and set('ABCD')<=set(data.arm):contrasts.append(('B','D'))
    rows=[]
    for a,b in contrasts:
        for metric,group in METRIC_GROUPS:
            rows+=compare(data[data.arm==a],data[data.arm==b],metric,f'{b}-{a}',phase,a,b,group)
    if not fresh and set('ABCD')<=set(data.arm):
        for metric,group in METRIC_GROUPS:
            pivot=data.pivot(index=PAIR+['is_healthy'],columns='arm',values=metric).reset_index()
            pivot[list('ABCD')]=pivot[list('ABCD')].astype(float)
            pivot['baseline']=pivot.C-pivot.A;pivot['intervention']=pivot.D-pivot.B
            # Required interaction is (C-A)-(D-B).
            pivot['baseline'],pivot['intervention']=pivot['intervention'].copy(),pivot['baseline'].copy()
            rows+=estimate(pivot,metric,'(C-A)-(D-B)',phase,'D-B','C-A',group)
    summary=data.groupby(['method_key','arm','is_healthy'],sort=True)[list(METRICS)].mean().reset_index()
    summary.to_csv(run/(phase+'_arm_summary.csv'),index=False)
    config=data.groupby(['configuration_id','method_key','arm','is_healthy'],sort=True)[list(METRICS)].mean().reset_index()
    config.to_csv(run/(phase+'_per_configuration.csv'),index=False)
    data.groupby(['method_key','policy_seed','arm','is_healthy'],sort=True)[list(METRICS)].mean().reset_index().to_csv(run/(phase+'_per_seed.csv'),index=False)
    data[PAIR+['arm','state','is_healthy']+list(METRICS)].to_csv(run/('fresh_case_confirmation.csv' if fresh else 'discovery_case_metrics.csv'),index=False)
    return rows


def reread(run):
    path=run/'fixed_transcript_reread.csv'
    if not path.exists():return []
    data=pd.read_csv(path);rows=[]
    assert data.diagnostic_reader.nunique()==2
    for reader,part in data.groupby('diagnostic_reader'):
        assert len(part)==5952 and part.final_is_actual_terminal.all(), 'Complete actual-trajectory denominator required'
        part=part.copy();part['is_healthy']=part.true_target_key=='healthy'
        for metric,old,new,group in [('healthy_false_alarm','original_false_alarm','reread_false_alarm','healthy'),
                                    ('fault_miss','original_missed_fault','reread_missed_fault','faulty'),
                                    ('final_nll_bits','original_nll_bits','nll_bits','all'),
                                    ('final_nll_bits','original_nll_bits','nll_bits','faulty')]:
            paired=part[PAIR+['is_healthy',old,new]].rename(columns={old:'baseline',new:'intervention'})
            rows+=estimate(paired,metric,f'{reader}-original','same_transcript','original',reader,group)
    return rows


def calibration(run):
    path=run/'calibration_same_records.csv'
    if not path.exists():return []
    data=pd.read_csv(path)
    assert len(data)==384*3 and set(data.variant)=={'retain','remove','supplement_to_two_pairs'}
    assert not data[PAIR+['variant']].duplicated().any()
    rows=[]
    for variant in ('remove','supplement_to_two_pairs'):
        for metric,group in METRIC_GROUPS:
            if metric not in data:continue
            rows+=compare(data[data.variant=='retain'],data[data.variant==variant],metric,
                          variant+'-retain','same_record_calibration','retain',variant,group)
    return rows


def runtime_attribution(run):
    if not (run/'intervention_outcomes.parquet').exists() or not (run/'fixed_transcript_reread.csv').exists():return []
    current=pd.read_parquet(run/'intervention_outcomes.parquet')
    current=current[current.arm=='A'].copy()
    fixed=pd.read_csv(run/'fixed_transcript_reread.csv')
    fixed=fixed[fixed.diagnostic_reader=='main_gnn']
    paired=current.merge(fixed,on=PAIR,how='left',validate='one_to_one',suffixes=('','_reread'))
    assert len(paired)==384 and paired.original_predicted_target_key.notna().all()
    base=paired[PAIR+['is_healthy']].copy()
    same=base.copy()
    base['healthy_false_alarm']=paired.original_false_alarm
    base['fault_miss']=paired.original_missed_fault
    base['correct_joint']=paired.original_predicted_target_key==paired.true_target_key
    base['final_nll_bits']=paired.original_nll_bits
    same['healthy_false_alarm']=paired.reread_false_alarm
    same['fault_miss']=paired.reread_missed_fault
    same['correct_joint']=paired.reread_predicted_target_key==paired.true_target_key
    same['final_nll_bits']=paired.nll_bits
    rows=[]
    for a,b,an,bn in ((base,same,'original','same_records_main_gnn'),(base,current,'original','full_unlimited_A'),
                      (same,current,'same_records_main_gnn','full_unlimited_A')):
        for metric,group in METRIC_GROUPS:
            if metric not in a or metric not in b:continue
            rows+=compare(a,b,metric,bn+'-'+an,'runtime_discovery',an,bn,group)
    paired[PAIR+['is_healthy','original_predicted_target_key','reread_predicted_target_key','predicted_target_key',
                 'original_false_alarm','reread_false_alarm','healthy_false_alarm',
                 'original_missed_fault','reread_missed_fault','fault_miss',
                 'original_timeout','original_prediction_prefix','actual_observation_prefix','unread_observations',
                 'source_final_history_hash','evidence_trace_hash','state']].to_csv(run/'runtime_attribution.csv',index=False)
    return rows


def stopping_existing(run):
    path=run/'fixed_path_reading_points.csv'
    if not path.exists():return []
    data=pd.read_csv(path).rename(columns={'nll_bits':'final_nll_bits'})
    data=data[data.diagnostic_reader=='main_gnn']
    left=data[data.reading_point=='actual_terminal'];right=data[data.reading_point=='first_actual_execution']
    rows=[]
    for metric,group in METRIC_GROUPS:
        if metric in data:
            rows+=compare(left,right,metric,'STOP_after_first_execution-final',
                          'stopping_existing_trajectories','actual_terminal','STOP_after_first_execution',group)
    return rows


def run(run):
    run=Path(run);rows=interventions(run)+interventions(run,True)+reread(run)+calibration(run)+runtime_attribution(run)+stopping_existing(run)
    temporary=run/'paired_effects.csv.tmp';pd.DataFrame(rows).to_csv(temporary,index=False)
    temporary.replace(run/'paired_effects.csv')
    print(f'Paired effects: {len(rows)} rows',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();run(a.run_dir)
