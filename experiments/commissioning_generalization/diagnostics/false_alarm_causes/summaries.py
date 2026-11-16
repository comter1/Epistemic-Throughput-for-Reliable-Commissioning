"""Descriptive summaries; no new model calls or outcome-dependent sample choice."""
from pathlib import Path
import argparse
import pandas as pd


def run(run):
    run=Path(run)
    routes=pd.read_csv(run/'budget_path_divergence.csv')
    summaries=[]
    for method,part in routes.groupby('method_key'):
        a=part[part.both_have_first_action];e=part[part.both_have_first_execution]
        summaries.append(dict(method_key=method,total_pairs=len(part),pairs_with_first_actions=len(a),
            first_action_changed=int(a.first_action_changed.sum()),first_action_change_fraction=a.first_action_changed.mean(),
            pairs_with_first_executions=len(e),first_execution_changed=int(e.first_execution_action_changed.sum()),
            first_execution_change_fraction=e.first_execution_action_changed.mean()))
    pd.DataFrame(summaries).to_csv(run/'budget_path_summary.csv',index=False)
    frame=pd.read_parquet(run/'prefix_predictions.parquet')
    keys=['configuration_id','case_id','method_key','policy_seed','diagnostic_reader']
    rows=[]
    for identity,part in frame.groupby(keys,sort=True):
        part=part.sort_values('decision_index');executed=part[part.executions_used>=1]
        selected=[('initial',part.iloc[0]),('first_actual_execution',executed.iloc[0] if len(executed) else part.iloc[-1]),
                  ('actual_terminal',part.iloc[-1])]
        for name,row in selected:
            rows.append(dict(zip(keys,identity))|dict(reading_point=name,decision_index=row.decision_index,
                executions_used=row.executions_used,calibrations_used=row.calibrations_used,bytes_used=row.bytes_used,
                is_healthy=row.is_healthy,healthy_false_alarm=row.false_alarm,fault_miss=row.missed_fault,
                correct_joint=row.correct_joint,nll_bits=row.nll_bits,p_healthy=row.p_healthy,
                no_actual_execution=len(executed)==0,evidence_prefix_hash=row.evidence_prefix_hash))
    data=pd.DataFrame(rows);data.to_csv(run/'fixed_path_reading_points.csv',index=False)
    fields=['healthy_false_alarm','fault_miss','correct_joint','nll_bits','p_healthy','executions_used','calibrations_used','bytes_used']
    data.groupby(['method_key','diagnostic_reader','reading_point','is_healthy'])[fields].mean().reset_index().to_csv(run/'fixed_path_reading_summary.csv',index=False)
    values=pd.read_csv(run/'action_value_checks.csv')
    k=['configuration_id','case_id','method_key','policy_seed','decision_index','evidence_prefix_hash','horizon','is_healthy','action_key','policy_selected']
    means=values.groupby(k)[['predicted_q','matched_horizon_sample_gain_bits','immediate_sample_gain_bits']].mean().reset_index()
    means['negative_mean_continuation_gain']=means.matched_horizon_sample_gain_bits<0
    means.to_csv(run/'action_value_mean_over_noise.csv',index=False)
    chosen=means[means.policy_selected].copy()
    branch_keys=k[:-2]
    alternatives=means.groupby(branch_keys).matched_horizon_sample_gain_bits.max().rename('best_sampled_alternative_gain').reset_index()
    chosen=chosen.merge(alternatives,on=branch_keys,validate='one_to_one')
    chosen['better_sampled_alternative']=chosen.best_sampled_alternative_gain>chosen.matched_horizon_sample_gain_bits+1e-6
    chosen.groupby(['method_key','is_healthy'])[['predicted_q','matched_horizon_sample_gain_bits',
        'negative_mean_continuation_gain','better_sampled_alternative']].mean().reset_index().to_csv(run/'action_value_summary.csv',index=False)
    print('Descriptive summaries complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();run(a.run_dir)
