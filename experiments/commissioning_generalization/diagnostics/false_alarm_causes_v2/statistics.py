"""V2 paired statistics: average seeds, then stratified case/config bootstrap.

All uncertainty is conditional on the frozen models and the declared 16 existing
test configurations. Three policy seeds are repeated runs, not physical systems.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from cgx.provenance import atomic_json, read_json

PAIR = ['configuration_id', 'case_id', 'method_key', 'policy_seed']
METRICS = [('healthy_false_alarm', 'healthy'), ('fault_miss', 'faulty'),
           ('correct_joint', 'faulty'), ('final_nll_bits', 'all'),
           ('final_nll_bits', 'healthy'), ('final_nll_bits', 'faulty'),
           ('executions_used', 'all'), ('calibrations_used', 'all'),
           ('decisions_used', 'all'), ('bytes_used', 'all')]


def hierarchical(frame, columns, repetitions=2000, seed=671932):
    """Resample paired column vectors, preserving healthy/fault case strata.

    Every configuration keeps its original stratum sizes in each replicate.
    Seed averaging occurs BEFORE either level is resampled.
    """
    assert frame.groupby(['case_id']).configuration_id.nunique().max() == 1
    assert frame[columns].notna().all().all(), 'Missing outcomes cannot silently leave the denominator'
    cases = frame.groupby(['configuration_id', 'case_id', 'is_healthy'], sort=True)[columns].mean().reset_index()
    groups = [p for _, p in cases.groupby('configuration_id', sort=True)]
    point = np.mean([p[columns].to_numpy(float).mean(0) for p in groups], axis=0)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(len(groups), size=(repetitions, len(groups)))
    draws = np.zeros((repetitions, len(groups), len(columns)))
    for index, config in enumerate(groups):
        rr, ss = np.where(chosen == index)
        value = np.zeros((len(rr), len(columns)))
        for _, stratum in config.groupby('is_healthy', sort=True):
            array = stratum[columns].to_numpy(float)
            sampled = rng.integers(len(array), size=(len(rr), len(array)))
            value += array[sampled].sum(1) / len(config)
        draws[rr, ss] = value
    return point, draws.mean(1), cases


def estimate(paired, metric, contrast, group='all'):
    results = []
    for method, part in paired.groupby('method_key', sort=True):
        if group == 'healthy': part = part[part.is_healthy]
        if group == 'faulty': part = part[~part.is_healthy]
        assert not part[PAIR].duplicated().any()
        assert set(part.policy_seed) == {11, 23, 37}
        assert (part.groupby('case_id').policy_seed.nunique() == 3).all()
        point, draws, cases = hierarchical(part, ['baseline', 'intervention'])
        delta = draws[:, 1] - draws[:, 0]
        results.append(dict(method_key=method, contrast=contrast, metric=metric,
            subgroup=group, baseline=float(point[0]), intervention=float(point[1]),
            difference=float(point[1]-point[0]),
            paired_ci_low=float(np.quantile(delta, .025)), paired_ci_high=float(np.quantile(delta, .975)),
            baseline_ci_low=float(np.quantile(draws[:, 0], .025)), baseline_ci_high=float(np.quantile(draws[:, 0], .975)),
            intervention_ci_low=float(np.quantile(draws[:, 1], .025)), intervention_ci_high=float(np.quantile(draws[:, 1], .975)),
            physical_cases=len(cases), configurations=cases.configuration_id.nunique(),
            paired_seed_runs=len(part), bootstrap_repetitions=2000, bootstrap_seed=671932,
            bootstrap_unit='configuration then physical case; seed-averaged; healthy/fault strata retained'))
    return results


def contrast_rows(data, a, b, metric, group):
    left = data[data.arm == a][PAIR+['is_healthy', metric]].rename(columns={metric:'baseline'})
    right = data[data.arm == b][PAIR+[metric]].rename(columns={metric:'intervention'})
    paired = left.merge(right, on=PAIR, how='outer', validate='one_to_one', indicator=True)
    assert (paired._merge == 'both').all(), f'Incomplete pairing for {b}-{a}'
    return estimate(paired, metric, f'{b}-{a}', group)


def rate_rows(data, grouping):
    output = []
    for identity, part in data.groupby(grouping, sort=True):
        if not isinstance(identity, tuple): identity = (identity,)
        names = dict(zip(grouping, identity))
        for metric, subgroup in [('healthy_false_alarm','healthy'),('fault_miss','faulty'),('correct_joint','faulty')]:
            s = part[part.is_healthy] if subgroup == 'healthy' else part[~part.is_healthy]
            output.append(dict(names, metric=metric, subgroup=subgroup,
                numerator=int(s[metric].sum()), denominator=len(s), rate=float(s[metric].mean()),
                distinct_physical_cases=s.case_id.nunique(), configurations=s.configuration_id.nunique(),
                seeds=s.policy_seed.nunique(), failures=int((s.state != 'passed').sum())))
    return pd.DataFrame(output)


def two_by_two(run):
    """Matched-update teacher/input effects; these never choose a test winner."""
    results=[]
    for phase in ('source_validation','fresh_replay'):
        data=pd.read_parquet(run/f'correction_2x2_{phase}_per_case.parquet')
        data=data[data.step.eq(2000)&data.arm.isin(['T00','T01','T10','T11'])]
        expected_cases=64 if phase=='source_validation' else 128
        assert len(data)==expected_cases*6*4
        assert not data[PAIR+['arm']].duplicated().any()
        for metric,group in METRICS:
            effects=[]
            for a,b in [('T00','T11'),('T00','T10'),('T01','T11'),('T00','T01'),('T10','T11')]:
                effects+=contrast_rows(data,a,b,metric,group)
            pivot=data.pivot(index=PAIR+['is_healthy'],columns='arm',values=metric).reset_index()
            pivot[['T00','T01','T10','T11']]=pivot[['T00','T01','T10','T11']].astype(float)
            for contrast,baseline,intervention in [
                ('mean_FINAL_label-minus-OOF_label',(pivot.T00+pivot.T01)/2,(pivot.T10+pivot.T11)/2),
                ('mean_FINAL_q_training-minus-OOF_q_training',(pivot.T00+pivot.T10)/2,(pivot.T01+pivot.T11)/2),
                ('(T11-T01)-(T10-T00)',pivot.T10-pivot.T00,pivot.T11-pivot.T01)]:
                paired=pivot[PAIR+['is_healthy']].copy()
                paired['baseline']=baseline;paired['intervention']=intervention
                effects+=estimate(paired,metric,contrast,group)
            results.extend(dict(row,phase=phase,matched_updates=2000,all_deployment_q='FINAL') for row in effects)
    pd.DataFrame(results).to_csv(run/'correction_2x2_paired_effects.csv',index=False)


def run(run_dir):
    run = Path(run_dir)
    data = pd.read_csv(run/'confirmation_per_case.csv')
    selected=read_json(run/'manifests/source_variant_selection.json')
    if selected['optional_live']['enabled']:
        assert read_json(run/'trajectories/live/complete.json')['state']=='passed'
        main=data[~data.arm.isin(['R-live','RG-live'])]
        live=pd.read_csv(run/'live_confirmation_per_case.csv')
        data=pd.concat([main,live],ignore_index=True)
    cases = read_json(run/'manifests/fresh_cases.json')
    case_map = {c['case_id']:c for c in cases}
    assert len(case_map) == 128 and sum(c['fault_family']=='healthy' for c in cases) == 64
    assert set(data.case_id) == set(case_map)
    assert set(data.method_key) == {'gnn_adaptive','gnn_depth2'}
    assert {'A0','S1','G1'} <= set(data.arm)
    assert not data[PAIR+['arm']].duplicated().any()
    expected = {(c['configuration_id'], c['case_id'], m, s) for c in cases
                for m in ('gnn_adaptive','gnn_depth2') for s in (11,23,37)}
    for arm, part in data.groupby('arm'):
        assert set(part[PAIR].itertuples(index=False, name=None)) == expected, f'Incomplete arm {arm}'
    data['fault_family'] = data.case_id.map(lambda c:case_map[c]['fault_family'])
    assert (data.is_healthy == (data.fault_family=='healthy')).all()
    failures = data[data.state != 'passed']
    failures.to_csv(run/'confirmation_failures.csv', index=False)
    if len(failures):
        atomic_json(run/'tests/confirmation_statistics.json', dict(state='blocked_by_retained_failures',
            failed_rows=len(failures), expected_rows=len(expected)*data.arm.nunique(), present_rows=len(data)))
        raise RuntimeError('All failed rows retained. Resolve or explicitly report their missing diagnostic outcomes before primary statistics.')
    assert data[[m for m,_ in METRICS]].notna().all().all()
    assert data.executions_used.between(0,4).all() and data.calibrations_used.between(0,2).all()
    assert data.decisions_used.between(0,6).all()
    data.to_csv(run/'confirmation_per_case.csv', index=False)
    rate_rows(data, ['method_key','arm']).to_csv(run/'confirmation_counts_and_rates.csv', index=False)
    rate_rows(data, ['configuration_id','method_key','arm']).to_csv(run/'confirmation_per_configuration.csv', index=False)
    rate_rows(data, ['method_key','policy_seed','arm']).to_csv(run/'confirmation_per_seed.csv', index=False)
    rate_rows(data, ['method_key','arm','fault_family']).to_csv(run/'confirmation_per_fault_family.csv', index=False)
    metric_columns = list(dict.fromkeys(m for m,_ in METRICS))
    config = data.groupby(['configuration_id','method_key','arm','is_healthy'], sort=True)[metric_columns].mean().reset_index()
    config.groupby(['method_key','arm','is_healthy'], sort=True)[metric_columns].mean().reset_index().to_csv(run/'confirmation_mean_metrics.csv', index=False)
    comparisons = [('A0','G1'),('S1','G1')]
    if 'R1' in set(data.arm): comparisons += [('A0','R1'),('A0','RG1')]
    comparisons += [('A0',arm) for arm in ('R-live','RG-live') if arm in set(data.arm)]
    effects = []
    for a,b in comparisons:
        for metric,group in METRICS:
            effects += contrast_rows(data,a,b,metric,group)
    if {'A0','R1','G1','RG1'} <= set(data.arm):
        for metric,group in METRICS:
            pivot = data[data.arm.isin(['A0','R1','G1','RG1'])].pivot(index=PAIR+['is_healthy'], columns='arm', values=metric).reset_index()
            pivot['baseline'] = pivot.R1.astype(float)-pivot.A0.astype(float)
            pivot['intervention'] = pivot.RG1.astype(float)-pivot.G1.astype(float)
            effects += estimate(pivot,metric,'(RG1-G1)-(R1-A0)',group)
    pd.DataFrame(effects).to_csv(run/'paired_effects.csv',index=False)
    two_by_two(run)
    remaining = data[(data.arm!='A0') & ~data.correct_joint.astype(bool)].copy()
    original = data[data.arm=='A0'][PAIR+['predicted_target_key','correct_joint']].rename(
        columns={'predicted_target_key':'original_predicted_target_key','correct_joint':'original_correct_joint'})
    remaining = remaining.merge(original,on=PAIR,validate='many_to_one')
    remaining['error_category'] = np.where(remaining.is_healthy,'healthy_false_alarm',np.where(remaining.fault_miss,'fault_missed_as_healthy','wrong_fault_type_or_location'))
    remaining['new_error_relative_to_A0'] = remaining.original_correct_joint
    remaining.to_csv(run/'remaining_errors.csv',index=False)
    atomic_json(run/'tests/confirmation_statistics.json',dict(state='passed',physical_cases=128,
        healthy_physical_cases=64,faulty_physical_cases=64,rows=len(data),arms=sorted(data.arm.unique()),
        seed_runs_per_method_arm=384,conditional_denominator_per_method_arm=192,
        pairing_complete=True,failed_rows_retained=len(failures),bootstrap_replicates=2000,
        seed_averaging_before_resampling=True,healthy_faulty_strata_preserved=True,
        topology_claim='fresh physical cases on already examined held-out topologies'))
    print(f'Confirmation statistics complete: {len(data)} rows, {len(effects)} paired estimates',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True)
    run(parser.parse_args().run_dir)
