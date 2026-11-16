"""Offline reports: physical-case paired statistics, honest denominators and figures.

This module may read evaluator truth only after execution; it is never imported by
an acquisition policy. Missing metrics stay missing, and failed runs remain rows.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .storage import file_hash, write_json

METHODS = ('original_task_once', 'fixed_four_pose', 'fixed_mixed_suite',
           'adaptive_task_checks', 'keep_existing_settings', 'true_payload_settings_reference')
LABELS = ('Original task once', 'Fixed four poses', 'Fixed mixed suite',
          'Adaptive checks', 'Existing settings', 'True payload reference')
COLORS = ('#4477AA', '#228833', '#66CCEE', '#AA3377', '#BBBBBB', '#EE7733')


def dump_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def number(value, default=float('nan')):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def mean(values):
    values = np.asarray([number(v) for v in values], dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else None


def nested(record, *keys, default=None):
    for key in keys:
        value = record
        for part in key.split('.'):
            value = value.get(part) if isinstance(value, dict) else None
        if value is not None:
            return value
    return default


def truth_index(root, split):
    path = root / 'private' / (split + '.json')
    if not path.exists():
        return {}
    content = json.loads(path.read_text())
    if isinstance(content, dict):
        content = content.get('cases', content)
    if isinstance(content, dict):
        return content
    return {str(row['case_id']): row for row in content}


def read_runs(root, split='test'):
    rows = []
    for path in sorted((root / 'runs' / split).glob('*/*.json')):
        row = json.loads(path.read_text())
        if 'case_id' not in row:
            continue
        row = dict(row)
        row['method_id'] = row.get('method_id', row.get('method', path.parent.name))
        row['record_path'] = str(path.relative_to(root))
        row['record_sha256'] = file_hash(path)
        rows.append(row)
    return rows


def normalize(row, private):
    truth = private.get(str(row['case_id']), {})
    released = row.get('released_status', 'hold_unresolved') in ('pass_at_full_speed', 'pass_at_half_speed')
    speed = number(row.get('final_speed'), 1.)
    count = int(row.get('audit_pass_count', 0))
    total = int(row.get('audit_total', 0))
    audited = total > 0
    all_pass = audited and count == total and bool(row.get('all_audits_pass', True))
    verification = row.get('public_verification_result', {})
    if not isinstance(verification, dict):
        verification = {'passed': bool(verification)}
    public_pass = bool(nested(verification, 'passed', 'pass', 'task_pass', default=False))
    motion = row.get('verification_private_metrics') or verification
    true_payload = nested(truth, 'true_payload', 'actual_payload', 'payload_true', default={})
    old_payload = nested(truth, 'old_payload', 'previous_payload', default={})
    cfg = row.get('final_payload_config', {})
    initial = row.get('initial_payload_config', {})
    estimate = nested(row, 'payload_estimate', 'final_estimate.payload', 'final_estimate.payload_estimate', default={})
    if not isinstance(estimate, dict):
        estimate = {}
    final_estimate = row.get('final_estimate') or {}
    group = nested(truth, 'group', 'change_group', 'stratum', default=row.get('group', 'unknown'))
    group = {'unchanged_correct': 'G0', 'changed_correct': 'G1', 'stale_mass': 'G2', 'stale_com': 'G3'}.get(group, group)
    family = row.get('object_family_id', truth.get('object_family_id', truth.get('family_id', 'unknown')))
    ptest, pverify = number(row.get('physical_test_seconds'), 0.), number(nested(row,'public_verification_seconds','verification_seconds'), 0.)
    fit, select = number(nested(row,'fitting_cpu_seconds','fit_cpu_seconds'), 0.), number(row.get('selection_cpu_seconds'), 0.)
    result = {k: row.get(k) for k in ('case_id', 'noise_seed', 'method_id', 'split', 'controller_hash',
              'public_config_hash', 'estimator_hash', 'workload_hash', 'record_path', 'record_sha256')}
    result.update(object_family_id=family, group=group, released=released, final_speed=speed,
                  released_status=row.get('released_status', 'hold_unresolved'), public_verification_pass=public_pass,
                  audit_pass_count=count, audit_total=total, all_audits_pass=all_pass,
                  correct_full_release=bool(released and public_pass and all_pass and speed == 1.),
                  correct_half_release=bool(released and public_pass and all_pass and speed == .5),
                  false_release=bool(released and audited and not all_pass),
                  unresolved=not released, incomplete_audit=not audited,
                  physical_test_seconds=ptest, public_verification_seconds=pverify,
                  online_physical_seconds=ptest+pverify, fitting_cpu_seconds=fit, selection_cpu_seconds=select,
                  setting_wait_seconds=number(nested(row,'setting_wait_seconds','setting_operation_wall_seconds'), 0.),
                  serial_deployment_seconds=ptest+pverify+fit+select+number(nested(row,'setting_wait_seconds','setting_operation_wall_seconds'), 0.),
                  setting_updated=bool(row.get('setting_updated', False)),
                  recorded_bytes=nested(row,'recorded_bytes','bytes_uncompressed_identification'), selected_test_ids=row.get('selected_test_ids', []),
                  stop_reason=row.get('stop_reason', 'missing'), error=row.get('error'),
                  saturation_seconds=nested(motion, 'saturation_seconds', 'actuator_saturation_seconds'),
                  unexpected_contacts=nested(motion, 'unexpected_contacts', 'unexpected_contact', 'prohibited_contacts'),
                  hard_abort=nested(motion, 'hard_abort', 'aborted'),
                  position_rms_m=nested(motion, 'position_rms_m', 'rms_error_m', 'ee_rms_error_m', 'rms_position_error_m'),
                  position_peak_m=nested(motion, 'position_peak_m', 'peak_error_m', 'ee_peak_error_m', 'peak_position_error_m'),
                  settled_position_error_m=nested(motion, 'settled_position_error_m', 'final_error_m', 'hold_max_error_m'),
                  torque_peak_Nm=nested(motion, 'torque_peak_Nm', 'max_torque_Nm'),
                  velocity_peak_rad_s=nested(motion, 'velocity_peak_rad_s', 'max_velocity_rad_s'),
                  joint_path_length_rad=row.get('joint_path_length_rad'),
                  virtual_model_calculations=row.get('virtual_model_calculations'),
                  true_mass_kg=true_payload.get('mass'),
                  final_payload_config=cfg, initial_payload_config=initial, payload_estimate=estimate)
    if true_payload.get('mass') is not None:
        for label, payload in [('estimate', estimate), ('config', cfg), ('initial_config', initial)]:
            if payload.get('mass') is not None:
                result[label+'_mass_abs_error_kg'] = abs(float(payload['mass']) - float(true_payload['mass']))
            if payload.get('com') is not None:
                result[label+'_com_error_mm'] = float(np.linalg.norm(np.asarray(payload['com'])-np.asarray(true_payload['com']))*1000)
            if payload.get('inertia_com') is not None:
                result[label+'_inertia_error_kg_m2'] = float(np.linalg.norm(np.asarray(payload['inertia_com'])-np.asarray(true_payload['inertia_com'])))
    if true_payload.get('com') is not None and old_payload.get('com') is not None:
        result['com_change_mm'] = float(np.linalg.norm(np.asarray(true_payload['com'])-np.asarray(old_payload['com']))*1000)
    if true_payload and estimate and final_estimate:
        z = 2.5758293035489004
        mass_sd = number(final_estimate.get('mass_sd'))
        if np.isfinite(mass_sd):
            result['mass_interval_covers_truth'] = abs(float(estimate['mass'])-float(true_payload['mass'])) <= z*mass_sd
        com_sd = final_estimate.get('com_sd')
        if com_sd is not None:
            covered = np.abs(np.asarray(estimate['com'])-np.asarray(true_payload['com'])) <= z*np.asarray(com_sd)
            result['com_interval_covers_truth'] = bool(np.all(covered))
            for axis,value in zip('xyz',covered):
                result['com_'+axis+'_interval_covers_truth'] = bool(value)
        result['interval_definition'] = 'Approximate pointwise99%; COM is intersection of three coordinate intervals, not a simultaneous99% region.'
    for key in ('mass_interval_covers_truth', 'com_interval_covers_truth', 'workload_torque_interval_coverage',
                'heldout_torque_prediction_rmse_Nm', 'residual_rms_Nm', 'regressor_rank', 'regressor_min_singular_value'):
        value = nested(row, key, 'evaluation.'+key, 'final_estimate.'+key)
        if key == 'heldout_torque_prediction_rmse_Nm':
            value = nested(row,key,'held_out_torque_prediction.rmse_Nm')
        if value is not None or key not in result:
            result[key] = value
    singular = final_estimate.get('regressor_singular_values', [])
    if singular:
        result['regressor_min_singular_value'] = min(singular)
    result['residual_score'] = final_estimate.get('residual_score')
    result['last_block_residual_score'] = final_estimate.get('last_block_residual_score')
    result['identification_recorded_bytes']=number(nested(row,'bytes_uncompressed_identification','recorded_bytes'),0.)
    verify_bytes=0
    paths=row.get('verification_paths') or {}
    if paths.get('metadata'):
        verify_bytes=json.loads(Path(paths['metadata']).read_text()).get('bytes_uncompressed',0)
    result['verification_recorded_bytes']=verify_bytes
    result['recorded_bytes']=result['identification_recorded_bytes']+verify_bytes
    online_paths=[h.get('paths',{}).get('public') for h in row.get('history',[])]+[paths.get('public')]
    result['online_compressed_public_record_bytes']=sum(Path(p).stat().st_size for p in online_paths if p)
    result['recorded_bytes_definition']='Uncompressed acquired public NumPy-array bytes, identification plus verification; not a network-wire size. Offline audits separate.'
    return result


def summarize(rows, extra=None):
    n = len(rows)
    released = sum(r['released'] for r in rows)
    false = sum(r['false_release'] for r in rows)
    result = dict(extra or {})
    result.update(runs=n, physical_cases=len({r['case_id'] for r in rows}),
                  families=len({r['object_family_id'] for r in rows}),
                  correct_full_release=sum(r['correct_full_release'] for r in rows),
                  correct_half_release=sum(r['correct_half_release'] for r in rows),
                  released_runs=released, false_release=false,
                  false_release_per_released=f'{false}/{released}', false_release_per_all=f'{false}/{n}',
                  unresolved=sum(r['unresolved'] for r in rows), incomplete_audit=sum(r['incomplete_audit'] for r in rows),
                  audit_task_pass_count=sum(r['audit_pass_count'] for r in rows),
                  audit_task_total=sum(r['audit_total'] for r in rows))
    for speed in (1., .5):
        label = 'full' if speed == 1. else 'half'
        subset = [r for r in rows if r['final_speed'] == speed]
        den = sum(r['released'] for r in subset)
        num = sum(r['false_release'] for r in subset)
        result[f'{label}_false_release_per_released'] = f'{num}/{den}'
        result[f'{label}_false_release_per_all'] = f'{num}/{n}'
    for field in ('physical_test_seconds', 'public_verification_seconds', 'online_physical_seconds',
                  'fitting_cpu_seconds', 'selection_cpu_seconds', 'serial_deployment_seconds', 'setting_updated',
                  'recorded_bytes', 'position_rms_m', 'position_peak_m', 'saturation_seconds',
                  'estimate_mass_abs_error_kg', 'estimate_com_error_mm', 'config_mass_abs_error_kg',
                  'config_com_error_mm', 'heldout_torque_prediction_rmse_Nm', 'workload_torque_interval_coverage'):
        result['mean_'+field] = mean(r.get(field) for r in rows)
        result['observed_'+field+'_n'] = sum(np.isfinite(number(r.get(field))) for r in rows)
    return result


def completeness(rows):
    failures = []
    keys = [(r['case_id'], r['noise_seed'], r['method_id']) for r in rows]
    if len(set(keys)) != len(keys):
        failures.append('duplicate case/seed/method records')
    for method in METHODS:
        group = [r for r in rows if r['method_id'] == method]
        if len(group) != 576:
            failures.append(f'{method}: {len(group)}/576 runs')
        if len({r['case_id'] for r in group}) != 192:
            failures.append(f'{method}: physical case count differs from 192')
        if {r['noise_seed'] for r in group} != {11,23,37}:
            failures.append(f'{method}: noise seed set differs')
        if len({r['object_family_id'] for r in group}) != 8:
            failures.append(f'{method}: family count differs from 8')
        for label in ('G0','G1','G2','G3'):
            if sum(r['group'] == label for r in group) != 144:
                failures.append(f'{method}/{label}: group count differs from 144')
        if any(r['audit_total'] != 8 for r in group):
            failures.append(f'{method}: missing eight private audits')
    return {'passed': not failures, 'expected_runs': 3456, 'observed_runs': len(rows), 'failures': failures}


def paired_bootstrap(rows, repetitions=2000, seed=631917):
    """Average measurement seeds inside case, then resample families and cases."""
    metrics = ('correct_full_release','correct_half_release','false_release','unresolved','online_physical_seconds')
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['object_family_id'],row['case_id'],row['method_id'])].append(row)
    case_values = {k: {metric: mean(r[metric] for r in group) for metric in metrics}
                   for k,group in grouped.items()}
    summaries, directions = [], []
    rng = np.random.default_rng(seed)
    for a,b in itertools.combinations(METHODS[:4],2):
        pairs = sorted((family,case) for family,case,method in case_values
                       if method == a and (family,case,b) in case_values)
        families = sorted({f for f,c in pairs})
        if not pairs:
            continue
        deltas = np.asarray([[case_values[(f,c,a)][m]-case_values[(f,c,b)][m] for m in metrics] for f,c in pairs])
        indices = {f: np.asarray([i for i,(ff,c) in enumerate(pairs) if ff == f]) for f in families}
        boots = np.empty((repetitions,len(metrics)))
        for rep in range(repetitions):
            selected = []
            for family_index in rng.integers(0,len(families),len(families)):
                ix=indices[families[family_index]]
                selected.extend(rng.choice(ix,len(ix),replace=True))
            boots[rep] = deltas[selected].mean(axis=0)
        for j,metric in enumerate(metrics):
            lo,hi=np.quantile(boots[:,j],[.025,.975])
            summaries.append({'method_a':a,'method_b':b,'metric':metric,
                              'difference_a_minus_b':float(deltas[:,j].mean()),
                              'ci95_low':float(lo),'ci95_high':float(hi),
                              'paired_physical_cases':len(pairs),'families':len(families),
                              'bootstrap_repetitions':repetitions,'seeds_averaged_before_resampling':True})
            for family,ix in indices.items():
                delta=float(deltas[ix,j].mean())
                directions.append({'method_a':a,'method_b':b,'metric':metric,'object_family_id':family,
                                   'physical_cases':len(ix),'difference_a_minus_b':delta,
                                   'direction':'positive' if delta>0 else 'negative' if delta<0 else 'equal'})
    return summaries,directions


def _history(row):
    return row.get('history', row.get('test_history', row.get('parameter_estimates_after_each_test', [])))


def history_table(row):
    rows=[]
    elapsed=0.
    for i,item in enumerate(_history(row)):
        if not isinstance(item,dict):
            continue
        action=nested(item,'test_id','action_id','action',default=(row.get('selected_test_ids') or ['unknown']*100)[i])
        duration=number(nested(item,'duration_s','physical_seconds','simulated_seconds'))
        elapsed+=duration
        estimate=nested(item,'estimate','payload_estimate','posterior',default=item)
        std=nested(estimate,'std','parameter_std',default=[])
        cov=nested(estimate,'covariance10','covariance','parameter_covariance')
        if cov is not None:
            cov=np.asarray(cov)
            if cov.ndim==2:
                std=np.sqrt(np.maximum(0,np.diag(cov))).tolist()
        result={'step':i,'action':str(action),'duration_s':duration,'elapsed_s':elapsed,
                'workload_uncertainty_fraction':nested(item,'workload_uncertainty_fraction','decision_relevant_uncertainty',
                                                       'estimate.workload_uncertainty_fraction','assessment.torque_interval_ratio','decision.torque_interval_ratio','stop.uncertainty_fraction'),
                'regressor_rank':nested(item,'regressor_rank','estimate.regressor_rank','estimate.rank','rank'),
                'residual_rms_Nm':nested(item,'residual_rms_Nm','estimate.residual_rms_Nm'),
                'residual_score':nested(item,'residual_score','estimate.residual_score'),
                'selection_reason':nested(item,'selection_reason','reason_for_next_test','reason')}
        for j,name in enumerate(('mass','mx','my','mz','Ixx','Iyy','Izz','Ixy','Ixz','Iyz')):
            result[name+'_std']=std[j] if len(std)>j else None
        com_sd=estimate.get('com_sd',[]) if isinstance(estimate,dict) else []
        for j,axis in enumerate('xyz'):
            result['com_'+axis+'_std_mm'] = float(com_sd[j])*1000 if len(com_sd)>j else None
        assessment = item.get('assessments_by_threshold',{}).get(str(row.get('threshold')), {})
        if result['workload_uncertainty_fraction'] is None:
            result['workload_uncertainty_fraction'] = assessment.get('torque_interval_ratio')
        rows.append(result)
    return rows


def _plot_settings():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})
    return plt


def _tex_plot(path, csv_name, x, ys, labels=None, xlabel='', ylabel='', bar=False):
    options=['width=12cm','height=6cm','grid=major','xlabel={'+xlabel+'}', 'ylabel={'+ylabel+'}']
    if bar: options.extend(['ybar','bar width=5pt'])
    plots=[]
    for index,y in enumerate(ys):
        plots.append('\\addplot table[x='+x+',y='+y+',col sep=comma]{'+csv_name+'};')
        if labels: plots.append('\\addlegendentry{'+labels[index].replace('_','\\_')+'}')
    Path(path).write_text('\\documentclass[tikz,border=2pt]{standalone}\n\\usepackage{pgfplots}\n'
                         '\\pgfplotsset{compat=1.18}\n\\begin{document}\n\\begin{tikzpicture}\n'
                         '\\begin{axis}['+','.join(options)+']\n'+'\n'.join(plots)+'\n'
                         '\\end{axis}\n\\end{tikzpicture}\n\\end{document}\n')


def workload_coverage(root, raw, rows, private):
    """Offline coverage on the frozen public work grid; no private ID call."""
    path=root/'configs/design_bank.json'
    if not path.exists():
        return
    bank=json.loads(path.read_text())
    y=np.asarray(bank['Y_full']).reshape(-1,10)
    for record,row in zip(raw,rows):
        est=record.get('final_estimate') or {}
        if 'parameter_mean10' not in est or 'covariance10' not in est:
            continue
        true=private[str(record['case_id'])]['true_payload']
        mass=float(true['mass']);com=np.asarray(true['com']);ic=np.asarray(true['inertia_com'])
        io=ic+mass*(np.dot(com,com)*np.eye(3)-np.outer(com,com))
        pi=np.r_[mass,mass*com,io[0,0],io[1,1],io[2,2],io[0,1],io[0,2],io[1,2]]
        delta=y@(np.asarray(est['parameter_mean10'])-pi)
        sd=np.sqrt(np.maximum(0,np.einsum('ni,ij,nj->n',y,np.asarray(est['covariance10']),y)))
        covered=np.abs(delta)<=float(bank['joint_interval_radius'])*sd
        row['workload_torque_interval_coverage']=float(np.mean(covered))
        row['workload_torque_interval_all_grid_points_cover']=bool(np.all(covered))
        row['workload_grid_payload_prediction_rmse_Nm']=float(np.sqrt(np.mean(delta**2)))
        row['workload_grid_note']='Frozen public reference grid, joint10D99% parameter-ellipsoid torque envelope; evaluator true parameter label; distinct from held-out measured-trajectory RMSE.'


def split_residual_table(root, raw):
    rows=[]
    for path in sorted((root/'source_validation/source_records').glob('*.json')):
        record=json.loads(path.read_text());cal=record.get('calibration',{})
        for joint in range(7):
            count=cal.get('count',0)
            if count:
                rows.append({'split':'source','case_id':record['case_id'],'joint':joint,
                             'noise_seed':5,'processed_vectors':count,
                             'label_based_residual_mean_Nm':cal['residual_mean'][joint],
                             'label_based_centered_residual_sd_Nm':math.sqrt(cal['centered_sum_squares'][joint]/count),
                             'label_based_abs_residual_q99_Nm':cal['absolute_residual_q99'][joint],
                             'note':'Source calibration truth label minus public measured-signal dynamics; no private sensor error draws.'})
    run_paths=sorted((root/'runs/validation').glob('*/*/*.json'))
    all_records=[json.loads(path.read_text()) for path in run_paths]+raw
    for record in all_records:
        est=record.get('final_estimate') or {}
        rows.append({'split':record.get('split'),'case_id':record['case_id'],'noise_seed':record['noise_seed'],
                     'method_id':record.get('method_id',record.get('method')),'threshold':record.get('threshold'),
                     'standardized_fit_residual_score':est.get('residual_score'),
                     'standardized_last_block_residual_score':est.get('last_block_residual_score'),
                     'heldout_torque_prediction_rmse_Nm':nested(record,'heldout_torque_prediction_rmse_Nm','held_out_torque_prediction.rmse_Nm'),
                     'note':'Fit scores are dimensionless prewhitened residual means; do not confuse with source Nm residuals.'})
    return rows


def physical_effect_figure(root, plt):
    """Use actual paired source interventions, with identical task and sensor seed."""
    base=root/'source_validation/payload_effects_final'
    folder=root/'figures'
    labels={'old_old':'Old plant / old config','new_old':'New plant / stale config','new_new':'New plant / updated config'}
    columns={};costs={}
    for condition in labels:
        path=base/condition/'private.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as data:
                columns[condition]={'time':data['time'].copy(),'error':data['error'].copy(),
                                    'torque':np.max(np.abs(data['torque']),axis=1)}
            costs[condition]=json.loads((base/condition/'metadata.json').read_text())['physical_seconds']
    if not columns:
        return
    rows=[]
    for i in range(max(len(c['time']) for c in columns.values())):
        row={'sample_index':i}
        for condition,data in columns.items():
            if i<len(data['time']):
                row.update({condition+'_time_s':data['time'][i],condition+'_error_m':data['error'][i],
                            condition+'_max_abs_joint_torque_Nm':data['torque'][i]})
        rows.append(row)
    dump_csv(folder/'04b_paired_source_interventions.csv',rows)
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    for condition,data in columns.items():
        axes[0].plot(data['time'],data['error'],label=labels[condition])
        axes[1].plot(data['time'],data['torque'],label=labels[condition])
    axes[0].set_ylabel('Actual end-effector tracking error (m)')
    axes[1].set_ylabel('Maximum absolute joint execution torque (Nm)')
    for ax in axes:ax.set_xlabel('Physical time (s)');ax.legend(fontsize=7)
    fig.suptitle('Paired source engineering intervention; unchanged task, no test-outcome selection')
    fig.tight_layout();fig.savefig(folder/'04b_paired_source_interventions.pdf');fig.savefig(folder/'04b_paired_source_interventions.png',dpi=140);plt.close(fig)
    conditions=list(columns)
    _tex_plot(folder/'04b_paired_source_interventions.tex','04b_paired_source_interventions.csv',
              conditions[0]+'_time_s',[c+'_error_m' for c in conditions],[labels[c] for c in conditions],
              xlabel='Physical seconds',ylabel='Actual position error (m)')
    write_json(folder/'04b_intervention_provenance.json',{'conditions':labels,'physical_seconds':costs,
               'source_only':True,'not_reselected_on_test':True,'trace_sampling_hz':500,
               'input_hashes':{c:file_hash(base/c/'private.npz') for c in columns},
               'controller_config_sha256':file_hash(root/'configs/controller.json'),
               'supersedes_initial_gain_illustration':True})


def export_example_traces(root, raw):
    """Small complete traces inside the ZIP, not inaccessible server-only examples."""
    selections=[]
    for group in ('G0','G2','G3'):
        record=next((r for r in raw if r['method_id']=='original_task_once' and r.get('group')==group),None)
        if record:
            selections.append((f'test_{group}_original_once',record,'Formal test; deterministic first saved case in this group.'))
    selected_path=root/'examples/selected_cases.json'
    if selected_path.exists():
        for example in json.loads(selected_path.read_text()):
            chosen=example.get('record')
            if not chosen:
                continue
            method=chosen.get('method_id',chosen.get('method'))
            record=next((r for r in raw if r['case_id']==chosen['case_id'] and r['noise_seed']==chosen['noise_seed'] and r['method_id']==method),None)
            if record:
                selections.append(('test_'+example['type'],record,'Exact formal test example discussed in FINDINGS_CN.md.'))
    candidates=[]
    for path in sorted((root/'runs/validation/threshold_0.02/fixed_mixed_suite').glob('*.json')):
        record=json.loads(path.read_text())
        if len(record['history'])>1:
            candidates.append((len(record['history']),str(path),record))
    if candidates:
        _,path,record=max(candidates,key=lambda x:(x[0],x[1]))
        selections.append(('validation_threshold002_multitest',record,
                           'Illustrative validation candidate threshold0.02; NOT the selected0.10 formal test method.'))
    source_path=root/'source_validation/source_sequential_illustration.json'
    if source_path.exists():
        selections.append(('source_all_library_illustration',json.loads(source_path.read_text()),
                           'Source replay of actual acquired library blocks; NOT a test run or60s-budget online method.'))
    output=root/'examples/traces';output.mkdir(parents=True,exist_ok=True)
    index=[]
    for label,record,note in selections:
        summary_path=root/'examples'/f'{label}.json'
        write_json(summary_path,{'interpretation':note,'record':record})
        blocks=[(f'identification_{i:02}_{h["test_id"]}',h['paths']) for i,h in enumerate(record.get('history',[]))]
        if record.get('verification_paths'):
            blocks.append(('verification',record['verification_paths']))
        if record.get('private_audit_results'):
            blocks.append(('private_audit_00',record['private_audit_results'][0]['paths']))
        for block_label,paths in blocks:
            source=Path(paths['public'])
            if not source.exists():
                raise FileNotFoundError('Selected example evidence missing: '+str(source))
            target=output/f'{label}__{block_label}.csv.gz'
            with np.load(source,allow_pickle=False) as archive:
                arrays={k:archive[k] for k in archive.files}
            n=len(arrays['time']);names=[];columns=[]
            for key,array in arrays.items():
                if array.ndim==0:
                    continue
                if key=='event_id':
                    names.append('measurement_event_id_hex');columns.append([bytes(v).hex() for v in array])
                elif key=='software_message_id':
                    for port in range(array.shape[1]):
                        names.append(f'software_port_{port}_message_id_hex');columns.append([bytes(v[port]).hex() for v in array])
                else:
                    for component in np.ndindex(array.shape[1:]):
                        names.append(key+''.join('_'+str(i) for i in component))
                        columns.append(array[(slice(None),)+component])
            with gzip.open(target,'wt',encoding='utf-8',newline='',compresslevel=6) as stream:
                writer=csv.writer(stream);writer.writerow(names)
                writer.writerows(zip(*columns))
            metadata=json.loads(Path(paths['metadata']).read_text()) if paths.get('metadata') else {}
            write_json(target.with_name(target.name+'.metadata.json'),{'interpretation':note,'case_id':record.get('case_id'),
                       'noise_seed':record.get('noise_seed'),'samples':n,'sampling_hz':100,'no_samples_or_channels_dropped':True,
                       'array_column_indices_are_zero_based':True,'execution_metadata':metadata})
            index.append({'example':label,'block':block_label,'interpretation':note,
                          'included_csv':str(target.relative_to(root)),'csv_sha256':file_hash(target),
                          'source_npz':str(source),'source_npz_sha256':file_hash(source),'samples':n,
                          'columns':len(names),'full_100Hz_public_stream':True})
    write_json(root/'examples/trace_index.json',{'traces':index,'full_raw_store_retained_external':True,
               'inside_zip_traces_include_all_software_ports_and_dynamic_channels':True,
               'all_history_estimates_covariances_and_singular_values_in_example_json':True})
    return index


def execution_documents(root, summaries):
    """Summarize recorded commands/exit codes and measured, attributable resources."""
    from datetime import datetime
    stages=[json.loads(p.read_text()) for p in sorted((root/'stages').glob('*.json')) if p.stem!='package']
    ledger=json.loads((root/'resources/physics_ledger.json').read_text())
    by_kind=defaultdict(lambda:{'blocks':0,'simulated_seconds':0.})
    with (root/'resources/physics_events.jsonl').open() as stream:
        for line in stream:
            event=json.loads(line)
            if event['event']=='complete':
                by_kind[event['kind']]['blocks']+=1
                by_kind[event['kind']]['simulated_seconds']+=event['actual_seconds']
    telemetry=defaultdict(list)
    telemetry_path=root/'resources/live_cgroup_samples.jsonl'
    if telemetry_path.exists():
        for line in telemetry_path.read_text().splitlines():
            row=json.loads(line)
            if row.get('returncode')==0:
                telemetry[row['port']].append(row['measurement'])
    resource={'physics_ledger':ledger,'simulated_seconds_by_kind':dict(by_kind),
              'ten_second_equivalents':ledger['charged_seconds']/10,
              'limit_ten_second_equivalents':100000,
              'formal_test_workers':{'2379':40,'2982':80,'2404':160},
              'source_workers':{'2379':28,'2982':55,'2404':109},
              'blas_threads_per_worker':1,'gpu_requested':False,
              'physical_simulation_and_large_fits_location':'authorized remote CPU nodes',
              'local_work':'lightweight SSH orchestration, code and small aggregate/design matrices',
              'cgroup_observations':{},'stages':[]}
    for label,path in [('source_mechanism','source_validation/repeat_vs_pose_mechanism.json'),
                       ('fresh_output_preparation','checks/fresh_output_preparation.json')]:
        if (root/path).exists():
            detail=json.loads((root/path).read_text())
            resource[label]={'record':path,'not_formal_test_conditions':True,
                             'physical_seconds':detail.get('total_physical_seconds',detail.get('simulated_seconds',180. if label=='fresh_output_preparation' else None)),
                             'included_in_top_level_physics_ledger':True}
    for port,samples in telemetry.items():
        samples.sort(key=lambda x:x['time'])
        def cpu(sample):
            return dict(line.split() for line in sample['cpu.stat'].splitlines())
        first,last=samples[0],samples[-1]
        resource['cgroup_observations'][str(port)]={
            'samples':len(samples),'observed_interval_seconds':last['time']-first['time'],
            'maximum_observed_memory_current_bytes':max(int(r['memory.current']) for r in samples),
            'cpu_usage_seconds_delta':(int(cpu(last)['usage_usec'])-int(cpu(first)['usage_usec']))/1e6,
            'cpu_quota':first['cpu.max'],
            'note':'Shared node cgroup delta/current samples, not exclusive process attribution. Historical memory.peak is not attributed to this run.'}
    for stage in stages:
        duration=None
        if stage.get('finished_utc') and stage.get('started_utc'):
            duration=(datetime.fromisoformat(stage['finished_utc'])-datetime.fromisoformat(stage['started_utc'])).total_seconds()
        resource['stages'].append({k:stage.get(k) for k in ('stage','started_utc','finished_utc','exit_code','completed')}|{'wall_seconds':duration})
    write_json(root/'resource_usage.json',resource)
    commands=[]
    for path in sorted((root/'jobs').glob('*.sh')):
        commands.append('# '+str(path.relative_to(root))+'\n'+path.read_text())
    for path in sorted((root/'commands').glob('*.txt')):
        commands.append('# '+str(path.relative_to(root))+'\n'+path.read_text())
    for stage in stages:
        commands.append('# stage '+str(stage.get('stage'))+' exit='+str(stage.get('exit_code'))+'\n'+str(stage.get('command'))+'\n')
    (root/'commands.txt').write_text('\n'.join(commands))
    log=['# Executed run log','',
         'Scientific inputs were frozen before test; reporting and packaging do not change decisions. Actual job scripts, stderr/stdout and exit markers remain under jobs/ and stages/.','',
         '| Stage | Start UTC | Finish UTC | Exit | Wall seconds |','|---|---|---|---:|---:|']
    for row in resource['stages']:
        log.append(f"| {row['stage']} | {row['started_utc']} | {row['finished_utc']} | {row['exit_code']} | {row['wall_seconds']} |")
    log.extend(['','## Engineering failures and corrections before formal test','',
        '- Initial remote dispatcher lacked the required SSH host-key/key context and failed before launching physics. Lightweight dispatch was moved to the local orchestrator; no private key was copied. This retry consumed no simulation time.',
        '- The first source attempt had167/192 L0 hard aborts under high feedback gains interacting with the fixed torque-rate limit. A source restart also reset persistent sensor bias. These records, settings and failures remain in engineering_attempt_01; ATTEMPT_STATUS.json maps original paths to retained locations. They are not mixed into final source fitting.',
        '- Source-only lower-gain qualification, persistent-bias continuation and measured torque/velocity time alignment were corrected before validation/test. All attempted physics remains charged. The final source dataset has192cases×11blocks; final validation uses64cases×4methods×3threshold candidates with exact candidate reuse only where state/config/RNG match.',
        '- Validation selected commonthreshold0.10 before test. All3456 registered test/reference runs and eight private tasks per run completed. No test outcome changed the controller, estimator, library or threshold.',
        '- Seven old Panda regression executions passed (70 simulated seconds). The five original finite benchmark unit tests also passed; they integrate no MuJoCo physics. A generated old bytecode file was restored to its original tracked content; protected hashes are checked again.',
        '- After formal decisions were frozen,12source families contributed one hash-selected case each to an explicitly posthoc forced-continuation comparison: commonL0+S1 then true repeatedL0/S5/D4 branches. Its528simulated seconds are charged; no registered method or test result was changed.',
        '- The post-run fresh-root preparation wrapper was actually exercised for20source profiling blocks (180simulated seconds), without validation/test execution. replication_preparation_check contains this subordinate run; its physics is reserved once in the top-level ledger and must not be added again from the child ledger.',
        '',f"Global physics: {ledger['charged_seconds']:.3f} simulated seconds = {ledger['charged_seconds']/10:.4f}/100000 ten-second equivalents, including engineering failures and retries. Outstanding reservations: {len(ledger.get('reservations',{}))}.",
        'Source collection used28+55+109workers; formal test used40+80+160. Every worker used one BLAS thread. This experiment requested no GPU because simulation and small-matrix estimation are CPU work.',
        'Telemetry reports maximum observed memory.current and cgroup CPU deltas during the sampled interval. Historical cgroup memory.peak includes other/past work and is not attributed to this experiment.',
        '', 'Independent final audit: checks/final_result_audit.json checks every3456row, all raw references, and recomputed500Hz FK/acceptance on216blocks (1.08million samples); no new simulation is needed for this read-only audit.',
        'Plot PDFs and PNG previews use Matplotlib. Standalone TikZ/PGFPlots and CSVs are supplied; LaTeX compilation was not available, and no environment upgrade was performed.'])
    (root/'CODEX_RUN_LOG.md').write_text('\n'.join(log)+'\n')
    (root/'README.md').write_text('''# CGX-PAYLOAD-COMMISSIONING-V1

Start with FINDINGS_CN.md, then results/summary.csv and results/grouped_by_change.csv.
The complete matrix is3456runs:192physical cases×3measurement seeds×6conditions.
The true-parameter reference is privileged, and eight audit paths are not eight independent objects.

The old settings already pass every declared smooth-work audit in this study.
L0-once releases570/576; fixed and adaptive continuation release576/576. The6runs needing9additional posture blocks resolve residual inconsistency in2physical cases; adaptive saves no physical
time over the equally early-stopping fixed plans. No method needs half speed.
This is not evidence that load identification is universally unnecessary.

L0 preserves the original100ms reference transitions. Final verification/audits use
the declared smooth10s workload; its full-speed release does not certify the original
fast transients. Controller/model differences and source engineering failures are
explicit in IMPLEMENTATION_AUDIT.md and CONTROLLER_DECISION.md.

Package navigation:
- protocol_locked.yaml and configs/: exact frozen scientific inputs.
- results/per_case.csv and runs/test/: every run, including unresolved decisions.
- results/private_audit_tasks.csv: all27648actual private work tasks.
- examples/selected_cases.json: real illustrative cases; absent failure types marked.
- examples/traces/: full100Hz CSV.gz traces (including all software ports), metadata,
  parameter means/covariances and singular values, directly readable without server access.
- figures/: vectorPDF, PNGpreview, standaloneTikZ/PGFPlots and underlyingCSV.
- checks/final_result_audit.json: independent row/reference/FK/acceptance audit.
- CODEX_RUN_LOG.md, resource_usage.json, commands.txt: executed stages and resources.
- raw_record_index.jsonl.gz: full server-held raw records, SHA256 and current absolute paths.
- engineering_attempt_01/ATTEMPT_STATUS.json: relocation map for archived failed-source paths.

The final package completion marker and delivery.json (archive SHA256, size and CRC)
remain outside the ZIP to avoid embedding a transient in-progress self-marker.
Full raw files, including failures, remain external; representative actual traces
are embedded here. The archive is checked below500,000,000bytes and CRC-tested.
Source full-library and validation0.02 illustrations are explicitly separate from
the selected0.10 test protocol. No success-only filtering or inferred fake traces.

Reproduce from the recorded Git base/commit and pinned environment in resources/.
For a fresh output root, use the explicitly post-run preparation convenience wrapper:
PYTHONPATH=.:src python -m payload_change_v1.prepare --run-all --run-root ABSOLUTE_ROOT
It prepares source engineering/library/controller inputs and profiles20blocks before
calling the original frozen stage runner. It was added after the reported experiment,
and was not the program that produced the already frozen formal results.
For an already prepared root or resume:
PYTHONPATH=.:src python -m payload_change_v1.run --stage all --run-root ABSOLUTE_ROOT
Run in experiments/commissioning_generalization. Resume uses stage/input hashes;
all generated datasets stay outside Git. Only the new payload namespace is edited.
''')
    return resource


def figures(root, raw, rows, summaries):
    plt=_plot_settings()
    folder=root/'figures';folder.mkdir(exist_ok=True)
    # Explicit evaluator boundaries; the audit has no return edge to the method.
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(13,5));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    nodes={'truth':(.08,.80,'True payload'),'plant':(.25,.80,'Private plant'),
           'measure':(.46,.80,'Acquired ID\nmeasurements'),'estimate':(.66,.80,'Payload estimator'),
           'choice':(.87,.80,'Choose next test /\nsubmit configuration'),
           'config':(.67,.46,'Public configuration\nand test reference'),
           'controller':(.46,.46,'Fixed torque\ncontroller'),
           'verify':(.25,.14,'Public verification'),'release':(.46,.14,'Release / hold'),
           'audit':(.86,.36,'Independent actual\nprivate work audits'),
           'report':(.86,.12,'Offline report only')}
    for key,(x,y,label) in nodes.items():
        ax.text(x,y,label,ha='center',va='center',fontsize=8.5,
                bbox={'boxstyle':'round,pad=.5','fc':'#FBE9E7' if key in ('truth','plant','audit') else '#EEF3F8','ec':'#667788'})
    def arrow(a,b,style='-',label=None):
        x,y,_=nodes[a];xx,yy,_=nodes[b]
        shrink=20 if abs(yy-y)>abs(xx-x) else 42
        ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','shrinkA':shrink,'shrinkB':shrink,'linestyle':style,'color':'#445566'})
        if label:ax.text((x+xx)/2,(y+yy)/2+.045,label,fontsize=7,ha='center')
    for a,b in [('truth','plant'),('plant','measure'),('measure','estimate'),('estimate','choice'),
                ('choice','config'),('config','controller'),('measure','controller'),('controller','plant'),
                ('plant','verify'),('verify','release'),('audit','report')]:arrow(a,b)
    arrow('config','audit','--')
    ax.text(.73,.29,'final submitted\nsettings',fontsize=7,ha='center')
    ax.add_patch(Rectangle((.01,.65),.33,.29,fill=False,linestyle='--',edgecolor='#AA3377'))
    ax.text(.175,.96,'PRIVATE PHYSICS',ha='center',fontsize=8,color='#AA3377')
    ax.add_patch(Rectangle((.75,.025),.24,.53,fill=False,linestyle='--',edgecolor='#AA3377'))
    ax.text(.87,.58,'PRIVATE EVALUATION',ha='center',fontsize=8,color='#AA3377')
    ax.text(.01,.02,'Private audit results flow only to the report. Verification records are not used to refit or reselect.',fontsize=8)
    fig.savefig(folder/'01_information_boundary.pdf');fig.savefig(folder/'01_information_boundary.png',dpi=140);plt.close(fig)
    (folder/'01_information_boundary.tex').write_text(r'''\documentclass[tikz,border=3pt]{standalone}
\usetikzlibrary{positioning,arrows.meta,fit}
\begin{document}\begin{tikzpicture}[node distance=15mm and10mm,box/.style={draw,rounded corners,align=center,font=\small},>=Stealth]
\node[box] (t) {Private true payload};\node[box,right=of t] (p) {Private plant};
\node[box,right=of p] (o) {Acquired ID\\measurements};\node[box,right=of o] (e) {Estimator};
\node[box,right=of e] (a) {Choose next test /\\submit settings};
\node[box,below=of e] (c) {Public configuration\\and reference};\node[box,below=of o] (u) {Fixed torque controller};
\node[box,below=of p] (v) {Public verification};\node[box,below=of u] (r) {Release / hold};
\node[box,below=of a] (h) {Independent private\\actual work audits};\node[box,below=of h] (f) {Offline report only};
\foreach\x/\y in {t/p,p/o,o/e,e/a,a/c,c/u,o/u,u/p,p/v,v/r,h/f}{\draw[->] (\x)--(\y);}
\draw[->,dashed] (c)--node[above,font=\scriptsize]{final settings}(h);
\node[draw,dashed,fit=(t)(p),label=above:{Private physics}] {};
\node[draw,dashed,fit=(h)(f),label=right:{Private evaluation}] {};
\end{tikzpicture}\end{document}
''')
    table=[]
    for index,method in enumerate(METHODS):
        s=next((r for r in summaries if r['method_id']==method),None)
        if s:
            table.append({'method_index':index,'method_id':method,'full_correct_rate':s['correct_full_release']/s['runs'],
                          'half_correct_rate':s['correct_half_release']/s['runs'],'false_release_all_rate':s['false_release']/s['runs'],
                          'unresolved_rate':s['unresolved']/s['runs'],'mean_online_seconds':s['mean_online_physical_seconds'],
                          'false_release_numerator':s['false_release'],'release_denominator':s['released_runs'],'runs_denominator':s['runs']})
    dump_csv(folder/'02_release_and_time.csv',table)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for row in table:
        i=row['method_index'];axes[0].bar(i,row['full_correct_rate'],color=COLORS[i])
        axes[0].bar(i,row['half_correct_rate'],bottom=row['full_correct_rate'],color=COLORS[i],alpha=.4)
        axes[1].bar(i,row['mean_online_seconds'],color=COLORS[i])
        axes[0].text(i,1.02,f"{round(row['full_correct_rate']*row['runs_denominator'])}/{row['runs_denominator']}\nFR {row['false_release_numerator']}/{row['release_denominator']}",ha='center',fontsize=7)
    for ax in axes:ax.set_xticks(range(6),LABELS,rotation=45,ha='right')
    axes[0].set_title('Correct full-speed x/n; FR = false / released',fontsize=9);axes[0].set_ylabel('Correct release / all runs');axes[0].set_ylim(0,1.2);axes[1].set_ylabel('Online physical seconds (ID + verification)')
    fig.tight_layout();fig.savefig(folder/'02_release_and_time.pdf');fig.savefig(folder/'02_release_and_time.png',dpi=140);plt.close(fig)
    _tex_plot(folder/'02_release_and_time.tex','02_release_and_time.csv','method_index',
              ['full_correct_rate','half_correct_rate','false_release_all_rate','unresolved_rate'],
              ['Full correct','Half correct','False / all','Unresolved'],xlabel='Method index (see CSV)',ylabel='Fraction of all runs',bar=True)
    _tex_plot(folder/'02b_physical_time.tex','02_release_and_time.csv','method_index',['mean_online_seconds'],xlabel='Method index',ylabel='Physical seconds',bar=True)
    example=max((r for r in raw if r['method_id']=='fixed_mixed_suite'),key=lambda r:len(_history(r)),default=None)
    history=history_table(example) if example else []
    dump_csv(folder/'03_sequential_uncertainty.csv',history,fields=None if history else ['step','elapsed_s','mass_std','workload_uncertainty_fraction'])
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    for ax,key,label in [(axes[0],'mass_std','Mass posterior SD (kg)'),(axes[1],'workload_uncertainty_fraction','Task torque uncertainty / limit')]:
        available=[r for r in history if np.isfinite(number(r.get(key)))]
        if available:ax.plot([r['elapsed_s'] for r in available],[r[key] for r in available],'o-',color=COLORS[2])
        else:ax.text(.5,.5,'Metric not present in saved history',ha='center',transform=ax.transAxes)
        ax.set_xlabel('Identification physical seconds');ax.set_ylabel(label)
    axes[1].axhline(.10,color='gray',linestyle='--',label='Common threshold0.10');axes[1].legend(fontsize=7)
    available=[r for r in history if r.get('residual_score') is not None]
    axes[2].plot([r['elapsed_s'] for r in available],[r['residual_score'] for r in available],'o-',color=COLORS[3])
    axes[2].axhline(4.,color='red',linestyle='--',label='Consistency limit4')
    axes[2].set_xlabel('Identification physical seconds');axes[2].set_ylabel('Standardized fit residual score');axes[2].legend(fontsize=7)
    axes[2].set_title('Residual consistency gates continuation',fontsize=8)
    fig.suptitle(('Case '+str(example['case_id'])+'; seed '+str(example['noise_seed'])) if example else 'No completed mixed-suite history')
    fig.tight_layout();fig.savefig(folder/'03_sequential_uncertainty.pdf');fig.savefig(folder/'03_sequential_uncertainty.png',dpi=140);plt.close(fig)
    _tex_plot(folder/'03_sequential_uncertainty.tex','03_sequential_uncertainty.csv','elapsed_s',['mass_std'],xlabel='Physical seconds',ylabel='Mass posterior SD (kg)')
    _tex_plot(folder/'03c_residual_consistency.tex','03_sequential_uncertainty.csv','elapsed_s',['residual_score'],xlabel='Physical seconds',ylabel='Residual consistency score (limit4)')
    source_example=root/'source_validation/source_sequential_illustration.json'
    if source_example.exists():
        source_record=json.loads(source_example.read_text())
        source_history=history_table(source_record)
        for row,h in zip(source_history,source_record['history']):
            row['source_only_illustration']=True
            row['inertia_sd_norm']=float(np.linalg.norm([row[k+'_std'] for k in ('Ixx','Iyy','Izz','Ixy','Ixz','Iyz')]))
            row['settled_only_inertia_rank']=h['regressor_analysis']['settled_only_inertia_columns']['rank']
            row['all_inertia_rank']=h['regressor_analysis']['all_inertia_columns']['rank']
        dump_csv(folder/'03b_source_full_library_illustration.csv',source_history)
        fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        for axis,color in zip('xyz',COLORS):
            axes[0].plot([r['elapsed_s'] for r in source_history],[r['com_'+axis+'_std_mm'] for r in source_history],'o-',color=color,label=axis)
        axes[0].set_ylabel('COM approximate posterior SD (mm)');axes[0].legend()
        axes[1].plot([r['elapsed_s'] for r in source_history],[r['inertia_sd_norm'] for r in source_history],'o-')
        axes[1].set_ylabel('Norm of six inertia SDs (kg m²)')
        axes[2].plot([r['elapsed_s'] for r in source_history],[r['workload_uncertainty_fraction'] for r in source_history],'o-')
        axes[2].set_ylabel('Joint99% task-torque width / limit')
        for ax in axes:ax.set_xlabel('Existing source physical seconds')
        fig.suptitle('Source-only full-library illustration (90 s); not a formal online test method')
        fig.tight_layout();fig.savefig(folder/'03b_source_full_library_illustration.pdf');fig.savefig(folder/'03b_source_full_library_illustration.png',dpi=140);plt.close(fig)
        _tex_plot(folder/'03b_source_full_library_illustration.tex','03b_source_full_library_illustration.csv','elapsed_s',
                  ['com_x_std_mm','com_y_std_mm','com_z_std_mm'],['x','y','z'],xlabel='Source physical seconds',ylabel='COM SD (mm)')
    motion=[]
    for i,method in enumerate(METHODS):
        for group_index,group in enumerate(('G0','G1','G2','G3')):
            sub=[r for r in rows if r['method_id']==method and r['group']==group]
            motion.append({'method_index':i,'group_index':group_index,'group':group,
                           'position_rms_m':mean(r.get('position_rms_m') for r in sub),
                           'position_peak_m':mean(r.get('position_peak_m') for r in sub),
                           'config_mass_abs_error_kg':mean(r.get('config_mass_abs_error_kg') for r in sub),
                           'config_com_error_mm':mean(r.get('config_com_error_mm') for r in sub)})
    dump_csv(folder/'04_configuration_and_motion.csv',motion)
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    for group_index,group in enumerate(('G0','G1','G2','G3')):
        sub=[r for r in motion if r['group']==group]
        for ax,key in [(axes[0],'position_rms_m'),(axes[1],'config_com_error_mm')]:
            vals=[number(r[key]) for r in sub];ax.plot(range(len(sub)),vals,'o-',label=group)
    for ax in axes:ax.set_xticks(range(6),LABELS,rotation=45,ha='right');ax.legend()
    axes[0].set_ylabel('Verification position RMS (m)');axes[1].set_ylabel('Final configured COM error (mm)')
    fig.tight_layout();fig.savefig(folder/'04_configuration_and_motion.pdf');fig.savefig(folder/'04_configuration_and_motion.png',dpi=140);plt.close(fig)
    _tex_plot(folder/'04_configuration_and_motion.tex','04_configuration_and_motion.csv','method_index',['position_rms_m'],xlabel='Method index',ylabel='Position RMS (m)')
    physical_effect_figure(root,plt)
    key=(example['case_id'],example['noise_seed']) if example else None
    timelines=[]
    for r in raw:
        if key and (r['case_id'],r['noise_seed'])==key and r['method_id'] in METHODS[:4]:
            for h in history_table(r):timelines.append(dict(h,method_id=r['method_id'],method_index=METHODS.index(r['method_id'])))
    dump_csv(folder/'05_action_timelines.csv',timelines,fields=None if timelines else ['method_id','method_index','action','elapsed_s','duration_s'])
    fig,ax=plt.subplots(figsize=(11,3.5))
    for r in timelines:
        i=r['method_index'];start=r['elapsed_s']-r['duration_s']
        ax.broken_barh([(start,r['duration_s'])],(i-.3,.6),facecolors=COLORS[i],alpha=.7,edgecolors='white',linewidth=1.5)
        ax.text(start+r['duration_s']/2,i,r['action'],ha='center',va='center',fontsize=7)
    ax.set_yticks(range(4),LABELS[:4]);ax.set_xlabel('Actual identification physical seconds, including motion and return')
    ax.set_title('Same case and seed: '+str(key));fig.tight_layout();fig.savefig(folder/'05_action_timelines.pdf');fig.savefig(folder/'05_action_timelines.png',dpi=140);plt.close(fig)
    (folder/'05_action_timelines.tex').write_text(r'''\documentclass[tikz,border=2pt]{standalone}
\usepackage{pgfplots,pgfplotstable}\pgfplotsset{compat=1.18}
\begin{document}\begin{tikzpicture}\begin{axis}[width=14cm,height=5cm,xlabel={Elapsed physical seconds},ylabel={Method index}]
\addplot[only marks] table[x=elapsed_s,y=method_index,col sep=comma]{05_action_timelines.csv};
\end{axis}\end{tikzpicture}\end{document}
''')
    write_json(folder/'rendering.json',{'pdf_backend':'matplotlib vector PDF','standalone_tex_sources':True,
               'latex_compilation':'not invoked; standalone sources supplied','uncertainty_example':key,
               'missing_metrics_are_not_imputed':True})


def findings(root, raw, rows, summaries, checks):
    lookup={s['method_id']:s for s in summaries}
    by_method={m:[r for r in rows if r['method_id']==m] for m in METHODS}
    raw_by_method={m:[r for r in raw if r['method_id']==m] for m in METHODS}
    original=by_method['original_task_once'];fixed=by_method['fixed_four_pose'];adaptive=by_method['adaptive_task_checks']
    keep=by_method['keep_existing_settings'];truth=by_method['true_payload_settings_reference']
    more=[r for r in raw_by_method['fixed_four_pose'] if len(r['history'])>1]
    physical_more=len({r['case_id'] for r in more})
    all_same=all(next(x for x in raw_by_method['fixed_four_pose'] if (x['case_id'],x['noise_seed'])==(r['case_id'],r['noise_seed']))['selected_test_ids']==r['selected_test_ids'] for r in raw_by_method['adaptive_task_checks'])
    l0=raw_by_method['original_task_once']
    l0pass=sum(r['history'][0]['private_metrics']['pass'] for r in l0)
    l0peak=sum(r['history'][0]['private_metrics']['peak_error_m']>.05 for r in l0)
    l0sat=sum(r['history'][0]['private_metrics']['saturation_fraction']>.01 for r in l0)
    updates={group:sum(r['setting_updated'] for r in original if r['group']==group) for group in ('G0','G1','G2','G3')}
    threshold=json.loads((root/'source_validation/threshold_selection.json').read_text())
    dependence=json.loads((root/'checks/actual_source_reading_dependence.json').read_text())
    text=['# 负载变化调试实验结果','',
          '正式矩阵完整：4个主方法与2个参考，共3456次运行。每方法192个物理案例、3个测量种子，共576次；每次8条私有工作审计。',
          '',f"**最重要的结果：保持原设置参考 {sum(r['correct_full_release'] for r in keep)}/{len(keep)} 在声明的原速度工作中正确放行。** 本轮负载变化确实改变物理力矩和误差，但这些变化没有使该工作范围内的旧设置失去可用性。参数辨识与更新提高部分数值精度，却不能据此声称修复了原本无法工作的机器人。",'',
          f"L0一次记录的方法正确放行 {sum(r['correct_full_release'] for r in original)}/{len(original)}；固定四姿态、固定混合和自适应均 {len(fixed)}/{len(fixed)}。额外检查只发生在 {physical_more} 个物理案例的 {len(more)} 个噪声运行。固定和自适应实际动作序列{'完全相同' if all_same else '存在差异'}，没有观察到自适应节省物理测试时间。",'',
          '**任务范围必须区分：** L0严格保留旧三点任务的100ms参考过渡；最终验证/审计采用预先声明的平滑10s工作轨迹（近处移动、较远伸展、工具转向），speed=1.0相对于这份工作定义。通过后者不代表恢复了旧100ms过渡的全部约束。',
          f"测试中L0本身只 {l0pass}/{len(l0)} 满足全部运动验收；{l0peak}/{len(l0)} 峰值位置误差超过5cm，{l0sat}/{len(l0)} 饱和占比超过1%（可重叠）。L0仍可提供辨识读数；不得把采证有用写成该轨迹已经验收通过。",'',
          '| 方法 | 原速度正确放行 | 半速正确放行 | 错误/放行 | 错误/全部 | 未解决 | 辨识秒 | 复测秒 | 更新次数 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        s=lookup[method];n=s['runs'];update_n=sum(r['setting_updated'] for r in by_method[method])
        text.append(f"| {method} | {s['correct_full_release']}/{n} | {s['correct_half_release']}/{n} | {s['false_release_per_released']} | {s['false_release_per_all']} | {s['unresolved']}/{n} | {s['mean_physical_test_seconds']:.3f} | {s['mean_public_verification_seconds']:.3f} | {update_n}/{n} |")
    text.extend(['','每方法私有任务通过数为 '+', '.join(m+': '+str(lookup[m]['audit_task_pass_count'])+'/'+str(lookup[m]['audit_task_total']) for m in METHODS)+'。',
                 '“8条全部通过”只是本轮审计结果。零观测误放行不构成连续工作空间安全保证，也不是独立576件物体的风险证明。',
                 'G0/G1/G2/G3、家族、质量和COM变化幅度分别在 results/grouped_by_change.csv、grouped_strata.csv 中给出。','',
                 '## 参数、预测与停止依据',''])
    for method in METHODS[:4]:
        group=by_method[method];s=lookup[method]
        mass_cover=sum(bool(r.get('mass_interval_covers_truth')) for r in group)
        com_cover=sum(bool(r.get('com_interval_covers_truth')) for r in group)
        text.append(f"- {method}：质量绝对误差均值 {s['mean_estimate_mass_abs_error_kg']:.6f} kg，COM欧氏误差 {s['mean_estimate_com_error_mm']:.3f} mm；质量边际99%区间覆盖 {mass_cover}/{len(group)}，三个COM坐标边际区间交集覆盖 {com_cover}/{len(group)}；独立公共复测力矩预测RMSE {s['mean_heldout_torque_prediction_rmse_Nm']:.6f} Nm。")
    text.extend(['','质量/COM区间用边际正态z=2.575829；COM三坐标交集不是同时99%的置信区域。停止判断使用联合10维近似99%椭球（半径4.817598）在公开工作网格投影后的最大力矩全宽，不能混称逐点99%。source标定协方差膨胀12.2819；上述test覆盖独立报告，未再调参。',
                 f"三个阈值在64个validation物理案例、种子7、四方法上比较；选定共同阈值 {threshold['selected_threshold']:.2f}。候选正确放行数为 "+', '.join(f"{c['threshold']:.2f}: {c['correct_releases']}/{c['runs']}" for c in threshold['candidates'])+'，候选均无观测误放行。选择发生在test之前。','',
                 '## 原因分析（12项）','',
        f"1. **L0已经解决绝大多数判定。** {sum(r['correct_full_release'] for r in original)}/{len(original)} 正确放行，但这不等于所有惯量分量均被准确测定。L0运动可提供非零惯性信息；质量/COM误差与区间覆盖见上表。L0自身严格运动通过仅{l0pass}/{len(l0)}，与平滑新任务通过分开。",
        f"2. **原设置本来就能完成本轮工作。** 保持设置参考G0/G1/G2/G3各144/144，全部576/576正确放行；其中负载变化的G1/G2/G3为432/432。G2/G3不因标签为stale就自动记成故障。正确参数参考也全部通过，因此没有观察到“原配置不行、更新后修好”的必要性证据。",
        f"3. **补姿态主要解决残差不一致。** {physical_more}个G3案例、{len(more)}次运行在L0后未放行。其任务力矩区间宽度约0.0319，已经低于0.10；阻止停止的是标准化残差超过共同上限4。S5以及必要时S6改变工具方向，使COM估计更接近实际值并降低残差。没有另跑重复L0科学对照，不能声称已实测“多姿态优于增加同方向独立样本”；事件复制不增证据由工程检查验证。",
        '4. **本轮正式策略没有需要动态补测。** 原L0本身包含快速运动；6个补测运行只用了静态S5/S6。静止观测的惯量列为零及未观测方向保持不确定由工程检查验证；额外动态块的信息变化仅用独立标明的source全库顺序重放展示，不冒充正式test的必要收益。',
        f"5. **强固定方法与自适应在本轮相同。** 固定四姿态、固定混合、自适应的逐案例动作与放行结果{'完全一致' if all_same else '并非完全相同'}。三者均570次L0后停、3次追加S5、3次追加S5与S6，平均在线物理20.125s。配对bootstrap先在192物理案例内平均三种子，再按8家族和案例重采样；区间和逐家族方向见paired_comparisons.csv、family_directions.csv。",
        '6. **没有观测到自适应专属省时。** 三种可继续方法相对L0方法平均多0.125物理秒，解决6次未解决判定。固定方法同样早停；不能把自适应实际20.125s与固定计划假定全部42/58s比较为节省。完整固定计划成本仅作计划上界，非实际发生成本。',
        f"7. **选择函数可依赖读数，但不等于部署优势。** 冻结先验/配置/设计的192个实际source L0记录替换检查，假设忽略STOP继续时动作分布为 {dependence['action_counts']}；理由为 {dependence['reason_counts']}。此检查包括测得运动导致的协方差变化，不能证明均值驱动的信息增益更好。正式6次继续与固定顺序一致。",
        f"8. **真参数仍不能完成声明原速度工作的例子未出现。** 真实参数参考 {sum(r['correct_full_release'] for r in truth)}/{len(truth)} 正确放行、4608/4608私有任务通过；无半速必要性证据。旧L0的快速过渡失败另列，不能替代本项同工作范围的失败例子。",
        '9. **本轮未解决来自估计器残差门槛，而非实际工作失败。** 原L0方法6次未解决的候选配置离线8条审计仍全部通过。原设置参考同样可行；额外姿态使估计一致性达标。早期source工程的高增益/力矩速率限制过冲和重启bias错误已在test前修正并完整保存，不能归因于选择器。',
        '10. **没有数据支持此范围内增加神经选择器。** 若目的只是确认本轮声明平滑工作是否可执行，保持现配置加独立公共复测已达到相同观察放行覆盖、只需10s。若需要同时估计负载参数，L0加共同停止与固定S5/S6补测足够；结论限于该负载/工作/传感范围，不外推到更激烈或接触任务。',
        f"11. **结果没有靠降速或大量拒绝换取。** 全方法半速放行0；三个可继续主方法未解决0，L0仅6/576。共同0.10阈值在validation选定，test未改变。主方法更新38/576，其中 {updates}；G0/G1无无故更新。38次更新依据5%预计力矩影响，不能称为38次必要维修，因为旧设置的实际工作仍全部可行。",
        '12. **成本来自实际积分运动。** L0为10s，S5/S6各8s，独立公共复测10s；进入、停稳、返回均计费。仿真墙钟、拟合/选择CPU、设置等待另列。8条私有审计属于离线计算，不算在线已花测试时间。拆分API不能制造额外物理代价。',''])
    mechanism_path=root/'source_validation/repeat_vs_pose_mechanism.json'
    if mechanism_path.exists():
        mechanism=json.loads(mechanism_path.read_text());summary=mechanism['summary'];directions=mechanism['paired_direction_counts_out_of_12']
        q3=(f"3. **补姿态解决实际test的残差问题，并有单独source机制对照。** 正式{physical_more}个G3案例、{len(more)}次运行，L0任务区间约0.0319已经低于0.10，继续由残差>4触发。另在12个source家族各按固定hash选1例，先实跑L0+S1，再从完整相同状态克隆，分别真实执行新L0(10s)、S5(8s)、D4(8s)，新增{mechanism['total_physical_seconds']:.0f}物理秒。S5对比重复L0，每秒COM方差迹下降更好{directions['new_pose_S5_vs_repeat_L0']['COM_trace_reduction_per_second']}/12，每秒任务力矩方差下降更好{directions['new_pose_S5_vs_repeat_L0']['task_torque_variance_reduction_per_second']}/12。此为冻结后SOURCE强制继续机制分析，不是新增正式策略或预算优化。")
        q4=(f"4. **不能说缓和动态块创造新可辨识方向或本轮必须有动态补测。** 正式继续只用了S5/S6；source对照的共同前缀已有rank10，L0自身快速运动已提供惯性激励。D4对比重复L0的每秒惯量协方差迹下降仅{directions['new_dynamic_D4_vs_repeat_L0']['inertia_covariance_trace_reduction_per_second']}/12更好，均值{summary['new_dynamic_D4']['inertia_covariance_trace_reduction_per_second']['mean']:.3g}对{summary['repeat_L0']['inertia_covariance_trace_reduction_per_second']['mean']:.3g}，不能据此主张动态必需。静态S5块包含进入/返回运动，不能把整个8秒块等同纯静止观测；纯停稳行惯量列为零。")
        text=[q3 if paragraph.startswith('3. **') else q4 if paragraph.startswith('4. **') else paragraph for paragraph in text]
        text.extend(['## SOURCE重复任务/新姿态/动态机制对照（事后强制继续）','',
                     '| 分支 | 真实分支秒数/例 | COM方差迹下降/秒均值 | 任务力矩方差下降/秒均值 | 运动验收通过 |',
                     '|---|---:|---:|---:|---:|'])
        mechanism_rows=[]
        for label,seconds in [('repeat_L0',10),('new_pose_S5',8),('new_dynamic_D4',8)]:
            m=summary[label]
            text.append(f"| {label} | {seconds} | {m['COM_trace_reduction_per_second']['mean']:.6g} | {m['task_torque_variance_reduction_per_second']['mean']:.6g} | {m['motion_passes']}/{m['cases']} |")
        for case in mechanism['cases']:
            for branch,data in case['branches'].items():
                mechanism_rows.append({'case_id':case['case_id'],'family_id':case['family_id'],'branch':branch,
                    'scope':'posthoc source forced continuation; not a formal test method',
                    **{key:value for key,value in data.items() if not isinstance(value,dict)}})
        dump_csv(root/'results/source_repeat_pose_dynamic.csv',mechanism_rows)
        text.extend(['','预定首例e2a55e61111f76dd843f中，最不确定COM主方向为hand-frame[-0.489,0.195,-0.850]（z为主并混合x）；S5每秒此方向方差下降1.687e-6m²/s，重复L0为8.488e-7m²/s。共同前缀COM标准差[x,y,z]=[3.927,3.355,4.701]mm，重复L0后[3.286,2.778,3.860]mm，S5后[2.951,2.166,3.097]mm。',
                     '这些是冻结近似估计器的方差变化，不是精确贝叶斯信息；受物理投影影响，个别惯量方差增量可以为负。12例不代表新的held-out性能分母。完整配对方向、原始记录与脚本见source_validation/repeat_vs_pose_mechanism.json、results/source_repeat_pose_dynamic.csv和commands/source_repeat_vs_pose.py。',''])
    examples=[]
    good=next((r for r in original if r['correct_full_release'] and r.get('mass_interval_covers_truth') and r.get('com_interval_covers_truth')),None)
    examples.append({'type':'simple_sufficient','record':good,'absence':'在本轮未出现' if good is None else None})
    improved=next((r for r in sorted(more,key=lambda x:(-len(x['history']),x['case_id'],x['noise_seed']))),None)
    examples.append({'type':'multipose_resolves_residual_inconsistency','record':improved,'absence':'在本轮未出现' if improved is None else None})
    examples.append({'type':'true_parameters_insufficient_at_full_work_speed','record':None,'absence':'在本轮声明工作范围未出现；真实参数参考全部原速度通过。未执行不必要的半速反事实。'})
    write_json(root/'examples/selected_cases.json',examples)
    text.extend(['## 具体案例',''])
    if good:
        text.append(f"**L0已足够：** case={good['case_id']}，seed={good['noise_seed']}；估计质量误差{good['estimate_mass_abs_error_kg']:.6f}kg、COM误差{good['estimate_com_error_mm']:.3f}mm，质量和COM坐标区间覆盖实际值，公共复测与8/8私有审计通过。")
    if improved:
        true=np.asarray(improved['true_payload']['com'])*1000
        text.extend(['',f"**多姿态使残差一致：** case={improved['case_id']}，seed={improved['noise_seed']}，G3，真COM={true.tolist()}mm。",'',
                     '| 前缀 | COM估计 x/y/z mm | COM标准差 x/y/z mm | 标准化残差 | 任务区间/力矩限制 |',
                     '|---|---|---|---:|---:|'])
        for h in improved['history']:
            e=h['estimate'];a=h['assessments_by_threshold'][str(improved['threshold'])]
            cm=', '.join(f'{v*1000:.3f}' for v in e['payload_estimate']['com']);sd=', '.join(f'{v*1000:.3f}' for v in e['com_sd'])
            text.append(f"| {h['test_id']} | {cm} | {sd} | {e['residual_score']:.3f} | {a['torque_interval_ratio']:.5f} |")
        text.extend(['','在此案例，L0把x方向第一质量矩对应的COM估到了负侧；S5/S6的不同姿态使估计转回真值所在正侧，y/z误差也减少。最初区间判据已满足，实际阻碍是残差一致性。最终配置即使保持原设置也能完成声明工作，所以这是“证据质量改善”，不是“修复不可执行任务”的例子。'])
    text.extend(['','**已知参数仍无法完成声明原速度工作：在本轮未出现。** 不用早期错误增益的工程失败或不同的L0瞬态条件冒充该例子。',
                 '', '包内examples/traces含少量完整100Hz公开传感/软件端口CSV.gz与估计均值、协方差、奇异值JSON，GPT Pro可直接分析。source全库和validation0.02示例明确标记，不能当作选定0.10的正式测试方法。所有完整原始记录含失败仍在服务器，带SHA256索引。'])
    (root/'FINDINGS_CN.md').write_text('\n'.join(text)+'\n')

def report(run_root, strict=True):
    root=Path(run_root);result=root/'results';result.mkdir(exist_ok=True)
    raw=read_runs(root);private=truth_index(root,'test')
    rows=[normalize(r,private) for r in raw]
    workload_coverage(root,raw,rows,private)
    check=completeness(rows);write_json(root/'checks/report_completeness.json',check)
    summaries=[summarize([r for r in rows if r['method_id']==m],{'method_id':m}) for m in METHODS
               if any(r['method_id']==m for r in rows)]
    dump_csv(result/'per_case.csv',rows)
    dump_csv(result/'summary.csv',summaries)
    dump_csv(result/'split_residuals.csv',split_residual_table(root,raw))
    audit_rows=[]
    for record in raw:
        for audit in record.get('private_audit_results',[]):
            audit_rows.append({'case_id':record['case_id'],'object_family_id':record.get('family_id'),
                               'noise_seed':record['noise_seed'],'method_id':record['method_id'],
                               'group':record.get('group'),'final_speed':record['final_speed'],
                               'task_index':audit['task_index'],'physical_seconds':audit['physical_seconds'],
                               'paths':audit.get('paths'),**audit['metrics']})
    dump_csv(result/'private_audit_tasks.csv',audit_rows)
    grouped=[]
    for method in METHODS:
        for group in ('G0','G1','G2','G3'):
            sub=[r for r in rows if r['method_id']==method and r['group']==group]
            if sub:grouped.append(summarize(sub,{'method_id':method,'group':group}))
    dump_csv(result/'grouped_by_change.csv',grouped)
    dump_csv(result/'per_seed.csv',[summarize([r for r in rows if r['method_id']==method and r['noise_seed']==seed],
                                             {'method_id':method,'noise_seed':seed})
                                   for method in METHODS for seed in (11,23,37)
                                   if any(r['method_id']==method and r['noise_seed']==seed for r in rows)])
    strata=[]
    for method in METHODS:
        subset=[r for r in rows if r['method_id']==method]
        for field in ('object_family_id','true_mass_bin','com_change_bin'):
            bins=defaultdict(list)
            for row in subset:
                value=row.get('object_family_id') if field=='object_family_id' else (
                    ('0.2–0.6' if number(row.get('true_mass_kg'))<.6 else '0.6–1.0' if number(row.get('true_mass_kg'))<1. else '1.0–1.5')
                    if field=='true_mass_bin' and np.isfinite(number(row.get('true_mass_kg'))) else
                    ('0–15' if number(row.get('com_change_mm'))<15 else '15–30' if number(row.get('com_change_mm'))<30 else '30–60')
                    if field=='com_change_bin' and np.isfinite(number(row.get('com_change_mm'))) else 'unknown')
                bins[value].append(row)
            for value,sub in bins.items():strata.append(summarize(sub,{'method_id':method,'stratum':field,'value':value}))
    dump_csv(result/'grouped_strata.csv',strata)
    common=['case_id','object_family_id','noise_seed','method_id','group']
    for name,fields in [
        ('parameter_errors',common+['estimate_mass_abs_error_kg','estimate_com_error_mm','estimate_inertia_error_kg_m2',
         'config_mass_abs_error_kg','config_com_error_mm','config_inertia_error_kg_m2','mass_interval_covers_truth','com_interval_covers_truth','regressor_rank','regressor_min_singular_value']),
        ('prediction_errors',common+['heldout_torque_prediction_rmse_Nm','workload_torque_interval_coverage',
         'workload_torque_interval_all_grid_points_cover','workload_grid_payload_prediction_rmse_Nm',
         'residual_rms_Nm','residual_score','last_block_residual_score']),
        ('task_outcomes',common+['released_status','final_speed','public_verification_pass','audit_pass_count','audit_total','all_audits_pass',
         'position_rms_m','position_peak_m','settled_position_error_m','saturation_seconds','unexpected_contacts','hard_abort','torque_peak_Nm','velocity_peak_rad_s']),
        ('resource_breakdown',common+['physical_test_seconds','public_verification_seconds','online_physical_seconds','fitting_cpu_seconds','selection_cpu_seconds',
         'setting_wait_seconds','serial_deployment_seconds','recorded_bytes','identification_recorded_bytes',
         'verification_recorded_bytes','online_compressed_public_record_bytes','recorded_bytes_definition',
         'joint_path_length_rad','virtual_model_calculations','setting_updated'])]:
        dump_csv(result/(name+'.csv'),rows,fields)
    paired,directions=paired_bootstrap(rows)
    dump_csv(result/'paired_comparisons.csv',paired);dump_csv(result/'family_directions.csv',directions)
    truth_lookup={(r['case_id'],r['noise_seed']):r for r in rows if r['method_id']=='true_payload_settings_reference'}
    opportunities=[]
    for method in METHODS[:4]:
        sub=[r for r in rows if r['method_id']==method and (r['case_id'],r['noise_seed']) in truth_lookup]
        possible=[r for r in sub if truth_lookup[(r['case_id'],r['noise_seed'])]['all_audits_pass']]
        opportunities.append({'method_id':method,'true_reference_feasible_runs':len(possible),
                              'not_released_despite_feasible_reference':sum(not r['released'] for r in possible),
                              'paired_runs':len(sub)})
    dump_csv(result/'missed_feasible_release.csv',opportunities)
    all_fields=list(dict.fromkeys(k for r in rows for k in r))
    write_json(result/'metric_availability.json',{field:{'present':sum(r.get(field) is not None for r in rows),'total':len(rows)} for field in all_fields})
    if rows:
        figures(root,raw,rows,summaries);findings(root,raw,rows,summaries,check)
        export_example_traces(root,raw)
        execution_documents(root,summaries)
    write_json(result/'report_status.json',{'completed_matrix':check['passed'],'rows':len(rows),'methods':len(summaries),
               'bootstrap_unit':'physical case means across noise seeds, then paired family/case resampling',
               'private_truth_read_only_offline':True,'strict':strict})
    if strict and not check['passed']:
        raise RuntimeError('Formal report incomplete: '+ '; '.join(check['failures'][:12]))
    return {'rows':len(rows),'complete':check['passed'],'summary_path':str(result/'summary.csv')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',required=True);parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args();print(json.dumps(report(args.run_root,strict=not args.allow_incomplete)))


if __name__=='__main__':main()
