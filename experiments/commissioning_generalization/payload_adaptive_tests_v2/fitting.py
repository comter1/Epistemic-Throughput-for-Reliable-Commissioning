"""Source fitting and disjoint-object calibration; no validation/test fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from payload_change_v1.estimator import (fit_parameters,source_coverage_record,fit_uncertainty_inflation,
    PayloadEvidenceEstimator,payload_to_pi,SCALE)
from .protocol import write_json, digest
from .worker import context
from .prediction import PublicPredictionModel

def logs_for(root,case):
    record=json.loads((Path(root)/'jobs/source'/(case['case_id']+'.json')).read_text())['result']
    blocks=[]
    for row in record['blocks']:
        if 'path' not in row: continue
        p=Path(root)/row['path']; meta=json.loads((p/'metadata.json').read_text())
        with np.load(p/'public.npz') as z:
            log={k:z[k] for k in ('q','dq','torque','time','event_id','unexpected_contact_flags')}
        log['execution_id']=meta['execution_id']; blocks.append((log,meta,row))
    return blocks

def initial_fit(root):
    root=Path(root); records=[]
    for p in (root/'jobs/source').glob('*.json'):
        row=json.loads(p.read_text())['result']
        if row['object_index']<6: records.append(row['calibration'])
    pars=fit_parameters(records)
    pars['fit_partition']='source objects with index 0..5; all three contexts retained together'
    write_json(root/'configs/estimator_base.json',pars)
    return pars

def fit_case(root,case):
    root=Path(root); pars=json.loads((root/'configs/estimator_base.json').read_text())
    ctx,_=context(root,case); blocks=logs_for(root,case)
    estimator=PayloadEvidenceEstimator(pars,geometry=case['configured_payload'])
    predictor=PublicPredictionModel(root,ctx,case['configured_payload'])
    coverage=source_coverage_record([x[0] for x in blocks],case['true_payload'],pars)
    pred=[]; true_pi=payload_to_pi(case['true_payload'])
    for i,(log,meta,row) in enumerate(blocks):
        mean,cov,_=estimator.posterior()
        k=predictor.kernel(meta['configured_payload'],meta['action'],meta['measured_start_reference'])
        expected=predictor.summary(k,true_pi,np.zeros((10,10)))
        public=predictor.summary(k,mean,cov)
        actual=meta['private_metrics']
        pred.append(dict(case_id=case['case_id'],object_id=case['object_id'],object_index=case['object_index'],
            context=case['context'],block=i,motion_kind=meta['action']['kind'],speed=meta['action']['speed'],
            actual_peak_m=actual['peak_error_m'],actual_rms_m=actual['rms_error_m'],actual_hold_m=actual['final_hold_peak_m'],
            predicted_true_peak_m=expected['mean_peak_m'],predicted_true_rms_m=expected['rms_m'],predicted_true_hold_m=expected['hold_m'],
            predicted_bound_peak_m=public['peak_m'],
            actual_torque_Nm=float(max(actual['peak_requested_Nm'])),predicted_true_torque_Nm=float(max(expected['mean_torque_Nm'])),
            torque_positive_error_Nm=float(max(np.asarray(actual['peak_requested_Nm'])-np.asarray(expected['mean_torque_Nm']))),
            actual_clearance_m=actual['minimum_clearance_m'],predicted_true_clearance_m=expected['mean_clearance_m'],
            parameter_distance=expected['parameter_distance'],hard_abort=actual['hard_abort'],unsupported=expected['unsupported']))
        estimator.ingest(log)
    # The declared job is a separate motion class: validate it explicitly too.
    for j,speed in enumerate((1.,.5)):
        meta=json.loads((root/'raw/source_qualification'/case['case_id']/str(speed)/'metadata.json').read_text())
        k=predictor.kernel(meta['configured_payload'],meta['action'],meta['measured_start_reference'])
        expected=predictor.summary(k,true_pi,np.zeros((10,10)))
        mean,cov,_=estimator.posterior(); public=predictor.summary(k,mean,cov)
        actual=meta['private_metrics']
        pred.append(dict(case_id=case['case_id'],object_id=case['object_id'],object_index=case['object_index'],
            context=case['context'],block=1000+j,motion_kind='work',speed=speed,
            actual_peak_m=actual['peak_error_m'],actual_rms_m=actual['rms_error_m'],actual_hold_m=actual['final_hold_peak_m'],
            predicted_true_peak_m=expected['mean_peak_m'],predicted_true_rms_m=expected['rms_m'],predicted_true_hold_m=expected['hold_m'],
            predicted_bound_peak_m=public['peak_m'],actual_torque_Nm=float(max(actual['peak_requested_Nm'])),
            predicted_true_torque_Nm=float(max(expected['mean_torque_Nm'])),
            torque_positive_error_Nm=float(max(np.asarray(actual['peak_requested_Nm'])-np.asarray(expected['mean_torque_Nm']))),
            actual_clearance_m=actual['minimum_clearance_m'],predicted_true_clearance_m=expected['mean_clearance_m'],
            parameter_distance=expected['parameter_distance'],hard_abort=actual['hard_abort'],unsupported=expected['unsupported']))
    return dict(case_id=case['case_id'],object_index=case['object_index'],context=case['context'],coverage=coverage,prediction=pred)

def finish_fit(root):
    root=Path(root); rows=[json.loads(p.read_text())['result'] for p in (root/'jobs/fit_case').glob('*.json')]
    assert len(rows)==288
    calibration=[c for row in rows if row['object_index']>=6 for c in row['coverage']]
    pars=json.loads((root/'configs/estimator_base.json').read_text())
    calibrated=fit_uncertainty_inflation(calibration,pars)
    # Parent helper returns (parameters, report); accept its explicit return type.
    if isinstance(calibrated,tuple): pars,report=calibrated
    else: pars=calibrated; report={}
    write_json(root/'configs/estimator_parameters.json',pars)
    write_json(root/'checks/uncertainty_calibration.json',report)
    predictions=pd.DataFrame([p for row in rows for p in row['prediction']])
    predictions.to_csv(root/'source_prediction_validation.csv',index=False)
    hold=predictions[predictions.object_index>=6]
    margins=dict(peak_m=max(0.,float((hold.actual_peak_m-hold.predicted_true_peak_m).max())),
        rms_m=max(0.,float((hold.actual_rms_m-hold.predicted_true_rms_m).max())),
        hold_m=max(0.,float((hold.actual_hold_m-hold.predicted_true_hold_m).max())),
        torque_Nm=max(0.,float(hold.torque_positive_error_Nm.max())),
        clearance_m=max(0.,float((hold.predicted_true_clearance_m-hold.actual_clearance_m).max())),
        scaled_linear_radius=float(hold.parameter_distance.max()),validated=True,
        rule='maximum positive actual-minus-first-order residual over disjoint held-out source objects; pooled across contexts/classes/speeds',
        source_objects=24, source_context_cases=72, linear_validation_rows=int((~hold.unsupported).sum()))
    write_json(root/'configs/prediction_margins.json',margins)
    coverage=pd.DataFrame([dict(case_id=row['case_id'],context=row['context'],object_index=row['object_index'],block=i,**c)
                          for row in rows for i,c in enumerate(row['coverage'])])
    coverage['mass_covered_base']=coverage.mass_abs_z<=2.5758293035489004*np.sqrt(pars['posterior_covariance_inflation'])
    coverage['com_covered_base']=coverage.com_max_abs_z<=2.5758293035489004*np.sqrt(pars['posterior_covariance_inflation'])
    coverage['joint_covered_base']=coverage.parameter_mahalanobis_squared<=23.209251158954356*pars['posterior_covariance_inflation']
    # Context/class/speed are aligned by actual acquired prefix order, not planned indices.
    coverage=coverage.merge(predictions[['case_id','block','motion_kind','speed']],on=['case_id','block'])
    coverage.to_csv(root/'source_uncertainty_coverage.csv',index=False)
    coverage.groupby(['context','motion_kind','speed'])[['mass_covered_base','com_covered_base','joint_covered_base']].agg(['mean','count']).to_csv(root/'source_coverage_strata.csv')
    return dict(parameters=pars,margins=margins)
