"""One admissibility/speed wrapper shared by every selector."""
import numpy as np
from payload_change_v1.estimator import whiten_features
from .prediction import DecisionTimeout
from .workstations import kinematic
from .protocol import digest

def class_alarms(estimator,history_kinds):
    mean,_,_=estimator.posterior(); values={}
    for execution,features in estimator._features.items():
        h,y=whiten_features(features,estimator.parameters)
        kind=history_kinds.get(execution,'unknown')
        cls='static' if kind in ('initial','static') else kind
        if len(y): values[cls]=max(values.get(cls,0.),float(np.mean((y-h@mean)**2)))
    threshold=estimator.parameters['residual_consistency_max']
    return {k:dict(score=v,threshold=threshold,blocked=v>threshold) for k,v in values.items()}

def candidates(predictor,bank,configured,start,mean,cov,remaining,alarms,clock):
    admitted=[]; screens=[]
    for base in sorted({v['action']['base_id'] for v in bank['actions'].values()}):
        for speed in (1.,.75,.5):
            row=bank['actions'][f'{base}@{speed}']; a=row['action']
            screen=dict(base_id=base,speed=speed,duration=a['duration'],admitted=False)
            if a['duration']>remaining+1e-8:
                screens.append(screen|dict(reason='remaining_identification_time')); continue
            cls=a['kind']
            if alarms.get(cls,{}).get('blocked',False):
                screens.append(screen|dict(reason='motion_class_residual_alarm',alarm=alarms[cls])); continue
            try:
                clock.check()
                # Configuration does not affect collision geometry. The key records
                # the exact measured entry state and public exterior nevertheless.
                key=('kinematic',digest([predictor.geometry,predictor.context,a,np.asarray(start).tolist()]))
                if key not in predictor.memory:
                    predictor.memory[key]=kinematic(predictor.geometry,predictor.context,a,start)
                geom=predictor.memory[key]
                if not geom['admissible']:
                    screens.append(screen|dict(reason='public_kinematics',geometry=geom)); continue
                kernel=predictor.kernel(configured,a,start,clock)
                prediction=predictor.summary(kernel,mean,cov)
                screen.update(prediction=prediction,geometry=geom)
                if prediction['motion_pass']:
                    screen.update(admitted=True,reason='joint_ellipsoid_tracking_and_torque_screen')
                    admitted.append(dict(a,information17=row['information17'],prediction=prediction,configured=configured))
                    screens.append(screen); break
                screen['reason']='uncertain_or_unsupported_motion'
                screens.append(screen)
            except DecisionTimeout:
                screens.append(screen|dict(reason='prediction_budget_unresolved'))
                return admitted,screens,True
    return admitted,screens,False

def secondary_check(predictor,choice,start,mean,cov,clock,work_requirement):
    """Only the selected ambiguous operation receives the 17-model check."""
    rows=predictor.secondary(choice['configured'],choice['action'],start,mean,cov,clock)
    m=predictor.margins
    checks=[]
    for r in rows:
        okay=not r['collision'] and r['clearance_m']>m['clearance_m'] and np.all(np.asarray(r['torque_Nm'])+m['torque_Nm']<np.array([87]*4+[12]*3))
        if work_requirement:
            okay=okay and r['rms_m']+m['rms_m']<=.005 and r['peak_m']+m['peak_m']<=.010 and r['hold_m']+m['hold_m']<=.002
        checks.append(bool(okay))
    return bool(all(checks)),rows
