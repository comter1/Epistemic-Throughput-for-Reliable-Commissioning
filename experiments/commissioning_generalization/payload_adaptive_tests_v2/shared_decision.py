"""Equal update, work assessment, verification, and stopping boundary."""
import copy
import numpy as np
from payload_change_v1.estimator import payload_to_pi
from .protocol import digest
from .trajectories import work
from .prediction import DecisionTimeout
from .motion_screen import secondary_check

def deployment_choices(predictor,initial,estimated,start,mean,cov,clock,consistent=True,all_choices=False):
    choices=[]; seen=set(); eligible=None; errors=[]
    for speed in (1.,.5):
        for name,config in [('keep_initial',initial),('use_estimate',estimated)]:
            key=(tuple(np.round(payload_to_pi(config),14)),speed)
            if key in seen: continue
            seen.add(key); action=work(predictor.context,speed)
            clock.check(); kernel=predictor.kernel(config,action,start,clock)
            prediction=predictor.summary(kernel,mean,cov)
            row=dict(name=name,speed=speed,configured=copy.deepcopy(config),action=action,kernel=kernel,prediction=prediction)
            row['eligible']=bool(consistent and prediction['work_pass'])
            choices.append(row)
            if row['eligible'] and eligible is None:
                eligible=row
                if not all_choices: return choices,eligible,errors
    # Saturation/nonlinear extrapolation is not silently approved by a linear model.
    # At most one selected ambiguous alternative gets a secondary check this call.
    if eligible is None and consistent:
        ambiguous=[c for c in choices if not c['prediction']['linear_supported']]
        if ambiguous:
            candidate=ambiguous[0]
            try:
                okay,rows=secondary_check(predictor,candidate,start,mean,cov,clock,True)
                candidate['secondary']=rows
                p=candidate['prediction']
                linear_agrees=(p['rms_m']<=.005 and p['peak_m']<=.010 and p['hold_m']<=.002
                    and p['clearance_m']>0 and np.all(np.asarray(p['torque_Nm'])<np.array([87]*4+[12]*3)))
                if okay and linear_agrees:
                    candidate['eligible']=True; eligible=candidate
                else: errors.append('secondary_disagreement_or_linear_screen_unresolved')
            except DecisionTimeout: errors.append('secondary_prediction_timeout')
    return choices,eligible,errors

def public_verification_pass(block):
    return bool(block.private['public_metrics']['pass'])

def can_update_at_rest(public_log):
    return bool(len(public_log['dq'])>=20 and np.max(np.abs(public_log['dq'][-20:]))<.02)

def serialize_choice(choice):
    return {k:v for k,v in choice.items() if k!='kernel'}
