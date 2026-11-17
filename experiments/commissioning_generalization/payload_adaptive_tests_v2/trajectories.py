"""Public smooth paths, time scaling, and measured-state entry transitions."""
import copy
import numpy as np
from cgx.env.system import HOME
from payload_change_v1.trajectories import action_library, _smooth, reference as v1_reference

def library():
    return [dict(a,base_id=a['id']) for a in action_library()]

def scaled(action,speed):
    a=copy.deepcopy(action); a.update(speed=float(speed),duration=float(np.ceil(float(a.get('base_duration',8.))/speed*100-1e-9)/100))
    a['base_duration']=float(action.get('base_duration',8.))
    return a

def b0(sign=1):
    offset=np.array([0,0,0,0,.15,-.12,0])*sign
    return dict(id='B0' if sign==1 else ('B0_mirror' if sign==-1 else 'B0_hold'),
                base_id='B0',kind='initial',speed=1.,duration=8.,pose=(HOME+offset).tolist())

def reference(action,t,start_q=None):
    start=HOME if start_q is None else np.asarray(start_q)
    speed=float(action.get('speed',1.)); u=t*speed
    if action['kind']=='initial':
        pose=np.asarray(action['pose'])
        if u<2: return _smooth(start,HOME,u,2.)
        if u<4: return _smooth(HOME,pose,u-2,2.)
        if u<5: return pose,np.zeros(7),np.zeros(7)
        if u<7: return _smooth(pose,HOME,u-5,2.)
        return HOME.copy(),np.zeros(7),np.zeros(7)
    if action['kind']=='work':
        return v1_reference(action,t,start)
    q,dq,ddq=v1_reference(dict(action,speed=1.),u,start)
    return q,dq*speed,ddq*speed*speed

def grid(action,start_q=None,hz=100):
    ts=np.arange(round(action['duration']*hz)+1)/hz
    q,dq,ddq=map(np.asarray,zip(*(reference(action,t,start_q) for t in ts)))
    return dict(time=ts,q=q,dq=dq,ddq=ddq)

def work(context,speed=1.,variation_seed=None):
    waypoints=np.asarray(context['waypoints'])
    if variation_seed is not None:
        rng=np.random.default_rng(int(variation_seed))
        waypoints=waypoints+rng.uniform(-.01,.01,waypoints.shape)
    return dict(id='work',kind='work',base_duration=10.,duration=10./speed,
                speed=float(speed),waypoints=waypoints.tolist(),context=context['id'])
