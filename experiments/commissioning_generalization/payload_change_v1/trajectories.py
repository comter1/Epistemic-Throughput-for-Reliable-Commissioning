"""Public, deterministic references. No trajectory uses unknown payload fields."""
from __future__ import annotations
import numpy as np
from cgx.env.system import HOME, generate_task, reference_at


def original_task(seed):
    return {'id': 'L0', 'kind': 'original', 'duration': 10.0,
            'task': generate_task(int(seed)), 'speed': 1.0}


def action_library():
    # Complementary wrist orientations and shoulder/elbow configurations.
    offsets = [[0, .20, -.25, .25, .60, -.35, .50],
               [.25, -.25, .20, -.15, -.60, .35, -.50],
               [-.30, .10, .35, .30, .40, .55, -.35],
               [.30, -.10, -.35, -.30, -.40, -.45, .35],
               [-.15, -.35, -.20, .40, .70, .20, -.60],
               [.15, .30, .20, -.40, -.70, -.20, .60]]
    result = [{'id': f'S{i+1}', 'kind': 'static', 'duration': 8.0,
               'pose': (HOME + np.array(o)).tolist(), 'speed': 1.0}
              for i, o in enumerate(offsets)]
    axes = [[.20, -.18, .15, .15, 0, 0, 0],
            [0, 0, 0, 0, .40, .30, -.35],
            [-.15, .18, .15, -.20, .20, 0, .20],
            [0, 0, .15, 0, -.35, .30, .40]]
    for i, offset in enumerate(axes):
        result.append({'id': f'D{i+1}', 'kind': 'dynamic', 'duration': 8.0,
                       'pose': (HOME + .5*np.array(offsets[i])).tolist(),
                       'offset': offset, 'speed': 1.0})
    return result


def workload(seed, speed=1.0):
    if speed not in (1.0, .5):
        raise ValueError('Only speed 1.0 and 0.5 are registered')
    rng = np.random.default_rng(int(seed))
    near = HOME + rng.uniform(-1, 1, 7)*np.array([.15, .12, .15, .12, .15, .12, .15])
    far = HOME + rng.uniform(-1, 1, 7)*np.array([.30, .30, .25, .35, .25, .30, .25])
    turn = far + rng.uniform(-1, 1, 7)*np.array([.08, .08, .08, .08, .40, .35, .40])
    return {'id': 'work', 'kind': 'work', 'duration': 10.0/speed,
            'waypoints': [near.tolist(), far.tolist(), turn.tolist()],
            'speed': float(speed)}


def workload_spec():
    return {'task_seconds_full_speed': 10, 'speeds': [1.0, .5],
            'classes': ['near_translation', 'farther_reach', 'tool_rotation'],
            'home': HOME.tolist(), 'near_joint_halfwidth': [.15,.12,.15,.12,.15,.12,.15],
            'far_joint_halfwidth': [.30,.30,.25,.35,.25,.30,.25],
            'turn_from_far_halfwidth': [.08,.08,.08,.08,.40,.35,.40],
            'smooth_transition_seconds': [2.5,3.0,3.0], 'final_hold_seconds': 1.5,
            'position_rms_m_max': .02, 'position_peak_m_max': .05,
            'final_error_m_max': .02, 'final_hold_s_min': .5,
            'continuous_saturation_s_max': .1, 'saturation_fraction_max': .01,
            'joint_limit_tolerance_rad': .001,
            'reference_velocity_limit_rad_s': 1.5,
            'reference_acceleration_limit_rad_s2': 3.0,
            'hard_measured_velocity_limit_rad_s': 6.0,
            'torque_slew_limit_Nm_s': [1000]*7,
            'limits_note': 'Torque limits from pinned model; other limits are simulation requirements, not manufacturer specifications.'}


def _smooth(a, b, t, duration):
    u = float(np.clip(t/duration, 0, 1))
    s = u*u*u*(10+u*(-15+6*u))
    ds = 30*u*u*(1-u)*(1-u)/duration
    dds = 60*u*(1-u)*(1-2*u)/(duration*duration)
    delta = np.asarray(b)-np.asarray(a)
    return np.asarray(a)+s*delta, ds*delta, dds*delta


def reference(action, t, start_q=None):
    start = HOME if start_q is None else np.asarray(start_q)
    kind = action['kind']
    if kind == 'original':
        # q_ref is literally the legacy function. Its 100 ms transitions are not softened.
        q = reference_at(float(t), action['task'])
        home = np.asarray(action['task']['home'])
        previous, begin = home, .5
        for target, deadline in zip(action['task']['waypoints'], action['task']['deadlines_s']):
            target = np.asarray(target)
            if t <= deadline:
                u = (t-begin)/.1
                if 0 < u < 1:
                    delta = target-previous
                    return q, (6*u-6*u*u)/.1*delta, (6-12*u)/.01*delta
                return q, np.zeros(7), np.zeros(7)
            previous, begin = target, deadline
        return q, np.zeros(7), np.zeros(7)
    if kind == 'static':
        pose = np.asarray(action['pose'])
        if t < 3:
            return _smooth(start, pose, t, 3)
        if t < 5:
            return pose, np.zeros(7), np.zeros(7)
        return _smooth(pose, HOME, t-5, 3)
    if kind == 'dynamic':
        pose, offset = np.asarray(action['pose']), np.asarray(action['offset'])
        if t < 2:
            return _smooth(start, pose, t, 2)
        if t < 6:
            u, w = (t-2)/4*np.pi, np.pi/4
            s, c = np.sin(u), np.cos(u)
            return pose+offset*s**4, offset*4*s**3*c*w, offset*(12*s*s*c*c-4*s**4)*w*w
        return _smooth(pose, HOME, t-6, 2)
    if kind == 'work':
        speed = float(action.get('speed', 1))
        scaled = t*speed
        a = start
        begin = 0
        for target, duration in zip(action['waypoints'], [2.5,3,3]):
            if scaled < begin+duration:
                q,dq,ddq = _smooth(a,target,scaled-begin,duration)
                return q,dq*speed,ddq*speed*speed
            a, begin = np.asarray(target), begin+duration
        return np.asarray(action['waypoints'][-1]),np.zeros(7),np.zeros(7)
    raise ValueError(kind)


def reference_grid(action, start_q=None, hz=20):
    t = np.arange(int(round(float(action['duration'])*hz)))/hz
    rows = [reference(action,float(x),start_q) for x in t]
    q,dq,ddq = map(np.asarray,zip(*rows))
    return {'time': t, 'q': q, 'dq': dq, 'ddq': ddq}


library = action_library
