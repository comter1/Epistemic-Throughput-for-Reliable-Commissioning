"""Public geometry only. Fixtures are present in both physical and predictor models."""
from functools import lru_cache
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from cgx.env.system import HOME
from payload_change_v1.physics import model_xml, Dynamics, _nums
from .trajectories import grid, work, b0, library
from .protocol import REQUIREMENTS, write_json, digest

def xml(payload,context,dynamic=True):
    root=ET.fromstring(model_xml(payload,dynamic=dynamic))
    for fixture in context['fixtures']:
        ET.SubElement(root.find('worldbody'),'geom',name=fixture['name'],type='box',
            pos=_nums(fixture['center']),size=_nums(np.asarray(fixture['size'])/2),
            rgba='.3 .5 .7 1',contype='1',conaffinity='1')
    return ET.tostring(root,encoding='unicode')

@lru_cache(maxsize=32)
def compiled(payload_json,context_json,dynamic=True):
    return mujoco.MjModel.from_xml_string(xml(json.loads(payload_json),json.loads(context_json),dynamic))

def model(payload,context,dynamic=True):
    return compiled(json.dumps(payload,sort_keys=True),json.dumps(context,sort_keys=True),dynamic)

def pairs(m):
    fixtures=[i for i in range(m.ngeom) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith('fixture_')]
    robot=[i for i in range(m.ngeom) if m.geom_bodyid[i]>0 and (m.geom_contype[i] or m.geom_conaffinity[i])]
    return [(a,b) for a in robot for b in fixtures]

def clearance(m,d,geom_pairs):
    if not geom_pairs: return 1.,None
    best=1.; winner=None
    for a,b in geom_pairs:
        distance=mujoco.mj_geomDistance(m,d,a,b,1.,None)
        if distance<best: best=float(distance); winner=(a,b)
    return best,winner

def clearance_gradient(m,d,geom_pairs):
    distance,pair=clearance(m,d,geom_pairs)
    if pair is None: return distance,np.zeros(7)
    line=np.zeros(6)
    mujoco.mj_geomDistance(m,d,*pair,1.,line)
    direction=line[:3]-line[3:]; norm=np.linalg.norm(direction)
    if norm<1e-9: return distance,np.zeros(7)
    jac=np.zeros((3,m.nv))
    mujoco.mj_jac(m,d,jac,None,line[:3],int(m.geom_bodyid[pair[0]]))
    return distance,(direction/norm)@jac

def kinematic(payload,context,action,start=None,hz=20):
    m=model(payload,context); d=mujoco.MjData(m); ps=pairs(m)
    g=grid(action,start,hz); clear=1.; contacts=0
    for q in g['q']:
        d.qpos[:]=q; mujoco.mj_forward(m,d)
        c,_=clearance(m,d,ps); clear=min(clear,c); contacts=max(contacts,d.ncon)
    lower,upper=m.jnt_range.T
    joint=bool(np.all((g['q']>=lower)&(g['q']<=upper)))
    vel=float(np.max(abs(g['dq']))); acc=float(np.max(abs(g['ddq'])))
    return dict(admissible=bool(joint and not contacts and clear>=.03 and vel<=1.5 and acc<=3),
        joint_valid=joint,contacts=int(contacts),minimum_clearance_m=clear,
        reference_velocity=vel,reference_acceleration=acc,action_id=action['id'])

def build(root):
    """Geometry placement is a declared swept-envelope construction, never outcome fitting."""
    from pathlib import Path
    root=Path(root)
    # Includes the largest allowed exterior, so geometry does not vary with private load.
    payload=dict(mass=.8,com=[0,0,.08],inertia_com=(np.eye(3)*.002).tolist(),
                 box_size=[.16,.16,.065],box_center=[0,0,.095])
    offsets=[
      [[.08,.06,-.08,.06,.10,.05,.08],[.18,-.10,.12,.16,.10,-.10,.08],[.16,-.08,.12,.16,.20,-.15,.20]],
      [[-.12,-.15,.12,.12,.12,.08,-.10],[-.40,-.30,.28,.32,.15,.20,-.20],[-.38,-.28,.28,.32,.25,.15,-.15]],
      [[.12,-.10,.10,.15,.15,-.10,.12],[.24,-.18,.20,.25,.28,-.18,.20],[.24,-.18,.20,.25,.65,.20,-.35]],
    ]
    contexts={}; proposals=[]; fk=Dynamics(None)
    for i,o in enumerate(offsets):
        c=dict(id=f'W{i}',waypoints=(HOME+np.asarray(o)).tolist(),fixtures=[],
            label=['open near transfer','lateral jig with side fixture','orientation change below overhead fixture'][i])
        if i:
            m=model(payload,c); d=mujoco.MjData(m)
            axis=1 if i==1 else 2
            extent=-100.
            for action in [work(c),b0(),b0(-1)]:
                for q in grid(action,hz=25)['q']:
                    d.qpos[:]=q; mujoco.mj_forward(m,d)
                    for j in range(m.ngeom):
                        if m.geom_bodyid[j]>0 and (m.geom_contype[j] or m.geom_conaffinity[j]):
                            extent=max(extent,float(d.geom_xpos[j,axis]+m.geom_rbound[j]))
            center=np.array([.25,0,.55]); size=np.array([1.5,.10,1.2]) if i==1 else np.array([1.5,1.5,.10])
            center[axis]=extent+.040+size[axis]/2
            c['fixtures']=[dict(name=f'fixture_W{i}',center=center.tolist(),size=size.tolist())]
            c['placement_rule']='positive-axis outer bounding-sphere swept envelope of job and B0 plus 40 mm; fixed before physical trials'
        c['cartesian_waypoints']=[fk.fk(q).tolist() for q in c['waypoints']]
        check=kinematic(payload,c,work(c))
        proposals.append(dict(context=c['id'],proposal=0,**check))
        if not check['admissible']: raise RuntimeError('Declared nominal job is kinematically invalid: '+str(proposals[-1]))
        contexts[c['id']]=c
    write_json(root/'configs/workstations.json',contexts)
    write_json(root/'configs/work_requirements.json',REQUIREMENTS)
    write_json(root/'configs/test_library.json',library())
    write_json(root/'checks/geometry_proposals.json',proposals)
    return contexts
