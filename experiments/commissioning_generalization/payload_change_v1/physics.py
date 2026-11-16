"""Isolated MuJoCo plant, public inverse dynamics, and complete healthy routing.

The backend alone owns the unknown plant. Controllers only receive delivered
encoder measurements and a separately compiled model of the configured payload.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import copy
import hashlib
import json
import pickle
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from cgx.env.system import HOME
from cgx.env.modules import Module, Packet, merge_packets
from .trajectories import reference, reference_grid
from .storage import digest, write_json

ASSET = Path('/work/Users/leiyo/cgx_data_v2/assets/mujoco_menagerie/franka_emika_panda')
CONTROLLER_VERSION = 'payload-id-pd-v2-source-qualified'
# Fixed after nominal true-configuration source qualification. The legacy
# effective stiffness caused rate-limiter overshoot on the exact L0 reference.
DEFAULT_CONTROLLER = {'kp':[200,200,150,150,80,80,60],
                      'kd':[10,10,8,8,5,5,4],
                      'slew_limit':[1000]*7, 'version':CONTROLLER_VERSION}


def _nums(x):
    return ' '.join(format(float(y),'.17g') for y in np.asarray(x).ravel())


def payload_pi(payload):
    m = float(payload['mass']); c = np.asarray(payload['com'],float)
    ic = np.asarray(payload['inertia_com'],float).reshape(3,3)
    io = ic+m*((c@c)*np.eye(3)-np.outer(c,c))
    return np.r_[m,m*c,io[0,0],io[1,1],io[2,2],io[0,1],io[0,2],io[1,2]]


def model_xml(payload=None, *, dynamic=True):
    """Compile-time changes only; no runtime mass/inertia editing.

    Fingers are rigidly held at the original open position (0.04 m), retaining
    their geometry and inertial parameters exactly once as a known fixed tool.
    """
    root = ET.parse(ASSET/'panda.xml').getroot()
    root.find('compiler').set('meshdir',str(ASSET/'assets'))
    root.find('option').set('timestep','0.002')
    # Rendering is unused. Remove only visual geoms/assets, keeping collision mesh assets.
    for body in root.iter('body'):
        for geom in list(body.findall('geom')):
            if geom.get('class') == 'visual':
                body.remove(geom)
    mesh_names = {g.get('mesh') for g in root.iter('geom') if g.get('mesh')}
    assets = root.find('asset')
    for asset in list(assets):
        if asset.tag == 'mesh' and asset.get('name',Path(asset.get('file','')).stem) not in mesh_names:
            assets.remove(asset)
    for name in ('left_finger','right_finger'):
        body = root.find(f'.//body[@name="{name}"]')
        pos = np.fromstring(body.get('pos','0 0 0'),sep=' ')
        # Right finger has a 180 degree z rotation, hence its local +y slides -y.
        pos[1] += .04 if name == 'left_finger' else -.04
        body.set('pos',_nums(pos))
        for joint in list(body.findall('joint')):
            body.remove(joint)
    for tag in ('tendon','equality','keyframe'):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    actuator = root.find('actuator')
    inherited_force_range = root.find('./default/default[@class="panda"]/general').get('forcerange')
    for item in list(actuator):
        if item.get('name') == 'actuator8':
            actuator.remove(item)
            continue
        force_range = item.get('forcerange', inherited_force_range)
        item.set('gainprm','1 0 0'); item.set('biasprm','0 0 0')
        item.set('biastype','none'); item.set('dyntype','none')
        item.set('ctrlrange',force_range)
        item.set('forcerange',force_range)
        item.set('gear','1')
    world = root.find('worldbody')
    ET.SubElement(world,'geom',name='floor',type='plane',size='0 0 .05')
    if payload is not None:
        mass = float(payload['mass'])
        if mass <= 0:
            raise ValueError('positive payload mass required')
        c = np.asarray(payload['com'],float)
        inertia = np.asarray(payload['inertia_com'],float).reshape(3,3)
        eig = np.linalg.eigvalsh(inertia)
        if eig.min() <= 0 or 2*eig.max() > eig.sum()+1e-12:
            raise ValueError('payload inertia is not physically consistent')
        hand = root.find('.//body[@name="hand"]')
        body = ET.SubElement(hand,'body',name='payload',pos='0 0 0')
        ET.SubElement(body,'inertial',mass=str(mass),pos=_nums(c),fullinertia=_nums([
            inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]]))
        size = np.asarray(payload.get('box_size',[.08]*3),float)
        center = payload.get('box_center',[0,0,.08])
        ET.SubElement(body,'geom',name='payload_envelope',type='box',size=_nums(size/2),
                      pos=_nums(center),mass='0',rgba='.4 .4 .4 1')
        contact = root.find('contact')
        for name in ('hand','left_finger','right_finger'):
            ET.SubElement(contact,'exclude',body1='payload',body2=name)
    if not dynamic:
        # Inverse dynamics is explicitly unconstrained: no contacts/limit forces.
        flag = root.find('option/flag')
        if flag is None:
            flag = ET.SubElement(root.find('option'),'flag')
        flag.set('constraint','disable')
    return ET.tostring(root,encoding='unicode')


@lru_cache(maxsize=80)
def _compiled(spec_json, dynamic):
    return mujoco.MjModel.from_xml_string(model_xml(json.loads(spec_json),dynamic=dynamic))


class Dynamics:
    """Known public model only. Independent data; immutable cached model."""
    def __init__(self, payload=None):
        self.payload = copy.deepcopy(payload)
        self.model = _compiled(json.dumps(payload,sort_keys=True),False)
        self.data = mujoco.MjData(self.model)
        self.hand = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'hand')
        self.limits = self.model.actuator_forcerange[:,1].copy()
        self.joint_range = self.model.jnt_range.copy()
        self.calls = 0

    def inverse(self,q,dq,ddq):
        self.data.qpos[:] = q; self.data.qvel[:] = dq; self.data.qacc[:] = ddq
        mujoco.mj_inverse(self.model,self.data)
        self.calls += 1
        # qfrc_inverse = M qacc + bias - passive - constraint; constraint disabled.
        # It already contains armature and compensates the known joint damping.
        return self.data.qfrc_inverse.copy()

    def fk(self,q):
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model,self.data)
        return self.data.xpos[self.hand].copy()

    def orientation(self,q):
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model,self.data)
        return self.data.xmat[self.hand].reshape(3,3).copy()


class HealthyRouting:
    """Execute the existing Packet/Module operations, recording every real port."""
    def __init__(self,graph):
        import networkx as nx
        self.graph = copy.deepcopy(graph)
        self.nodes = {n['id']:n for n in graph['nodes']}
        dag = nx.DiGraph(); dag.add_nodes_from(self.nodes)
        dag.add_edges_from((e['source'],e['target']) for e in graph['edges'])
        self.order = list(nx.topological_sort(dag))
        self.modules = {i:Module(n) for i,n in self.nodes.items()}
        self.incoming = {i:[] for i in self.nodes}
        for edge in graph['edges']:
            self.incoming[edge['target']].append(edge)
        self.ports = [(n['id'],p['name']) for n in graph['nodes'] for p in n['ports']]
        self.index = {p:i for i,p in enumerate(self.ports)}
        self.channels = [next(p['channels'] for p in self.nodes[n]['ports'] if p['name']==port)
                         for n,port in self.ports]

    def step(self,q,dq,tick,token,controller):
        outputs = {}; port_values = np.zeros((len(self.ports),14)); tokens = np.empty((len(self.ports),16),np.uint8)
        def record(n,p,packet):
            i = self.index[(n,p)]
            idx = list(packet.channels)
            if packet.state: idx += [j+7 for j in packet.channels]
            port_values[i,idx] = packet.values[idx]
            tokens[i] = np.frombuffer(packet.token,dtype=np.uint8)
        command = None
        for nid in self.order:
            node = self.nodes[nid]; module = self.modules[nid]
            inputs = {}
            for edge in self.incoming[nid]:
                packet = outputs[(edge['source'],edge['source_port'])]
                inputs[edge['target_port']] = packet
                record(nid,edge['target_port'],packet)
            kind = node['type']
            if kind == 'sensor':
                out = Packet(np.r_[q,dq],token,tuple(range(7)),True)
            elif kind == 'merger':
                out = merge_packets(list(inputs.values()))
            else:
                packet = next(iter(inputs.values()))
                if kind == 'cache': out = module.cache(packet,tick)
                elif kind == 'adapter': out = module.adapter(packet)
                elif kind == 'queue': out = module.queue(packet,tick,packet)
                elif kind == 'splitter': out = packet
                elif kind == 'controller':
                    measured = packet.canonical()
                    u = controller(measured[:7],measured[7:])
                    ct = hashlib.blake2b(token+b'torque-command',digest_size=16).digest()
                    out = Packet(np.r_[u,np.zeros(7)],ct,tuple(range(7)),False)
                elif kind == 'actuator':
                    command = packet.canonical()[:7]
                    out = packet
                else: raise ValueError('unsupported healthy node '+kind)
            for port in node['ports']:
                if port['direction'] == 'out':
                    packet = out.subset(port['channels'])
                    outputs[(nid,port['name'])] = packet
                    record(nid,port['name'],packet)
        if command is None:
            raise ValueError('graph has no actuator')
        return command,port_values,tokens


def _metric(q, dq, error, saturation, contacts, lower, upper, duration, aborted):
    count = len(error); dt = duration/max(count,1)
    sat = np.any(saturation,axis=1) if count else np.array([],bool)
    maxrun = 0
    for j in range(7):
        current = 0
        for value in saturation[:,j]:
            current = current+1 if value else 0
            maxrun = max(maxrun,current)
    violation = bool(np.any((q<lower-.001)|(q>upper+.001))) if count else True
    last = max(1,int(round(.5/dt)))
    held = bool(count >= last and np.all(error[-last:] <= .02))
    rms = float(np.sqrt(np.mean(error**2))) if count else 1e9
    peak = float(np.max(error)) if count else 1e9
    ratio = float(np.mean(sat)) if count else 1
    continuous = maxrun*dt
    passed = (held and rms<=.02 and peak<=.05 and not np.any(contacts)
              and not violation and not aborted and ratio<=.01+1e-12 and continuous<=.1+1e-12)
    return {'pass':bool(passed),'rms_error_m':rms,'peak_error_m':peak,'final_hold_pass':held,
            'final_error_m':float(error[-1]) if count else 1e9,
            'saturation_fraction':ratio,'saturation_seconds':float(sat.sum()*dt),
            'max_continuous_saturation_seconds':float(continuous),'unexpected_contact':bool(np.any(contacts)),
            'contact_samples':int(np.count_nonzero(contacts)), 'joint_limit_violation':violation,
            'hard_abort':bool(aborted),'velocity_peak_rad_s':np.max(np.abs(dq),axis=0).tolist() if count else [0]*7}


@dataclass
class BlockResult:
    public: dict
    private: dict
    metadata: dict
    physical_seconds: float

    def save(self,path):
        path = Path(path); path.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path/'public.npz',**self.public)
        np.savez_compressed(path/'private.npz',**self.private['arrays'])
        write_json(path/'metadata.json',self.metadata | {'physical_seconds':self.physical_seconds,
                   'public_metrics':self.private['public_metrics'],'private_metrics':self.private['metrics']})
        return {'public':str(path/'public.npz'),'private':str(path/'private.npz'),
                'metadata':str(path/'metadata.json')}


class PayloadPlant:
    def __init__(self,true_payload,configured_payload,noise_seed,graph=None,ledger=None,controller=None):
        if ledger is None:
            raise ValueError('PhysicsLedger is mandatory')
        self.ledger = ledger
        self._plant_spec_hash = digest(true_payload)
        self._model = _compiled(json.dumps(true_payload,sort_keys=True),True)
        self._data = mujoco.MjData(self._model)
        self._data.qpos[:] = HOME
        self._rng = np.random.default_rng(int(noise_seed))
        self._data.qpos[:] += self._rng.normal(0,1e-4,7)
        self._bias = self._rng.normal(0,.02,7)
        self._ar = self._rng.normal(0,.05,7)
        self._step = 0
        self._event_namespace = hashlib.sha256(str(noise_seed).encode()).hexdigest()[:16]
        self.controller_config = copy.deepcopy(controller or DEFAULT_CONTROLLER)
        self.configure(configured_payload)
        self.kinematics = Dynamics(None)
        self.limits = self.controller.limits.copy()
        self.lower,self.upper = self._model.jnt_range.T.copy()
        self.routing = HealthyRouting(graph) if graph is not None else None
        self._hand = mujoco.mj_name2id(self._model,mujoco.mjtObj.mjOBJ_BODY,'hand')
        # Initial hold is computed at the PUBLIC rest reference, never exact
        # hidden perturbed qpos. The first 100 Hz control uses noisy encoders.
        self._previous_applied = self.controller.inverse(HOME,np.zeros(7),np.zeros(7))
        self._data.ctrl[:] = np.clip(self._previous_applied,-self.limits,self.limits)
        mujoco.mj_forward(self._model,self._data)

    def configure(self,payload):
        self.configured_payload = copy.deepcopy(payload)
        self.controller = Dynamics(payload)

    def state(self):
        """Evaluator-only readable state summary, not a complete cache identity."""
        return {'q':self._data.qpos.tolist(),'dq':self._data.qvel.tolist(),
                'time':float(self._data.time),'ctrl':self._data.ctrl.tolist()}

    def cache_identity(self):
        """Evaluator-only complete state fingerprint; never a policy input.

        A reusable execution additionally keys its full action and execution
        identity. Integration includes solver warmstart, controls and time.
        Module histories, sensor streams and persistent errors are also hashed.
        """
        signature = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self._model,signature))
        mujoco.mj_getState(self._model,self._data,state,signature)
        payload = {'plant_spec':self._plant_spec_hash,
                   'integration_state':state,
                   'sensor_rng':self._rng.bit_generator.state,
                   'torque_bias':self._bias, 'torque_ar':self._ar,
                   'sensor_namespace':self._event_namespace,'sample_step':self._step,
                   'previous_applied':self._previous_applied,
                   'configured_payload':self.configured_payload,
                   'controller_config':self.controller_config,
                   'routing_graph':self.routing.graph if self.routing else None,
                   'routing_modules':self.routing.modules if self.routing else None,
                   'backend_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'reference_sha256':hashlib.sha256((Path(__file__).parent/'trajectories.py').read_bytes()).hexdigest(),
                   'mujoco_version':mujoco.__version__}
        return hashlib.sha256(pickle.dumps(payload,protocol=5)).hexdigest()

    def clone(self):
        """Evaluator-only exact branch for shared validation histories.

        Copies all integration, sensor RNG, persistent bias, command, and routing
        state. This does not integrate physics or reset the robot. Every later
        execute reserves and charges its own actual physical time normally.
        """
        other = object.__new__(PayloadPlant)
        other.__dict__ = self.__dict__.copy()
        other._data = mujoco.MjData(self._model)
        mujoco.mj_copyData(other._data,self._model,self._data)
        other._rng = np.random.default_rng()
        other._rng.bit_generator.state = copy.deepcopy(self._rng.bit_generator.state)
        other._bias=self._bias.copy(); other._ar=self._ar.copy()
        other._previous_applied=self._previous_applied.copy()
        other.controller_config=copy.deepcopy(self.controller_config)
        other.configured_payload=copy.deepcopy(self.configured_payload)
        other.controller=Dynamics(self.configured_payload)
        mujoco.mj_copyData(other.controller.data,other.controller.model,self.controller.data)
        other.controller.calls=self.controller.calls
        other.kinematics=Dynamics(None)
        mujoco.mj_copyData(other.kinematics.data,other.kinematics.model,self.kinematics.data)
        other.routing=copy.deepcopy(self.routing)
        return other

    def execute(self,action,execution_id,*,kind='physics'):
        duration = float(action['duration'])
        token = self.ledger.reserve(duration,unit=str(execution_id),kind=kind)
        begin = time.perf_counter(); start_state = self.state()
        data,model = self._data,self._model
        n = int(round(duration/.01)); nphys = n*5
        keys = ['q','dq','torque','requested','applied','qref','dqref','ddqref']
        public = {k:np.zeros((n,7)) for k in keys}
        public.update(time=np.zeros(n),sample_index=np.arange(n,dtype=np.int32),
                      event_id=np.empty((n,16),np.uint8),tracking_flags=np.zeros(n,bool),
                      joint_limit_flags=np.zeros(n,bool),unexpected_contact_flags=np.zeros(n,bool),
                      actuator_saturation_flags=np.zeros((n,7),bool),
                      torque_slew_limited_flags=np.zeros((n,7),bool),
                      configured_payload_mass=np.full(n,float(self.configured_payload['mass'])),
                      configured_payload_com=np.tile(self.configured_payload['com'],(n,1)),
                      configured_payload_inertia=np.tile(np.asarray(self.configured_payload['inertia_com']).reshape(1,9),(n,1)))
        if self.routing:
            public['software_values'] = np.zeros((n,len(self.routing.ports),14))
            public['software_message_id'] = np.zeros((n,len(self.routing.ports),16),np.uint8)
        private = {k:np.zeros((nphys,7)) for k in ('q','dq','torque')}
        private.update(time=np.zeros(nphys),error=np.zeros(nphys),contacts=np.zeros(nphys,np.int32),
                       saturation=np.zeros((nphys,7),bool))
        public_error = np.zeros(n)
        start_q = data.qpos.copy()
        # This starting reference is an encoder observation, not a reset/teleport.
        start_ref = start_q+self._rng.normal(0,1e-4,7)
        aborted = False; abort_reason = ''; count = 0; pc = 0
        kp = np.asarray(self.controller_config['kp']); kd = np.asarray(self.controller_config['kd'])
        slew = np.asarray(self.controller_config['slew_limit'])*.01
        for i in range(n):
            t = i*.01
            q = data.qpos.copy()+self._rng.normal(0,1e-4,7)
            dq = data.qvel.copy()+self._rng.normal(0,1e-3,7)
            self._ar = .6*self._ar + .05*np.sqrt(1-.6**2)*self._rng.normal(size=7)
            # Sample actuator force at the same instant as q/dq, BEFORE the new command.
            torque = data.qfrc_actuator.copy()+self._bias+self._ar
            qref,dqref,ddqref = reference(action,t,start_ref)
            requested = None
            def control(qdel,dqdel):
                nonlocal requested
                requested = self.controller.inverse(qdel,dqdel,ddqref)+kp*(qref-qdel)+kd*(dqref-dqdel)
                bounded = np.clip(requested,-self.limits,self.limits)
                return np.clip(bounded,self._previous_applied-slew,self._previous_applied+slew)
            event = hashlib.blake2b((str(execution_id)+':'+self._event_namespace+':'+str(self._step)).encode(),digest_size=16).digest()
            if self.routing:
                applied,values,tokens = self.routing.step(q,dq,self._step,event,control)
                public['software_values'][i] = values
                public['software_message_id'][i] = tokens
            else:
                applied = control(q,dq)
            self._previous_applied = applied.copy(); data.ctrl[:] = applied
            saturated = np.abs(requested)>self.limits+1e-10
            public['q'][i]=q; public['dq'][i]=dq; public['torque'][i]=torque
            public['requested'][i]=requested; public['applied'][i]=applied
            public['qref'][i]=qref; public['dqref'][i]=dqref; public['ddqref'][i]=ddqref
            public['time'][i]=data.time; public['event_id'][i]=np.frombuffer(event,np.uint8)
            public['actuator_saturation_flags'][i]=saturated
            public['torque_slew_limited_flags'][i]=np.abs(applied-np.clip(requested,-self.limits,self.limits))>1e-10
            public_error[i]=np.linalg.norm(self.kinematics.fk(q)-self.kinematics.fk(qref))
            public['tracking_flags'][i]=public_error[i]>.02
            public['joint_limit_flags'][i]=bool(np.any((q<self.lower-.001)|(q>self.upper+.001)))
            contact_now=False
            for s in range(5):
                mujoco.mj_step(model,data)
                mujoco.mj_kinematics(model,data)
                rp,_,_ = reference(action,min(t+(s+1)*.002,duration),start_ref)
                private['q'][pc]=data.qpos; private['dq'][pc]=data.qvel
                private['torque'][pc]=data.qfrc_actuator; private['time'][pc]=data.time
                private['error'][pc]=np.linalg.norm(data.xpos[self._hand]-self.kinematics.fk(rp))
                private['contacts'][pc]=data.ncon; private['saturation'][pc]=saturated
                contact_now |= data.ncon>0
                pc += 1
                if not np.all(np.isfinite(data.qpos)) or np.any((data.qpos<self.lower-.001)|(data.qpos>self.upper+.001)):
                    aborted=True; abort_reason='joint_limit_or_nonfinite'
                if np.any(np.abs(data.qvel)>6):
                    aborted=True; abort_reason='hard_velocity'
                if data.ncon:
                    aborted=True; abort_reason='unexpected_contact'
                if aborted: break
            public['unexpected_contact_flags'][i]=contact_now
            count+=1; self._step+=1
            if aborted: break
        elapsed = pc*.002
        public = {k:v[:count] for k,v in public.items()}
        private = {k:v[:pc] for k,v in private.items()}
        m = _metric(private['q'],private['dq'],private['error'],private['saturation'],
                    private['contacts'],self.lower,self.upper,elapsed,aborted)
        pm = _metric(public['q'],public['dq'],public_error[:count],public['actuator_saturation_flags'],
                     public['unexpected_contact_flags'],self.lower,self.upper,count*.01,aborted)
        m['torque_peak_Nm']=np.max(np.abs(private['torque']),axis=0).tolist() if pc else [0]*7
        m['torque_min_margin_Nm']=(self.limits-np.max(np.abs(private['torque']),axis=0)).tolist() if pc else self.limits.tolist()
        m['joint_path_length_rad']=float(np.sum(np.abs(np.diff(private['q'],axis=0))))
        self.ledger.complete(token,elapsed,status='hard_abort' if aborted else 'completed')
        metadata={'execution_id':str(execution_id),'action':copy.deepcopy(action),'start_state':start_state,
                  'end_state':self.state(),'controller_version':self.controller_config['version'],
                  'controller_hash':digest(self.controller_config),'configured_payload':self.configured_payload,
                  'wall_seconds':time.perf_counter()-begin,'abort_reason':abort_reason,
                  'software_ports':self.routing.ports if self.routing else [],
                  'software_channels':self.routing.channels if self.routing else [],
                  'sensor_definition':'qfrc_actuator + persistent bias + stationary AR1; Nm, joint sign, unit gear; pre-command simultaneous encoder sample',
                  'bytes_uncompressed':sum(v.nbytes for v in public.values()),'ledger_token':token}
        return BlockResult(public,{'arrays':private,'metrics':m,'public_metrics':pm},metadata,elapsed)
