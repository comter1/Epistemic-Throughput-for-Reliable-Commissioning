"""Versioned V1 execution adapter; new fixtures, references, and precision metrics.

No parent module globals are modified. Only this evaluator owns private MuJoCo
state; selectors receive acquired logs and declared geometry.
"""
import copy, hashlib, json, pickle, time
from pathlib import Path
import numpy as np
import mujoco
from cgx.env.system import HOME
from payload_change_v1.physics import Dynamics, HealthyRouting, BlockResult, DEFAULT_CONTROLLER, _metric as parent_metric
from .protocol import digest
from .trajectories import reference
from .workstations import model as work_model, pairs, clearance


def _metric(q,dq,error,saturation,contacts,lower,upper,duration,aborted):
    m=parent_metric(q,dq,error,saturation,contacts,lower,upper,duration,aborted)
    count=len(error); last=min(count,250) if count else 0
    # Private arrays are 500 Hz; public arrays are 100 Hz. Derive the full half-second interval.
    n=int(round(.5*count/max(duration,1e-12)))
    held=bool(count>=n and n>0 and np.all(error[-n:]<=.002))
    m['final_hold_peak_m']=float(np.max(error[-n:])) if n and count else 1e9
    m['final_hold_pass']=held
    m['pass']=bool(held and m['rms_error_m']<=.005 and m['peak_error_m']<=.010
        and not m['unexpected_contact'] and not m['joint_limit_violation'] and not aborted
        and m['saturation_fraction']<=.01+1e-12 and m['max_continuous_saturation_seconds']<=.1+1e-12)
    return m

class PayloadPlant:
    def __init__(self,true_payload,configured_payload,noise_seed,context,graph=None,ledger=None,controller=None):
        if ledger is None:
            raise ValueError('PhysicsLedger is mandatory')
        self.ledger = ledger
        self._plant_spec_hash = digest(true_payload)
        self.context=copy.deepcopy(context)
        self._model=work_model(true_payload,context)
        self._clearance_pairs=pairs(self._model)
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
        payload = {'plant_spec':self._plant_spec_hash, 'public_context':self.context,
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
        token = self.ledger.reserve(round(duration/.01)*.01,unit=str(execution_id),kind=kind)
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
                       saturation=np.zeros((nphys,7),bool),clearance=np.ones(nphys))
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
            public['tracking_flags'][i]=public_error[i]>.010
            public['joint_limit_flags'][i]=bool(np.any((q<self.lower-.001)|(q>self.upper+.001)))
            contact_now=False
            for s in range(5):
                mujoco.mj_step(model,data)
                mujoco.mj_kinematics(model,data)
                rp,_,_ = reference(action,min(t+(s+1)*.002,duration),start_ref)
                private['q'][pc]=data.qpos; private['dq'][pc]=data.qvel
                private['torque'][pc]=data.qfrc_actuator; private['time'][pc]=data.time
                private['error'][pc]=np.linalg.norm(data.xpos[self._hand]-self.kinematics.fk(rp))
                private['clearance'][pc]=clearance(model,data,self._clearance_pairs)[0]
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
        m['minimum_clearance_m']=float(np.min(private['clearance'])) if pc else 0.
        m['peak_requested_Nm']=np.max(np.abs(public['requested']),axis=0).tolist() if count else [0]*7
        m['slew_limited']=bool(np.any(public['torque_slew_limited_flags']))
        m['torque_peak_Nm']=np.max(np.abs(private['torque']),axis=0).tolist() if pc else [0]*7
        m['torque_min_margin_Nm']=(self.limits-np.max(np.abs(private['torque']),axis=0)).tolist() if pc else self.limits.tolist()
        m['joint_path_length_rad']=float(np.sum(np.abs(np.diff(private['q'],axis=0))))
        self.ledger.complete(token,elapsed,status='hard_abort' if aborted else 'completed')
        metadata={'execution_id':str(execution_id),'action':copy.deepcopy(action),'start_state':start_state,
                  'end_state':self.state(),'measured_start_reference':start_ref.tolist(),'controller_version':self.controller_config['version'],
                  'controller_hash':digest(self.controller_config),'configured_payload':self.configured_payload,
                  'wall_seconds':time.perf_counter()-begin,'abort_reason':abort_reason,
                  'software_ports':self.routing.ports if self.routing else [],
                  'software_channels':self.routing.channels if self.routing else [],
                  'sensor_definition':'qfrc_actuator + persistent bias + stationary AR1; Nm, joint sign, unit gear; pre-command simultaneous encoder sample',
                  'bytes_uncompressed':sum(v.nbytes for v in public.values()),'ledger_token':token}
        return BlockResult(public,{'arrays':private,'metrics':m,'public_metrics':pm},metadata,elapsed)
