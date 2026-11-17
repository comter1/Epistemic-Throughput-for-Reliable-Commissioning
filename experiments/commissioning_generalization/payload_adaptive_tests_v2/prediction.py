"""Public closed-loop integrations and sampled-control first-order sensitivities.

This API has no plant argument. All models are compiled from the supplied public
configuration or explicit, physically projected parameter hypotheses.
"""
from __future__ import annotations
import fcntl
import json
import time
from pathlib import Path
import numpy as np
import mujoco
from scipy.stats import qmc, norm
from cgx.env.system import HOME
from payload_change_v1.physics import Dynamics, DEFAULT_CONTROLLER
from payload_change_v1.estimator import default_regressor, payload_to_pi, pi_to_payload, project_physical, SCALE
from .protocol import Ledger, digest, file_hash, write_json, REQUIREMENTS
from .trajectories import grid
from .workstations import model, pairs, clearance_gradient, clearance

class DecisionTimeout(RuntimeError): pass

class Clock:
    def __init__(self,limit=10.): self.start=time.process_time(); self.wall_start=time.perf_counter(); self.limit=limit
    def check(self):
        # Leave 20 ms for exception unwinding/recording. Check inside MC batches,
        # rather than allowing an entire candidate evaluation to overrun the cap.
        if time.process_time()-self.start>=max(0.,self.limit-.02): raise DecisionTimeout('ten-second serial CPU prediction/planning cap')
    @property
    def used(self): return time.process_time()-self.start
    @property
    def wall_used(self): return time.perf_counter()-self.wall_start

def sqrt_psd(c):
    v,u=np.linalg.eigh((c+c.T)/2)
    return u*np.sqrt(np.maximum(v,0))

class PublicPredictionModel:
    def __init__(self,root,context,geometry,margins=None):
        self.root=Path(root); self.context=context; self.geometry=geometry
        self.margins=margins or dict(peak_m=0.,rms_m=0.,hold_m=0.,torque_Nm=0.,clearance_m=0.,
                                     scaled_linear_radius=1e6,validated=False)
        self.ledger=Ledger(root,'prediction'); self.memory={}; self.cache_hits=0
        self.code_hash=digest({p.name:file_hash(p) for p in (Path(__file__),Path(__file__).with_name('trajectories.py'),Path(__file__).with_name('workstations.py'))})
    def kernel(self,configured,action,start=None,clock=None):
        start=np.asarray(HOME if start is None else start)
        key=digest(dict(configured=configured,context=self.context,action=action,start=start.tolist(),
                        code=self.code_hash,controller=DEFAULT_CONTROLLER,mujoco=mujoco.__version__))
        if key in self.memory:
            self.cache_hits+=1; return self.memory[key]
        path=self.root/'cache/public_prediction'/key[:2]/(key+'.npz'); path.parent.mkdir(parents=True,exist_ok=True)
        if clock: clock.check()
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if path.exists():
                with np.load(path) as z: result={k:z[k] for k in z.files}
                self.ledger.cache_hit(key,float(action['duration']),'deterministic public configured model')
                self.cache_hits+=1
            else:
                result=self._integrate(configured,configured,action,start,True,clock)
                np.savez_compressed(path,**result)
        self.memory[key]=result
        return result

    def _integrate(self,physical,configured,action,start,sensitivities,clock=None):
        token=self.ledger.reserve(round(action['duration']*100)/100,unit=digest([physical,configured,action,start.tolist()]),
                                  kind='linearized_nominal' if sensitivities else 'secondary_hypothesis')
        m=model(physical,self.context); d=mujoco.MjData(m); d.qpos[:]=start
        ctrl=Dynamics(configured); kin=Dynamics(None); hand=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'hand')
        gp=pairs(m); limits=m.actuator_forcerange[:,1]; kp=np.asarray(DEFAULT_CONTROLLER['kp']); kd=np.asarray(DEFAULT_CONTROLLER['kd'])
        refs=grid(action,start,hz=500); n=int(round(action['duration']*100))
        ee=np.zeros((n,3)); jee=np.zeros((n,3,10)); torque=np.zeros((n,7)); jt=np.zeros((n,7,10))
        cs=np.ones(n); jc=np.zeros((n,10)); qout=np.zeros((n,7)); dqout=np.zeros((n,7))
        A=np.eye(14); B=np.zeros((14,7)); Gq=np.zeros((7,7)); Gv=np.zeros((7,7)); Y=np.zeros((7,10))
        S=np.zeros((14,10)); U=np.zeros((7,10)); J=np.zeros((3,7))
        Fq=np.zeros((7,7)); Fv=np.zeros((7,7)); reg=default_regressor() if sensitivities else None
        prev=ctrl.inverse(start,np.zeros(7),np.zeros(7)); d.ctrl[:]=np.clip(prev,-limits,limits)
        mujoco.mj_forward(m,d)
        unsupported=False; collision=False; pc=0
        try:
            for i in range(n):
                if clock and i%10==0: clock.check()
                ix=i*5; qr,vr,ar=(refs[k][ix] for k in ('q','dq','ddq'))
                u=ctrl.inverse(d.qpos,d.qvel,ar)+kp*(qr-d.qpos)+kd*(vr-d.qvel)
                applied=np.clip(np.clip(u,-limits,limits),prev-10,prev+10)
                unsupported |= bool(np.any(np.abs(applied-u)>1e-8))
                d.ctrl[:]=applied; prev=applied.copy(); mujoco.mj_forward(m,d)
                if sensitivities and i%5==0:
                    # These derivatives belong to an independently compiled PUBLIC model.
                    mujoco.mjd_transitionFD(m,d,1e-6,False,A,B,None,None)
                    ctrl.inverse(d.qpos,d.qvel,ar)
                    mujoco.mjd_inverseFD(ctrl.model,ctrl.data,1e-6,False,Fq,Fv,None,None,None,None,None)
                    Gq=Fq.T-np.diag(kp); Gv=Fv.T-np.diag(kd)
                    _,Y=reg.at(d.qpos,d.qvel,d.qacc)
                if sensitivities: U=Gq@S[:7]+Gv@S[7:]
                torque[i]=u; jt[i]=U
                for sub in range(5):
                    if sensitivities: S=A@S+B@(U-Y)
                    mujoco.mj_step(m,d); pc+=1
                mujoco.mj_kinematics(m,d)
                ee[i]=d.xpos[hand]-kin.fk(refs['q'][ix+5])
                qout[i]=d.qpos; dqout[i]=d.qvel
                if sensitivities:
                    mujoco.mj_jacBody(m,d,J,None,hand); jee[i]=J@S[:7]
                    cs[i],g=clearance_gradient(m,d,gp); jc[i]=g@S[:7]
                else: cs[i]=clearance(m,d,gp)[0]
                collision |= bool(d.ncon)
        except BaseException:
            self.ledger.complete(token,pc*.002,'interrupted_public_prediction')
            raise
        self.ledger.complete(token,pc*.002)
        return dict(ee=ee,jee=jee,torque=torque,jtorque=jt,clearance=cs,jclearance=jc,
            q=qout,dq=dqout,configured_pi=payload_to_pi(configured),duration=np.array(action['duration']),
            unsupported=np.array(unsupported or collision),collision=np.array(collision),
            reference_velocity=np.array(np.max(abs(refs['dq']))),reference_acceleration=np.array(np.max(abs(refs['ddq']))))

    def summary(self,k,mean,cov,*,radius=4.817598069):
        delta=np.asarray(mean)[:10]-k['configured_pi']; c=np.asarray(cov)[:10,:10]
        e=k['ee']+np.einsum('tij,j->ti',k['jee'],delta)
        # A norm bound over ONE joint ellipsoid, not independent per-time quantiles.
        errvar=np.einsum('tki,ij,tkj->t',k['jee'],c,k['jee'])
        bound=np.linalg.norm(e,axis=1)+radius*np.sqrt(np.maximum(errvar,0))
        u=k['torque']+np.einsum('tij,j->ti',k['jtorque'],delta)
        uv=np.einsum('tki,ij,tkj->tk',k['jtorque'],c,k['jtorque'])
        ub=np.abs(u)+radius*np.sqrt(np.maximum(uv,0))+self.margins['torque_Nm']
        cm=k['clearance']+k['jclearance']@delta
        cv=np.einsum('ti,ij,tj->t',k['jclearance'],c,k['jclearance'])
        cb=cm-radius*np.sqrt(np.maximum(cv,0))-self.margins['clearance_m']
        r=dict(peak_m=float(bound.max()+self.margins['peak_m']),
            rms_m=float(np.sqrt(np.mean(bound**2))+self.margins['rms_m']),
            hold_m=float(bound[-50:].max()+self.margins['hold_m']),
            nominal_peak_m=float(np.linalg.norm(k['ee'],axis=1).max()),
            mean_peak_m=float(np.linalg.norm(e,axis=1).max()),
            mean_torque_Nm=np.max(abs(u),axis=0).tolist(), torque_Nm=np.max(ub,axis=0).tolist(),
            clearance_m=float(cb.min()), mean_clearance_m=float(cm.min()),
            unsupported=bool(k['unsupported']),reference_velocity=float(k['reference_velocity']),
            reference_acceleration=float(k['reference_acceleration']),
            parameter_distance=float(np.linalg.norm(delta/SCALE[:10])),
            source_validated=bool(self.margins.get('validated',False)))
        r['linear_supported']=bool(not r['unsupported'] and r['parameter_distance']<=self.margins['scaled_linear_radius'])
        r['motion_pass']=bool(r['linear_supported'] and r['clearance_m']>0 and np.all(np.asarray(r['torque_Nm'])<[87]*4+[12]*3)
                              and r['reference_velocity']<=1.5 and r['reference_acceleration']<=3)
        r['work_pass']=bool(r['motion_pass'] and r['rms_m']<=.005 and r['peak_m']<=.010 and r['hold_m']<=.002)
        return r

    def samples_pass(self,k,samples,clock=None):
        """Fast decision-state map. The deployment configuration is fixed by k."""
        # Use all reference samples; batching bounds memory for prospective outcomes.
        values=[]
        for a in np.array_split(samples,max(1,int(np.ceil(len(samples)/128)))):
            if clock: clock.check()
            delta=a[:,:10]-k['configured_pi']
            # 20 Hz score grid is calibrated independently; deployment uses all 100 Hz rows.
            e=k['ee'][::5,None,:]+np.einsum('tij,nj->tni',k['jee'][::5],delta)
            err=np.linalg.norm(e,axis=2)
            u=k['torque'][::5,None,:]+np.einsum('tij,nj->tni',k['jtorque'][::5],delta)
            cl=k['clearance'][::5,None]+k['jclearance'][::5]@delta.T
            ok=(np.sqrt(np.mean(err**2,axis=0))+self.margins['rms_m']<=.005)
            ok&=np.max(err,axis=0)+self.margins['peak_m']<=.010
            ok&=np.max(err[-10:],axis=0)+self.margins['hold_m']<=.002
            ok&=np.all(np.max(abs(u),axis=0)+self.margins['torque_Nm']<np.array([87]*4+[12]*3),axis=1)
            ok&=cl.min(axis=0)-self.margins['clearance_m']>0
            if bool(k['unsupported']): ok[:]=False
            values.extend(ok.tolist())
        return np.array(values,bool)

    def secondary(self,configured,action,start,mean,cov,clock):
        z=norm.ppf(np.clip(qmc.Sobol(10,scramble=False).random_base2(4),1e-3,1-1e-3))
        samples=np.vstack([mean[:10],mean[:10]+z@sqrt_psd(cov[:10,:10]).T])
        records=[]
        for pi in samples:
            clock.check(); pi,_=project_physical(pi,self.geometry)
            hypothesis=pi_to_payload(pi,self.geometry)
            k=self._integrate(hypothesis,configured,action,np.asarray(start),False,clock)
            e=np.linalg.norm(k['ee'],axis=1)
            records.append(dict(hypothesis=pi.tolist(),peak_m=float(e.max()),rms_m=float(np.sqrt(np.mean(e**2))),
                hold_m=float(e[-50:].max()),torque_Nm=np.max(abs(k['torque']),axis=0).tolist(),
                clearance_m=float(k['clearance'].min()),collision=bool(k['collision']),
                unsupported=bool(k['unsupported'])))
        return records
