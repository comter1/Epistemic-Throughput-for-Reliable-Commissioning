"""Atomic V2 attempt ledger. V1 and original ledgers are never opened for writing."""
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import json
import os
import socket
import time


class BudgetExceeded(RuntimeError):pass


class Ledger:
    LIMITS={'robot_execution':16000,'calibration':10000,'model_and_features':200000,
            'head_fit':24,'updates_per_fit':2000,'training_gpu_seconds':48*3600}
    KINDS=('robot_execution','calibration','model_forward','feature_build','reader_partial_forward',
           'head_fit','head_update','head_forward','cache_hit','training_gpu_seconds')
    def __init__(self,run):
        self.run=Path(run);self.run.mkdir(parents=True,exist_ok=True)
        self.path=self.run/'resource_ledger.json';self.lock=self.run/'resource_ledger.lock'

    def _read(self):
        return json.loads(self.path.read_text()) if self.path.exists() else dict(
            counts={k:0 for k in self.KINDS},successful={'robot_execution':0,'calibration':0},
            limits=self.LIMITS,by_detail={},fits={},attempts_include_failures=True)

    def _save(self,state):
        state.update(updated_at=datetime.now(timezone.utc).isoformat(),last_host=socket.gethostname())
        temporary=self.path.with_name(f'.ledger.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n');temporary.replace(self.path)

    def reserve(self,kind,count=1,detail=''):
        if kind not in self.KINDS or count<0 or (kind!='training_gpu_seconds' and type(count) is not int):
            raise ValueError('Invalid resource reservation')
        with self.lock.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=self._read();proposed=state['counts'].copy();proposed[kind]+=count
            for key in ('robot_execution','calibration','head_fit','training_gpu_seconds'):
                if proposed[key]>self.LIMITS[key]:raise BudgetExceeded(f'V2 {key} hard cap reached')
            total=sum(proposed[k] for k in ('model_forward','feature_build','reader_partial_forward'))
            if total>self.LIMITS['model_and_features']:raise BudgetExceeded('V2 combined model/feature cap reached')
            if kind=='head_update':
                if not detail:raise ValueError('head_update requires exact fit ID in detail')
                fit=state['fits'].setdefault(detail,{'updates':0,'registered':False})
                if not fit['registered']:raise ValueError('Register fit before updates')
                if fit['updates']+count>2000:raise BudgetExceeded('V2 2000 updates per fit hard cap reached')
                fit['updates']+=count
            state['counts']=proposed
            label=detail.split(':',1)[0] or 'unspecified'
            by=state['by_detail'].setdefault(label,{k:0 for k in self.KINDS});by[kind]+=count
            self._save(state);return proposed

    def register_fit(self,fit_id):
        with self.lock.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=self._read()
            fit=state['fits'].setdefault(fit_id,{'updates':0,'registered':False})
            if not fit['registered']:
                if state['counts']['head_fit']>=24:raise BudgetExceeded('V2 24 fit cap reached')
                fit['registered']=True;state['counts']['head_fit']+=1;self._save(state)
            return dict(fit)

    def success(self,kind,count=1,detail=''):
        if kind not in ('robot_execution','calibration'):raise ValueError('Success only for physical operations')
        with self.lock.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=self._read()
            state['successful'][kind]+=count
            if state['successful'][kind]>state['counts'][kind]:raise ValueError('Success without reserved attempt')
            self._save(state)

    def snapshot(self):
        with self.lock.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_SH);return self._read()


class TimedStage:
    def __init__(self,run,name):self.run,self.name=Path(run),name
    def __enter__(self):self.wall,self.cpu=time.monotonic(),time.process_time();return self
    def __exit__(self,kind,value,tb):
        row=dict(stage=self.name,hostname=socket.gethostname(),pid=os.getpid(),wall_seconds=time.monotonic()-self.wall,
                 process_cpu_seconds=time.process_time()-self.cpu,state='passed' if kind is None else 'failed',
                 error=None if kind is None else f'{kind.__name__}: {value}')
        with (self.run/'stage_timings.jsonl').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX);handle.write(json.dumps(row)+'\n')
