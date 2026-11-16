"""Exact physical identities, immutable cache reads and a durable fresh-work cap."""
from contextlib import contextmanager
import fcntl
import gzip
import json
import os
from pathlib import Path
import socket
import time

from cgx.data.store import _atomic_gzip
from cgx.provenance import atomic_json, now, read_json
from structured_evidence_v1.rollout import CachedSystem

FRESH_EXECUTION_CAP=190000


class FreshExecutionCapExceeded(RuntimeError):
    pass


@contextmanager
def file_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def reserve_attempt(run_root,identity,robots,calibrations):
    """Count before calling physics; never refund failed/interrupted attempts."""
    directory=Path(run_root)/'resources';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'fresh_attempt_budget.json'
    with file_lock(directory/'fresh_attempt_budget.lock'):
        state=read_json(path) if path.exists() else dict(cap=FRESH_EXECUTION_CAP,
            robot_execution_attempts=0,calibration_attempts=0,physical_operations=0,created_at=now())
        if state['cap']!=FRESH_EXECUTION_CAP:raise ValueError('Registered fresh-work cap changed')
        if state['robot_execution_attempts']+robots>state['cap']:
            raise FreshExecutionCapExceeded('Extension reached its durable190000 fresh robot-attempt cap')
        state['robot_execution_attempts']+=robots
        state['calibration_attempts']+=calibrations
        state['physical_operations']+=1
        state['updated_at']=now()
        event=dict(sequence=state['physical_operations'],identity=identity,robots=robots,
            calibrations=calibrations,host=socket.gethostname(),pid=os.getpid(),reserved_at=now())
        # Counter commits first. A crash before journaling is conservatively counted.
        atomic_json(path,state)
        with (directory/'fresh_attempt_reservations.jsonl').open('a') as stream:
            stream.write(json.dumps(event,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
        return event


class ExtensionCachedSystem(CachedSystem):
    """Only the evaluator receives this object or its private System instance."""
    def __init__(self,*args,readonly_cache_roots=(),attempt_root=None,**kwargs):
        super().__init__(*args,**kwargs)
        if self.cache_root is None or attempt_root is None:raise ValueError('Durable extension cache/cap required')
        self.readonly_cache_roots=tuple(Path(p).resolve() for p in readonly_cache_roots)
        if self.cache_root.resolve() in self.readonly_cache_roots:raise ValueError('Immutable cache cannot be writable')
        self.attempt_root=Path(attempt_root)
        self.imported_cache_hits=self.extension_cache_hits=0

    def _execute(self,kind,parameters,call,robots,calibrations):
        from cgx.env.system import ExecutionResult
        key=self.identity(kind,parameters);self.last_acquisition_id=key
        relative=Path(key[:2])/(key+'.json.gz');path=self.cache_root/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        with file_lock(path.with_suffix('.lock')):
            for root,imported in [(self.cache_root,False)]+[(p,True) for p in self.readonly_cache_roots]:
                candidate=root/relative
                if candidate.is_file():
                    with gzip.open(candidate,'rt') as stream:payload=json.load(stream)
                    if payload['identity']!=key:raise ValueError('Exact physical-cache identity mismatch')
                    self.cache_hits+=1
                    if imported:self.imported_cache_hits+=1
                    else:self.extension_cache_hits+=1
                    self.physical_seconds+=payload['physical_seconds']
                    return ExecutionResult(payload['observation'],b'',payload['metrics'])
            reservation=reserve_attempt(self.attempt_root,key,robots,calibrations)
            self.robot_attempts+=robots;self.calibration_attempts+=calibrations
            journal=path.with_suffix('.attempts.json')
            attempts=read_json(journal)['attempts'] if journal.exists() else []
            event=dict(state='started',started_at=now(),host=socket.gethostname(),pid=os.getpid(),
                execution_attempts=robots,calibration_attempts=calibrations,reservation=reservation)
            attempts.append(event);atomic_json(journal,dict(identity=key,attempts=attempts))
            started=time.monotonic()
            try:result=call()
            except BaseException as exc:
                seconds=time.monotonic()-started;self.actual_physical_seconds+=seconds
                event.update(state='failed',finished_at=now(),seconds=seconds,error=f'{type(exc).__name__}:{exc}')
                atomic_json(journal,dict(identity=key,attempts=attempts));raise
            seconds=time.monotonic()-started
            self.actual_physical_seconds+=seconds;self.physical_seconds+=seconds
            event.update(state='passed',finished_at=now(),seconds=seconds)
            atomic_json(journal,dict(identity=key,attempts=attempts))
            payload=dict(identity=key,observation=result.observation,metrics=result.metrics,
                physical_seconds=seconds,execution_attempts=robots,calibration_attempts=calibrations)
            _atomic_gzip(path,json.dumps(payload,separators=(',',':'),sort_keys=True).encode())
            return result
