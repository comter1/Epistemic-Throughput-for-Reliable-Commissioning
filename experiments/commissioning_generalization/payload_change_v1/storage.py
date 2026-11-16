"""Atomic provenance and a process-safe, conservative physics budget ledger."""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


class PhysicsLedger:
    """Reserve before stepping; interrupted reservations retain their entire charge.

    The cap includes source, engineering checks, failed runs, retries and audit.
    Model compilation/instantaneous inverse dynamics is counted separately by callers.
    Locks are short and cover only metadata, never simulator execution.
    """
    def __init__(self, run_root, cap_seconds=1_000_000.0):
        self.root = Path(run_root) / 'resources'
        self.root.mkdir(parents=True, exist_ok=True)
        self.cap = float(cap_seconds)

    @contextlib.contextmanager
    def _locked(self):
        with open(self.root / 'physics.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / 'physics_ledger.json'
            state = json.loads(path.read_text()) if path.exists() else {
                'cap_seconds': self.cap, 'charged_seconds': 0.0,
                'completed_seconds': 0.0, 'reservations': {}, 'completed_blocks': 0}
            yield state
            write_json(path, state)
            fcntl.flock(lock, fcntl.LOCK_UN)

    def reserve(self, seconds, unit='unspecified', kind='physics'):
        seconds = float(seconds)
        if seconds <= 0:
            raise ValueError('reservation must be positive')
        token = uuid.uuid4().hex
        record = {'seconds': seconds, 'unit': str(unit), 'kind': str(kind),
                  'reserved_at': time.time(), 'pid': os.getpid()}
        with self._locked() as state:
            if state['charged_seconds'] + seconds > state['cap_seconds'] + 1e-8:
                raise RuntimeError('100,000 ten-second execution-equivalent budget exhausted')
            state['charged_seconds'] += seconds
            state['reservations'][token] = record
            with open(self.root / 'physics_events.jsonl', 'a') as events:
                events.write(canonical({'event': 'reserve', 'token': token, **record}) + '\n')
        return token

    def complete(self, token, actual_seconds, status='completed'):
        actual = float(actual_seconds)
        with self._locked() as state:
            record = state['reservations'].pop(token)
            if actual < 0 or actual > record['seconds'] + 1e-6:
                state['reservations'][token] = record
                raise ValueError('actual physics exceeds reserved amount')
            state['charged_seconds'] += actual - record['seconds']
            state['completed_seconds'] += actual
            state['completed_blocks'] += 1
            with open(self.root / 'physics_events.jsonl', 'a') as events:
                events.write(canonical({'event': 'complete', 'token': token, **record,
                                        'actual_seconds': actual, 'status': status,
                                        'finished_at': time.time()}) + '\n')

    def summary(self):
        with self._locked() as state:
            return {k: v for k, v in state.items() if k != 'reservations'} | {
                'pending_blocks': len(state['reservations']),
                'ten_second_equivalents': state['charged_seconds'] / 10}
