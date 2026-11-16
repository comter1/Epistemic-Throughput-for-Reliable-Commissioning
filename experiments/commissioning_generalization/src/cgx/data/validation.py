"""Parallel strict content verification; no evidence blobs are sampled or skipped."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import OrderedDict
import itertools
import os
import time
import multiprocessing as mp
from cgx.provenance import atomic_json
from cgx.config import allocated_cpu_count

_STORE=None
_SEEN=None


def _initialize(root):
    global _STORE,_SEEN
    from .store import Store
    _STORE=Store(root);_SEEN=OrderedDict()


def _verify(digest):
    packed=_STORE.history_references(digest)
    blobs=[('graphs',packed['graph_ref']),('tasks',packed['task_ref'])]
    blobs += [('observations',ref) for ref in packed['observation_refs']]
    for key in blobs:
        if key not in _SEEN:
            _STORE._get(*key)
            _SEEN[key]=None
            if len(_SEEN)>65536:_SEEN.popitem(last=False)
    return digest


def verify_histories(root,refs,split):
    workers=min(max(1,int(os.environ.get('CGX_DATA_WORKERS','16'))),allocated_cpu_count())
    started=time.monotonic();iterator=iter(refs);completed=0
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_initialize,initargs=(str(root),)) as pool:
        while batch:=list(itertools.islice(iterator,4096)):
            for _ in pool.map(_verify,batch,chunksize=32):completed+=1
            atomic_json(Path(root)/'reports'/f'blob_validation_{split}.json',dict(state='running',completed=completed,total=len(refs),workers=workers,elapsed_seconds=time.monotonic()-started))
    report=dict(state='passed',verified_histories=completed,total=len(refs),workers=workers,elapsed_seconds=time.monotonic()-started)
    atomic_json(Path(root)/'reports'/f'blob_validation_{split}.json',report)
    return report
