"""Compile acquired evidence once, using CPU workers and immutable feature caches."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import shutil
import time
from cgx.models.features import FEATURE_VERSION
from cgx.provenance import require_protocol_lock,atomic_json
from .replay import Replay,canonical_hash,hash_file

_WORKER=None


def _initialize(root,feature_directory):
    global _WORKER
    from cgx.data.store import Store
    _WORKER=Replay.__new__(Replay);_WORKER.root=Path(root);_WORKER.store=Store(root);_WORKER.feature_directory=Path(feature_directory)


def _compile(ref):
    path=_WORKER.feature_directory/ref[:2]/(ref+'.npz')
    if shutil.disk_usage(_WORKER.root).free<10*1024**3:raise RuntimeError('Feature preparation stopped before disk-full')
    fast=path.with_suffix('.cgxf')
    if not path.exists() or not fast.exists():_WORKER._features(ref)
    return dict(history_hash=ref,relative_path=str(path.relative_to(_WORKER.root)),sha256=hash_file(path),bytes=path.stat().st_size,
                fast_relative_path=str(fast.relative_to(_WORKER.root)),fast_sha256=hash_file(fast),fast_bytes=fast.stat().st_size)


def prepare_replay(root,config,workers=16,resume=False):
    root=Path(root);lock=require_protocol_lock(root,config);started=time.perf_counter()
    replays=[Replay(root,s) for s in ('train','validation')];directory=replays[0].feature_directory
    identity=dict(protocol_lock_hash=canonical_hash(lock),replays={r.split:r.source_hash for r in replays},feature_version=FEATURE_VERSION,feature_directory=str(directory.relative_to(root)),storage_version='lossless_cgxf_v1')
    marker=root/'training'/'features_complete.json';manifest=root/'training'/'feature_hashes.jsonl'
    if resume and marker.exists():
        previous=json.loads(marker.read_text())
        if previous['identity']!=identity or previous['manifest_sha256']!=hash_file(manifest):raise RuntimeError('Prepared replay identity changed')
        with open(manifest) as stream:
            for line in stream:
                row=json.loads(line);path=root/row['relative_path']
                if not path.exists() or hash_file(path)!=row['sha256']:raise RuntimeError(f'Prepared feature artifact corrupt: {path}')
                fast=root/row['fast_relative_path']
                if not fast.exists() or hash_file(fast)!=row['fast_sha256']:raise RuntimeError(f'Fast feature artifact corrupt: {fast}')
        return previous
    plan_report={r.split:{'cases':len(r.case_rows),'eligible_nonstop_transitions':sum(r.plan_counts.values()),'missing_case_ids':[c[0] for c in r.case_rows if not r.plan_counts.get(c[0],0)],'per_case_counts':r.plan_counts} for r in replays}
    atomic_json(root/'training/fixed_plan_support.json',plan_report)
    if plan_report['train']['missing_case_ids']:raise RuntimeError('Source cases lack feedback-independent plan support')
    refs=set()
    for replay in replays:
        for row in replay.rows():refs.update([row['history_hash'],row['next_history_hash']])
    refs=sorted(refs);allocated=len(os.sched_getaffinity(0));workers=min(int(workers),max(1,allocated//2));temporary=manifest.with_suffix('.building.jsonl')
    total_bytes=0;fast_bytes=0;completed=0
    with open(temporary,'w') as output,ProcessPoolExecutor(max_workers=workers,initializer=_initialize,initargs=(str(root),str(directory))) as pool:
        for row in pool.map(_compile,refs,chunksize=16):
            output.write(json.dumps(row,sort_keys=True)+'\n');total_bytes+=row['bytes'];completed+=1
            fast_bytes+=row['fast_bytes']
            if completed%1000==0:
                progress=dict(completed=completed,total=len(refs),elapsed_seconds=time.perf_counter()-started,bytes_written=total_bytes,
                              projected_bytes=total_bytes/max(1,completed)*len(refs),workers=workers,feature_version=FEATURE_VERSION)
                atomic_json(root/'training'/'feature_progress.json',progress);print(json.dumps(progress),flush=True)
        output.flush();os.fsync(output.fileno())
    os.replace(temporary,manifest)
    raw_bytes=sum(p.stat().st_size for p in (root/'public'/'observations').glob('*.json.gz'))
    result=dict(state='passed',identity=identity,unique_acquired_histories=completed,feature_cache_bytes=total_bytes+fast_bytes,compressed_feature_bytes=total_bytes,fast_feature_bytes=fast_bytes,
                retained_compressed_observation_bytes=raw_bytes,workers=workers,elapsed_seconds=time.perf_counter()-started,
                manifest_sha256=hash_file(manifest),technical_smoke=bool(lock.get('technical_smoke',False)),raw_retention='lossless compressed observations preserved')
    atomic_json(marker,result);return result
