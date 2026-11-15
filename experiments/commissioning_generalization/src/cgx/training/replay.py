"""A shared disk-backed replay index with uniform-case sampling and hash checks."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from functools import lru_cache
import numpy as np
from cgx.models.features import tensorize_history, FEATURE_VERSION, GraphFeatures


def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def canonical_hash(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class Replay:
    def __init__(self,root,split='train',create=True):
        # A cached mmap holds an OS descriptor. SSH sessions may inherit a1024
        # soft limit despite a much larger hard limit; two replays need more.
        import resource
        soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
        desired=16384 if hard==resource.RLIM_INFINITY else min(16384,hard)
        if soft<desired:resource.setrlimit(resource.RLIMIT_NOFILE,(desired,hard))
        from cgx.data.store import Store
        self.root=Path(root); self.split=split; self.store=Store(root)
        self.path=self.root/'training'/f'replay_{split}.sqlite'; self.path.parent.mkdir(parents=True,exist_ok=True)
        source=self.root/'private'/'transitions'/split
        self.source_hash=canonical_hash([(p.name,hash_file(p)) for p in sorted(source.glob('*.jsonl.gz'))])
        if not source.exists() or not any(source.glob('*.jsonl.gz')): raise RuntimeError(f'No collected {split} transitions: {source}')
        with open(self.path.with_suffix('.lock'),'a') as lock_file:
            fcntl.flock(lock_file,fcntl.LOCK_EX)
            if not self.path.exists():
                if not create: raise FileNotFoundError(self.path)
                temp=self.path.with_suffix('.building.sqlite')
                if temp.exists(): temp.unlink()
                db=sqlite3.connect(temp)
                db.executescript('CREATE TABLE transitions(case_id TEXT, idx INTEGER, trajectory TEXT, fold INTEGER, data TEXT, PRIMARY KEY(case_id,idx)); CREATE TABLE cases(case_id TEXT PRIMARY KEY, n INTEGER, fold INTEGER, configuration TEXT); CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT); CREATE INDEX trajectory_index ON transitions(trajectory);')
                last=None; counts={}; case_meta={}
                for row in self.store.transitions(split):
                    cid=row['parent_case_id']; idx=counts.get(cid,0); counts[cid]=idx+1
                    case_meta[cid]=(row['source_fold'],row['configuration_id'])
                    db.execute('INSERT INTO transitions VALUES(?,?,?,?,?)',(cid,idx,row['trajectory_id'],row['source_fold'],json.dumps(row,separators=(',',':'))))
                if not counts: raise RuntimeError('Collected replay has no transitions')
                db.executemany('INSERT INTO cases VALUES(?,?,?,?)',[(k,n,*case_meta[k]) for k,n in counts.items()])
                db.execute('INSERT INTO meta VALUES(?,?)',('source_hash',self.source_hash)); db.commit(); db.close(); os.replace(temp,self.path)
        self.db=sqlite3.connect(self.path)
        stored=self.db.execute("SELECT value FROM meta WHERE key='source_hash'").fetchone()[0]
        if stored!=self.source_hash: raise RuntimeError('Replay input changed; use new protocol/run directory, not stale resume')
        with open(self.path.with_suffix('.lock'),'a') as lock_file:
            fcntl.flock(lock_file,fcntl.LOCK_EX)
            exists=self.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='plan_support'").fetchone()
            if not exists:
                self.db.execute('CREATE TABLE plan_support(case_id TEXT,sample_idx INTEGER,transition_idx INTEGER,PRIMARY KEY(case_id,sample_idx))')
                budgets={};rejected=set();counts={}
                for cid,idx,tid,data in self.db.execute('SELECT case_id,idx,trajectory,data FROM transitions ORDER BY case_id,idx'):
                    row=json.loads(data)
                    if row.get('behavior') not in ('random_legal','fixed_diverse_plan') and row.get('coverage_branch') is None:continue
                    action=row['action']
                    if action['kind']=='stop' or tid in rejected:continue
                    if tid not in budgets:budgets[tid]=dict(row['resources_before'])
                    budget=budgets[tid];cost=action['cost_bound']
                    if any(cost[k]>budget[k] for k in budget):rejected.add(tid);continue
                    sample=counts.get(cid,0);self.db.execute('INSERT INTO plan_support VALUES(?,?,?)',(cid,sample,idx));counts[cid]=sample+1
                    budgets[tid]={k:budget[k]-cost[k] for k in budget}
                self.db.commit()
        self.plan_counts=dict(self.db.execute('SELECT case_id,COUNT(*) FROM plan_support GROUP BY case_id').fetchall())
        self.case_rows=self.db.execute('SELECT case_id,n,fold,configuration FROM cases ORDER BY case_id').fetchall()
        # Preparation has committed the complete immutable index. Avoid shared
        # filesystem lock/unlock RPCs on each of millions of training SELECTs.
        self.db.close()
        self.db=sqlite3.connect(self.path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
        self.feature_directory=self.root/'encoded'/canonical_hash([FEATURE_VERSION,hash_file(Path(__file__).parents[1]/'models'/'features.py')])
        self.feature_directory.mkdir(parents=True,exist_ok=True)
        soft,_=resource.getrlimit(resource.RLIMIT_NOFILE)
        self.cache=lru_cache(maxsize=min(1024,max(16,soft//8)))(self._features)
        self._prefetch_pool=None
        self.state_cache=lru_cache(maxsize=8192)(self._state)
        self.graph_cache=lru_cache(maxsize=512)(lambda ref:self.store._get('graphs',ref))
        self.eligible={fold:[r for r in self.case_rows if r[2]!=fold] for fold in (None,0,1,2)}
    def rows(self):
        for (value,) in self.db.execute('SELECT data FROM transitions ORDER BY case_id,idx'): yield json.loads(value)
    def case_transitions(self,cid):
        return [json.loads(row[0]) for row in self.db.execute('SELECT data FROM transitions WHERE case_id=? ORDER BY idx',(cid,))]
    def trajectory(self,tid):
        return sorted([json.loads(row[0]) for row in self.db.execute('SELECT data FROM transitions WHERE trajectory=?',(tid,))],key=lambda r:r['step'])
    def sample(self,rng,n,excluded_fold=None,feedback_independent=False):
        eligible=self.eligible[excluded_fold]
        if not eligible: raise RuntimeError('Crossfit exclusion removed every case')
        if feedback_independent and any(not self.plan_counts.get(r[0],0) for r in eligible):raise RuntimeError('Missing feedback-independent source-case coverage; run shared fixed-plan support pass')
        out=[]
        for i in rng.integers(0,len(eligible),n):
            cid,count,fold,cfg=eligible[i]
            if feedback_independent:
                index=self.db.execute('SELECT transition_idx FROM plan_support WHERE case_id=? AND sample_idx=?',(cid,int(rng.integers(self.plan_counts[cid])))).fetchone()[0]
            else:index=int(rng.integers(count))
            row=self.db.execute('SELECT data FROM transitions WHERE case_id=? AND idx=?',(cid,index)).fetchone()[0]
            out.append(json.loads(row))
        return out
    def history(self,ref): return self.store.load_history(ref)
    def _state(self,ref):
        packed=self.store.history_references(ref)
        return dict(graph=self.graph_cache(packed['graph_ref']),remaining=packed['remaining'],initial_budget=packed['initial_budget'],action_keys=packed['action_keys'])
    def state(self,ref):
        import copy
        return copy.deepcopy(self.state_cache(ref))
    def training_state(self,ref):
        """Trainer-only view: immutable graph shared; mutable budgets copied.

        The public state() API remains defensively copied. Internal training
        never mutates topology, and index randomization operates on fresh arrays.
        """
        state=self.state_cache(ref)
        return {**state,'remaining':dict(state['remaining']),'initial_budget':dict(state['initial_budget']),
                'action_keys':list(state['action_keys'])}
    def _features(self,ref):
        path=self.feature_directory/ref[:2]/(ref+'.npz')
        from .feature_cache import read_features,write_features
        fast=path.with_suffix('.cgxf')
        if fast.exists():return read_features(fast)
        if path.exists():
            with np.load(path,allow_pickle=False) as f:
                metadata=json.loads(str(f['metadata']))
                sequences=np.split(f['sequences'],np.cumsum(f['lengths'])[:-1]) if len(f['lengths']) else []
                features=GraphFeatures(nodes=f['nodes'],edge_index=f['edge_index'],edge_types=f['edge_types'],edge_features=f['edge_features'],
                    sequences=sequences,resources=f['resources'],task=f['task'],**metadata)
            write_features(fast,features)
            return features
        features=tensorize_history(self.history(ref));path.parent.mkdir(parents=True,exist_ok=True)
        metadata={k:getattr(features,k) for k in ('sequence_owners','targets','target_nodes','target_families','prior','physical_count','node_lookup','clock_lookup')}
        temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
        with open(temporary,'wb') as stream:
            np.savez_compressed(stream,nodes=features.nodes,edge_index=features.edge_index,edge_types=features.edge_types,edge_features=features.edge_features,
                sequences=np.concatenate(features.sequences) if features.sequences else np.empty((0,64),np.float32),lengths=np.asarray([len(s) for s in features.sequences]),
                resources=features.resources,task=features.task,metadata=json.dumps(metadata,sort_keys=True))
        os.replace(temporary,path)
        write_features(fast,features)
        return features
    def features(self,ref): return self.cache(ref)

    def prefetch_features(self,refs):
        """Parallel file decoding only; sampling and augmentation keep their RNG order."""
        from concurrent.futures import ThreadPoolExecutor
        count=max(1,int(os.environ.get('CGX_FEATURE_READERS','4')))
        if count==1:
            for ref in dict.fromkeys(refs):self.features(ref)
            return
        if self._prefetch_pool is None:self._prefetch_pool=ThreadPoolExecutor(max_workers=count)
        for _ in self._prefetch_pool.map(self.features,dict.fromkeys(refs)):pass
