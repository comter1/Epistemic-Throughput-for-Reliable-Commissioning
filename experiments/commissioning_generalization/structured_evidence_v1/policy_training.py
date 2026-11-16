"""Original fitted-Q algorithm with newly computed structured OOF supervision.

The update loop is copied from the immutable original trainer; network widths,
losses, augmentation, random sampling, optimizer, finite horizons, checkpoint
selection and early stopping are unchanged. Only reader/replay paths differ.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import replace
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import time
import numpy as np
import torch
from torch import nn
from cgx.baselines import affordable
from cgx.config import load_config
from cgx.models import PolicyModel, collate_features, collate_actions
from cgx.models.features import RESOURCES
from cgx.provenance import atomic_json, read_json, file_sha256
from cgx.training.engine import (_scores_db, _device, _seed, _resume,
    _augment, _autocast, _state, _save, _selection_key, _mark_complete,
    transition_key)
from cgx.training.objectives import frozen
from cgx.training.replay import canonical_hash, hash_file
from cgx.training.stopping import should_stop
from .selectors import (ReadOnlyReplay, source_history_provenance, reader_fingerprint,
    ImmutableScoreLookup, consume_original_reader_initialization)


def load_reader(root,name='final',device='cpu'):
    from .reader import load_reader as implementation
    return implementation(root,name=name,device=device)


def require_training_gate(root):
    gate=read_json(Path(root)/'manifests/policy_training_authorized.json')
    assert gate['state']=='authorized_after_reader_replay', 'Complete reader-only replay before retraining a selector'
    replay=Path(gate['reader_replay_complete_path'])
    assert file_sha256(replay)==gate['reader_replay_complete_sha256']
    assert read_json(replay)['state']=='passed'
    return gate


def _identity(source_root,config,replay,extra=None):
    """Version the new experiment without rewriting the original source seal."""
    from cgx.config import EXPERIMENT
    from cgx.models.features import FEATURE_VERSION
    source_root=Path(source_root)
    lock=read_json(source_root/'protocol_lock.json')
    assert canonical_hash(config)==lock['config_hash'], 'Benchmark numerical configuration changed'
    amendment_path=replay.root/'manifests/source_version_amendment.json'
    amendment=read_json(amendment_path)
    assert amendment['state']=='passed' and amendment['active_config_exactly_matches_original']
    assert amendment['original_scientific_runtime_unchanged']
    assert amendment['original_protocol_lock_sha256']==file_sha256(source_root/'protocol_lock.json')
    inventory=read_json(source_root/'software/source_files.json')
    # The old global source seal predates display-name/reporting-only edits.
    # Scientific selector tensors and network implementations must still match.
    for name in ('src/cgx/models/features.py','src/cgx/models/networks.py'):
        assert file_sha256(EXPERIMENT/name)==inventory[name],f'Original scientific module changed: {name}'
    return dict(protocol_hash=canonical_hash(config),original_protocol_lock_hash=file_sha256(source_root/'protocol_lock.json'),
        source_version_amendment_hash=file_sha256(amendment_path),
        replay_hash=replay.source_hash,feature_version=FEATURE_VERSION,
        feature_code_hash=file_sha256(EXPERIMENT/'src/cgx/models/features.py'),
        network_code_hash=file_sha256(EXPERIMENT/'src/cgx/models/networks.py'),**(extra or {}))


def fit_policy(source_root,root,config,method,seed,resume=False,technical_smoke=False,validation_fn=None,device=None,technical_updates=2,technical_batch_size=2):
    from cgx.observation.actions import action_catalog
    if method not in config['methods']['learned']:raise ValueError(f'Unregistered learned method: {method}')
    if seed not in config['seeds']['training']:raise ValueError(f'Unregistered training seed: {seed}')
    if validation_fn is None:raise ValueError('Full source-validation rollout callback is required; TD loss cannot select checkpoints')
    root=Path(root);source_root=Path(source_root)
    require_training_gate(root)
    replay=ReadOnlyReplay(source_root,root,'train');scores,score_marker=_scores_db(root);device=_device(device);rng=_seed(seed)
    score_lookup=ImmutableScoreLookup(scores)
    reader=load_reader(root,'final',device='cpu');reader_before=reader_fingerprint(reader)
    initialization_control=consume_original_reader_initialization(source_root)
    model=PolicyModel(method).to(device);target=copy.deepcopy(model);frozen(target)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'])
    identity=_identity(source_root,config,replay,dict(method=method,seed=seed,reward_hash=score_marker['output_hash'],reader_hash=reader.checkpoint_hash,replay_view='feedback_independent_affordable_prefixes_v1' if method=='gnn_fixed_plan' else 'common_all_transitions',diagnostic_family='structured_evidence_v1',adapter_code_hash=hash_file(Path(__file__)),policy_initialization=initialization_control))
    directory=root/'models'/'policies'/f'{method}_seed{seed}';directory.mkdir(parents=True,exist_ok=True)
    if resume and (directory/'complete.json').exists():
        result=json.loads((directory/'complete.json').read_text())
        if result['identity']!=identity or hash_file(directory/result['selected_checkpoint'])!=result['selected_checkpoint_hash']:raise RuntimeError('Policy resume input mismatch')
        return result
    checkpoint=_resume(directory,model,optimizer,rng,identity,resume);start=checkpoint['iteration'];history=checkpoint['validation_history']
    if 'target_weights' in checkpoint:target.load_state_dict(checkpoint['target_weights'])
    cap=technical_updates if technical_smoke else config['training']['policy_update_cap'];interval=cap if technical_smoke else config['training']['validation_interval'];batch_size=technical_batch_size if technical_smoke else config['training']['batch_size'];micro=batch_size if technical_smoke else int(os.environ.get('CGX_MICROBATCH','128'))
    if technical_smoke and 'technical_smoke' not in str(root):raise ValueError('Technical smoke requires a separate technical_smoke root')
    catalog_cache={};trajectory_cache={}
    def catalog(graph):
        # Replay retains immutable graph objects; identity is a cache key only,
        # never a feature or a learned system-ID embedding.
        key=id(graph)
        if key not in catalog_cache:catalog_cache[key]=action_catalog(graph,config)
        return catalog_cache[key]
    def prediction(ref,fold):
        return score_lookup.prediction(ref,fold)
    def scored(row):
        return score_lookup.score(transition_key(row))
    def features_with_prediction(ref,h,p):
        f=replay.features(ref);nodes=f.nodes.copy();probabilities=dict(zip(p['target_keys'],np.exp2(p['log_probabilities_base2'])))
        for key,i in zip(f.targets[1:],f.target_nodes[1:]):nodes[i,27]=probabilities.get(key,0.)
        nodes[:,28]=probabilities.get('healthy',0.)
        if 'planned_actions' in h:
            counts=np.zeros(len(nodes))
            for a in h['planned_actions']:
                for port in a['ports']:
                    node=port.split(':')[0]
                    if node in f.node_lookup:counts[f.node_lookup[node]]+=1
            nodes[:,26]=np.log1p(counts)
        initial=h['initial_budget'];remaining=h['remaining']
        resources=np.asarray([remaining.get(k,0)/max(1,initial.get(k,0)) for k in RESOURCES]+[remaining.get('executions',0)/8,remaining.get('calibrations',0)/2,remaining.get('decisions',0)/10,math.log1p(initial.get('bytes',0))/20],np.float32)
        return replace(f,nodes=nodes,resources=resources)
    def training_item(row):
        current_ref=row['history_hash'];next_ref=row['next_history_hash']
        h=replay.training_state(current_ref);nh=replay.training_state(next_ref);acts=catalog(h['graph']);bykey={a['action_key']:a for a in acts};action=bykey[row['action_key']]
        reward,next_nll=scored(row);terminal=row['terminal'];fold=row['source_fold']
        p=prediction(row['history_hash'],fold);npred=prediction(row['next_history_hash'],fold)
        if method=='gnn_fixed_plan':
            tid=row['trajectory_id']
            if tid not in trajectory_cache:
                allrows=replay.trajectory(tid);initial=replay.training_state(allrows[0]['history_hash']);planned=[];valid=[];state={**initial,'remaining':dict(initial['remaining'])}
                for r in allrows:
                    a=bykey[r['action_key']]
                    if a['kind']=='stop' or not affordable(a,state['remaining']):break
                    valid.append(r);planned.append(a);state['remaining']={k:state['remaining'][k]-a['cost_bound'][k] for k in RESOURCES}
                trajectory_cache[tid]=(allrows[0],valid,planned)
                if len(trajectory_cache)>1024:trajectory_cache.pop(next(iter(trajectory_cache)))
            first,valid,planned=trajectory_cache[tid];position=next((i for i,r in enumerate(valid) if r['step']==row['step']),None)
            if position is None:return None
            current_ref=next_ref=first['history_hash']
            h=replay.training_state(current_ref);h['planned_actions']=planned[:position]
            for a in planned[:position]:h['remaining']={k:h['remaining'][k]-a['cost_bound'][k] for k in RESOURCES}
            nh={**h,'planned_actions':planned[:position+1],'remaining':{k:h['remaining'][k]-action['cost_bound'][k] for k in RESOURCES}}
            p=prediction(first['history_hash'],fold);npred=p;terminal=position==len(valid)-1
            # Terminal gain belongs to the last real acquisition, not to zero-valued STOP.
            reward=sum(scored(r)[0] for r in valid) if terminal else 0.
        if method in ('gnn_depth2','gnn_myopic'):
            horizon=1 if method=='gnn_myopic' else int(rng.integers(1,3))
            h['remaining']['decisions']=min(horizon,h['remaining']['decisions'])
            nh['remaining']['decisions']=max(0,h['remaining']['decisions']-1)
        current=features_with_prediction(current_ref,h,p);nxt=features_with_prediction(next_ref,nh,npred)
        current=_augment(current,rng);nxt=_augment(nxt,rng)
        nextactions=[a for a in acts if affordable(a,nh['remaining'])]
        return current,nxt,action,nextactions,float(reward),float(next_nll),bool(terminal),nh['remaining']['decisions']
    started=time.perf_counter();interval_started=started
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    iteration=start
    effective_cap=start if not technical_smoke and should_stop(history,config['training']) else cap
    for iteration in range(start+1,effective_cap+1):
        model.train();optimizer.zero_grad(set_to_none=True);items=[];attempts=0
        while len(items)<batch_size:
            sampled=replay.sample(rng,batch_size-len(items),feedback_independent=method=='gnn_fixed_plan')
            replay.prefetch_features([r[k] for r in sampled for k in ('history_hash','next_history_hash')])
            for row in sampled:
                value=training_item(row);attempts+=1
                if value is not None:items.append(value)
            if attempts>100*batch_size:raise RuntimeError('No affordable no-feedback replay support')
        loss_total=0.
        for off in range(0,batch_size,micro):
            subset=items[off:off+micro];batch=collate_features([x[0] for x in subset],device);actions=collate_actions(batch,[[x[2]] for x in subset],device)
            rewards=torch.tensor([x[4] for x in subset],device=device);terminal=torch.tensor([x[6] for x in subset],device=device);horizon=torch.tensor([x[7] for x in subset],device=device)
            with _autocast(device):
                values,nextloss=model(batch,actions,return_next_loss=True)
                if method=='gnn_myopic':targets=rewards
                else:
                    nb=collate_features([x[1] for x in subset],device);na=collate_actions(nb,[x[3] for x in subset],device)
                    with torch.no_grad():
                        online_next=model(nb,na);target_next=target(nb,na)
                        online_chunks=online_next.split(na['counts']);target_chunks=target_next.split(na['counts']);continuation=torch.stack([t[o.argmax()] for o,t in zip(online_chunks,target_chunks)])
                        targets=rewards+torch.where(terminal|(horizon<=0),torch.zeros_like(continuation),continuation)
                loss=nn.functional.smooth_l1_loss(values.float(),targets.float())
                if method=='gnn_myopic':loss=loss+nn.functional.smooth_l1_loss(nextloss.float(),torch.tensor([x[5] for x in subset],device=device))
                loss=loss*len(subset)/batch_size
            loss.backward();loss_total+=float(loss.detach())
        norm=nn.utils.clip_grad_norm_(model.parameters(),config['training']['grad_clip_norm'])
        if not math.isfinite(loss_total) or not torch.isfinite(norm):raise FloatingPointError('Policy nonfinite update')
        optimizer.step()
        if iteration%config['training']['target_update_interval']==0:target.load_state_dict(model.state_dict())
        if iteration%interval==0 or iteration==cap:
            update_seconds=time.perf_counter()-interval_started;validation_started=time.perf_counter()
            model.eval();metric=validation_fn(model,reader,method)
            if not math.isfinite(metric['mean_final_nll_bits']):raise FloatingPointError('Nonfinite full-rollout validation')
            interval_updates=iteration-(history[-1]['iteration'] if history else start)
            metric.update(iteration=iteration,training_huber_loss=loss_total,update_interval_seconds=update_seconds,
                interval_updates=interval_updates,updates_per_second_excluding_validation=interval_updates/update_seconds,
                validation_seconds=time.perf_counter()-validation_started);history.append(metric)
            filename=f'checkpoint_{iteration:07d}.pt';state=_state(model,optimizer,rng,iteration,identity,history,dict(method=method,target_weights=target.state_dict(),microbatch=micro))
            _save(directory/filename,state);_save(directory/'last.pt',state);atomic_json(directory/'validation.json',history);print(json.dumps({'method':method,'seed':seed,**metric}),flush=True)
            if not technical_smoke and should_stop(history,config['training']):break
            interval_started=time.perf_counter()
    after=reader_fingerprint(reader)
    if reader_before!=after:raise RuntimeError('Frozen common reader mutated during policy updates')
    best=min(history,key=_selection_key);return _mark_complete(directory,identity,f"checkpoint_{best['iteration']:07d}.pt",iteration,started,model,technical_smoke,cap,config['training'])


_SCORING=None


def _scoring_initialize(source_root,run_root):
    global _SCORING
    from functools import lru_cache
    torch.set_num_threads(1)
    replay=ReadOnlyReplay(source_root,run_root,'train')
    # Repeated prefixes of a case share observations; retain a bounded decoded
    # working set in each worker, without changing their provenance identities.
    replay.store.load_observation=lru_cache(maxsize=24)(replay.store.load_observation)
    readers={i:load_reader(run_root,f'crossfit_{i}',device='cpu') for i in range(3)}
    _SCORING=(replay,readers)


def _score_case(job):
    cid,_,fold,cfg=job;replay,readers=_SCORING;reader=readers[fold]
    if cfg in set(reader.training_configurations):raise RuntimeError('Structured OOF reader trained on excluded configuration')
    rows=replay.case_transitions(cid);provenance=source_history_provenance(rows)
    predictions={};outputs=[]
    for ref in sorted(provenance):
        history=replay.history(ref)
        ids={**provenance[ref],'observation_refs':replay.store.history_references(ref)['observation_refs']}
        assert len(history['observations'])==len(ids['observation_ids'])
        p=reader.predict(history,provenance=ids)
        values=np.asarray(p['log_probabilities_base2'],float)
        assert np.isfinite(values).all() and abs(np.exp2(values).sum()-1)<1e-6
        assert rows[0]['true_target_key'] in p['target_keys']
        # Store the explicit envelope with each prediction. Cache identity
        # includes provenance; numeric equality alone never defines an event.
        p={**p,'event_provenance':ids,'event_provenance_hash':canonical_hash(ids),
           'reader_hash':reader.checkpoint_hash,'prediction_cache_key':canonical_hash([ref,ids,reader.checkpoint_hash])}
        predictions[ref]=p
        outputs.append((ref,fold,json.dumps(p,separators=(',',':'))))
    scores=[]
    for row in rows:
        before=predictions[row['history_hash']];after=predictions[row['next_history_hash']]
        key=row['true_target_key']
        bp=before['log_probabilities_base2'][before['target_keys'].index(key)]
        ap=after['log_probabilities_base2'][after['target_keys'].index(key)]
        reward=ap-bp
        if row['action']['kind']=='stop':assert row['history_hash']==row['next_history_hash'] and reward==0.
        scores.append((transition_key(row),reward,-ap,reader.checkpoint_hash,row['trajectory_id'],row['step'],row['history_hash'],row['next_history_hash']))
    return cid,outputs,scores


def score_transitions(source_root,run_root,workers=32,resume=True):
    """Replace every old reward with same-history structured OOF log-score gain."""
    import fcntl
    run=Path(run_root);directory=run/'training';directory.mkdir(parents=True,exist_ok=True)
    replay=ReadOnlyReplay(source_root,run,'train');path=directory/'scores.sqlite';marker=directory/'scores_complete.json'
    reader_hashes={str(i):load_reader(run,f'crossfit_{i}',device='cpu').checkpoint_hash for i in range(3)}
    identity=_identity(source_root,load_config(),replay,dict(readers=reader_hashes,
        reward_definition='structured_oof_log2_true_probability_after_minus_before',old_reward_labels_read=False,
        score_adapter_sha256=file_sha256(Path(__file__))))
    if marker.exists():
        result=read_json(marker)
        assert result['identity']==identity and result['output_hash']==file_sha256(path)
        return result
    with (directory/'score_driver.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        db=sqlite3.connect(path)
        db.executescript('CREATE TABLE IF NOT EXISTS predictions(history TEXT,fold INTEGER,data TEXT,PRIMARY KEY(history,fold)); CREATE TABLE IF NOT EXISTS scores(id TEXT PRIMARY KEY,reward REAL,next_nll REAL,reader_hash TEXT,trajectory TEXT,step INTEGER,history TEXT,next_history TEXT); CREATE INDEX IF NOT EXISTS trajectory ON scores(trajectory); CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE IF NOT EXISTS completed_cases(case_id TEXT PRIMARY KEY);')
        old=db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
        if old:assert json.loads(old[0])==identity
        else:db.execute('INSERT INTO meta VALUES(?,?)',('identity',json.dumps(identity)));db.commit()
        done={r[0] for r in db.execute('SELECT case_id FROM completed_cases')}
        if done and not resume:raise RuntimeError('Existing score work requires explicit resume')
        pending=[r for r in replay.case_rows if r[0] not in done]
        started=time.monotonic();count=len(done)
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_scoring_initialize,
                initargs=(str(source_root),str(run))) as pool:
            for cid,predictions,scores in pool.map(_score_case,pending,chunksize=1):
                db.executemany('INSERT OR REPLACE INTO predictions VALUES(?,?,?)',predictions)
                db.executemany('INSERT OR REPLACE INTO scores VALUES(?,?,?,?,?,?,?,?)',scores)
                db.execute('INSERT INTO completed_cases VALUES(?)',(cid,));db.commit();count+=1
                if count%32==0:
                    progress=dict(state='running',cases=count,total=len(replay.case_rows),elapsed_seconds=time.monotonic()-started)
                    atomic_json(directory/'score_progress.json',progress);print(json.dumps(progress),flush=True)
        total=db.execute('SELECT COUNT(*) FROM scores').fetchone()[0]
        expected=replay.db.execute('SELECT COUNT(*) FROM transitions').fetchone()[0]
        assert total==expected and count==len(replay.case_rows)
        unique=db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
        # Labels were used only here, after every public-history prediction.
        db.close()
        result=dict(state='passed',identity=identity,transitions=total,unique_predictions=unique,
            output_hash=file_sha256(path),cases=count,source_split='train',technical_smoke=False,
            old_reward_labels_read=False,all_topology_exclusion_checks_passed=True,
            elapsed_seconds=time.monotonic()-started,workers=workers)
        atomic_json(marker,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['score','fit'])
    p.add_argument('--source-root',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--method');p.add_argument('--seed',type=int);p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,default=32);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.command=='score':result=score_transitions(a.source_root,a.run_dir,a.workers,a.resume)
    else:
        from .evaluation import validation_callback
        directory=Path(a.run_dir)/'validation'/f'{a.method}_seed{a.seed}'
        callback=validation_callback(a.source_root,a.run_dir,load_config(),a.workers,directory)
        result=fit_policy(a.source_root,a.run_dir,load_config(),a.method,a.seed,resume=a.resume,
            validation_fn=callback,device=a.device)
    print(json.dumps(result,sort_keys=True),flush=True)
