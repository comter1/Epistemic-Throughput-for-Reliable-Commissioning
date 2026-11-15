"""Resumable reader and fitted-Q training with topology cross-fitting.

Formal caps cannot be shortened through a convenience argument. Small runs require
technical_smoke=True and are isolated under technical_smoke/models.
"""
from __future__ import annotations
import copy
import gzip
import hashlib
import json
import math
import os
import random
import sqlite3
import time
from pathlib import Path
from dataclasses import replace
from collections import defaultdict
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
from cgx.models import DiagnosticReader,PolicyModel,collate_features,collate_actions,tensorize_history,FEATURE_VERSION,parameter_count
from cgx.models.features import RESOURCES
from cgx.baselines import affordable
from .objectives import joint_nll_bits,double_q_target,frozen
from .replay import Replay,canonical_hash,hash_file
from .stopping import should_stop


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True));os.replace(temp,path)


def _save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    torch.save(value,temp);os.replace(temp,path)


def _torch_load(path,device): return torch.load(path,map_location=device,weights_only=False)


def _seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    return np.random.default_rng(seed)


def _device(device=None):
    from cgx.runtime.determinism import configure_torch
    configure_torch()
    return torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))


def _identity(root,config,replay,extra=None):
    from cgx.provenance import require_protocol_lock
    lock=require_protocol_lock(root,config)
    return dict(protocol_hash=canonical_hash(config),protocol_lock_hash=canonical_hash(lock),replay_hash=replay.source_hash,
                feature_version=FEATURE_VERSION,feature_code_hash=hash_file(Path(__file__).parents[1]/'models'/'features.py'),
                network_code_hash=hash_file(Path(__file__).parents[1]/'models'/'networks.py'),**(extra or {}))


def _rng_state(rng):
    return dict(numpy_generator=rng.bit_generator.state,python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_rng(state,rng):
    rng.bit_generator.state=state['numpy_generator'];random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def _state(model,optimizer,rng,iteration,identity,validation,extra=None):
    return dict(weights=model.state_dict(),optimizer=optimizer.state_dict(),iteration=iteration,identity=identity,rng=_rng_state(rng),
                validation_history=validation,torch_version=torch.__version__,parameter_count=parameter_count(model),**(extra or {}))


def _resume(directory,model,optimizer,rng,identity,resume):
    path=directory/'last.pt'
    if path.exists():
        if not resume: raise RuntimeError(f'Run already exists; explicitly --resume: {directory}')
        checkpoint=_torch_load(path,next(model.parameters()).device)
        if checkpoint['identity']!=identity:raise RuntimeError(f'Resume input/code/config hash mismatch: {directory}')
        model.load_state_dict(checkpoint['weights']);optimizer.load_state_dict(checkpoint['optimizer']);_restore_rng(checkpoint['rng'],rng)
        return checkpoint
    return dict(iteration=0,validation_history=[])


def _augment(f,rng):
    """Random index relabeling including clocks, preserving semantic channels."""
    order=rng.permutation(len(f.nodes)); inverse=np.empty_like(order);inverse[order]=np.arange(len(order))
    return replace(f,nodes=f.nodes[order],edge_index=inverse[f.edge_index],
                   sequence_owners=[int(inverse[i]) if i>=0 else -1 for i in f.sequence_owners],
                   target_nodes=[int(inverse[i]) if i>=0 else -1 for i in f.target_nodes],
                   node_lookup={k:int(inverse[v]) for k,v in f.node_lookup.items()},clock_lookup={k:int(inverse[v]) for k,v in f.clock_lookup.items()})


def _autocast(device): return torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()


def _reader_validation(model,replay,microbatch=128,smoke=False):
    model.eval(); device=next(model.parameters()).device; per_configuration=defaultdict(list)
    cases=replay.case_rows[:2] if smoke else replay.case_rows
    with torch.inference_mode():
        for cid,_,_,cfg in cases:
            rows=replay.case_transitions(cid); refs=sorted(set(r[k] for r in rows for k in ('history_hash','next_history_hash')))
            target=rows[0]['true_target_key']; losses=[]
            for j in range(0,len(refs),microbatch):
                replay.prefetch_features(refs[j:j+microbatch])
                features=[replay.features(ref) for ref in refs[j:j+microbatch]]; batch=collate_features(features,device)
                lp=model(batch).cpu()
                losses.extend(float(-lp[i,f.targets.index(target)]) for i,f in enumerate(features))
            per_configuration[cfg].append(float(np.mean(losses)))
    return dict(mean_final_nll_bits=float(np.mean([np.mean(v) for v in per_configuration.values()])),
                mean_bytes_used=0.,configurations=len(per_configuration),cases=len(cases),kind='all_recorded_validation_histories_equal_case_configuration')


def _selection_key(entry): return entry['mean_final_nll_bits'],entry.get('mean_bytes_used',0),entry['iteration']


def _mark_complete(directory,identity,selected,updates,started,model,technical_smoke,registered_cap=None,training=None):
    marker=dict(state='passed',identity=identity,actual_updates=updates,selected_checkpoint=selected,
                selected_checkpoint_hash=hash_file(directory/selected),elapsed_seconds=time.perf_counter()-started,
                peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                parameter_count=parameter_count(model),technical_smoke=technical_smoke,
                registered_update_cap=registered_cap or updates,
                termination_reason='validation_patience' if registered_cap and updates<registered_cap else 'update_cap',
                early_stopping_rule=(training or {}).get('early_stopping'))
    atomic_json(directory/'complete.json',marker);return marker


def fit_readers(root,config,crossfit=True,final=True,independent=True,resume=False,technical_smoke=False,device=None,only=None,technical_updates=2,technical_batch_size=2):
    root=Path(root)
    if technical_smoke and 'technical_smoke' not in str(root):raise ValueError('Technical smoke requires isolated root')
    train=Replay(root,'train');validation=Replay(root,'validation');results={};device=_device(device)
    names=([f'crossfit_{i}' for i in range(config['training']['crossfit_folds'])] if crossfit else [])+(['final'] if final else [])+(['independent'] if independent else [])
    if only is not None:names=[only]
    for name in names:
        excluded=int(name.split('_')[1]) if name.startswith('crossfit_') else None
        seed=config['seeds']['training'][0]+(1009 if name=='independent' else 0);rng=_seed(seed)
        architecture='transformer' if name=='independent' else 'gnn';width=256 if name=='independent' else 128
        model=DiagnosticReader(architecture,width).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'])
        identity=_identity(root,config,train,dict(reader=name,excluded_fold=excluded,validation_replay_hash=validation.source_hash,seed=seed))
        directory=root/'models'/'readers'/name;directory.mkdir(parents=True,exist_ok=True)
        complete=directory/'complete.json'
        if resume and complete.exists():
            result=json.loads(complete.read_text())
            if result['identity']!=identity or hash_file(directory/result['selected_checkpoint'])!=result['selected_checkpoint_hash']:raise RuntimeError('Completed reader hash mismatch')
            results[name]=result;continue
        checkpoint=_resume(directory,model,optimizer,rng,identity,resume);start=checkpoint['iteration'];history=checkpoint['validation_history']
        cap=technical_updates if technical_smoke else config['training']['predictor_update_cap'];interval=cap if technical_smoke else config['training']['validation_interval'];batch_size=technical_batch_size if technical_smoke else config['training']['batch_size'];micro=batch_size if technical_smoke else int(os.environ.get('CGX_MICROBATCH','128'))
        started=time.perf_counter();interval_started=started
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        training_cfgs=sorted({r[3] for r in train.case_rows if r[2]!=excluded})
        iteration=start
        effective_cap=start if not technical_smoke and should_stop(history,config['training']) else cap
        for iteration in range(start+1,effective_cap+1):
            model.train();optimizer.zero_grad(set_to_none=True);rows=train.sample(rng,batch_size,excluded)
            loss_total=0.
            for off in range(0,batch_size,micro):
                subset=rows[off:off+micro];refs=[r['next_history_hash'] if rng.random()<.5 else r['history_hash'] for r in subset]
                train.prefetch_features(refs)
                features=[_augment(train.features(ref),rng) for ref in refs];batch=collate_features(features,device)
                labels=torch.tensor([f.targets.index(r['true_target_key']) for f,r in zip(features,subset)],device=device)
                with _autocast(device):loss=joint_nll_bits(model(batch),labels)*len(subset)/batch_size
                loss.backward();loss_total+=float(loss.detach())
            norm=nn.utils.clip_grad_norm_(model.parameters(),config['training']['grad_clip_norm'])
            if not math.isfinite(loss_total) or not torch.isfinite(norm):raise FloatingPointError('Reader nonfinite training update')
            optimizer.step()
            if iteration%interval==0 or iteration==cap:
                update_seconds=time.perf_counter()-interval_started
                validation_started=time.perf_counter()
                metric=_reader_validation(model,validation,smoke=technical_smoke);metric.update(iteration=iteration,training_loss_bits=loss_total)
                interval_updates=iteration-(history[-1]['iteration'] if history else start)
                metric.update(update_interval_seconds=update_seconds,interval_updates=interval_updates,
                    updates_per_second_excluding_validation=interval_updates/update_seconds,
                    validation_seconds=time.perf_counter()-validation_started)
                if not math.isfinite(metric['mean_final_nll_bits']):raise FloatingPointError('Nonfinite reader validation')
                history.append(metric);filename=f'checkpoint_{iteration:07d}.pt'
                state=_state(model,optimizer,rng,iteration,identity,history,dict(architecture=architecture,width=width,training_configurations=training_cfgs,excluded_fold=excluded))
                _save(directory/filename,state);_save(directory/'last.pt',state);atomic_json(directory/'validation.json',history)
                print(json.dumps({'reader':name,**metric}),flush=True)
                if not technical_smoke and should_stop(history,config['training']):break
                interval_started=time.perf_counter()
        best=min(history,key=_selection_key);selected=f"checkpoint_{best['iteration']:07d}.pt"
        results[name]=_mark_complete(directory,identity,selected,iteration,started,model,technical_smoke,cap,config['training'])
    return results


def load_reader(root,name='final',device=None,technical_smoke=False):
    directory=Path(root)/'models'/'readers'/name
    marker=json.loads((directory/'complete.json').read_text())
    if marker['state']!='passed':raise RuntimeError('Reader is not completed')
    path=directory/marker['selected_checkpoint']
    if hash_file(path)!=marker['selected_checkpoint_hash']:raise RuntimeError('Reader checkpoint hash mismatch')
    checkpoint=_torch_load(path,_device(device));model=DiagnosticReader(checkpoint['architecture'],checkpoint['width']).to(_device(device));model.load_state_dict(checkpoint['weights'])
    model.checkpoint_hash=marker['selected_checkpoint_hash'];model.training_configurations=checkpoint['training_configurations'];model.identity=checkpoint['identity']
    return frozen(model)


def load_policy(root,method,seed,device=None,technical_smoke=False):
    directory=Path(root)/'models'/'policies'/f'{method}_seed{seed}'
    marker=json.loads((directory/'complete.json').read_text());path=directory/marker['selected_checkpoint']
    if marker['state']!='passed' or hash_file(path)!=marker['selected_checkpoint_hash']:raise RuntimeError('Policy checkpoint not completed or hash mismatch')
    checkpoint=_torch_load(path,_device(device));model=PolicyModel(method).to(_device(device));model.load_state_dict(checkpoint['weights']);model.checkpoint_hash=marker['selected_checkpoint_hash'];model.identity=checkpoint['identity']
    return frozen(model)


def transition_key(row):return canonical_hash([row['trajectory_id'],row['step'],row['history_hash'],row['action_key'],row['next_history_hash']])


def score_transitions(root,config,crossfit_only=True,resume=False,technical_smoke=False,device=None):
    if not crossfit_only:raise ValueError('Formal reward scoring must be out of fold')
    root=Path(root);replay=Replay(root,'train');device=_device(device)
    readers={i:load_reader(root,f'crossfit_{i}',device,technical_smoke) for i in range(config['training']['crossfit_folds'])}
    identity=_identity(root,config,replay,dict(readers={str(i):m.checkpoint_hash for i,m in readers.items()}))
    directory=root/'training';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'scores.sqlite'; marker=directory/'scores_complete.json'
    if resume and marker.exists():
        result=json.loads(marker.read_text())
        if result['identity']!=identity or result['output_hash']!=hash_file(path):raise RuntimeError('Score resume hash mismatch')
        return result
    db=sqlite3.connect(path)
    db.executescript('CREATE TABLE IF NOT EXISTS predictions(history TEXT,fold INTEGER,data TEXT,PRIMARY KEY(history,fold)); CREATE TABLE IF NOT EXISTS scores(id TEXT PRIMARY KEY,reward REAL,next_nll REAL,reader_hash TEXT,trajectory TEXT,step INTEGER,history TEXT,next_history TEXT); CREATE INDEX IF NOT EXISTS trajectory ON scores(trajectory); CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);')
    old=db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
    if old and json.loads(old[0])!=identity:raise RuntimeError('Crossfit score inputs changed')
    db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',('identity',json.dumps(identity)));db.commit()
    def predict(ref,fold,cfg):
        model=readers[fold]
        if cfg in model.training_configurations:raise RuntimeError('OOF violation: reward reader trained on source topology')
        found=db.execute('SELECT data FROM predictions WHERE history=? AND fold=?',(ref,fold)).fetchone()
        if found:return json.loads(found[0])
        # Cached features contain precisely the acquired-history representation.
        f=replay.features(ref)
        with torch.inference_mode():lp=model(collate_features([f],device))[0,:len(f.targets)].cpu().tolist()
        pred={'target_keys':list(f.targets),'log_probabilities_base2':lp}
        db.execute('INSERT INTO predictions VALUES(?,?,?)',(ref,fold,json.dumps(pred)))
        return pred
    count=0
    # Batched out-of-fold predictions avoid decoding the same raw evidence at
    # every transition. Every history retains its topology-exclusion check.
    pending={i:{} for i in readers}
    def flush(fold):
        refs=list(pending[fold])
        if not refs:return
        replay.prefetch_features(refs)
        features=[replay.features(ref) for ref in refs]
        with torch.inference_mode():lp=readers[fold](collate_features(features,device)).cpu()
        db.executemany('INSERT OR REPLACE INTO predictions VALUES(?,?,?)',[
            (ref,fold,json.dumps({'target_keys':list(f.targets),'log_probabilities_base2':lp[i,:len(f.targets)].tolist()}))
            for i,(ref,f) in enumerate(zip(refs,features))])
        pending[fold].clear()
    for row in replay.rows():
        fold=row['source_fold']
        if row['configuration_id'] in readers[fold].training_configurations:raise RuntimeError('OOF violation')
        for ref in (row['history_hash'],row['next_history_hash']):
            if ref not in pending[fold] and not db.execute('SELECT 1 FROM predictions WHERE history=? AND fold=?',(ref,fold)).fetchone():pending[fold][ref]=None
        if len(pending[fold])>=128:flush(fold)
    for fold in readers:flush(fold)
    db.commit()
    for row in replay.rows():
        key=transition_key(row)
        if resume and db.execute('SELECT 1 FROM scores WHERE id=?',(key,)).fetchone():continue
        fold=row['source_fold'];before=predict(row['history_hash'],fold,row['configuration_id']);after=predict(row['next_history_hash'],fold,row['configuration_id']);target=row['true_target_key']
        bp=before['log_probabilities_base2'][before['target_keys'].index(target)];ap=after['log_probabilities_base2'][after['target_keys'].index(target)]
        db.execute('INSERT OR REPLACE INTO scores VALUES(?,?,?,?,?,?,?,?)',(key,ap-bp,-ap,readers[fold].checkpoint_hash,row['trajectory_id'],row['step'],row['history_hash'],row['next_history_hash']))
        count+=1
        if count%1000==0:db.commit();print(json.dumps({'scored_transitions':count}),flush=True)
    db.commit();total=db.execute('SELECT COUNT(*) FROM scores').fetchone()[0];db.close()
    result=dict(state='passed',identity=identity,transitions=total,output_hash=hash_file(path),technical_smoke=technical_smoke)
    atomic_json(marker,result);return result


def _scores_db(root,technical_smoke=False):
    directory=Path(root)/'training';marker=json.loads((directory/'scores_complete.json').read_text());path=directory/'scores.sqlite'
    if marker['state']!='passed' or hash_file(path)!=marker['output_hash']:raise RuntimeError('Reward scores are incomplete or changed')
    return sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True),marker


def fit_policy(root,config,method,seed,resume=False,technical_smoke=False,validation_fn=None,device=None,technical_updates=2,technical_batch_size=2):
    from cgx.observation.actions import action_catalog
    if method not in config['methods']['learned']:raise ValueError(f'Unregistered learned method: {method}')
    if seed not in config['seeds']['training']:raise ValueError(f'Unregistered training seed: {seed}')
    if validation_fn is None:raise ValueError('Full source-validation rollout callback is required; TD loss cannot select checkpoints')
    root=Path(root);replay=Replay(root,'train');scores,score_marker=_scores_db(root);device=_device(device);rng=_seed(seed)
    reader=load_reader(root,'final',device);reader_before=canonical_hash({k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in reader.state_dict().items()})
    model=PolicyModel(method).to(device);target=copy.deepcopy(model);frozen(target)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'])
    identity=_identity(root,config,replay,dict(method=method,seed=seed,reward_hash=score_marker['output_hash'],reader_hash=reader.checkpoint_hash,replay_view='feedback_independent_affordable_prefixes_v1' if method=='gnn_fixed_plan' else 'common_all_transitions'))
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
        row=scores.execute('SELECT data FROM predictions WHERE history=? AND fold=?',(ref,fold)).fetchone()
        if not row:raise RuntimeError(f'Missing out-of-fold prediction {ref}')
        return json.loads(row[0])
    def scored(row):
        record=scores.execute('SELECT reward,next_nll FROM scores WHERE id=?',(transition_key(row),)).fetchone()
        if record is None:raise RuntimeError('Unscored replay transition')
        return record
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
    after=canonical_hash({k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in reader.state_dict().items()})
    if reader_before!=after:raise RuntimeError('Frozen common reader mutated during policy updates')
    best=min(history,key=_selection_key);return _mark_complete(directory,identity,f"checkpoint_{best['iteration']:07d}.pt",iteration,started,model,technical_smoke,cap,config['training'])


def select_checkpoints(root,config):
    """Verify every registered trained run; selection uses source validation only."""
    root=Path(root);selected={}
    names=[f'crossfit_{i}' for i in range(config['training']['crossfit_folds'])]+['final','independent']
    dirs=[root/'models'/'readers'/name for name in names]+[root/'models'/'policies'/f'{method}_seed{seed}' for method in config['methods']['learned'] for seed in config['seeds']['training']]
    for directory in dirs:
        marker=json.loads((directory/'complete.json').read_text());history=json.loads((directory/'validation.json').read_text());best=min(history,key=_selection_key);filename=f"checkpoint_{best['iteration']:07d}.pt"
        if marker['selected_checkpoint']!=filename or hash_file(directory/filename)!=marker['selected_checkpoint_hash']:raise RuntimeError(f'Selection/hash mismatch: {directory}')
        selected[str(directory.relative_to(root))]=dict(checkpoint=filename,sha256=marker['selected_checkpoint_hash'],validation=best,technical_smoke=marker['technical_smoke'])
    output=dict(state='passed',selection_split='validation',models=selected);atomic_json(root/'models'/'selection.json',output);return output
