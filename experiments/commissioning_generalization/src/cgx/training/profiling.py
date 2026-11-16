"""Capacity measurements on acquired technical evidence, excluded from results."""
from pathlib import Path
from collections import defaultdict
import copy
import json
import time
import numpy as np
import torch
from cgx.models import DiagnosticReader,PolicyModel,collate_features,collate_actions,parameter_count
from cgx.observation.actions import action_catalog
from .engine import atomic_json,_autocast,_device,_seed
from .replay import Replay
from .objectives import joint_nll_bits,frozen


def profile_updates(root,config,n=1000,device=None):
    root=Path(root)
    if 'technical_smoke' not in str(root):raise ValueError('Use acquired technical_smoke evidence for capacity profiling')
    if n<1000:raise ValueError('Protocol capacity profile requires at least1000 representative updates')
    replay=Replay(root,'train');device=_device(device);rng=_seed(991);rows=replay.sample(rng,config['training']['batch_size'])
    started=time.perf_counter();features=[replay.features(r['history_hash']) for r in rows];next_features=[replay.features(r['next_history_hash']) for r in rows]
    preprocessing=time.perf_counter()-started
    b=collate_features(features,device);nb=collate_features(next_features,device)
    catalogs=[action_catalog(replay.state(r['history_hash'])['graph'],config) for r in rows]
    selected=[[next(a for a in cat if a['action_key']==r['action_key'])] for cat,r in zip(catalogs,rows)]
    ca=collate_actions(b,selected,device);na=collate_actions(nb,catalogs,device)
    labels=torch.tensor([f.targets.index(r['true_target_key']) for f,r in zip(features,rows)],device=device)
    modes=['reader_gnn','reader_transformer','policy_gnn','policy_transformer'];reports=[]
    for mode in modes:
        reader_mode=mode.startswith('reader');architecture=mode.split('_')[1]
        model=(DiagnosticReader(architecture,256 if architecture=='transformer' else 128) if reader_mode else PolicyModel('transformer_adaptive' if architecture=='transformer' else 'gnn_adaptive')).to(device)
        target=frozen(copy.deepcopy(model));optimizer=torch.optim.AdamW(model.parameters(),lr=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'])
        durations=[];count=n//4+(modes.index(mode)<n%4)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        for i in range(count+5):
            if device.type=='cuda':torch.cuda.synchronize()
            tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                if reader_mode:loss=joint_nll_bits(model(b),labels)
                else:
                    current=model(b,ca)
                    with torch.no_grad():
                        online=model(nb,na);old=target(nb,na);goal=torch.stack([o[v.argmax()] for o,v in zip(old.split(na['counts']),online.split(na['counts']))])
                    # Profiling reuses actual states; zero technical reward is disclosed.
                    loss=torch.nn.functional.smooth_l1_loss(current.float(),goal.float())
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config['training']['grad_clip_norm']);optimizer.step()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite profiling update')
            if device.type=='cuda':torch.cuda.synchronize()
            if i>=5:durations.append(time.perf_counter()-tick)
            if i and i%50==0:print(json.dumps({'profile_mode':mode,'updates':i,'target':count}),flush=True)
        reports.append(dict(mode=mode,actual_updates=count,warmup_updates=5,batch_size=len(rows),p50_seconds=float(np.median(durations)),p95_seconds=float(np.percentile(durations,95)),total_seconds=sum(durations),peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,parameters=parameter_count(model)))
        del model,target,optimizer
        if device.type=='cuda':torch.cuda.empty_cache()
    result=dict(state='passed',technical_smoke=True,actual_updates=sum(r['actual_updates'] for r in reports),profiles=reports,preprocessing_seconds=preprocessing,
                replay_hash=replay.source_hash,device=str(device),event_sequences=len(b['events']),maximum_sequence_tokens=b['events'].shape[1],
                boundary='Forward/backward, optimizer and target forward included; preprocessing separately measured. Reused acquired technical states; policy timing uses zero technical rewards, never research results.')
    atomic_json(root/'reports/learning_profile.json',result);return result
