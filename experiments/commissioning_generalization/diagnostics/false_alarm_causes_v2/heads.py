"""Matched source-only action-head fits and exact frozen-trajectory stopping replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from cgx.models.networks import PolicyModel
from cgx.provenance import atomic_json, file_sha256, hash_json, read_json
from cgx.seeds import seed_for
from cgx.training.engine import load_policy
from .head_data import plain, prediction
from .ledger import Ledger
from .stop_audit import weight_digest


CELLS={'T00':('oof','oof'),'T01':('final','oof'),'T10':('oof','final'),'T11':('final','final')}
STEPS=(0,500,1000,1500,2000)
METHODS=('gnn_adaptive','gnn_depth2')
SEEDS=(11,23,37)


def configure():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def matched_schedule(frame,seed,updates=2000,batch_size=128):
    """Configuration, then case, then eligible prefix, each uniformly sampled."""
    train=frame[frame.source_split.eq('train')]
    assert len(train)>0 and train.eligible_for_fit.all()
    tree={str(config):{str(case):part.index.to_numpy() for case,part in group.groupby('case_id',sort=True)}
          for config,group in train.groupby('configuration_id',sort=True)}
    configs=sorted(tree);cases={config:sorted(tree[config]) for config in configs}
    rng=np.random.default_rng(seed);indices=np.empty((updates,batch_size),dtype=np.int64)
    for i in range(updates):
        for j in range(batch_size):
            config=configs[int(rng.integers(len(configs)))];keys=cases[config]
            case=keys[int(rng.integers(len(keys)))];eligible=tree[config][case]
            indices[i,j]=eligible[int(rng.integers(len(eligible)))]
    assert frame.iloc[indices.reshape(-1)].source_split.eq('train').all()
    return indices


def checkpoint_path(run,method,seed,cell,step):
    return Path(run)/'head_checkpoints'/method/f'seed{seed}'/cell/f'step_{step:04d}.pt'


def fit(root,run_dir,method,seed,device='cuda:1'):
    configure();run=Path(run_dir);ledger=Ledger(run);seed=int(seed)
    assert method in METHODS and seed in SEEDS
    assert read_json(run/'head_data_complete.json')['state']=='passed'
    name=f'{method}_seed{seed}';meta_path=run/'head_data'/f'{name}.parquet';array_path=run/'head_data'/f'{name}.npz'
    frame=pd.read_parquet(meta_path);arrays=np.load(array_path)
    assert set(frame.source_split)=={'train','validation'}
    assert len(frame)==len(arrays['x_oof'])==len(arrays['x_final'])
    assert frame[['configuration_id','case_id','decision_index']].duplicated().sum()==0
    batch_seed=seed_for(671931,method,seed)
    schedule=matched_schedule(frame,batch_seed)
    schedule_hash=hashlib.sha256(schedule.tobytes()).hexdigest()
    schedule_path=run/'head_data'/f'{name}_batch_indices.npy';np.save(schedule_path,schedule)
    model=load_policy(Path(root),method,seed,device=device)
    assert all(not p.requires_grad for p in model.parameters())
    frozen_hash=weight_digest(model);initial_hash=weight_digest(model.action_head)
    heads={cell:deepcopy(model.action_head).train() for cell in CELLS}
    for head in heads.values():
        for parameter in head.parameters():parameter.requires_grad_(True)
        assert weight_digest(head)==initial_hash
    assert not set(id(p) for p in model.parameters())&set(id(p) for h in heads.values() for p in h.parameters())
    optimizers={cell:torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=1e-4) for cell,head in heads.items()}
    tensors={name:torch.as_tensor(arrays[name],dtype=torch.float32,device=device) for name in ('x_oof','x_final','y_oof','y_final')}
    indices=torch.as_tensor(schedule,device=device)
    ids={cell:f'{method}:seed{seed}:{cell}' for cell in CELLS}
    metadata=dict(method_key=method,policy_seed=seed,original_checkpoint_hash=model.checkpoint_hash,
        original_model_tensor_hash=frozen_hash,initial_head_hash=initial_hash,batch_schedule_sha256=schedule_hash,
        matched_batch_seed=671931,derived_batch_seed=batch_seed,batch_seed_derivation='seed_for(671931,method,policy_seed)',
        training_meta_sha256=file_sha256(meta_path),training_arrays_sha256=file_sha256(array_path),
        head_fit_code_sha256=file_sha256(Path(__file__)),
        source_manifest_sha256=file_sha256(run/'manifests/source_cases.json'),
        input_width=281,batch_size=128,updates=2000,learning_rate=1e-4,weight_decay=1e-4,
        loss='Huber smooth_l1_loss beta=1 reduction=mean',optimizer='AdamW',precision='FP32',
        threshold=0.,threshold_rule='strictly greater than zero; original STOP retained',
        training_support='all non-STOP source_train proposed continuations; no return-sign filtering',
        test_or_v1_labels_read=False,all_deployment_inputs='FINAL',backbone_frozen=True)
    metadata['support_counts']={split:dict(prefixes=len(part),cases=part.case_id.nunique(),
        configurations=part.configuration_id.nunique(),oof_positive=int(part.return_oof.gt(0).sum()),
        oof_negative=int(part.return_oof.lt(0).sum()),final_positive=int(part.return_final.gt(0).sum()),
        final_negative=int(part.return_final.lt(0).sum())) for split,part in frame.groupby('source_split')}
    def save(cell,step):
        path=checkpoint_path(run,method,seed,cell,step);path.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(metadata,cell=cell,step=step,q_input_training=CELLS[cell][0],return_reader=CELLS[cell][1],
                     weights={key:value.detach().cpu().clone() for key,value in heads[cell].state_dict().items()},
                     optimizer=optimizers[cell].state_dict())
        temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)
    for cell in CELLS:
        registered=ledger.register_fit(ids[cell]);assert registered['updates']==0,'No untracked duplicate fitting or retries'
        save(cell,0)
    start=time.perf_counter();cpu=time.process_time();logs=[]
    try:
        for update,batch in enumerate(indices,1):
            lossrow=dict(update=update)
            for cell,(input_reader,label_reader) in CELLS.items():
                ledger.reserve('head_update',detail=ids[cell])
                ledger.reserve('head_forward',detail='head_fit:'+ids[cell])
                optimizer=optimizers[cell];optimizer.zero_grad(set_to_none=True)
                predicted=heads[cell](tensors['x_'+input_reader][batch]).squeeze(-1)
                loss=torch.nn.functional.smooth_l1_loss(predicted,tensors['y_'+label_reader][batch])
                assert torch.isfinite(loss);loss.backward();optimizer.step()
                lossrow[cell]=float(loss.detach())
            logs.append(lossrow)
            if update in STEPS:
                for cell in CELLS:save(cell,update)
                print(json.dumps(dict(stage='head_fit',method=method,seed=seed,update=update,
                                      wall_seconds=time.perf_counter()-start,losses=lossrow)),flush=True)
                if update==500:
                    projected=(time.perf_counter()-start)/500*2000*6
                    atomic_json(run/'head_data'/f'{name}_training_profile.json',dict(measured_updates_per_cell=500,
                        simultaneous_independent_cells=4,measured_wall_seconds=time.perf_counter()-start,
                        projected_all_24_head_gpu_seconds=projected,allowed_gpu_seconds=48*3600,
                        status='within_cap' if projected<=48*3600 else 'exceeds_cap'))
                    assert projected<=48*3600,'Measured projection exceeds declared training cap'
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    finally:
        elapsed=time.perf_counter()-start
        if str(device).startswith('cuda'):ledger.reserve('training_gpu_seconds',elapsed,detail='head_fit:'+name)
    assert weight_digest(model)==frozen_hash
    assert all(p.grad is None for p in model.parameters())
    pd.DataFrame(logs).to_csv(run/'head_data'/f'{name}_training_loss.csv',index=False)
    atomic_json(run/'head_data'/f'{name}_fit_complete.json',dict(metadata,state='passed',cells=list(CELLS),
        exact_updates_per_cell=2000,all_cells_same_batch_order=True,original_weights_unchanged=True,
        frozen_backbone_gradients_none=True,training_wall_seconds=elapsed,cpu_seconds=time.process_time()-cpu,
        device=device,head_forward_batches=8000,head_forward_examples=8000*128))


def load_head(run,method,seed,cell,step,device='cpu'):
    path=checkpoint_path(run,method,int(seed),cell,int(step));saved=torch.load(path,map_location=device,weights_only=False)
    head=PolicyModel(method).action_head.to(device);head.load_state_dict(saved['weights']);head.eval()
    for p in head.parameters():p.requires_grad_(False)
    assert saved['cell']==cell and saved['step']==step and saved['policy_seed']==seed
    return head,saved


def outcome(row,arm,cell=None,step=None,gate_score=None):
    q=prediction(row);keys=q['target_keys'];lp=np.asarray(q['log_probabilities_base2'],float)
    truth=row['true_target_key'];predicted=keys[int(np.argmax(lp))];healthy=truth=='healthy'
    initial=row['initial_budget'];remaining=row['remaining']
    result={k:row[k] for k in ('case_id','configuration_id','method_key','policy_seed','history_hash','decision_index')}
    result.update(arm=arm,cell=cell,step=step,final_prediction=q,predicted_target_key=predicted,
        true_target_key=truth,is_healthy=healthy,fault_family='healthy' if healthy else truth.rsplit(':',1)[1],
        healthy_false_alarm=float(predicted!='healthy') if healthy else None,
        fault_miss=float(predicted=='healthy') if not healthy else None,
        correct_joint=predicted==truth,final_nll_bits=-float(lp[keys.index(truth)]),
        p_healthy=float(np.exp2(lp[keys.index('healthy')])),
        executions_used=int(row['executions_used']),calibrations_used=int(row['calibrations_used']),
        bytes_used=initial['bytes']-remaining['bytes'],decisions_used=initial['decisions']-remaining['decisions'],
        source_split=row.get('source_split'),termination_reason=row.get('terminal_reason') or 'prefix_stopping_gate',
        gate_score=gate_score,physical_wall_seconds=None,
        replay_scope='exact retained prefix; logical costs; no separately timed physical rollout')
    return result


def replay(run_dir,phase='source',device='cpu'):
    """Read all head inputs under FINAL; select only on source validation."""
    configure();run=Path(run_dir);ledger=Ledger(run)
    assert phase in ('source','fresh')
    if phase=='source':audit_fits(run)
    if phase=='fresh':assert (run/'head_selection.json').exists(),'Freeze source selection before fresh replay'
    frame=pd.read_parquet(run/f'{phase}_prefixes.parquet')
    if phase=='source':frame=frame[frame.source_split.eq('validation')]
    rows=[plain(row) for row in frame.to_dict('records')];outputs=[];all_scores=[];reconstructions=[]
    for method in METHODS:
        for seed in SEEDS:
            group=[r for r in rows if r['method_key']==method and r['policy_seed']==seed]
            usable=[r for r in group if r['actionable'] and r['chosen_action']['kind']!='stop']
            inputs=torch.as_tensor(np.asarray([r['chosen_head_input_final'] for r in usable],np.float32),device=device)
            assert inputs.ndim==2 and inputs.shape[1]==281
            snapshots=[(cell,step) for cell in CELLS for step in (STEPS if phase=='source' else (2000,))]
            if phase=='fresh':
                choice=read_json(run/'head_selection.json')['methods'][method]
                if (choice['cell'],choice['step']) not in snapshots:snapshots.append((choice['cell'],choice['step']))
            predictions={}
            for cell,step in snapshots:
                head,_=load_head(run,method,seed,cell,step,device)
                ledger.reserve('head_forward',detail=f'head_replay:{phase}')
                with torch.inference_mode():scores=head(inputs).squeeze(-1).cpu().numpy()
                assert np.isfinite(scores).all(),'Nonfinite correction scores cannot silently continue'
                if step==0:
                    original=np.asarray([r['chosen_score'] for r in usable])
                    maximum=float(np.abs(scores-original).max())
                    sign_changes=int(np.count_nonzero((scores>0)!=(original>0)))
                    reconstructions.append(dict(method_key=method,policy_seed=seed,cell=cell,
                        max_absolute_score_error=maximum,tolerance=1e-6,threshold_sign_changes=sign_changes,
                        original_zero_nonstop_scores=int(np.count_nonzero(original==0))))
                    assert maximum<1e-6 and sign_changes==0,'Investigate cached-head reconstruction/device rounding before selection'
                predictions[cell,step]={(r['case_id'],r['decision_index']):float(score) for r,score in zip(usable,scores)}
                all_scores.extend(dict(case_id=r['case_id'],configuration_id=r['configuration_id'],method_key=method,
                    policy_seed=seed,decision_index=r['decision_index'],history_hash=r['history_hash'],
                    cell=cell,step=step,score=float(score),allow=bool(score>0)) for r,score in zip(usable,scores))
            trajectories=defaultdict(list)
            for row in group:trajectories[row['case_id']].append(row)
            for case,trajectory in trajectories.items():
                trajectory.sort(key=lambda r:r['decision_index'])
                outputs.append(outcome(trajectory[-1],'A0'))
                first=next((r for r in trajectory if r['executions_used']>=1),trajectory[-1])
                outputs.append(outcome(first,'S1'))
                for cell,step in snapshots:
                    retained=trajectory[-1];score=None
                    for row in trajectory:
                        if not row['actionable'] or row['chosen_action']['kind']=='stop':retained=row;score=None;break
                        score=predictions[cell,step][case,row['decision_index']]
                        if score<=0:retained=row;break
                    outputs.append(outcome(retained,cell,cell,step,score))
    result=pd.DataFrame(outputs);scores=pd.DataFrame(all_scores)
    prefix='source_validation' if phase=='source' else 'fresh_replay'
    result.to_parquet(run/f'correction_2x2_{prefix}_per_case.parquet',index=False)
    result.drop(columns=['final_prediction']).to_csv(run/f'correction_2x2_{prefix}_per_case.csv',index=False)
    scores.to_parquet(run/f'correction_scores_{prefix}.parquet',index=False)
    summary=summarize(result);summary.to_csv(run/f'correction_2x2_{prefix}.csv',index=False)
    if phase=='source':select(run,summary)
    else:
        selected=read_json(run/'head_selection.json')['methods'];g1=[]
        for method,choice in selected.items():
            arm=result[result.method_key.eq(method)&result.cell.eq(choice['cell'])&result.step.eq(choice['step'])].copy()
            arm['arm']='G1';arm['selected_as_deployable_improvement']=choice['guardrails_passed'];g1.append(arm)
        pd.concat(g1,ignore_index=True).to_parquet(run/'fresh_G1_prefixes.parquet',index=False)
    if reconstructions:atomic_json(run/'tests/head_step_zero_reconstruction.json',dict(state='passed',checks=reconstructions))
    atomic_json(run/f'head_replay_{phase}_complete.json',dict(state='passed',rows=len(result),score_rows=len(scores),
        physical_executions=0,full_model_forwards=0,head_only=True,
        deployment_q_input='FINAL for every cell',source_selection_read_before_fresh=phase=='fresh'))
    return result


def audit_fits(run):
    run=Path(run);ledger=Ledger(run).snapshot();groups=[]
    for method in METHODS:
        for seed in SEEDS:
            report=read_json(run/'head_data'/f'{method}_seed{seed}_fit_complete.json')
            assert report['state']=='passed' and report['exact_updates_per_cell']==2000
            cells=[]
            for cell in CELLS:
                fit_id=f'{method}:seed{seed}:{cell}'
                assert ledger['fits'][fit_id]['updates']==2000 and ledger['fits'][fit_id]['registered']
                artifacts=[]
                for step in STEPS:
                    path=checkpoint_path(run,method,seed,cell,step)
                    saved=torch.load(path,map_location='cpu',weights_only=False)
                    for key in ('batch_schedule_sha256','training_meta_sha256','training_arrays_sha256',
                                'initial_head_hash','original_model_tensor_hash','original_checkpoint_hash'):
                        assert saved[key]==report[key],f'Matched-cell violation: {key}'
                    assert saved['step']==step and saved['cell']==cell
                    assert saved['q_input_training']==CELLS[cell][0] and saved['return_reader']==CELLS[cell][1]
                    artifacts.append(dict(step=step,sha256=file_sha256(path),path=str(path)))
                cells.append(dict(cell=cell,fit_id=fit_id,updates=2000,artifacts=artifacts))
            groups.append(dict(method_key=method,policy_seed=seed,cells=cells,
                batch_schedule_sha256=report['batch_schedule_sha256'],initial_head_hash=report['initial_head_hash'],
                support_counts=report['support_counts']))
    assert ledger['counts']['head_fit']==24 and ledger['counts']['head_update']==48000
    atomic_json(run/'four_cell_match_audit.json',dict(state='passed',groups=groups,total_fits=24,
        total_updates=48000,all_comparisons_at_update=2000,snapshots=list(STEPS),
        same_support_actions_initialization_and_batch_order=True,all_backbones_frozen=True,
        all_deployment_q_inputs='FINAL',new_features=False,fresh_or_v1_labels_used_for_fitting=False))


def summarize(frame):
    metrics=['healthy_false_alarm','fault_miss','correct_joint','final_nll_bits','executions_used','calibrations_used','bytes_used']
    data=frame.copy();data['correct_joint']=data.correct_joint.astype(float)
    data.loc[data.is_healthy,'correct_joint']=np.nan
    groups=['method_key','arm','cell','step'];rows=[]
    for key,part in data.groupby(groups,dropna=False,sort=True):
        # Collapse policy seeds and physical cases inside each configuration, then
        # configurations equally. Seed-wise values remain in the per-case file.
        equal=part.groupby('configuration_id')[metrics].mean().mean()
        row=dict(zip(groups,key));row.update(equal.to_dict())
        row.update(configurations=part.configuration_id.nunique(),unique_cases=part.case_id.nunique(),
            trajectories=len(part),healthy_denominator=int(part.is_healthy.sum()),
            false_alarms=int(part.healthy_false_alarm.sum()),faulty_denominator=int((~part.is_healthy).sum()),
            missed_faults=int(part.fault_miss.sum()),joint_correct_faults=int(part.correct_joint.sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def select(run,summary):
    selected={}
    for method in METHODS:
        baseline=summary[summary.method_key.eq(method)&summary.arm.eq('A0')].iloc[0]
        # Step zero is the unchanged-head control, not a trained correction.
        # This clarification was sealed before any new SOURCE outcome existed.
        candidates=summary[summary.method_key.eq(method)&summary.arm.isin(CELLS)&summary.step.gt(0)].copy()
        candidates['guardrails_passed']=(candidates.healthy_false_alarm.le(baseline.healthy_false_alarm+1e-12)&
            candidates.fault_miss.le(baseline.fault_miss+.02+1e-12)&
            candidates.correct_joint.ge(baseline.correct_joint-.02-1e-12))
        good=candidates[candidates.guardrails_passed]
        if len(good):
            tied=good[good.final_nll_bits.le(good.final_nll_bits.min()+.005)]
            chosen=tied.sort_values(['executions_used','step','cell']).iloc[0]
        else:chosen=candidates[candidates.cell.eq('T11')&candidates.step.eq(2000)].iloc[0]
        selected[method]=dict(cell=chosen.cell,step=int(chosen.step),guardrails_passed=bool(chosen.guardrails_passed),
            shared_across_policy_seeds=list(SEEDS),status='source_selected' if chosen.guardrails_passed else 'experimental_not_deployable',
            equal_configuration_nll_bits=float(chosen.final_nll_bits),healthy_false_alarm=float(chosen.healthy_false_alarm),
            fault_miss=float(chosen.fault_miss),joint_fault_accuracy=float(chosen.correct_joint),
            checkpoints={str(seed):dict(path=str(checkpoint_path(run,method,seed,chosen.cell,int(chosen.step))),
                sha256=file_sha256(checkpoint_path(run,method,seed,chosen.cell,int(chosen.step)))) for seed in SEEDS})
        candidates.to_csv(Path(run)/f'head_selection_candidates_{method}.csv',index=False)
    atomic_json(Path(run)/'head_selection.json',dict(methods=selected,state='frozen_after_source_validation_before_fresh',
        source_cases_sha256=file_sha256(Path(run)/'manifests/source_cases.json'),
        source_validation_results_sha256=file_sha256(Path(run)/'correction_2x2_source_validation.csv'),
        selection_scope='one cell and checkpoint shared across three seeds separately per method',
        correction_selection_updates=[500,1000,1500,2000],step_zero='reported unchanged-head control; not a correction candidate',
        fixed_threshold=0.,all_cells_at_2000_reported=True,heldout_outcomes_read=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('fit');p.add_argument('--root',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--method',required=True,choices=METHODS);p.add_argument('--seed',required=True,type=int,choices=SEEDS)
    p.add_argument('--device',default='cuda:1')
    p=sub.add_parser('replay');p.add_argument('--run-dir',required=True);p.add_argument('--phase',choices=['source','fresh'],default='source')
    p.add_argument('--device',default='cpu');args=parser.parse_args();kwargs=vars(args);command=kwargs.pop('command')
    if command=='fit':fit(kwargs.pop('root'),kwargs.pop('run_dir'),**kwargs)
    else:replay(kwargs.pop('run_dir'),**kwargs)
