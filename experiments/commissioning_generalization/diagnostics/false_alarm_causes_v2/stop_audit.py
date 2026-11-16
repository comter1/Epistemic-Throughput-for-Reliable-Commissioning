"""Actual-checkpoint STOP gradients, without any optimizer update or new evidence."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from cgx.config import load_config
from cgx.data.store import Store
from cgx.models.features import tensorize_history, collate_features, collate_actions
from cgx.observation.actions import action_catalog, feasible_actions
from cgx.provenance import atomic_json, hash_json, read_json
from cgx.training.engine import load_policy
from cgx.training.objectives import double_q_target
from diagnostics.false_alarm_causes.interventions import load_blob
from .ledger import Ledger


def weight_digest(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def gradients(model):
    return {name:torch.zeros_like(p).reshape(-1).cpu() if p.grad is None else p.grad.detach().reshape(-1).cpu().clone()
            for name,p in model.named_parameters()}


def gradient_summary(model, values):
    result={}
    for prefix in ('backbone','action_head','next_loss_head'):
        pieces=[v for name,v in values.items() if name.startswith(prefix+'.')]
        vector=torch.cat(pieces)
        result[prefix]={'l2':float(torch.linalg.vector_norm(vector)), 'max_abs':float(vector.abs().max()),
                        'parameters_with_grad':sum(p.grad is not None for name,p in model.named_parameters() if name.startswith(prefix+'.'))}
    return result


def max_difference(left,right,factor=1.):
    return max(float((left[k]-factor*right[k]).abs().max()) for k in left)


def audit_one(root,run,v1,case,method,seed=11,device='cpu'):
    ledger=Ledger(run);model=load_policy(root,method,seed,device=device)
    for p in model.parameters():p.requires_grad_(True)
    model.eval();before=weight_digest(model)
    source=v1/'interventions/confirmation/cases'/f"{hash_json([case['case_id'],method,seed,'A'])}.json.gz"
    item=load_blob(source);history=Store(v1).load_history(item['row']['evidence_trace_hash'])
    history=deepcopy(history);history['observations']=history['observations'][:1]
    history['remaining']=dict(history['initial_budget']);history['action_keys']=[]
    if method=='gnn_depth2':history['remaining']['decisions']=min(2,history['remaining']['decisions'])
    catalog=action_catalog(history['graph'],load_config());legal=feasible_actions(catalog,history['remaining'])
    stop=next(a for a in legal if a['kind']=='stop')
    continuation=[a for a in legal if a['kind']!='stop'][:2]
    assert len(continuation)==2
    ledger.reserve('feature_build',detail='stop_gradient_audit')
    features=tensorize_history(history,prediction=item['row']['initial_prediction'])
    def forward(actions):
        ledger.reserve('model_forward',len(actions),detail='stop_gradient_audit')
        batch=collate_features([features]*len(actions),device)
        candidates=collate_actions(batch,[[a] for a in actions],device)
        return model(batch,candidates,return_next_loss=True)
    model.zero_grad(set_to_none=True)
    score,_=forward([stop]);zero_loss=torch.nn.functional.smooth_l1_loss(score,torch.zeros_like(score))
    zero_loss.backward();stop_grads=gradients(model);stop_summary=gradient_summary(model,stop_grads)
    assert score.item()==0. and zero_loss.item()==0.
    assert max(float(x.abs().max()) for x in stop_grads.values())==0.
    model.zero_grad(set_to_none=True)
    values,_=forward(continuation)
    # Deliberately artificial technical targets produce nonzero gradients. They
    # are identical in all gradient comparisons and are never research rewards.
    targets=values.detach()+torch.tensor([-.25,.5],device=device)
    loss=torch.nn.functional.smooth_l1_loss(values,targets,reduction='sum')/len(continuation)
    loss.backward();baseline=gradients(model);baseline_summary=gradient_summary(model,baseline)
    mixed_actions=continuation+[stop]*4
    mixed_targets=torch.cat((targets,torch.zeros(4,device=device)))
    model.zero_grad(set_to_none=True)
    values,_=forward(mixed_actions)
    same_normalization=torch.nn.functional.smooth_l1_loss(values,mixed_targets,reduction='sum')/len(continuation)
    same_normalization.backward();same=gradients(model)
    model.zero_grad(set_to_none=True)
    values,_=forward(mixed_actions)
    diluted_loss=torch.nn.functional.smooth_l1_loss(values,mixed_targets,reduction='mean')
    diluted_loss.backward();diluted=gradients(model)
    normalized_difference=max_difference(same,baseline)
    dilution_difference=max_difference(diluted,baseline,len(continuation)/len(mixed_actions))
    assert normalized_difference<1e-6,(method,normalized_difference)
    assert dilution_difference<1e-6,(method,dilution_difference)
    terminal=double_q_target(torch.tensor([0.,0.]),torch.tensor([[5.,0.],[5.,0.]]),
        torch.tensor([[6.,0.],[6.,0.]]),torch.ones(2,2,dtype=torch.bool),
        torch.tensor([True,False]),torch.tensor([3,0]))
    assert torch.equal(terminal,torch.zeros_like(terminal))
    zero_resources={k:0 for k in history['remaining']}
    assert any(a['kind']=='stop' for a in feasible_actions(catalog,zero_resources))
    assert before==weight_digest(model)
    return {'method_key':method,'policy_seed':seed,'checkpoint_hash':model.checkpoint_hash,'device':device,
        'state':'passed','source_case_id':case['case_id'],'engineering_history_only':True,
        'technical_continuation_targets':'original score plus fixed [-0.25,+0.5] offsets; not empirical rewards or fitted outcomes',
        'stop_score':float(score.detach()),'stop_loss':float(zero_loss.detach()),'stop_gradients':stop_summary,
        'continuation_gradients':baseline_summary,'continuation_examples':2,'added_stop_examples':4,
        'same_continuation_normalization_max_gradient_difference':normalized_difference,
        'mixed_batch_mean_expected_scale':len(continuation)/len(mixed_actions),
        'mixed_batch_mean_max_gradient_difference_from_expected_scale':dilution_difference,
        'gradient_absolute_tolerance':1e-6,'terminal_and_zero_horizon_targets':terminal.tolist(),
        'stop_in_legal_successors':True,'stop_legal_with_zero_resources':True,
        'optimizer_steps':0,'weights_before':before,'weights_after':weight_digest(model),
        'checkpoint_modified':False}


def run(root,run,v1,device='cpu'):
    root,run,v1=map(Path,(root,run,v1));torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    case=min(read_json(v1/'manifests/fresh_cases.json'),key=lambda c:c['case_id'])
    start=time.perf_counter();cpu=time.process_time()
    results=[audit_one(root,run,v1,case,method,device=device) for method in ('gnn_adaptive','gnn_depth2')]
    report={'state':'passed','methods':results,'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
            'interpretation':'STOP=0 is already present; zero-reward terminal STOP rows add no Q learning signal. Mixed-batch dilution is distinct from learning stopping. No optimizer step was performed.'}
    atomic_json(run/'stop_gradient_audit.json',report)
    (run/'stop_gradient_audit.md').write_text('''# STOP 梯度审计

使用两个原选定checkpoint、实际PolicyModel与Huber训练损失；没有optimizer step，也没有改变模型权重。合法terminal STOP的评分、loss和Q梯度均为零。给固定的继续动作技术例子增加4条STOP，在保持继续动作归一化不变时，梯度保持一致；按混合batch平均时，梯度仅缩小到原来的2/6。辅助next-loss head不参与这两个方法的损失。STOP始终存在于合法后继动作中；terminal或剩余决策0均不带后续收益。

技术例子的target为原输出加固定偏移，只用于制造可检测梯度，不是研究reward，不用于拟合任何模型。该结果排除“单纯多放零reward STOP行即可教会这两个模型停止”的直接解释；后续应修正继续动作相对于已知0分STOP的收益估值。
''')
    print({'state':'passed','methods':len(results),'optimizer_steps':0},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--v1-run',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    run(a.root,a.run_dir,a.v1_run,a.device)
