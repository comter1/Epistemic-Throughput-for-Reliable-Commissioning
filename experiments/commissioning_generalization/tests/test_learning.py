import copy
import json
from pathlib import Path
import math
import numpy as np
import pytest
import torch
from cgx.models import DiagnosticReader,PolicyModel,tensorize_history,collate_features,collate_actions,parameter_count
from cgx.models.features import target_space
from cgx.baselines import RuntimePolicy
from cgx.training.objectives import joint_nll_bits,score_reward,double_q_target,frozen
from cgx.observation.actions import action_catalog,initial_budget

ROOT=Path(__file__).parents[1]

def fixture():
    g=json.loads((ROOT/'examples/public_graph.json').read_text());b=initial_budget(g,.25)
    records=[]
    for port,offset in [('n5:in',.003),('n5:out',.014)]:
        for i in range(6):records.append(dict(port=port,local_time_s=.103+i*.01+offset,message_token=f'opaque{i//2}',event_type='receive',values=[i*.1]*7,missing=False))
    h=dict(graph=g,observations=[dict(action_key='public_action',records=records,calibration_bursts=[],visible_status='ok',actual_cost=dict(bytes=1536,executions=1,decisions=1,calibrations=0))],remaining=b,initial_budget=b,action_keys=[])
    return h,action_catalog(g)


def rename(h,actions):
    h=copy.deepcopy(h);actions=copy.deepcopy(actions);g=h['graph'];mapping={n['id']:f'z{100-i}' for i,n in enumerate(g['nodes'])};cm={c:f'domain_{100-i}' for i,c in enumerate(g['clock_domains'])}
    for n in g['nodes']:n['id']=mapping[n['id']];n['clock_domain']=cm[n['clock_domain']]
    g['nodes'].reverse();g['clock_domains']=[cm[c] for c in reversed(g['clock_domains'])]
    for e in g['edges']:e['source']=mapping[e['source']];e['target']=mapping[e['target']]
    g['edges'].reverse()
    def port(p):
        n,pn=p.split(':',1);return mapping.get(n,n)+':'+pn
    for obs in h['observations']:
        for r in obs['records']:r['port']=port(r['port'])
    for a in actions:
        a['ports']=[port(p) for p in a['ports']]
        if a['calibration_pair']:a['calibration_pair']=[cm[c] for c in a['calibration_pair']]
    return h,actions,mapping


@pytest.mark.parametrize('architecture',['gnn','transformer'])
def test_D04_D07_L10_L11_permutation_variable_heads(architecture):
    torch.set_num_threads(1);torch.manual_seed(41);h,actions=fixture();rh,ra,mapping=rename(h,actions)
    reader=DiagnosticReader(architecture).eval();pred=reader.predict(h);renamed=reader.predict(rh)
    expected={('healthy' if k=='healthy' else mapping[k.split(':')[0]]+':'+k.split(':')[1]):v for k,v in zip(pred['target_keys'],pred['log_probabilities_base2'])}
    assert np.allclose([expected[k] for k in renamed['target_keys']],renamed['log_probabilities_base2'],atol=1e-6,rtol=0)
    assert abs(sum(np.exp2(pred['log_probabilities_base2']))-1)<1e-6
    method='transformer_adaptive' if architecture=='transformer' else 'gnn_adaptive';policy=PolicyModel(method).eval()
    with torch.no_grad():
        scores=policy.score(h,actions);rs=policy.score(rh,ra)
        assert torch.allclose(scores,rs,atol=1e-6,rtol=0)
        rev=policy.score(h,list(reversed(actions)))
        assert torch.allclose(scores,rev.flip(0),atol=1e-6,rtol=0)
        assert scores[[a['kind']=='stop' for a in actions]].item()==0


def test_D03_D05_O10_L04_private_poison_and_id_bijection():
    torch.manual_seed(31);h,_=fixture();reader=DiagnosticReader().eval();a=reader.predict(h)
    h['fault_family']='queue_hold';h['true_offsets']={'c0':100};h['label']=999;h['filename']='TEST';h['global_tick']=123
    for obs in h['observations']:
        for r in obs['records']:r['message_token']='RENAMED'+r['message_token']
    b=reader.predict(h)
    assert a==b


def test_L13_frozen_inference_has_no_target_statistics_updates():
    h,actions=fixture();reader=frozen(DiagnosticReader());policy=frozen(PolicyModel())
    before={name:{k:v.clone() for k,v in model.state_dict().items()} for name,model in [('reader',reader),('policy',policy)]}
    assert not any(isinstance(module,torch.nn.modules.batchnorm._BatchNorm) for model in (reader,policy) for module in model.modules())
    changed=copy.deepcopy(h);changed['observations'][0]['records'][0]['values']=[15.]*7
    for history in (h,changed):reader.predict(history);policy.score(history,actions)
    for name,model in [('reader',reader),('policy',policy)]:
        assert all(torch.equal(v,before[name][k]) for k,v in model.state_dict().items())
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def test_L13_strict_repeated_inference_settings():
    from cgx.runtime.determinism import configure_torch
    settings=configure_torch()
    assert settings['deterministic_algorithms'] and torch.are_deterministic_algorithms_enabled()
    h,_=fixture();reader=DiagnosticReader().eval()
    assert reader.predict(h)==reader.predict(h)


def test_L01_L02_L03_reward_loss_and_telescoping():
    lp=torch.log2(torch.tensor([[.25,.5,.25],[.5,.25,.25]]));labels=torch.tensor([1,0])
    assert joint_nll_bits(lp,labels).item()==1
    paths=torch.tensor([-3.,-1.,-2.,-.5]);rewards=score_reward(paths[:-1],paths[1:])
    assert rewards[1]<0
    assert rewards.sum()==paths[-1]-paths[0]
    assert score_reward(paths,paths).eq(0).all()


def test_L06_terminal_stop_and_horizon():
    r=torch.tensor([1.,2.,3.,4.]);online=torch.tensor([[9.,0.],[1.,0.],[-1.,0.],[8.,0.]])
    target=torch.tensor([[7.,0.],[5.,0.],[6.,0.],[9.,0.]]);mask=torch.ones_like(online,dtype=torch.bool);terminal=torch.tensor([True,False,False,False]);horizon=torch.tensor([2,0,1,2])
    result=double_q_target(r,online,target,mask,terminal,horizon)
    assert result.tolist()==[1,2,3,13]


def test_L07_delayed_payoff_fitted_value_differs_from_myopic():
    torch.manual_seed(3);torch.set_num_threads(1)
    # Four public state/action encodings: immediate=.2, invest=0, later payoff=1, STOP=0.
    x=torch.eye(4);reward=torch.tensor([.2,0.,1.,0.]);terminal=torch.tensor([True,False,True,True])
    learned=[]
    for myopic in (False,True):
        net=torch.nn.Linear(4,1,bias=False);target=copy.deepcopy(net);opt=torch.optim.AdamW(net.parameters(),lr=.05,weight_decay=0)
        for step in range(250):
            with torch.no_grad():
                values=target(x).flatten();bootstrap=torch.tensor([0.,float(max(0,values[2])),0.,0.]);goal=reward if myopic else reward+bootstrap
            loss=torch.nn.functional.smooth_l1_loss(net(x).flatten(),goal);opt.zero_grad();loss.backward();opt.step()
            if step%10==0:target.load_state_dict(net.state_dict())
        learned.append(net(x).detach().flatten())
    assert learned[0][:2].argmax().item()==1 and learned[1][:2].argmax().item()==0


def test_L05_L08_L09_fixed_plan_and_frozen_reader():
    h,actions=fixture();reader=frozen(DiagnosticReader());before={k:v.clone() for k,v in reader.state_dict().items()};m=PolicyModel('gnn_fixed_plan');policy=RuntimePolicy('gnn_fixed_plan',m)
    # Runtime wrapper must not disable training when used for validation.
    assert all(p.requires_grad for p in m.parameters())
    p1=policy.plan(h,actions,reader.predict(h));p2=policy.plan(copy.deepcopy(h),actions,reader.predict(h))
    assert [a['action_key'] for a in p1]==[a['action_key'] for a in p2]
    changed=copy.deepcopy(h);changed['observations'][0]['records'][0]['values']=[100.]*7
    with torch.no_grad(): assert not torch.allclose(m.score(h,actions),m.score(changed,actions))
    for k,v in reader.state_dict().items():assert torch.equal(v,before[k])


def test_L12_parameter_counts_within_twenty_percent():
    g=parameter_count(PolicyModel());t=parameter_count(PolicyModel('transformer_adaptive'))
    assert abs(t-g)/g<.2


def test_batched_graphs_equal_single_predictions():
    torch.manual_seed(2);h,_=fixture();h2=copy.deepcopy(h);h2['observations']=[]
    model=DiagnosticReader().eval();fs=[tensorize_history(h),tensorize_history(h2)]
    with torch.no_grad():
        both=model(collate_features(fs,'cpu'))
        for i,f in enumerate(fs):assert torch.allclose(both[i],model(collate_features([f],'cpu'))[0],atol=1e-6,rtol=0)


def test_D07_heldout_size_uses_same_weights_and_valid_joint_softmax():
    from cgx.graphs import generate_graph
    reader=DiagnosticReader().eval();policy=PolicyModel('transformer_adaptive').eval()
    shapes={name:tuple(p.shape) for name,p in reader.named_parameters()}
    cardinalities=[]
    for size in (8,16,24,32):
        graph=generate_graph(100+size,size,'chain',2);budget=initial_budget(graph,.25)
        history=dict(graph=graph,observations=[],remaining=budget,initial_budget=budget,action_keys=[])
        prediction=reader.predict(history);cardinalities.append(len(prediction['target_keys']))
        assert np.isfinite(prediction['log_probabilities_base2']).all()
        assert sum(np.exp2(prediction['log_probabilities_base2']))==pytest.approx(1.,abs=1e-6)
        actions=action_catalog(graph);scores=policy.score(history,actions)
        assert scores.shape==(len(actions),) and torch.isfinite(scores).all()
        assert shapes=={name:tuple(p.shape) for name,p in reader.named_parameters()}
    assert len(set(cardinalities))>1


def test_fixed_plan_has_each_decision_deadline(monkeypatch):
    import importlib
    from contextlib import contextmanager
    module=importlib.import_module('cgx.evaluation.rollout');calls=[]
    @contextmanager
    def clock(seconds):
        calls.append(seconds);yield
    monkeypatch.setattr(module,'call_deadline',clock)
    h,actions=fixture();policy=RuntimePolicy('full_recording_feasible');plan=policy.plan(h,actions)
    assert plan and len(calls)>=len(plan)
    assert all(value==2. for value in calls)
    assert len(policy.last_plan_decision_seconds)==len(calls)
