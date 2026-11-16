"""Executable public-graph/selector controls; no physical simulation or fitting."""
from copy import deepcopy
from collections import Counter
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

from cgx.graphs import generate_graph, legal_targets
from cgx.graphs.generator import relabel_graph
from cgx.observation.actions import action_catalog, initial_budget, charge, can_afford
from cgx.provenance import atomic_json, file_sha256, hash_json, now
from .policies import SimplePolicy, TEMPLATES, build_fixed_plan, public_structure, calibration_pair_ranking


def _history(graph,cap=4):
    budget=initial_budget(graph,1.,cap)
    return dict(graph=deepcopy(graph),observations=[],remaining=dict(budget),
                initial_budget=dict(budget),action_keys=[])


def _prediction(graph,target=None):
    import numpy as np
    keys,prior=legal_targets(graph)
    if target is not None:prior=[1.-1e-8*(len(keys)-1) if k==target else 1e-8 for k in keys]
    return dict(target_keys=keys,log_probabilities_base2=np.log2(prior).tolist())


def _fixtures():
    return [generate_graph(9001+i,component_count=n,motif=motif,branches=2)
            for i,(n,motif) in enumerate(((8,'chain'),(12,'chain'),(12,'single'),(16,'single'),
                                          (16,'double'),(20,'double'),(24,'single'),(24,'double')))]


def _specs():
    return ([dict(method='full_once_100'),dict(method='full_once_100_cal2'),dict(method='coverage_forward')]+
        [dict(method='fixed_template_validation',template_id=t) for t in TEMPLATES]+
        [dict(method='random_legal',choice_seed=s,case_id='opaque-engineering-case') for s in (11,23,37)]+
        [dict(method='interface_rule_se',stop_threshold=t) for t in (None,.99,.999)])


def control_plan_legality_and_charging(graphs):
    count=0;calibrated_plans=0
    for graph in graphs:
        catalog=action_catalog(graph);registered={a['action_key']:a for a in catalog}
        for cap in (1,2,4):
            h=_history(graph,cap)
            for template in TEMPLATES:
                policy=SimplePolicy(dict(method='fixed_template_validation',template_id=template))
                plan=policy.plan(h,catalog,_prediction(graph));remaining=dict(h['remaining'])
                for action in plan:
                    assert action==registered[action['action_key']]
                    assert can_afford(remaining,action['cost_bound'])
                    remaining=charge(remaining,action['cost_bound'])
                assert remaining==policy.plan_metadata['planned_remaining']
                used={k:h['initial_budget'][k]-remaining[k] for k in remaining}
                assert used['executions']<=cap and used['calibrations']<=2 and used['decisions']<=cap+2
                assert used['bytes']==sum(a['cost_bound']['bytes'] for a in plan)
                if template in ('F0','F1'):
                    assert used['executions']==1
                    assert plan[-1]['kind']=='full_trace' and plan[-1]['sample_hz']==100
                    assert all(a['kind']=='calibration_only' for a in plan[:-1])
                    assert len({tuple(a['calibration_pair']) for a in plan[:-1]})==len(plan)-1
                    if template=='F0':assert used['calibrations']==0 and used['decisions']==1
                    else:
                        assert used['calibrations']==len(plan)-1
                        assert used['bytes']==plan[-1]['cost_bound']['bytes']+160*used['calibrations']
                        calibrated_plans+=used['calibrations']>0
                else:
                    expected_rate=TEMPLATES[template]['sample_hz']
                    assert all(a['sample_hz']==expected_rate for a in plan if a['ports'])
                    requests=policy.plan_metadata['intended_requests'];turns=[r['turn'] for r in requests]
                    assert turns==sorted(turns)
                    first=[r for r in requests if r['turn']==0]
                    eligible=[n['id'] for n in graph['nodes'] if n['type'] in ('cache','queue','adapter')]
                    assert {r['component'] for r in first}==set(eligible)
                    assert all(r['request_kind']=='module_pair' for r in first)
                    keys=[r['action_key'] for r in requests if r['available']]
                    assert len(keys)==len(set(keys))
                    if template=='F7':assert used['calibrations']==0
                count+=1
    assert calibrated_plans
    return dict(template_case_budget_checks=count,calibrated_one_pass_plans=calibrated_plans,
                no_physical_execution=True,charge_basis='original declared bounds; real actual-cost tests owned by runner')


def control_fixed_information_and_warmup(graphs):
    class ForbiddenRead:
        def __iter__(self):raise AssertionError('Fixed plan inspected measurements or prediction')
        def __getitem__(self,key):raise AssertionError('Fixed plan inspected measurements or prediction')
        def __bool__(self):raise AssertionError('Fixed plan inspected measurements or prediction')
    checks=0
    for graph in graphs:
        h=_history(graph);catalog=action_catalog(graph)
        for template in TEMPLATES:
            p=SimplePolicy(dict(method='fixed_template_validation',template_id=template))
            first=p.plan(h,catalog,_prediction(graph))
            perturbed={**h,'observations':ForbiddenRead()}
            second=p.plan(perturbed,catalog,ForbiddenRead())
            warmed=deepcopy(p);assert warmed.plan(h,catalog,_prediction(graph))==first
            assert second==first==p.plan(h,catalog,_prediction(graph))
            checks+=1
    # One-pass choose interface also stops after the sole charged execution.
    graph=graphs[0];h=_history(graph,4);catalog=action_catalog(graph)
    p=SimplePolicy(dict(method='full_once_100'));a=p.choose(h,catalog)
    h['remaining']=charge(h['remaining'],a['cost_bound']);h['action_keys'].append(a['action_key'])
    assert p.choose(h,catalog)['kind']=='stop'
    return dict(fixed_plan_invariance_checks=checks,warmup_consumed_choices=0,one_pass_stop_verified=True)


def _mapped_action(action,inverse_nodes,inverse_clocks):
    return dict(kind=action['kind'],sample_hz=action['sample_hz'],
        ports=sorted(inverse_nodes[p.split(':',1)[0]]+':'+p.split(':',1)[1] for p in action['ports']),
        calibration_pair=sorted(inverse_clocks[d] for d in action['calibration_pair']) if action['calibration_pair'] else None,
        cost_bound=action['cost_bound'])


def control_structural_reindexing(graphs):
    checks=0;symmetric=[]
    for i,graph in enumerate(graphs):
        changed,node_map=relabel_graph(graph,8811+i)
        old_nodes={n['id']:n for n in graph['nodes']};new_nodes={n['id']:n for n in changed['nodes']}
        clock_map={old_nodes[n]['clock_domain']:new_nodes[new]['clock_domain'] for n,new in node_map.items()}
        inverse_nodes={v:k for k,v in node_map.items()};inverse_clocks={v:k for k,v in clock_map.items()}
        identity_nodes={n:n for n in old_nodes};identity_clocks={d:d for d in graph['clock_domains']}
        structure=public_structure(graph)
        if structure['node_ties'] or structure['clock_ties']:
            symmetric.append(dict(fixture=i,node_ties=structure['node_ties'],clock_ties=structure['clock_ties']))
            continue
        for template in TEMPLATES:
            a=build_fixed_plan(graph,initial_budget(graph,1.,4),action_catalog(graph),template)[0]
            b=build_fixed_plan(changed,initial_budget(changed,1.,4),action_catalog(changed),template)[0]
            assert [_mapped_action(x,identity_nodes,identity_clocks) for x in a]==[
                _mapped_action(x,inverse_nodes,inverse_clocks) for x in b],(i,template)
            checks+=1
        spec=dict(method='random_legal',choice_seed=11,case_id='same-opaque-case')
        h=_history(graph);h2=_history(changed)
        for step in range(4):
            a=SimplePolicy(spec).choose(h,action_catalog(graph));b=SimplePolicy(spec).choose(h2,action_catalog(changed))
            assert _mapped_action(a,identity_nodes,identity_clocks)==_mapped_action(b,inverse_nodes,inverse_clocks)
            h['remaining']=charge(h['remaining'],a['cost_bound']);h2['remaining']=charge(h2['remaining'],b['cost_bound'])
            h['action_keys'].append(a['action_key']);h2['action_keys'].append(b['action_key']);checks+=1
    assert checks
    # Node-array/edge-array order preserves every new ordering even when names
    # stay fixed. This is the transformation actually used by the old evaluator.
    for graph in graphs:
        reordered=deepcopy(graph);reordered['nodes'].reverse();reordered['edges'].reverse()
        for t in TEMPLATES:
            a=build_fixed_plan(graph,initial_budget(graph,1.,4),action_catalog(graph),t)[0]
            b=build_fixed_plan(reordered,initial_budget(reordered,1.,4),action_catalog(reordered),t)[0]
            assert a==b
    return dict(semantic_relabel_checks=checks,node_array_checks=len(graphs)*len(TEMPLATES),
        exact_symmetry_exceptions=symmetric,interface_rule_equivariance_claimed=False)


def control_calibration_and_mismatch(graphs):
    checks=0
    for graph in graphs:
        catalog=action_catalog(graph);ranking=calibration_pair_ranking(graph,catalog)
        nodes={n['id']:n for n in graph['nodes']};counts=Counter()
        for e in graph['edges']:
            a,b=nodes[e['source']]['clock_domain'],nodes[e['target']]['clock_domain']
            if a!=b:counts[frozenset((a,b))]+=1
        assert [r['additional_cross_clock_connections'] for r in ranking]==sorted(counts.values(),reverse=True)
        assert len({tuple(r['pair']) for r in ranking})==len(ranking)
        assert all(len(r['pair'])==2 and r['pair'][0]!=r['pair'][1] for r in ranking)
        checks+=1
    graph=graphs[0];catalog=action_catalog(graph);budget=initial_budget(graph,1.,1)
    for actions,b in [([a for a in catalog if not(a['kind']=='full_trace' and a['sample_hz']==100)],budget),
                      (catalog,{**budget,'bytes':1})]:
        try:build_fixed_plan(graph,b,actions,'F0')
        except ValueError as exc:assert 'Protocol mismatch' in str(exc)
        else:raise AssertionError('Missing/unaffordable required full trace silently substituted')
    return dict(public_pair_rankings_checked=checks,required_full_trace_fail_closed_checks=2)


def control_random_support_and_repeated_calls(graphs):
    graph=graphs[0];h=_history(graph);catalog=action_catalog(graph)
    legal=[a for a in catalog if a['kind']!='stop' and can_afford(h['remaining'],a['cost_bound'])]
    p=SimplePolicy(dict(method='random_legal',choice_seed=11,case_id='opaque-random-fixture'))
    before=p.choose(h,catalog,_prediction(graph))
    for _ in range(8):assert p.choose(h,list(reversed(catalog)),_prediction(graph,'healthy'))==before
    assert p.choose(h,catalog+deepcopy(catalog))==before
    assert deepcopy(p).choose(h,catalog)==before
    class ForcedIndex:
        def __init__(self,index):self.index=index
        def randrange(self,n):assert n==len(legal);return self.index
    reached=[]
    for index in range(len(legal)):
        with patch('diagnostics.simple_selector_controls_v1.policies.random.Random',return_value=ForcedIndex(index)):
            reached.append(p.choose(h,catalog)['action_key'])
    assert set(reached)=={a['action_key'] for a in legal} and len(reached)==len(set(reached))
    assert any(a['kind']=='full_trace' for a in legal)
    assert {a['sample_hz'] for a in legal if a['ports']}=={10,50,100}
    chosen=before;h['remaining']=charge(h['remaining'],chosen['cost_bound']);h['action_keys'].append(chosen['action_key'])
    assert any(a['action_key']==chosen['action_key'] for a in catalog if can_afford(h['remaining'],a['cost_bound']))
    exhausted={**h,'remaining':dict(executions=0,bytes=0,calibrations=0,decisions=0)}
    assert p.choose(exhausted,catalog)['kind']=='stop'
    return dict(affordable_distinct_nonstop_keys=len(legal),exhaustive_sampler_support_keys=len(reached),
        uniformity_basis='one standard-library randrange over all distinct affordable non-STOP keys',
        full_trace_rates=[10,50,100],warmup_and_repeated_calls_consume_no_draws=True,repeats_remain_available=True)


def control_interface_rule_and_stop(graphs):
    from cgx.baselines.policies import interface_rule
    exact=0;ordering_changes=[];posterior_choices=set();measured_change=None
    for i,graph in enumerate(graphs):
        h=_history(graph);catalog=action_catalog(graph);prediction=_prediction(graph)
        p=SimplePolicy(dict(method='interface_rule_se'))
        expected=interface_rule(h,catalog,prediction);assert p.choose(h,catalog,prediction)==expected;exact+=1
        reverse=interface_rule(h,list(reversed(catalog)),prediction)
        if reverse['action_key']!=expected['action_key']:
            ordering_changes.append(dict(fixture=i,original=expected['action_key'],reversed=reverse['action_key'],
                interpretation='Inherited first-max equal-score catalog tie; not claimed graph-automorphism invariant'))
        for target in legal_targets(graph)[0]:posterior_choices.add(p.choose(h,catalog,_prediction(graph,target))['action_key'])
        baseline=p.choose(h,catalog,_prediction(graph,'healthy'))
        baseline_nodes={port.split(':',1)[0] for port in baseline['ports']}
        for node in graph['nodes']:
            if node['type'] not in ('cache','queue') or node['id'] in baseline_nodes:continue
            obs={'records':[dict(port=node['id']+':'+direction,local_time_s=t,message_token='opaque-equal-token',
                values=[0.]*node['ports'][0]['width']) for direction,t in [('in',1.),('out',1.5)]], 'calibration_bursts':[]}
            updated={**h,'observations':[obs]};changed=p.choose(updated,catalog,_prediction(graph,'healthy'))
            if changed['action_key']!=baseline['action_key']:
                measured_change=dict(fixture=i,observed_component=node['id'],original=baseline['action_key'],updated=changed['action_key']);break
        for threshold in (.99,.999):
            stopping=SimplePolicy(dict(method='interface_rule_se',stop_threshold=threshold))
            confident=_prediction(graph,'healthy')
            assert stopping.choose(h,catalog,confident)['kind']!='stop'
            cal=next(a for a in catalog if a['kind']=='calibration_only')
            cal_h={**h,'remaining':charge(h['remaining'],cal['cost_bound']),'action_keys':[cal['action_key']]}
            assert stopping.choose(cal_h,catalog,confident)['kind']!='stop'
            execution=next(a for a in catalog if a['kind']=='module_pair')
            after={**h,'remaining':charge(h['remaining'],execution['cost_bound']),'action_keys':[execution['action_key']]}
            assert stopping.choose(after,catalog,confident)['kind']=='stop'
    assert len(posterior_choices)>1 and measured_change is not None
    return dict(original_formula_exact_matches=exact,confidence_stop_thresholds=[.99,.999],
        confidence_stop_before_first_execution=False,posterior_changes_action=True,
        acquired_values_change_action=measured_change,inherited_catalog_order_changes=ordering_changes,
        interface_rule_strict_relabel_equivariance='not claimed; original first-max tie behavior deliberately retained')


def control_no_selector_weights_or_truth(graphs):
    import torch
    import cgx.training.engine
    import cgx.models
    class PublicHistory(dict):
        def __getitem__(self,key):
            if key in ('case','fault_family','fault_node','evaluator','true_clock_offsets','severity','true_target_key'):
                raise AssertionError('Private evaluator information accessed')
            return super().__getitem__(key)
        def get(self,key,default=None):
            if key in ('case','fault_family','fault_node','evaluator','true_clock_offsets','severity','true_target_key'):
                raise AssertionError('Private evaluator information accessed')
            return super().get(key,default)
    count=0;g=graphs[0];h=PublicHistory(_history(g));actions=action_catalog(g);q=_prediction(g)
    with patch.object(torch,'load',side_effect=AssertionError('Checkpoint loaded by simple selector')), \
         patch.object(cgx.training.engine,'load_policy',side_effect=AssertionError('Learned selector loaded')), \
         patch.object(cgx.models.PolicyModel,'__init__',side_effect=AssertionError('Learned selector constructed')):
        for spec in _specs():
            p=SimplePolicy(spec)
            p.plan(h,actions,q) if p.fixed_plan else p.choose(h,actions,q)
            count+=1
    return dict(specifications_checked=count,checkpoint_loads=0,selector_constructions=0,private_history_reads=0,
                labels_used_for_scoring=False,physical_executions=0)


def run_controls(output):
    started=time.monotonic();graphs=_fixtures();results={}
    controls=[control_plan_legality_and_charging,control_fixed_information_and_warmup,
        control_structural_reindexing,control_calibration_and_mismatch,
        control_random_support_and_repeated_calls,control_interface_rule_and_stop,
        control_no_selector_weights_or_truth]
    for control in controls:
        tick=time.monotonic();result=control(graphs)
        results[control.__name__]=dict(state='passed',seconds=time.monotonic()-tick,**result)
    report=dict(state='passed',created_at=now(),public_fixture_graphs=len(graphs),
        fixture_hashes=[hash_json(g) for g in graphs],controls=results,wall_seconds=time.monotonic()-started,
        physical_execution_attempts=0,models_trained=0,scientific_outcomes_used=0,
        implementation_sha256=file_sha256(Path(__file__).with_name('policies.py')),
        tests_sha256=file_sha256(__file__),interface_rule_tie_limitation_explicit=True)
    atomic_json(output,report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(run_controls(args.output),indent=2),flush=True)
