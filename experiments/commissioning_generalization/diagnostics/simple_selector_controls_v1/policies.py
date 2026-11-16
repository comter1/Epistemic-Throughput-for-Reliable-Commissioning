"""Public-information inspection rules for CGX-SIMPLE-SELECTORS-V1.

No selector is fitted or loaded here. The existing interface_rule formula is
called unchanged. Fixed plans are functions only of public topology, action
catalog and resource bounds. Random choices are stateless keyed draws, so a
warm-up/repeated scoring call cannot advance the eventual choice.
"""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import time

from cgx.observation.actions import action_catalog, can_afford, charge
from cgx.provenance import file_sha256, hash_json


VERSION = 'cgx-simple-selector-rules-v1'
METHODS = ('full_once_100', 'full_once_100_cal2', 'random_legal',
           'coverage_forward', 'fixed_template_validation', 'interface_rule_se')
TEMPLATES = {
    'F0': dict(kind='full_once', sample_hz=100, calibration=False),
    'F1': dict(kind='full_once', sample_hz=100, calibration=True),
    'F2': dict(kind='coverage', order='forward', sample_hz=100, calibration=True),
    'F3': dict(kind='coverage', order='reverse', sample_hz=100, calibration=True),
    'F4': dict(kind='coverage', order='alternate_types', sample_hz=100, calibration=True),
    'F5': dict(kind='coverage', order='forward', sample_hz=50, calibration=True),
    'F6': dict(kind='coverage', order='alternate_types', sample_hz=50, calibration=True),
    'F7': dict(kind='coverage', order='forward', sample_hz=100, calibration=False),
}
STRUCTURAL_TIES = ('Topological sensor-to-actuator depth, then label-independent '
    'typed port/connection/clock-membership refinement. Only exact public-graph '
    'automorphism ties use lexicographic opaque IDs as a declared convention; '
    'renaming such symmetric ties may permute equivalent components.')
INTERFACE_TIES = ('The unchanged interface_rule chooses the first maximum-score '
    'action in the original evaluator catalog. Its inherited equal-score tie '
    'behavior is retained, including the original case-paired node-array shuffle; '
    'equal rule scores need not imply graph automorphism.')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _public_node_signature(node):
    return _canonical(dict(type=node['type'], nominal=node['nominal'],
        ports=sorted(node['ports'], key=lambda p: p['name'])))


def _edge_signature(edge):
    return (edge['source_port'], edge['target_port'], edge['relation'])


def _automorphic(graph, first, second, *, clock=False):
    """Fail closed on a structural-refinement collision outside a real symmetry."""
    import networkx as nx
    base=nx.MultiDiGraph()
    for node in graph['nodes']:
        base.add_node(('node', node['id']), public=_public_node_signature(node))
    for domain in graph['clock_domains']:
        base.add_node(('clock', domain), public='clock')
    for node in graph['nodes']:
        base.add_edge(('node',node['id']),('clock',node['clock_domain']), relation='clock_member')
    for edge in graph['edges']:
        base.add_edge(('node',edge['source']),('node',edge['target']), relation=_canonical(_edge_signature(edge)))
    left=base.copy();right=base.copy();kind='clock' if clock else 'node'
    nx.set_node_attributes(left,False,'root');nx.set_node_attributes(right,False,'root')
    left.nodes[(kind,first)]['root']=True;right.nodes[(kind,second)]['root']=True
    return nx.is_isomorphic(left,right,
        node_match=nx.algorithms.isomorphism.categorical_node_match(['public','root'],['',False]),
        edge_match=nx.algorithms.isomorphism.categorical_multiedge_match('relation',''))


@lru_cache(maxsize=128)
def _structure_from_json(encoded):
    graph=json.loads(encoded);nodes={n['id']:n for n in graph['nodes']}
    incoming=defaultdict(list);outgoing=defaultdict(list)
    for edge in graph['edges']:
        incoming[edge['target']].append(edge);outgoing[edge['source']].append(edge)
    pending=set(nodes);depth={}
    while pending:
        ready=[n for n in pending if all(e['source'] in depth for e in incoming[n])]
        if not ready:raise ValueError('Public graph must be a DAG')
        for n in ready:depth[n]=max((depth[e['source']]+1 for e in incoming[n]),default=0)
        pending.difference_update(ready)
    base={n:_public_node_signature(node) for n,node in nodes.items()}
    colors={n:hash_json(base[n]) for n in nodes}
    # Domain names never enter numerical/ordering signatures. Shared membership
    # does, through multisets of the public member-component signatures.
    for _ in range(len(nodes)+1):
        domains={d:hash_json(sorted(colors[n] for n in nodes if nodes[n]['clock_domain']==d))
                 for d in graph['clock_domains']}
        colors={n:hash_json([base[n],domains[nodes[n]['clock_domain']],
            sorted((_edge_signature(e),colors[e['source']]) for e in incoming[n]),
            sorted((_edge_signature(e),colors[e['target']]) for e in outgoing[n])]) for n in nodes}
    domains={d:hash_json(sorted(colors[n] for n in nodes if nodes[n]['clock_domain']==d))
             for d in graph['clock_domains']}
    node_ties=defaultdict(list);clock_ties=defaultdict(list)
    for n in nodes:node_ties[(depth[n],colors[n])].append(n)
    for d in domains:clock_ties[domains[d]].append(d)
    for group in node_ties.values():
        for other in group[1:]:
            if not _automorphic(graph,group[0],other):
                raise ValueError('Public component refinement collision outside a genuine automorphism')
    for group in clock_ties.values():
        for other in group[1:]:
            if not _automorphic(graph,group[0],other,clock=True):
                raise ValueError('Public clock refinement collision outside a genuine automorphism')
    order=sorted(nodes,key=lambda n:(depth[n],colors[n],n))
    clock_order=sorted(domains,key=lambda d:(domains[d],d))
    return dict(node_order=order,node_rank={n:i for i,n in enumerate(order)},
        clock_order=clock_order,clock_rank={d:i for i,d in enumerate(clock_order)},
        node_signature={n:(depth[n],colors[n]) for n in nodes},clock_signature=domains,
        node_ties=[sorted(v) for v in node_ties.values() if len(v)>1],
        clock_ties=[sorted(v) for v in clock_ties.values() if len(v)>1],tie_convention=STRUCTURAL_TIES)


def public_structure(graph):
    return deepcopy(_structure_from_json(_canonical(graph)))


def _stop(actions):
    return next(a for a in actions if a['kind']=='stop')


def _legal_distinct(actions, remaining):
    result={}
    for action in actions:
        key=action['action_key']
        if key in result and result[key]!=action:raise ValueError('Conflicting duplicate action key')
        if can_afford(remaining,action['cost_bound']):result[key]=action
    return list(result.values())


def _action_order(action,structure):
    ports=tuple(sorted((structure['node_rank'][p.split(':',1)[0]],p.split(':',1)[1])
                       for p in action['ports']))
    clocks=tuple(sorted(structure['clock_rank'][d] for d in action.get('calibration_pair') or ()))
    return (action['kind'],int(action['sample_hz']),ports,clocks,action['action_key'])


def calibration_pair_ranking(graph,actions):
    """Greedy coverage of actual public cross-clock links by legal clock pairs."""
    structure=public_structure(graph);nodes={n['id']:n for n in graph['nodes']}
    edges=defaultdict(list)
    for edge in graph['edges']:
        a=nodes[edge['source']]['clock_domain'];b=nodes[edge['target']]['clock_domain']
        if a!=b:edges[frozenset((a,b))].append(deepcopy(edge))
    candidates=[]
    for action in actions:
        if action['kind']=='calibration_only':
            pair=frozenset(action['calibration_pair'])
            candidates.append((action,edges[pair]))
    covered=set();ranking=[]
    while candidates:
        def key(item):
            action,links=item
            novelty=len({_canonical(e) for e in links}-covered)
            pair=tuple(sorted(structure['clock_rank'][d] for d in action['calibration_pair']))
            return (-novelty,pair,action['action_key'])
        action,links=min(candidates,key=key);candidates.remove((action,links))
        newly={_canonical(e) for e in links}-covered;covered.update(newly)
        ranking.append(dict(action_key=action['action_key'],pair=list(action['calibration_pair']),
            additional_cross_clock_connections=len(newly),cross_clock_connections=links))
    return ranking


def _coverage_requests(graph,actions,definition,structure):
    nodes={n['id']:n for n in graph['nodes']};order=[n for n in structure['node_order']
        if nodes[n]['type'] in ('cache','queue','adapter')]
    if definition['order']=='reverse':order.reverse()
    elif definition['order']=='alternate_types':
        queues={kind:deque(n for n in order if nodes[n]['type']==kind) for kind in ('cache','queue','adapter')}
        order=[]
        while any(queues.values()):
            for kind in ('cache','queue','adapter'):
                if queues[kind]:order.append(queues[kind].popleft())
    rate=definition['sample_hz']
    exact={(tuple(sorted(a['ports'])),a['sample_hz']):a for a in actions
           if a['kind'] in ('module_pair','link_pair') and a['calibration_pair'] is None}
    component_requests={}
    for n in order:
        module_ports=tuple(sorted(n+':'+p['name'] for p in nodes[n]['ports']))
        candidates=[dict(component=n,request_kind='module_pair',ports=list(module_ports))]
        adjacent=[e for e in graph['edges'] if n in (e['source'],e['target'])]
        adjacent.sort(key=lambda e:((e['target']==n) if definition['order']=='reverse' else (e['source']==n),
            structure['node_rank'][e['source']],structure['node_rank'][e['target']],_edge_signature(e)))
        for edge in adjacent:
            candidates.append(dict(component=n,request_kind='link_pair',ports=sorted([
                edge['source']+':'+edge['source_port'],edge['target']+':'+edge['target_port']])))
        component_requests[n]=candidates
    # Every eligible component gets its module-pair turn before anyone gets a
    # second local request. Adjacent-interface requests then use the same turns.
    seen=set();requests=[]
    for turn in range(max((len(v) for v in component_requests.values()),default=0)):
        for n in order:
            if turn>=len(component_requests[n]):continue
            request=component_requests[n][turn];action=exact.get((tuple(request['ports']),rate))
            if action is None:
                requests.append(dict(request,sample_hz=rate,available=False,action_key=None,turn=turn));continue
            if action['action_key'] in seen:continue
            seen.add(action['action_key'])
            requests.append(dict(request,sample_hz=rate,available=True,action_key=action['action_key'],turn=turn))
    return order,requests


def build_fixed_plan(graph,remaining,actions,template_id):
    """Build an entire upper-bound-feasible plan without acquired observations."""
    if template_id not in TEMPLATES:raise ValueError('Unknown fixed template')
    definition=TEMPLATES[template_id];structure=public_structure(graph)
    lookup={a['action_key']:a for a in actions};budget=dict(remaining);plan=[]
    metadata=dict(version=VERSION,template_id=template_id,definition=deepcopy(definition),
        information_used=['public graph','legal catalog','resource bounds'],
        structure=structure,initial_remaining=dict(remaining),intended_requests=[],skipped_requests=[],
        calibration_pair_ranking=[],selected_calibration_pairs=[],selection_timing_scope='complete fixed-plan construction')
    if definition['kind']=='full_once':
        full=[a for a in actions if a['kind']=='full_trace' and a['sample_hz']==100]
        if len(full)!=1 or not can_afford(budget,full[0]['cost_bound']):
            raise ValueError('Protocol mismatch: exactly one affordable original 100 Hz full_trace is required')
        full=full[0]
        if definition['calibration']:
            ranking=calibration_pair_ranking(graph,actions);metadata['calibration_pair_ranking']=ranking
            for candidate in ranking:
                if len(metadata['selected_calibration_pairs'])>=2:break
                action=lookup[candidate['action_key']]
                combined={k:action['cost_bound'][k]+full['cost_bound'][k] for k in budget}
                if can_afford(budget,combined):
                    plan.append(action);budget=charge(budget,action['cost_bound'])
                    metadata['selected_calibration_pairs'].append(list(action['calibration_pair']))
        plan.append(full);budget=charge(budget,full['cost_bound'])
        metadata['intended_requests']=[dict(action_key=a['action_key'],kind=a['kind'],ports=a['ports'],
            calibration_pair=a['calibration_pair'],sample_hz=a['sample_hz']) for a in plan]
    else:
        order,requests=_coverage_requests(graph,actions,definition,structure)
        metadata['component_order']=order;metadata['intended_requests']=requests
        nodes={n['id']:n for n in graph['nodes']};calibrated=set()
        for request in requests:
            if not request['available']:
                metadata['skipped_requests'].append(dict(request,reason='requested_catalog_action_absent'));continue
            action=lookup[request['action_key']]
            if not can_afford(budget,action['cost_bound']):
                metadata['skipped_requests'].append(dict(request,reason='resource_bounds'));continue
            domains=frozenset(nodes[p.split(':',1)[0]]['clock_domain'] for p in action['ports'])
            if definition['calibration'] and len(domains)==2 and domains not in calibrated:
                paired=[a for a in actions if a['kind']=='pair_plus_calibration' and
                    a['ports']==action['ports'] and a['sample_hz']==action['sample_hz'] and
                    frozenset(a['calibration_pair'])==domains and can_afford(budget,a['cost_bound'])]
                if paired:
                    if len(paired)!=1:raise ValueError('Ambiguous original paired-calibration action')
                    action=paired[0]
                else:
                    calibration=[a for a in actions if a['kind']=='calibration_only' and
                        frozenset(a['calibration_pair'])==domains and
                        can_afford(budget,{k:a['cost_bound'][k]+action['cost_bound'][k] for k in budget})]
                    if calibration:
                        if len(calibration)!=1:raise ValueError('Ambiguous original calibration-only action')
                        cal=calibration[0];plan.append(cal);budget=charge(budget,cal['cost_bound'])
                        calibrated.add(domains);metadata['selected_calibration_pairs'].append(list(cal['calibration_pair']))
            plan.append(action);budget=charge(budget,action['cost_bound'])
            if action['calibration_pair'] is not None:
                calibrated.add(frozenset(action['calibration_pair']))
                metadata['selected_calibration_pairs'].append(list(action['calibration_pair']))
    metadata['planned_action_keys']=[a['action_key'] for a in plan]
    metadata['planned_actions']=deepcopy(plan);metadata['planned_remaining']=budget
    return plan,metadata


@lru_cache(maxsize=1)
def implementation_fingerprint():
    here=Path(__file__);experiment=here.parents[2]
    return hash_json(dict(version=VERSION,files={str(p.relative_to(experiment)):file_sha256(p) for p in
        (here,experiment/'src/cgx/baselines/policies.py',experiment/'src/cgx/observation/actions.py')}))


class SimplePolicy:
    """RuntimePolicy-compatible selector without learned selector parameters."""
    def __init__(self,spec):
        allowed={'method','template_id','stop_threshold','choice_seed','case_id'}
        if set(spec)-allowed:raise ValueError('Unexpected simple-rule specification field')
        self.spec=deepcopy(spec);self.method=spec['method']
        if self.method not in METHODS:raise ValueError('Unknown simple method')
        self.fixed_plan=self.method not in ('random_legal','interface_rule_se')
        self.diagnosis_schedule='after_each' if self.method=='interface_rule_se' else 'final_only'
        self.template_id={'full_once_100':'F0','full_once_100_cal2':'F1','coverage_forward':'F2'}.get(
            self.method,spec.get('template_id'))
        if self.fixed_plan and self.template_id not in TEMPLATES:raise ValueError('A declared fixed template is required')
        self.stop_threshold=spec.get('stop_threshold')
        if self.stop_threshold not in (None,.99,.999):raise ValueError('Undeclared confidence stopping threshold')
        if self.method!='interface_rule_se' and self.stop_threshold is not None:raise ValueError('Confidence stop is only for the graph rule')
        self.choice_seed=spec.get('choice_seed');self.case_id=spec.get('case_id')
        if self.method=='random_legal' and (type(self.choice_seed) is not int or not isinstance(self.case_id,str)):
            raise ValueError('Random requires a choice seed and opaque public case ID')
        self.rule_hash=hash_json(dict(implementation=implementation_fingerprint(),
            spec={k:v for k,v in self.spec.items() if k!='case_id'}))
        self.last_plan_decision_seconds=[]
        self.plan_metadata=dict(version=VERSION,method=self.method,diagnosis_schedule=self.diagnosis_schedule,
            rule_hash=self.rule_hash,selector_weights_loaded=False,
            tie_convention=INTERFACE_TIES if self.method=='interface_rule_se' else STRUCTURAL_TIES)

    def plan(self,history,actions,prediction=None):
        if not self.fixed_plan:raise ValueError('This simple method is sequential, not a fixed planner')
        started=time.monotonic()
        plan,metadata=build_fixed_plan(history['graph'],history['remaining'],actions,self.template_id)
        self.last_plan_decision_seconds=[time.monotonic()-started]
        self.plan_metadata.update(metadata)
        return plan

    def choose(self,history,actions,prediction=None):
        legal=_legal_distinct(actions,history['remaining'])
        if self.fixed_plan:
            # The deployed path calls plan once. This interface is also stateless:
            # reconstruct the same original plan rather than replan from readings.
            initial={'graph':history['graph'],'remaining':history['initial_budget']}
            plan=self.plan(initial,action_catalog(history['graph']),prediction)
            used=set(history.get('action_keys',()))
            available={a['action_key']:a for a in legal}
            return next((available[a['action_key']] for a in plan
                if a['action_key'] not in used and a['action_key'] in available),_stop(legal))
        if self.method=='interface_rule_se':
            executed=history['initial_budget']['executions']-history['remaining']['executions']
            if self.stop_threshold is not None and executed>=1 and prediction:
                maximum=max(2.**float(x) for x in prediction['log_probabilities_base2'])
                if maximum>=self.stop_threshold:return _stop(legal)
            from cgx.baselines.policies import interface_rule
            # Preserve the original formula AND its existing input-order tie rule.
            return interface_rule(history,legal,prediction)
        non_stop=[a for a in legal if a['kind']!='stop']
        if not non_stop:return _stop(legal)
        structure=public_structure(history['graph'])
        non_stop.sort(key=lambda a:_action_order(a,structure))
        step=history['initial_budget']['decisions']-history['remaining']['decisions']
        key=hash_json(['cgx-simple-random-choice-v1',self.choice_seed,self.case_id,int(step)])
        index=random.Random(int(key,16)).randrange(len(non_stop))
        return non_stop[index]
