"""Actual simulator qualification, executed faults and ordinary repairs (S01–S09)."""
import copy
import numpy as np
import pytest
from cgx.graphs import generate_graph
from cgx.env import System, generate_task
from cgx.env.modules import Packet, Module, merge_packets
from cgx.observation.actions import trace_ports

def case(graph, family='healthy', severity=None):
    types = {'cache_stale':'cache','queue_hold':'queue','adapter_scale':'adapter'}
    node = next((n['id'] for n in graph['nodes'] if n['type'] == types.get(family)), None)
    return {'fault_family':family,'fault_node':node,'severity':severity,
            'observer_offsets_ms':{d:3.1+i for i,d in enumerate(graph['clock_domains'])},
            'streams':{'noise':1234,'task':123}}

@pytest.mark.parametrize('motif,n', [('chain',10),('single',12),('double',16)])
def test_S01_S07_S08_healthy_real_routing(motif,n):
    g=generate_graph(5,n,motif,2)
    result=System(g,case(g),generate_task(123)).run(debug=True)
    assert result.metrics['task_pass']
    assert result.metrics['simulated_seconds']==10
    assert len(result.debug['physical_q'])==1000
    assert len(result.debug['controller_inputs'])==1000

@pytest.mark.parametrize('family,severity', [('cache_stale',10),('queue_hold',10),('adapter_scale',.15)])
def test_S02_S03_S04_S05_executed_fault_path_and_S06_repair(family,severity):
    g=generate_graph(5,12,'chain',2)
    nominal=System(g,case(g),generate_task(123)).run(debug=True)
    private=case(g,family,severity)
    system=System(g,private,generate_task(123))
    faulty=system.run(debug=True)
    assert not np.allclose(nominal.debug['physical_q'],faulty.debug['physical_q'],atol=1e-8)
    if family=='cache_stale':
        assert not np.array_equal(nominal.debug['controller_inputs'],faulty.debug['controller_inputs'])
        assert not np.array_equal(nominal.debug['commands'],faulty.debug['commands'])
    if family=='queue_hold':
        assert np.array_equal(faulty.debug['actuator_commands'][:10],np.tile(generate_task(123)['home'],(10,1)))
    if family=='adapter_scale':
        assert system.graph == g
    wrong=next(n['id'] for n in g['nodes'] if n['type'] in ('cache','queue','adapter') and n['id']!=private['fault_node'])
    system.repair(wrong)
    wrong_result=system.run(debug=True)
    assert np.array_equal(wrong_result.debug['physical_q'],faulty.debug['physical_q'])
    system.repair(private['fault_node'])
    repaired=system.run(debug=True)
    assert np.array_equal(repaired.debug['physical_q'],nominal.debug['physical_q'])
    assert repaired.metrics['task_pass']

@pytest.mark.parametrize('family,severity', [('cache_stale',5),('queue_hold',5),('adapter_scale',-.1)])
def test_independent_B_physical_semantics(family,severity):
    g=generate_graph(5,12,'single',2)
    a=System(g,case(g,family,severity),generate_task(123),implementation='A').run(debug=True)
    b=System(g,case(g,family,severity),generate_task(123),implementation='B').run(debug=True)
    np.testing.assert_allclose(a.debug['physical_q'],b.debug['physical_q'],atol=1e-12,rtol=0)

def test_S09_recording_does_not_change_physics():
    g=generate_graph(5,12,'single',2)
    system=System(g,case(g,'cache_stale',5),generate_task(123))
    no_trace=system.run(debug=True)
    full=system.run(trace_ports(g),100,debug=True)
    assert np.array_equal(no_trace.debug['physical_q'],full.debug['physical_q'])
    assert len(full.wire)==len(full.observation['records'])*128

@pytest.mark.parametrize('implementation',['A','B'])
def test_delay_operator_reference_and_structural_subsets(implementation):
    node={'type':'cache','nominal':{}}
    cache=Module(node,implementation); cache.delay_ticks=2
    queue=Module({'type':'queue','nominal':{}},implementation); queue.delay_ticks=2
    packets=[Packet(np.full(14,i,dtype=float),bytes([i])*16,tuple(range(7)),True) for i in range(8)]
    for i,p in enumerate(packets):
        assert cache.cache(p,i).token==packets[max(0,i-2)].token
        expected=packets[i-2] if i>=2 else packets[0]
        assert queue.queue(p,i,packets[0]).token==expected.token
    merged=merge_packets([packets[1].subset(range(3)),packets[2].subset(range(3,7))])
    assert merged.channels==tuple(range(7))
    assert merged.token not in (packets[1].token,packets[2].token)
    assert np.all(merged.values[:3]==1) and np.all(merged.values[3:7]==2)

@pytest.mark.parametrize('implementation',['A','B'])
def test_S04_nominal_representation_and_injected_payload_gain(implementation):
    node={'type':'adapter','nominal':{'scale':.5,'offset':-1.}}
    module=Module(copy.deepcopy(node),implementation);module.extra_gain=1.15
    packet=Packet(np.arange(14,dtype=float),b'a'*16,tuple(range(7)),True,scale=2.,offset=1.)
    result=module.adapter(packet)
    np.testing.assert_allclose(result.values,(((np.arange(14)-1)/2)*.5-1)*1.15,atol=1e-14)
    assert module.node==node
    assert result.token==packet.token
    module.repair()
    nominal=module.adapter(packet)
    np.testing.assert_allclose(nominal.canonical(),packet.canonical(),atol=1e-14)

def test_S08_each_branch_fault_location_changes_executed_commands():
    graph=generate_graph(167,24,'double',3)
    nominal=System(graph,case(graph),generate_task(123)).run(debug=True)
    family={'cache':'cache_stale','queue':'queue_hold','adapter':'adapter_scale'}
    checked=0
    for node in graph['nodes']:
        if node['type'] not in family: continue
        private=case(graph,family[node['type']],.1 if node['type']=='adapter' else 10)
        private['fault_node']=node['id']
        faulty=System(graph,private,generate_task(123)).run(debug=True)
        assert not np.array_equal(faulty.debug['actuator_commands'],nominal.debug['actuator_commands']),node
        checked+=1
    assert checked>=6
