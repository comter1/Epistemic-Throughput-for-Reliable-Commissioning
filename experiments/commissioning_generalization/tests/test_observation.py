"""Wire, clock, isolation and budget invariants O01–O11."""
import copy
import json
from pathlib import Path
import numpy as np
import jsonschema
import pytest
from cgx.env import System, generate_task
from cgx.graphs import generate_graph
from cgx.observation.actions import action_catalog,initial_budget,feasible_actions,charge,full_run_bytes
from cgx.observation.clocks import calibration_burst
from cgx.observation.wire import EVENT,BURST,offset_estimate

ROOT=Path(__file__).resolve().parents[1]

def fixture():
    g=generate_graph(12,12,'single',2)
    c={'fault_family':'healthy','fault_node':None,'severity':None,'observer_offsets_ms':{d:2+i*3 for i,d in enumerate(g['clock_domains'])},'streams':{'noise':5}}
    return g,c,System(g,c,generate_task(12))

def test_O01_O05_O11_clocks_and_no_hidden_start():
    g,c,s=fixture(); a=next(a for a in action_catalog(g) if a['kind']=='module_pair' and a['sample_hz']==100)
    one=s.run(a['ports'],100,execution_index=1,debug=True)
    two=s.run(a['ports'],100,execution_index=2,debug=True)
    domains={f"{n['id']}:{p['name']}":n['clock_domain'] for n in g['nodes'] for p in n['ports']}
    domains['external:out']=g['clock_domains'][0]
    for event in one.debug['port_times']:
        assert event['local_time']==pytest.approx(event['true_time']+c['observer_offsets_ms'][domains[event['port']]]/1000+event['error'],abs=1e-14)
    assert [e['error'] for e in one.debug['port_times']] != [e['error'] for e in two.debug['port_times']]
    first={}
    for event in one.debug['port_times']: first.setdefault(event['port'],event['true_time'])
    assert all(t>0 for t in first.values())
    for rec in one.observation['records']:
        assert not ({'tick','true_time','severity','fault_family','offset','ordinal'} & set(rec))

def test_O02_O06_actual_wire_and_only_acquired_records():
    g,c,s=fixture(); a=next(a for a in action_catalog(g) if a['kind']=='single_port')
    result=s.acquire(a)
    assert EVENT.size==128 and BURST.size==160
    assert len(result.wire)==128*len(result.observation['records'])
    assert set(r['port'] for r in result.observation['records']) == set(a['ports'])|{'external:out'}
    assert result.observation['actual_cost']['bytes']==len(result.wire)
    jsonschema.validate(result.observation,json.loads((ROOT/'schemas/observation.schema.json').read_text()))

def test_O03_O04_calibration_algebra_variance_no_repair():
    g,c,s=fixture(); pair=g['clock_domains'][:2]
    b,duration=calibration_burst(pair,c['observer_offsets_ms'],4,noiseless=True,symmetric=True)
    true=(c['observer_offsets_ms'][pair[1]]-c['observer_offsets_ms'][pair[0]])/1000
    np.testing.assert_allclose(offset_estimate(b),true,atol=1e-14,rtol=0)
    estimates=[offset_estimate(calibration_burst(pair,c['observer_offsets_ms'],4,k)[0]).mean() for k in range(100)]
    assert np.var(estimates)>1e-9
    before=copy.deepcopy(s.offsets); before_params=[(m.delay_ticks,m.extra_gain) for m in s.modules.values()]
    r=s.calibrate(pair)
    assert len(r.wire)==160 and r.observation['actual_cost']['executions']==0
    assert before==s.offsets and before_params==[(m.delay_ticks,m.extra_gain) for m in s.modules.values()]

def test_O07_O08_caps_and_public_reference():
    g,c,s=fixture(); actions=action_catalog(g)
    schema=json.loads((ROOT/'schemas/action.schema.json').read_text())
    for a in actions: jsonschema.validate(a,schema)
    assert initial_budget(g,0)['bytes']==0
    assert [a['kind'] for a in feasible_actions(actions,initial_budget(g,0))]==['stop']
    remaining=initial_budget(g,.25)
    assert remaining['bytes']==full_run_bytes(g)
    chosen=[]
    while True:
        feasible=[a for a in feasible_actions(actions,remaining) if a['kind']!='stop']
        if not feasible: break
        action=next((a for a in feasible if a['kind']=='calibration_only'),feasible[0])
        result=s.acquire(action,execution_index=1+sum(a['cost_bound']['executions'] for a in chosen),burst_index=sum(a['cost_bound']['calibrations'] for a in chosen))
        remaining=charge(remaining,result.observation['actual_cost']); chosen.append(action)
        assert len(chosen)<=6
    assert len(chosen)<=6 and all(v>=0 for v in remaining.values())
    with pytest.raises(ValueError):charge(remaining,{'executions':5,'bytes':0,'calibrations':0,'decisions':1})

def test_O09_initial_same_for_methods():
    g,c,s=fixture()
    assert s.run().observation == s.run().observation

def test_public_verifier_has_no_hidden_evaluator_dependency():
    from cgx.env.verification import public_verification_pass
    g,c,s=fixture(); result=s.run()
    before=public_verification_pass(result.observation)
    result.metrics.update(task_pass=False,prohibited_contacts=1000000)
    result.debug['fault_family']='poison'
    assert public_verification_pass(result.observation)==before
    broken=copy.deepcopy(result.observation)
    for r in broken['records']: r['values'][7]=0.
    assert not public_verification_pass(broken)

def test_O06_wire_decoded_payload_is_exact_returned_float32():
    g,c,s=fixture(); action=next(a for a in action_catalog(g) if a['kind']=='single_port')
    result=s.acquire(action)
    for i,record in enumerate(result.observation['records']):
        unpacked=EVENT.unpack_from(result.wire,i*EVENT.size)
        assert unpacked[0]==record['local_time_s']
        assert unpacked[1].hex()==record['message_token']
        assert list(unpacked[6:6+len(record['values'])])==record['values']

def test_timestamp_noise_keyed_to_physical_event_across_sample_rates():
    g,c,s=fixture(); ports=next(a['ports'] for a in action_catalog(g) if a['kind']=='module_pair')
    a=s.run(ports,50,execution_index=3,debug=True)
    b=s.run(ports,100,execution_index=3,debug=True)
    left={(e['port'],e['true_time']):e['error'] for e in a.debug['port_times']}
    right={(e['port'],e['true_time']):e['error'] for e in b.debug['port_times']}
    shared=set(left)&set(right)
    assert shared
    assert all(left[k]==right[k] for k in shared)

def test_combined_and_full_acquisition_actual_cost_bounds():
    g,c,s=fixture(); actions=action_catalog(g)
    for kind in ('pair_plus_calibration','full_trace'):
        action=next(a for a in actions if a['kind']==kind and a['sample_hz']==100)
        result=s.acquire(action,execution_index=4,burst_index=1)
        assert result.observation['actual_cost']['executions']==1
        assert result.observation['actual_cost']['decisions']==1
        assert result.observation['actual_cost']['calibrations']==int(kind=='pair_plus_calibration')
        assert all(result.observation['actual_cost'][k]<=action['cost_bound'][k] for k in action['cost_bound'])
        assert len(result.wire)==128*len(result.observation['records'])+160*len(result.observation['calibration_bursts'])
