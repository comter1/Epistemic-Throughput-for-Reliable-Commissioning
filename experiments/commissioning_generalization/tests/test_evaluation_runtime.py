"""Acceptance fixtures, independent of whether any acquisition method wins."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from cgx.config import load_config
from cgx.graphs import generate_graph,legal_targets
from cgx.env.system import generate_task
from cgx.baselines import RuntimePolicy
from cgx.evaluation.rollout import rollout
from cgx.reporting.results import paired_bootstrap,holm
from cgx.provenance import atomic_json,require_protocol_lock,hash_json,verify_hashes,file_sha256

class PriorReader:
    def predict(self,history):
        keys,prior=legal_targets(history['graph'])
        return {'target_keys':keys,'log_probabilities_base2':np.log2(prior).tolist()}

def fixture():
    graph=generate_graph(324,12,'single',2)
    case={'case_id':'software_fixture','configuration_id':'fixture','split':'validation','fault_family':'healthy','fault_node':None,'severity':None,
          'observer_offsets_ms':{d:float(i) for i,d in enumerate(graph['clock_domains'])},'streams':{'task':321,'noise':322,'audit':323,'verification':324}}
    return graph,case,generate_task(321),load_config()

def test_O09_E06_zero_budget_common_prior_and_full_reference_label():
    graph,case,task,config=fixture(); results=[]
    for method in ['default_only','interface_rule','full_recording_feasible']:
        policy=RuntimePolicy(method)
        row,_,_,_=rollout(graph,case,task,config,PriorReader(),policy,0)
        results.append(row)
    assert all(r['final_nll_bits']==2 and r['bytes_used']==0 and r['executions_used']==0 for r in results)
    policy=RuntimePolicy('full_recording_unrestricted_reference')
    row,_,_,_=rollout(graph,case,task,config,PriorReader(),policy,0,unrestricted=True)
    assert not row['feasible'] and row['unrestricted_reference'] and row['executions_used']==4

def test_E03_nan_prediction_retains_case_and_prior():
    class BadReader:
        def predict(self,h):
            keys,_=legal_targets(h['graph']);return {'target_keys':keys,'log_probabilities_base2':[float('nan')]*len(keys)}
    graph,case,task,config=fixture()
    row,_,_,_=rollout(graph,case,task,config,BadReader(),RuntimePolicy('default_only'),.25)
    assert row['case_id']==case['case_id'] and row['abstained'] and row['final_nll_bits']==2

def test_E03_runtime_timeout_keeps_denominator():
    graph,case,task,config=fixture(); config['resources']['diagnosis_timeout_seconds_per_call']=0.
    row,_,_,_=rollout(graph,case,task,config,PriorReader(),RuntimePolicy('default_only'),.25)
    assert row['timeout'] and row['abstained'] and row['final_nll_bits']==2

def test_E04_system_bootstrap_equal_weight_and_pairing():
    rows=[]
    for cid,n,delta in [('a',2,1.),('b',20,3.)]:
        for case in range(n):
            for method,loss in [('first',4+delta),('second',4)]:
                rows.append({'configuration_id':cid,'case_id':str(case),'method':method,'final_nll_bits':loss,'policy_seed':11})
    result=paired_bootstrap(pd.DataFrame(rows),'first','second',2000,5)
    assert result['difference_nll_bits']==2. and result['configurations']==2 and result['cases']==22
    assert result['ci_low']<=2<=result['ci_high']
    with pytest.raises(ValueError): paired_bootstrap(pd.DataFrame(rows[:-1]),'first','second',10)
    assert holm([.01,.04,.03])==pytest.approx([.03,.06,.06])

def test_A05_A06_E11_missing_or_changed_prerequisite_never_passes(tmp_path):
    with pytest.raises(FileNotFoundError): require_protocol_lock(tmp_path,load_config())
    p=tmp_path/'data.json';atomic_json(p,{'value':1});digest=file_sha256(p)
    atomic_json(tmp_path/'protocol_lock.json',{'state':'passed','technical_smoke':True,'config_hash':hash_json(load_config()),'files':{'data.json':digest}})
    require_protocol_lock(tmp_path,load_config())
    atomic_json(p,{'value':2})
    with pytest.raises(ValueError): require_protocol_lock(tmp_path,load_config())

def test_E07_E08_verification_uses_independent_streams_and_no_hidden_feedback():
    from cgx.env.system import System
    from cgx.evaluation.repairs import verification
    graph,case,task,config=fixture();system=System(graph,case,task,config)
    original=deepcopy(system.task);original_targets=system.target_xyz.copy()
    result=verification(system,case,config,'healthy',.99,.5,audit=True)
    assert result['public_verification_count']==2
    assert result['audit_count'] in (0,20)
    assert result['audit_all_pass'] is None if result['audit_count']==0 else isinstance(result['audit_all_pass'],bool)
    assert system.task==original and np.array_equal(system.target_xyz,original_targets)
    assert 'risk_certificate' not in result


def test_E03_fixed_plan_exception_retained_in_denominator():
    class BrokenPlan:
        fixed_plan=True
        def plan(self,*args):raise KeyError("deliberate fixture")
    graph,case,task,config=fixture()
    row,_,_,_=rollout(graph,case,task,config,PriorReader(),BrokenPlan(),.25)
    assert row['case_id']==case['case_id'] and row['abstained'] and row['executions_used']==0
    assert any('KeyError' in e for e in row['errors'])


def test_E03_watchdog_interrupts_call_and_warmup_is_separate():
    import time
    from cgx.evaluation.rollout import call_deadline,CallDeadlineExceeded
    start=time.monotonic()
    with pytest.raises(CallDeadlineExceeded):
        with call_deadline(.02):time.sleep(.5)
    assert time.monotonic()-start < .3
    class ColdReader(PriorReader):
        def __init__(self):self.calls=0
        def predict(self,h):
            self.calls+=1
            if self.calls==1:time.sleep(.04)
            return super().predict(h)
    graph,case,task,config=fixture();reader=ColdReader()
    row,_,_,_=rollout(graph,case,task,config,reader,RuntimePolicy('default_only'),0)
    assert not row['timeout'] and row['warmup_seconds']>=.04 and row['inference_seconds']<.04
    row2,_,_,_=rollout(graph,case,task,config,reader,RuntimePolicy('default_only'),0)
    assert row2['warmup_seconds']==0 and row2['recorder_seconds']>0


def test_D04_eval_node_permutation_preserves_physics_and_pairing():
    graph,case,task,config=fixture()
    a,_,h1,system1=rollout(graph,case,task,config,PriorReader(),RuntimePolicy('default_only'),0)
    b,_,h2,system2=rollout(graph,case,task,config,PriorReader(),RuntimePolicy('interface_rule'),0)
    assert h1['graph']==h2['graph'] and h1['observations']==h2['observations']
    assert {n['id']:n['clock_domain'] for n in graph['nodes']}=={n['id']:n['clock_domain'] for n in h1['graph']['nodes']}
    assert system1.graph==graph==system2.graph
    assert [n['id'] for n in h1['graph']['nodes']] != [n['id'] for n in graph['nodes']]
    assert a['final_nll_bits']==b['final_nll_bits']


def test_E05_E11_result_inventory_missing_extra_and_tampering(tmp_path):
    from cgx.evaluation.matrix import result_identity,validate_result_artifact,verify_result_inventory
    row={'method':'gnn_adaptive','policy_seed':11,'execution_cap':4,'byte_fraction':.25,'case_id':'case'}
    payload={'row':row,'trace':[]};key=result_identity(row)
    item={'state':'passed','frozen_hash':'frozen','result_hash':hash_json(payload),**payload}
    validate_result_artifact(item,key,'frozen')
    with pytest.raises(ValueError):verify_result_inventory(tmp_path,{key})
    atomic_json(tmp_path/(key+'.json'),item)
    assert verify_result_inventory(tmp_path,{key})==1
    item['row']['case_id']='changed'
    with pytest.raises(ValueError):validate_result_artifact(item,key,'frozen')
    atomic_json(tmp_path/'extra.json',{})
    with pytest.raises(ValueError):verify_result_inventory(tmp_path,{key})


def test_D03_validation_callback_never_reads_test_private_files(tmp_path,monkeypatch):
    from cgx.graphs.manifests import write_jsonl
    from cgx.evaluation.matrix import validation_callback
    write_jsonl(tmp_path/'manifests/split_manifest.jsonl',[])
    write_jsonl(tmp_path/'manifests/public_graphs.jsonl',[])
    write_jsonl(tmp_path/'manifests/tasks.jsonl',[])
    write_jsonl(tmp_path/'private/cases/validation.jsonl',[])
    write_jsonl(tmp_path/'private/private_cases.jsonl',[{'fault_node':'poison'}])
    original=Path.open;opened=[]
    def checked(path,*args,**kwargs):
        opened.append(str(path))
        if path.name=='private_cases.jsonl' or path.name.startswith('test_'):raise AssertionError('Opened held-out labels')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',checked)
    callback=validation_callback(tmp_path,load_config())
    assert callable(callback) and any(p.endswith('validation.jsonl') for p in opened)


def test_E05_matrix_executes_every_cell_and_resumes_by_payload_hash(tmp_path,monkeypatch):
    """Small explicit test fixture, never the registered scientific matrix."""
    from cgx.graphs.manifests import write_jsonl
    import cgx.evaluation.matrix as matrix
    import cgx.training
    graph,case,task,config=fixture();case['split']='test_seen';config['protocol_id']='CGX-1.0'
    config['splits']={'test_seen':{'configurations':1,'cases_per_configuration':1}}
    config['methods']['learned']=[];config['methods']['nonlearned']=['default_only','interface_rule'];config['seeds']['training']=[]
    config['resources']['byte_fractions']=[0.,.25]
    write_jsonl(tmp_path/'manifests/split_manifest.jsonl',[{'configuration_id':case['configuration_id'],'split':'test_seen','implementation':'A'}])
    write_jsonl(tmp_path/'manifests/public_graphs.jsonl',[{'configuration_id':case['configuration_id'],'graph':graph}])
    write_jsonl(tmp_path/'manifests/tasks.jsonl',[{'case_id':case['case_id'],'task':task}])
    write_jsonl(tmp_path/'private/cases/test_seen.jsonl',[case])
    atomic_json(tmp_path/'environment.json',{'technical_test':True});atomic_json(tmp_path/'manifests/hashes.json',{})
    (tmp_path/'assets').symlink_to(Path('/home/leiyo/work/cgx_data/assets'),target_is_directory=True)
    frozen={'files':{},'protocol_hash':'test-only'}
    monkeypatch.setattr(matrix,'require_frozen',lambda *args:frozen)
    reader=PriorReader();reader.checkpoint_hash='prior-test-reader'
    monkeypatch.setattr(cgx.training,'load_reader',lambda *args,**kwargs:reader)
    result=matrix.evaluate(tmp_path,config)
    assert result['case_files']==4 and result['expected_for_requested_suites']==4
    resumed=matrix.evaluate(tmp_path,config,resume=True)
    assert resumed['case_files']==4
    path=next((tmp_path/'evaluation/main/cases').glob('*.json'))
    item=json.loads(path.read_text());item['row']['final_nll_bits']=99.;atomic_json(path,item)
    with pytest.raises(ValueError,match='content changed'):matrix.evaluate(tmp_path,config,resume=True)


def test_L08_fixed_plan_finishes_before_final_diagnosis_failure():
    class FixedTwo:
        fixed_plan=True
        _cgx_eval_warmed=True
        def plan(self,history,actions,prediction=None):
            action=next(a for a in actions if a['kind']=='single_port' and a['sample_hz']==10)
            return [action,deepcopy(action)]
    class FinalFailure(PriorReader):
        _cgx_eval_warmed=True
        def __init__(self):self.calls=0
        def predict(self,h):
            self.calls+=1
            if self.calls==2:raise FloatingPointError('Final diagnosis failed')
            return super().predict(h)
    graph,case,task,config=fixture();reader=FinalFailure()
    result,traces,history,_=rollout(graph,case,task,config,reader,FixedTwo(),.25)
    assert result['executions_used']==2 and len(history['observations'])==3
    assert reader.calls==2 and result['abstained']
    assert [t['diagnosis_updated'] for t in traces]==[False,True]
