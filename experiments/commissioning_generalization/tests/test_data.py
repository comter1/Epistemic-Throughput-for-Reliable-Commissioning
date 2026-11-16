from copy import deepcopy
import gzip
import json
import pytest
import jsonschema
from cgx.data.store import Store,public_requirements,check_public_history
from cgx.config import load_config
from cgx.graphs import generate_graph
from cgx.env.system import generate_task
from cgx.observation.actions import initial_budget
from cgx.provenance import atomic_json


def history():
    graph=generate_graph(3,8)
    obs={"action_key":"initial","records":[],"calibration_bursts":[],"visible_status":"ok","actual_cost":{"bytes":0,"executions":1,"calibrations":0,"decisions":0}}
    budget=initial_budget(graph,.25)
    return {"graph":graph,"observations":[obs],"remaining":budget,"initial_budget":budget,"action_keys":[],"nominal_task":generate_task(123),"requirements":public_requirements(load_config())}


def test_D03_O10_store_rejects_private_fields(tmp_path):
    store=Store(tmp_path);h=history();digest=store.put_history(h)
    assert store.load_history(digest)==h
    for location in ("root","graph","record","task"):
        bad=deepcopy(h)
        if location=="root":bad["fault_node"]="secret"
        elif location=="graph":bad["graph"]["nodes"][0]["severity"]=5
        elif location=="record":bad["observations"][0]["true_time"]=4
        else:bad["nominal_task"]["fault_family"]="healthy"
        with pytest.raises((ValueError,jsonschema.ValidationError)):store.put_history(bad)


def test_A06_corrupt_public_blob_cannot_resume_as_valid(tmp_path):
    store=Store(tmp_path);digest=store.put_history(history())
    p=tmp_path/"public/histories"/(digest+".json.gz")
    with gzip.open(p,"wt") as f:json.dump({},f)
    with pytest.raises(ValueError,match="Corrupt"):store.load_history(digest)


def test_A06_transition_requires_success_marker_and_matching_hash(tmp_path):
    store=Store(tmp_path);digest=store.write_transitions("train","case",[{"value":1}])
    assert list(store.transitions("train"))==[]
    atomic_json(store.completion_path("train","case"),{"state":"failed","transition_file_hash":digest})
    assert list(store.transitions("train"))==[]
    atomic_json(store.completion_path("train","case"),{"state":"passed","transition_file_hash":digest})
    assert list(store.transitions("train"))==[{"value":1}]
    store.write_transitions("train","case",[{"value":2}])
    with pytest.raises(ValueError,match="hash changed"):list(store.transitions("train"))


def test_O07_no_free_nonstop_budget_and_no_test_collection(tmp_path):
    from cgx.data.collection import collect
    with pytest.raises(ValueError,match="test split"):collect(load_config(),tmp_path,"test_seen",technical_smoke=True)


def test_D03_training_case_accessor_never_opens_heldout_truth(tmp_path,monkeypatch):
    from cgx.graphs.manifests import write_jsonl,load_cases
    write_jsonl(tmp_path/"private/cases/train.jsonl",[{"split":"train","case_id":"source"}])
    write_jsonl(tmp_path/"private/cases/test_seen.jsonl",[{"fault_node":"POISON"}])
    from pathlib import Path
    original=Path.open;opened=[]
    def checked(path,*args,**kwargs):
        opened.append(str(path))
        if "test_" in path.name or path.name=="private_cases.jsonl":raise AssertionError("Opened sealed labels")
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",checked)
    assert list(Store(tmp_path).cases("train"))==[{"split":"train","case_id":"source"}]
    assert all(p.endswith("train.jsonl") for p in opened)
    with pytest.raises(ValueError):list(load_cases(tmp_path,None))


def test_L08_fixed_plan_support_adds_branch_without_replacing_eight(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from cgx.data.collection import collect_case
    import cgx.env.system
    class InstrumentFixture:
        def __init__(self,*args,**kwargs):pass
        def run(self,**kwargs):
            return SimpleNamespace(observation={"action_key":"initial","records":[],"calibration_bursts":[],"visible_status":"ok","actual_cost":{"executions":1,"bytes":0,"calibrations":0,"decisions":0}},wire=b'')
        def acquire(self,action,**kwargs):
            return SimpleNamespace(observation={"action_key":action['action_key'],"records":[],"calibration_bursts":[],"visible_status":"ok","actual_cost":action['cost_bound']},wire=b'')
    monkeypatch.setattr(cgx.env.system,'System',InstrumentFixture)
    cfg=load_config();cfg['training']['collection_policy_mix']={'random_legal':0.,'structure_rule':1.,'fixed_diverse_plan':0.}
    cfg['training']['collection_execution_caps']=[1];cfg['training']['collection_execution_cap_probabilities']=[1.]
    case={'case_id':'unit-support','split':'train','configuration_id':'unit-graph','fault_family':'healthy'}
    graph=generate_graph(12,8)
    status=collect_case(cfg,tmp_path,case,graph,generate_task(10),{'implementation':'A','source_fold':0},technical_smoke=True)
    rows=list(Store(tmp_path).transitions('train'))
    assert status['fixed_plan_support_trajectories']==1 and status['trajectories']==9
    assert len({r['trajectory_id'] for r in rows if r['coverage_branch'] is None})==8
    support=[r for r in rows if r['coverage_branch']=='fixed_plan_support']
    assert support and all(r['behavior']=='fixed_diverse_plan' for r in support)
