#!/usr/bin/env python3
"""Validate specification consistency, contract examples and simple executable utility."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    p = json.loads((ROOT/"configs/protocol.json").read_text(encoding="utf-8"))
    assert p["protocol_id"] == "CGX-2.0"
    assert len(set(p["seeds"]["training"])) == 3
    assert len(p["methods"]["learned"]) == 5
    assert sum(p["targets"]["family_probabilities"]) == 1
    assert sum(p["training"]["collection_policy_mix"].values()) == 1
    assert abs(sum(p["training"]["collection_execution_cap_probabilities"])-1) < 1e-12
    assert p["resources"]["main_decision_cap"] == p["resources"]["main_execution_cap"] + p["resources"]["main_calibration_cap"]
    counts = {k:v["configurations"]*v["cases_per_configuration"] for k,v in p["splits"].items()}
    assert counts["train"] == 4096 and counts["validation"] == 512
    assert sum(v for k,v in counts.items() if k.startswith("test_")) == 3200
    assert p['resources']['main_execution_budgets']==[1,2,4]
    assert p['compute']['gpus']==3
    assert p["training"]["source_trajectories_per_case"] == len(p["training"]["main_byte_fractions_for_collection"])*p["training"]["replicates_per_byte_fraction"]
    assert p["training"]["gamma"] == 1 and p["training"]["reward_clip"] is None
    assert p["reproducibility"]["true_timestamps_never_in_public_stream"]
    assert not p["training"]["online_test_learning"]
    assert p["models"]["transformer_sees_relations"]
    assert not p["models"]["node_id_embeddings"]
    required=["00_PROTOCOL","01_SYSTEM_AND_FAULTS","02_GRAPHS_AND_SPLITS","03_OBSERVATIONS_AND_BUDGETS",
              "04_MODELS_AND_LEARNING","05_DATA_AND_EVALUATION","06_IMPLEMENTATION_AND_RUNBOOK",
              "07_COMPUTE_2XB200","08_ACCEPTANCE_TESTS","09_SOURCES_AND_DECISIONS"]
    for name in required:
        assert (ROOT/"docs"/(name+".md")).is_file(), name
    try:
        import jsonschema
    except ImportError as exc:
        raise SystemExit("Install jsonschema in an isolated environment to validate examples") from exc
    for name in ["public_graph","private_case","budget","action","observation"]:
        schema=json.loads((ROOT/"schemas"/(name+".schema.json")).read_text())
        example=json.loads((ROOT/"examples"/(name+".json")).read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(example, schema)
    graph=json.loads((ROOT/"examples/public_graph.json").read_text())
    ids={x['id'] for x in graph['nodes']}
    assert len(ids)==len(graph['nodes'])==8
    ports={(n['id'],a['name']):a for n in graph['nodes'] for a in n['ports']}
    for e in graph['edges']:
        src=ports[(e['source'],e['source_port'])]; dst=ports[(e['target'],e['target_port'])]
        assert src['direction']=='out' and dst['direction']=='in'
        assert src['width']==dst['width'] and src['unit']==dst['unit']
    poisoned=dict(graph); poisoned['true_fault_node']='n5'
    try:
        jsonschema.validate(poisoned,json.loads((ROOT/'schemas/public_graph.schema.json').read_text()))
    except jsonschema.ValidationError:
        pass
    else:
        raise AssertionError('Public schema accepted hidden truth')
    spec=importlib.util.spec_from_file_location('cgx_contracts',ROOT/'contracts/interfaces.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    b=mod.ResourceVector(4,1000,2,6)
    left=b.subtract(mod.ResourceVector(1,100,1,1))
    assert left==mod.ResourceVector(3,900,1,5)
    try: left.subtract(mod.ResourceVector(bytes=901))
    except ValueError: pass
    else: raise AssertionError('overrun accepted')
    try: mod.ResourceVector(bytes=-1)
    except ValueError: pass
    else: raise AssertionError('negative budget accepted')
    print(json.dumps({'status':'passed','validated':'handoff_only_not_MuJoCo_or_training',
                      'case_counts':counts,'policy_jobs':15,'example_schemas':5,'public_truth_rejection':True},indent=2))

if __name__=='__main__': main()
