from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import pytest
import jsonschema
from cgx.config import load_config
from cgx.graphs import generate_graph,legal_targets,relabel_graph,isomorphic,structure_hash,validate_graph
from cgx.graphs.generator import graph_statistics,matched_variant
from cgx.graphs.manifests import build_graph_records,_cases


@pytest.mark.parametrize("size,motif,branches",[(8,"chain",2),(12,"single",2),(14,"double",2),(16,"double",3),(32,"double",3)])
def test_S07_S08_typed_graph_channel_coverage(size,motif,branches):
    graph=generate_graph(42,size,motif,branches)
    assert len(graph["nodes"])==size
    assert validate_graph(graph)
    schema=json.loads((Path(__file__).parents[1]/"schemas/public_graph.schema.json").read_text())
    jsonschema.validate(graph,schema)
    changed=deepcopy(graph);changed["edges"][0]["target_port"]="nonexistent"
    with pytest.raises(ValueError):validate_graph(changed)


def test_D01_D09_names_and_continuous_parameters_do_not_change_identity():
    graph=generate_graph(44,14,"double",2)
    renamed,_=relabel_graph(graph,998)
    assert isomorphic(graph,renamed)
    assert isomorphic(graph,renamed,clocks=True)
    assert structure_hash(graph)==structure_hash(renamed)
    for n in renamed["nodes"]:
        if n["type"]=="adapter":n["nominal"]["scale"]=13.37
    assert isomorphic(graph,renamed)


def test_D07_legal_variable_targets_and_family_prior():
    sizes=[]
    for count in (8,12,28):
        g=generate_graph(12,count,"chain")
        keys,prior=legal_targets(g);sizes.append(len(keys))
        assert keys[0]=="healthy" and prior[0]==.25
        assert sum(prior)==pytest.approx(1.)
        for family in ("cache_stale","queue_hold","adapter_scale"):
            assert sum(p for k,p in zip(keys,prior) if k.endswith(":"+family))==pytest.approx(.25)
    assert len(set(sizes))>1


def test_D01_D02_D08_full_graph_manifest_and_matched_holdouts():
    cfg=load_config();rows,graphs,book=build_graph_records(cfg)
    assert len(rows)==72
    for split,spec in cfg["splits"].items():
        assert sum(r["split"]==split for r in rows)==spec["configurations"]
    for row in rows:
        if row["split"]=="test_connections":
            own=graph_statistics(graphs[row["configuration_id"]]);anchor=graph_statistics(graphs[row["anchor_configuration_id"]])
            assert all(own[k]==anchor[k] for k in ("components","type_counts","branch_count","degree_histogram"))
            assert not isomorphic(graphs[row["configuration_id"]],graphs[row["anchor_configuration_id"]])
        elif row["split"]=="test_clocks":
            assert isomorphic(graphs[row["configuration_id"]],graphs[row["anchor_configuration_id"]])
            assert not isomorphic(graphs[row["configuration_id"]],graphs[row["anchor_configuration_id"]],clocks=True)


def test_D06_balanced_cases_and_rng_separation():
    cfg=load_config();rows,graphs,_=build_graph_records(cfg)
    cases=list(_cases(cfg,rows[:2],graphs));seen=set()
    assert len(cases)==256
    for cid in {c["configuration_id"] for c in cases}:
        assert set(Counter(c["fault_family"] for c in cases if c["configuration_id"]==cid).values())=={32}
    for c in cases:
        for value in c["streams"].values():
            assert value not in seen;seen.add(value)
