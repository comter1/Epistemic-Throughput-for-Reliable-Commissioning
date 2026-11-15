import copy
import json
import os
from pathlib import Path
import numpy as np
import torch

from cgx.config import load_config,EXPERIMENT
from cgx.models import DiagnosticReader,PolicyModel
from cgx.evaluation.parallel import model_spec,_construct


def test_v2_profile_has_all_registered_cases_methods_and_seeds():
    p=load_config()
    assert p['seeds']['training']==[11,23,37]
    assert p['training']['predictor_update_cap']==100000
    assert p['training']['policy_update_cap']==100000
    assert p['training']['batch_size']==128
    assert p['training']['source_trajectories_per_case']==8
    assert p['resources']['main_execution_budgets']==[1,2,4]
    assert p['resources']['main_execution_cap']==4
    assert len(p['evaluation']['repair_methods'])==6
    assert sum(v['configurations']*v['cases_per_configuration'] for k,v in p['splits'].items() if k.startswith('test_'))==3200
    from cgx.evaluation.design import matrix_cells
    cells=matrix_cells(p)
    assert len(cells)==46 and len(set(cells))==46
    assert len([c for c in cells if c[0]=='full_recording_feasible'])==1
    assert not any(c[0]=='gnn_myopic' for c in cells)


def test_worker_weights_are_frozen_independent_cpu_copies():
    torch.set_num_threads(1)
    for model in (DiagnosticReader(),PolicyModel('transformer_adaptive')):
        specification=model_spec(model);worker=_construct(specification)
        assert not worker.training and not any(p.requires_grad for p in worker.parameters())
        for name,p in model.state_dict().items():
            assert torch.equal(p,worker.state_dict()[name])
            assert p.data_ptr()!=worker.state_dict()[name].data_ptr()


def test_parallel_blob_verification_rejects_corruption(tmp_path,monkeypatch):
    from cgx.data.validation import verify_histories
    from cgx.data.store import Store
    monkeypatch.setenv('CGX_DATA_WORKERS','2')
    store=Store(tmp_path)
    graph=store._put('graphs',{'public':'graph'});task=store._put('tasks',{'task':1});obs=store._put('observations',{'record':2})
    digest=store._put('histories',dict(graph_ref=graph,task_ref=task,observation_refs=[obs]))
    result=verify_histories(tmp_path,[digest],'train');assert result['verified_histories']==1
    (tmp_path/'public/observations'/f'{obs}.json.gz').write_bytes(b'corrupt')
    import pytest
    with pytest.raises(Exception):verify_histories(tmp_path,[digest],'train')


def test_parallel_frozen_matrix_covers_and_resumes_every_cell(tmp_path,monkeypatch):
    from test_evaluation_runtime import fixture
    from cgx.evaluation import parallel_matrix
    from cgx.graphs.manifests import write_jsonl
    from cgx.provenance import atomic_json
    graph,case,task,config=fixture();case['split']='test_seen'
    config['protocol_id']='CGX-1.0' # A four-cell legacy-format execution fixture.
    config['splits']={'test_seen':{'configurations':1,'cases_per_configuration':1}}
    config['methods']['learned']=[];config['methods']['nonlearned']=['default_only','interface_rule'];config['seeds']['training']=[]
    config['resources']['byte_fractions']=[0.,.25]
    write_jsonl(tmp_path/'manifests/split_manifest.jsonl',[{'configuration_id':case['configuration_id'],'split':'test_seen','implementation':'A'}])
    write_jsonl(tmp_path/'manifests/public_graphs.jsonl',[{'configuration_id':case['configuration_id'],'graph':graph}])
    write_jsonl(tmp_path/'manifests/tasks.jsonl',[{'case_id':case['case_id'],'task':task}])
    write_jsonl(tmp_path/'private/cases/test_seen.jsonl',[case])
    atomic_json(tmp_path/'environment.json',{'technical_test':True});atomic_json(tmp_path/'manifests/hashes.json',{})
    (tmp_path/'assets').symlink_to(Path('/home/leiyo/work/cgx_data/assets'),target_is_directory=True)
    monkeypatch.setenv('CGX_EVAL_WORKERS','2')
    monkeypatch.setattr(parallel_matrix,'require_frozen',lambda *args:{'files':{},'protocol_hash':'test-only'})
    reader=DiagnosticReader().eval();reader.checkpoint_hash='actual_fixture_weights'
    monkeypatch.setattr(parallel_matrix,'load_reader',lambda *args,**kwargs:reader)
    result=parallel_matrix.evaluate_parallel(tmp_path,config)
    assert result['case_files']==4
    files=list((tmp_path/'evaluation/main/cases').glob('*.json'))
    rows=[json.loads(p.read_text())['row'] for p in files]
    assert all(r['inference_device']=='cpu' and not r['abstained'] for r in rows)
    assert parallel_matrix.evaluate_parallel(tmp_path,config,resume=True)['case_files']==4
    changed=json.loads(files[0].read_text());changed['row']['bytes_used']+=1;files[0].write_text(json.dumps(changed))
    import pytest
    with pytest.raises(ValueError):parallel_matrix.evaluate_parallel(tmp_path,config,resume=True)


def test_remote_registry_requires_actual_hardware_evidence(tmp_path,monkeypatch):
    from cgx.runtime.remote import registered_workers
    from cgx.provenance import atomic_json,file_sha256
    import pytest
    hardware=tmp_path/'hardware.json';identity=tmp_path/'identity';identity.write_text('fixture—not a real key')
    atomic_json(hardware,dict(state='passed',actual_gpu_forward_backward_index_add=False,gpu_uuid='fixture-gpu'))
    registry=tmp_path/'workers.json';atomic_json(registry,{'workers':[{'hardware_report':str(hardware),'hardware_report_hash':file_sha256(hardware),
        'gpu_uuid':'fixture-gpu','shared_filesystem_verified':True,'identity_file':str(identity)}]})
    monkeypatch.setenv('CGX_REMOTE_WORKERS_FILE',str(registry))
    with pytest.raises(ValueError):registered_workers()


def test_early_stopping_requires_ten_actual_checks_and_keeps_small_improvements():
    from cgx.training.stopping import should_stop,validate_completion
    import pytest
    training=load_config()['training']
    history=[dict(iteration=2000*(i+1),mean_final_nll_bits=2.-i*.0001) for i in range(11)]
    assert not should_stop(history[:10],training)
    assert should_stop(history,training)
    marker=dict(actual_updates=22000,registered_update_cap=100000,termination_reason='validation_patience',early_stopping_rule=training['early_stopping'])
    assert validate_completion(marker,history,training,100000)
    with pytest.raises(ValueError):validate_completion({**marker,'actual_updates':20000},history,training,100000)
    improved=history+[dict(iteration=24000,mean_final_nll_bits=1.99)]
    assert not should_stop(improved,training)


def test_lossless_mmap_features_preserve_all_arrays(tmp_path):
    from test_learning import fixture
    from cgx.models.features import tensorize_history
    from cgx.training.feature_cache import write_features,read_features
    history,_=fixture();original=tensorize_history(history);path=tmp_path/'features.cgxf'
    write_features(path,original);loaded=read_features(path)
    for key in ('nodes','edge_index','edge_types','edge_features','resources','task'):
        assert np.array_equal(getattr(original,key),getattr(loaded,key))
    assert len(original.sequences)==len(loaded.sequences)
    assert all(np.array_equal(a,b) for a,b in zip(original.sequences,loaded.sequences))
    assert original.targets==loaded.targets and original.node_lookup==loaded.node_lookup
    path.write_bytes(path.read_bytes()[:-1])
    import pytest
    with pytest.raises(ValueError):read_features(path)
