from pathlib import Path
import json
import subprocess
import sys
import pytest
from cgx.config import load_config,data_root,EXPERIMENT,default_workers,allocated_cpu_count
from cgx.provenance import command_record,read_json

def test_A03_generated_artifacts_reject_entire_repository():
    for folder in ('finite_delay_benchmark_complete','finite-delay-plotting','result_review','experiments/commissioning_generalization'):
        with pytest.raises(ValueError):data_root(EXPERIMENT.parents[1]/folder)

def test_A04_registered_counts_seeds_priors_and_no_hidden_schema_fields():
    import jsonschema
    c=load_config()
    assert c['protocol_id']=='CGX-2.0'
    assert c['splits']['train']['configurations']==32 and c['splits']['validation']['configurations']==8
    assert [c['splits'][s]['configurations'] for s in c['splits'] if s.startswith('test_')]==[8,16,8]
    assert c['seeds']['training']==[11,23,37]
    assert c['targets']['family_probabilities']==[.25]*4
    assert c['training']['batch_size']==128 and c['training']['gamma']==1
    assert c['training']['predictor_update_cap']==100000 and c['training']['policy_update_cap']==100000
    assert c['training']['validation_interval']==2000 and c['training']['early_stopping']['patience_checks']==10
    assert c['compute']['gpus']==3 and c['resources']['main_execution_budgets']==[1,2,4]
    g=json.loads((EXPERIMENT/'examples/public_graph.json').read_text())
    g['fault_family']='healthy'
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate(g,json.loads((EXPERIMENT/'schemas/public_graph.schema.json').read_text()))

def test_A06_failed_command_records_failure_not_success(tmp_path):
    with pytest.raises(RuntimeError):
        with command_record(tmp_path,['explicit_failure_fixture'],load_config()):
            raise RuntimeError('intentional acceptance fixture')
    rows=[read_json(p) for p in (tmp_path/'commands').glob('*.json')]
    assert len(rows)==1 and rows[0]['state']=='failed' and rows[0]['exit_code']!=0

def test_A05_unimplemented_or_missing_prerequisite_cli_nonzero(tmp_path):
    p=subprocess.run([sys.executable,'-m','cgx.cli','freeze-models','--data-root',str(tmp_path)],capture_output=True,text=True)
    assert p.returncode!=0 and 'protocol_lock.json' in p.stderr
    assert not (tmp_path/'models/frozen.json').exists()

def test_allocated_worker_count_respects_affinity_and_cgroup():
    assert default_workers()==min(16,max(1,allocated_cpu_count()//2))
