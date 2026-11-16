"""Inventory immutable SED inputs and register the simple-selector extension."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from cgx.config import EXPERIMENT, load_config
from cgx.data.store import Store
from cgx.provenance import atomic_json, file_sha256, hash_json, now, read_json
from structured_evidence_v1.gates import verify_readers

NAMESPACE = Path(__file__).parent
EXPECTED_SPLITS = {'train': (32, 4096), 'validation': (8, 512),
    'test_seen': (8, 800), 'test_connections': (16, 1600), 'test_size': (8, 800)}
TEMPLATES = {
    'F0': dict(plan='one_full_trace', rate=100, calibration='none'),
    'F1': dict(plan='one_full_trace', rate=100, calibration='up_to_two_public_coverage_pairs'),
    'F2': dict(plan='forward_component_coverage', rate=100, calibration='requested_cross_clock_pairs'),
    'F3': dict(plan='reverse_component_coverage', rate=100, calibration='requested_cross_clock_pairs'),
    'F4': dict(plan='alternate_cache_queue_adapter', rate=100, calibration='requested_cross_clock_pairs'),
    'F5': dict(plan='forward_component_coverage', rate=50, calibration='requested_cross_clock_pairs'),
    'F6': dict(plan='alternate_cache_queue_adapter', rate=50, calibration='requested_cross_clock_pairs'),
    'F7': dict(plan='forward_component_coverage', rate=100, calibration='none'),
}


def write_once(path, obj):
    path=Path(path)
    if path.exists():
        raise RuntimeError(f'Existing artifact is preserved: {path}')
    atomic_json(path,obj)


def inventory(source_root, sed_run, run_dir, workers=32):
    source,sed,run=map(lambda p:Path(p).resolve(),(source_root,sed_run,run_dir))
    assert source!=run and sed!=run and sed not in run.parents
    assert not (run/'source_inventory.json').exists()
    config=load_config();store=Store(source);repo=EXPERIMENT.parents[1]
    verify_readers(sed);frozen=read_json(sed/'models/frozen.json')
    assert frozen['state']=='passed' and frozen['protocol_hash']==hash_json(config)
    for name,digest in frozen['scientific_code'].items():
        assert file_sha256(EXPERIMENT/'structured_evidence_v1'/name)==digest,name
    for name,digest in frozen['files'].items():assert file_sha256(sed/name)==digest,name
    assert file_sha256(sed/'models/readers/final/model.json')==frozen['reader_hash']
    old_checks=read_json(sed/'source_checks.json')
    for name,digest in old_checks['manifest_hashes'].items():assert file_sha256(source/name)==digest,name
    split_stats={};split_ids={};cases=[c for split in EXPECTED_SPLITS for c in store.cases(split)];by_split=defaultdict(list)
    for case in cases:by_split[case['split']].append(case)
    for split,(n_configs,n_cases) in EXPECTED_SPLITS.items():
        subset=by_split[split];configs=store.configurations(split)
        assert len(subset)==n_cases and len(configs)==n_configs,(split,len(subset),len(configs))
        assert len({c['case_id'] for c in subset})==n_cases
        families=Counter(c['fault_family'] for c in subset)
        assert set(families)==set(config['targets']['families']) and set(families.values())=={n_cases//4}
        for cfg in configs:
            group=[c for c in subset if c['configuration_id']==cfg['configuration_id']]
            assert len(group)==n_cases//n_configs
            assert set(Counter(c['fault_family'] for c in group).values())=={len(group)//4}
        split_stats[split]=dict(configurations=n_configs,cases=n_cases,families=dict(families))
        split_ids[split]=sorted(c['case_id'] for c in subset)
    for left in split_ids:
        for right in split_ids:
            if left!=right:assert not set(split_ids[left])&set(split_ids[right]),(left,right)
    assert config['system']['waypoints']==3 and config['targets']['faults_per_case_max']==1
    assert config['resources']['main_execution_budgets']==[1,2,4]
    assert config['resources']['main_calibration_cap']==2
    assert config['resources']['reference_executions_for_byte_cap']==4
    assert config['resources']['round_sensitivity_decision_cap_rule']=='execution_cap_plus_calibration_cap'
    assert config['resources']['selection_timeout_seconds_per_decision']==2
    assert config['resources']['diagnosis_timeout_seconds_per_call']==2
    assert config['observations']['sample_hz']==[10,50,100]
    assert config['seeds']['training']==[11,23,37]
    assert config['evaluation']['primary_metric']=='mean_final_log_loss_bits'
    required=['protocol.json','base_config.json','models/frozen.json','models/readers/frozen.json',
        'models/readers/parameters_frozen.json','models/readers/final/model.json',
        'evaluation/main/per_case_results.parquet','evaluation/main/complete.json',
        'runtime_audit/full_matrix_adjusted.parquet','runtime_audit/official_vs_untimed.parquet',
        'runtime_audit/complete.json','runtime_audit/official_cell_hashes.jsonl',
        'evidence_export/new_main/blobs.jsonl.gz','evidence_export/new_main/complete.json',
        'evidence_export/old_histories/blobs.jsonl.gz','evidence_export/old_histories/complete.json',
        'provenance/protected_before.json','provenance/original_evaluation_input_hashes.jsonl',
        'provenance/final_audit.json']
    assert all((sed/name).is_file() for name in required)
    assert (source/'public/observations').is_dir() and (sed/'public/observations').is_dir()
    assert (sed/'cache/physical').is_dir()
    assert (source/'assets/mujoco_menagerie/franka_emika_panda').is_dir()
    main=read_json(sed/'evaluation/main/complete.json');audit=read_json(sed/'runtime_audit/complete.json')
    assert main['state']=='passed' and main['rows']==main['expected']==147200
    assert audit['state']=='passed' and audit['full_matrix_rows']==147200
    assert audit['official_results_unchanged'] and audit['full_matrix_denominator_unchanged']
    for name,digest in audit['output_hashes'].items():assert file_sha256(sed/'runtime_audit'/name)==digest,name
    source_paths={name:dict(path=str(sed/name),sha256=file_sha256(sed/name),bytes=(sed/name).stat().st_size) for name in required}
    # Preserve all previously protected inputs, every tracked pre-existing source
    # file, selected weights and old/new SED official result inventories.
    protected=read_json(sed/'provenance/protected_before.json')
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=repo).decode().split('\0')
    for name in tracked:
        p=repo/name
        if name and p.is_file() and NAMESPACE not in p.parents:protected[str(p)]=file_sha256(p)
    protected.update({str(sed/name):digest for name,digest in frozen['files'].items()})
    protected.update({v['path']:v['sha256'] for v in source_paths.values()})
    for name in (sed/'provenance/original_evaluation_input_hashes.jsonl',sed/'runtime_audit/official_cell_hashes.jsonl'):
        for line in name.open():
            row=json.loads(line)
            path=source/row['path'] if 'path' in row else sed/'evaluation/main/cases'/(row['cell_key']+'.json')
            protected[str(path)]=row['sha256']
    def check(item):
        path,expected=item
        if not Path(path).is_file() or file_sha256(path)!=expected:return path
    with ThreadPoolExecutor(max_workers=workers) as pool:
        changed=[p for p in pool.map(check,protected.items()) if p]
    if changed:raise RuntimeError(f'Protected input mismatch: {changed[:10]}')
    write_once(run/'provenance/protected_before.json',protected)
    pilot=[]
    for configuration in sorted({c['configuration_id'] for c in by_split['validation']}):
        for family in config['targets']['families']:
            group=[c for c in by_split['validation'] if c['configuration_id']==configuration and c['fault_family']==family]
            pilot.append(min(group,key=lambda c:hash_json(['simple-selectors-throughput-v1',c['case_id']])))
    assert len(pilot)==32
    write_once(run/'validation/pilot_cases.json',dict(state='selected_before_validation_outcomes',case_ids=[c['case_id'] for c in pilot],
        strata=[dict(case_id=c['case_id'],configuration_id=c['configuration_id'],family=c['fault_family']) for c in pilot],
        selection='one hash-minimal case per validation configuration and fault-family stratum',
        useful_required_validation=True,subset_not_a_separate_scientific_pilot=True))
    write_once(run/'provenance/split_case_ids.json',split_ids)
    inventory=dict(state='passed',created_at=now(),source_root=str(source),sed_run=str(sed),run_dir=str(run),
        current_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip(),
        git_status_at_inventory=subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True),
        executable_config_path=str(EXPERIMENT/'configs/protocol.json'),executable_config_hash=hash_json(config),
        executable_config_sha256=file_sha256(EXPERIMENT/'configs/protocol.json'),splits=split_stats,
        reader_hash=frozen['reader_hash'],reader_frozen_sha256=file_sha256(sed/'models/readers/frozen.json'),
        available_selected_checkpoints={k:dict(v,path=str(sed/'models/policies'/k/v['checkpoint'])) for k,v in frozen['inventory'].items()},
        frozen_cost_aware_coefficients=frozen['cost_aware_beta'],importable_official_rows=147200,importable_primary_rows=73600,
        importable_completed_read_rows=147200,existing_diagnostic_timeout_rereads=audit['replayed_cells'],
        inputs=source_paths,public_stores=[str(source/'public'),str(sed/'public')],
        exact_physical_cache=str(sed/'cache/physical'),protected_files=len(protected),
        protected_manifest_sha256=file_sha256(run/'provenance/protected_before.json'),
        protocol_discrepancies=[],missing_required_artifacts=[],new_model_training=False,new_reader_fitting=False)
    write_once(run/'source_inventory.json',inventory)
    write_once(run/'base_config.json',config)
    protocol=dict(protocol_id='CGX-SIMPLE-SELECTORS-V1',state='declared_before_new_validation_and_test_outcomes',created_at=now(),
        source_inventory_sha256=file_sha256(run/'source_inventory.json'),request_sha256=file_sha256(run/'REQUEST.md'),
        source_root=str(source),sed_run=str(sed),reader_hash=frozen['reader_hash'],config_hash=hash_json(config),
        completed_read_watchdog_seconds=120,fresh_robot_execution_attempt_cap=190000,
        execution_budgets=[1,2,4],calibration_cap=2,decision_cap='B+2',byte_fraction=1.,
        byte_cap='four full-rate recordings, independent of B',selection_deadline_seconds=2.,diagnostic_deadline_seconds=2.,
        source_validation_cases=512,source_validation_logical_rows=16896,new_primary_rows=32000,new_transfer_rows=32000,new_test_rows=64000,
        imported_learned_and_four_run_reference_rows=147200,random_choice_seeds=[11,23,37],deterministic_seed=None,
        maximum_logical_additional_test_executions=140800,maximum_logical_additional_validation_executions=33280,
        maximum_initial_distinct_case_executions=3712,all_attempts_including_failed_and_engineering_counted=True,
        fixed_templates=TEMPLATES,heuristic_stop_options=[None,.99,.999],confidence_stop_requires_first_additional_execution=True,
        selection_rule=['configuration_balanced_mean_deployed_final_nll_bits','fewer_bytes','fewer_executions','fewer_calibrations','fixed_candidate_order'],
        principal_comparisons=[['gnn_adaptive','fixed_template_validation'],['gnn_adaptive','interface_rule_se'],
            ['transformer_adaptive','fixed_template_validation'],['transformer_adaptive','interface_rule_se']],
        statistics=dict(primary_suite='test_connections',primary_budget=4,bootstrap_replicates=2000,seed=671917,
            sampling='average seeds within physical case, then paired configurations and family-stratified cases',
            accuracy_noninferiority_tolerance=.01,fault_miss_noninferiority_tolerance=.01,
            comparability='both one-sided paired95% bounds meet the separately declared1pp tolerances',
            intervals='descriptive two-sided95% plus separately labeled one-sided95% comparability bounds',
            simultaneous_significance_claims=False,no_p_value_equivalence_claim=True),
        outcomes=dict(online='unchanged per-call deadlines and fallbacks',completed='same saved evidence, no recovered acquisitions,120s crash watchdog',
            non_timeout_predictions='reuse only verified final completed vectors',all_failed_cells_retained=True),
        stages=['engineering_controls','useful_validation32_throughput','all512_source_validation','selection_seal','primary32000_and_report','transfer32000','full_report_and_archive'],
        hardware=dict(authorized_ports=[2379,2982,2404],authority='explicit human authorization in this conversation',
            computation='remote CPU process pools; no GPU training or reservation',local='coordination/editing only',threads_per_worker=1,
            unrelated_jobs_preserved=True),delivery_maximum_bytes=500000000,
        previous_runs_immutable=True,no_new_selector_training=True,no_new_reader_fit=True,no_reward_regeneration=True,
        studied_test_cases_reused_for_paired_extension=True,new_untouched_test_claim=False)
    write_once(run/'protocol.json',protocol)
    return dict(state='passed',split_counts=split_stats,protected_files=len(protected),protocol_discrepancies=[],run_dir=str(run))


def seal_code(run_dir):
    run=Path(run_dir)
    if (run/'selection_manifest.json').exists():raise RuntimeError('Selection already sealed')
    science=['policies.py','rollout.py','evaluation.py','storage.py','cache.py']
    missing=[name for name in science if not (NAMESPACE/name).is_file()]
    if missing:raise FileNotFoundError(missing)
    record=dict(state='sealed_before_validation',created_at=now(),
        code_hashes={name:file_sha256(NAMESPACE/name) for name in science},
        protocol_sha256=file_sha256(run/'protocol.json'),source_inventory_sha256=file_sha256(run/'source_inventory.json'),
        fixed_templates=TEMPLATES,stopping_options=[None,.99,.999],reader_hash=read_json(run/'protocol.json')['reader_hash'])
    write_once(run/'provenance/validation_design_seal.json',record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['inventory','seal-code']);p.add_argument('--run-dir',required=True)
    p.add_argument('--source-root');p.add_argument('--sed-run');p.add_argument('--workers',type=int,default=32)
    a=p.parse_args();print(json.dumps(inventory(a.source_root,a.sed_run,a.run_dir,a.workers) if a.command=='inventory' else seal_code(a.run_dir)),flush=True)
