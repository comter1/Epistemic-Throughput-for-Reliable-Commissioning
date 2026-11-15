"""Capacity estimates from executed measurements, distinct from completion."""
from pathlib import Path
import shutil
from cgx.config import default_workers,EXPERIMENT
from cgx.provenance import read_json,atomic_json,file_sha256,now


def resource_forecast(root,config,learning_profile):
    from cgx.evaluation.design import matrix_cells,seeded_method
    root=Path(root);learning_profile=Path(learning_profile)
    physics_path=root/'reports/execution_profile.json';physics=read_json(physics_path);learning=read_json(learning_profile)
    if physics['executions']!=1000 or learning['actual_updates']!=1000 or learning['state']!='passed':
        raise ValueError('Forecast requires actual1000 physical executions and1000 measured optimizer updates')
    profiles={p['mode']:p for p in learning['profiles']};stats=physics['summary']
    count=lambda split:config['splits'][split]['configurations']*config['splits'][split]['cases_per_configuration']
    source=count('train');validation=count('validation');test=sum(count(s) for s in config['splits'] if s.startswith('test_'))
    seeds=len(config['seeds']['training']);policy_runs=len(config['methods']['learned'])*seeds
    reader_cap=config['training']['predictor_update_cap'];policy_cap=config['training']['policy_update_cap']
    updates={'reader_gnn':4*reader_cap,'reader_transformer':reader_cap,'policy_gnn':4*seeds*policy_cap,'policy_transformer':seeds*policy_cap}
    kernel_seconds=sum(updates[k]*profiles[k]['p50_seconds'] for k in updates)
    cells=matrix_cells(config);main_evaluations=test*len(cells);main_physical_upper=test*sum(1+cap for _,_,cap,_ in cells)
    interval=config['training']['validation_interval'];max_checks=policy_cap//interval
    min_checks=min(max_checks,1+config['training'].get('early_stopping',{}).get('patience_checks',max_checks-1))
    repair_pairs=sum(len(config['evaluation']['repair_training_seeds']) if seeded_method(m,config) else 1 for m in config['evaluation']['repair_methods'])
    repair_cases=sum(1 for s in config['splits'] if s.startswith('test_'))*config['evaluation']['repair_configurations_per_test_suite']*config['evaluation']['repair_cases_per_configuration']*repair_pairs
    inputs={str(physics_path):file_sha256(physics_path),str(learning_profile):file_sha256(learning_profile)}
    report=dict(state='passed',created_at=now(),planning_estimate_only=True,protocol_id=config['protocol_id'],gpus=config['compute']['gpus'],
        measured_executions=1000,measured_updates=1000,measured_physics=stats,measured_training=profiles,inputs=inputs,
        source_cases=source,validation_cases=validation,test_cases=test,reader_jobs=5,policy_jobs=policy_runs,
        planned_policy_updates=policy_runs*policy_cap,planned_reader_updates=5*reader_cap,
        optimizer_only_gpu_hours_p50=kernel_seconds/3600,optimizer_only_ideal_wall_hours=kernel_seconds/(config['compute']['gpus']*3600),
        main_evaluation_case_count=main_evaluations,main_physical_execution_upper_bound=main_physical_upper,
        full_policy_validation_case_count_range=[policy_runs*min_checks*validation,policy_runs*max_checks*validation],
        repair_audit_conditions=repair_cases,source_hard_execution_cap=config['training']['max_training_executions'],
        disk_free_bytes=shutil.disk_usage(root).free,workers=default_workers(),optional_appendix_excluded=config['evaluation'].get('optional_appendix_suites',[]))
    if config['protocol_id']=='CGX-2.0':
        throughput_path=root/'reports/v2_throughput.json';smoke_path=root/'reports/technical_smoke.json'
        throughput=read_json(throughput_path);smoke=read_json(smoke_path)
        if throughput['state']!='passed' or smoke['state']!='passed' or smoke.get('technical_batch_size')!=config['training']['batch_size']:
            raise ValueError('V2 requires actual collection, evaluation and full-batch training throughput')
        inputs.update({str(p):file_sha256(p) for p in (throughput_path,smoke_path)})
        speed={k:v['updates_per_second_excluding_validation'] for k,v in smoke['training_throughput'].items()}
        max_training_seconds=sum(reader_cap/speed['readers/'+name] for name in ['crossfit_0','crossfit_1','crossfit_2','final','independent'])
        max_training_seconds+=sum(seeds*policy_cap/speed['policies/'+m+'_seed11'] for m in config['methods']['learned'])
        minimum_ratio=min_checks*interval/policy_cap
        reused_path=root/'reports/reused_data.json'
        reused=read_json(reused_path)['cases_by_split'] if reused_path.exists() else {'train':0,'validation':0}
        total_collection_seconds=(source+validation-sum(reused.values()))/throughput['collection']['cases_per_second']
        eval_seconds={r['method']:1/r['cases_per_second'] for r in throughput['evaluation']}
        main_seconds=test*sum(eval_seconds[m] for m,_,_,_ in cells)
        validation_max_seconds=max_checks*validation*seeds*sum(eval_seconds[m] for m in config['methods']['main_learned_evaluation'])
        validation_min_seconds=validation_max_seconds*minimum_ratio
        beta_seconds=validation*seeds*len(config['methods']['cost_aware_adaptation']['beta_grid'])*eval_seconds['cost_aware_adaptation']
        threshold_seconds=validation*sum((seeds if seeded_method(m,config) else 1)*eval_seconds[m] for m in config['evaluation']['repair_methods'])
        public_audit_executions=(validation*repair_pairs+repair_cases)*(config['resources']['verification_executions_reserved']+config['resources']['audit_executions'])
        audit_seconds=public_audit_executions*stats['mean_seconds']/throughput['workers']
        gpu_count=config['compute']['gpus']
        estimates=[(max_training_seconds*ratio+vseconds)/gpu_count+total_collection_seconds+main_seconds+beta_seconds+threshold_seconds+audit_seconds
            for ratio,vseconds in [(minimum_ratio,validation_min_seconds),(1.,validation_max_seconds)]]
        report.update(measured_source_cases_per_second=throughput['collection']['cases_per_second'],
            measured_transitions_per_second=throughput['collection']['transitions_per_second'],
            measured_updates_per_second_with_replay=speed,measured_evaluation=throughput['evaluation'],verified_imported_cases=reused,
            maximum_update_loop_gpu_hours_from_full_batch_smoke=max_training_seconds/3600,
            modeled_core_wall_hours_range=[s/3600 for s in estimates],main_evaluation_hours_at_measured_b4_speed=main_seconds/3600,
            unmeasured_allowance_multiplier=[1.3,2.0],planning_wall_hours_with_explicit_allowance=[estimates[0]*1.3/3600,estimates[1]*2./3600],
            excluded_from_direct_measurement=['full-corpus cache cold misses','full-corpus OOF inference','reader validation over all512 cases','independent rereading','repair evidence I/O','queue tail imbalance'],
            interpretation='Measured technical workloads only. Core range spans earliest possible validation patience and all100k caps; 1.3-2x allowance is an assumption for unmeasured stages. Small warm replay is not full-corpus training. B=4 measured evaluation costs conservatively apply to B=1/2. Formal progress supersedes this forecast. All scientific cases remain mandatory.')
    atomic_json(root/'reports/resource_forecast.json',report);atomic_json(EXPERIMENT/'reports/resource_forecast.json',report)
    return report
