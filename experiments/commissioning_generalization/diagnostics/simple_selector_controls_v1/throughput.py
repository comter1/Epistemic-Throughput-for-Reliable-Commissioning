"""Measure the required source subset, without inspecting test outcomes."""
import argparse
import json
from pathlib import Path

from cgx.data.store import Store
from cgx.provenance import atomic_json, hash_json, now, read_json
from .evaluation import logical_key, planned_jobs


def measure(run_dir):
    run = Path(run_dir); protocol = read_json(run/'protocol.json')
    subset = read_json(run/'validation/pilot_cases.json')['case_ids']
    qualifier = hash_json(sorted(subset))[:12]
    marker = read_json(run/f'validation/validation_subset_{qualifier}_shard0of1_complete.json')
    if marker['state'] != 'passed' or marker['cells'] != 1056:
        raise ValueError('The entire32-case required validation subset must finish')
    jobs = planned_jobs(Store(protocol['source_root']), 'validation', case_subset=subset)
    unique = {}; rows = []
    for job in jobs:
        payload = read_json(run/'validation/cases'/(logical_key(job)+'.json'))
        expected = payload.pop('result_hash')
        if hash_json(payload) != expected:
            raise ValueError('Pilot cell content changed')
        row = payload['online']; rows.append(row)
        unique.setdefault(row['execution_source_key'], row)
    assert len(rows) == 1056
    summaries = {}
    for name in ('physical_cache_hits','imported_physical_cache_hits','extension_physical_cache_hits',
                 'new_robot_execution_attempts','new_calibration_attempts','actual_new_physical_seconds',
                 'reader_calls','reader_warmup_calls','selector_calls','selector_warmup_calls',
                 'selection_seconds','inference_seconds','completed_read_seconds'):
        values = [row.get(name) for row in unique.values()]
        summaries[name] = dict(sum=sum(x for x in values if x is not None),
                               unknown_rows=sum(x is None for x in values))
    rate = len(rows)/marker['elapsed_seconds']; scaled = rate * 280/marker['workers']
    main_lower = 64000/scaled
    result = dict(state='measured_before_new_test', created_at=now(),
        source_physical_cases=32, source_logical_rows=len(rows), canonical_rollouts=len(unique),
        measured_wall_seconds=marker['elapsed_seconds'], measured_cpu_workers=marker['workers'],
        logical_rows_per_second=rate, per_worker_threads=1, measurement_port=2379,
        planned_remote_cpu_workers={2379:40, 2982:80, 2404:160}, planned_total_cpu_workers=280,
        expected_validation_rows=16896, expected_primary_rows=32000, expected_transfer_rows=32000,
        expected_new_test_rows=64000, imported_rows=147200, maximum_initial_case_executions=3712,
        maximum_logical_additional_validation_executions=33280,
        maximum_logical_additional_test_executions=140800, fresh_attempt_cap=190000,
        observed_canonical_work=summaries, durable_all_attempts=read_json(run/'resources/fresh_attempt_budget.json'),
        main_evaluation_ideal_linear_seconds=main_lower,
        main_evaluation_planning_range_seconds=[2*main_lower, 6*main_lower],
        estimate_basis='Observed source logical rows/s scaled24 to280 workers; planningrange2-6xideal accounts for largergraphs, contention, cachemix and shardimbalance. Reporting/archiving are additional.',
        resource_warning='Cache-reused logical rows are not independent timing trials; this is operational scheduling, not an algorithm speedup claim.',
        no_test_outcomes_inspected=True)
    path = run/'resources/source_throughput.json'
    if path.exists():
        raise RuntimeError('Existing throughput measurement is preserved')
    atomic_json(path,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',required=True)
    print(json.dumps(measure(p.parse_args().run_dir)),flush=True)
