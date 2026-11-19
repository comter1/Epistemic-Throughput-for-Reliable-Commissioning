"""Executed provenance checks; no simulator truth is returned to a method."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .storage import PhysicsLedger, file_hash, write_json


def protected_snapshot(run_root, repo_root=None, data_root=None):
    root = Path(run_root)
    repo = Path(repo_root or Path(__file__).resolve().parents[3])
    data = Path(data_root or root.parents[1])
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=repo).decode().split('\0')
    paths = {repo / p for p in tracked if p and '/payload_change_v1/' not in p}
    asset = data / 'assets/mujoco_menagerie/franka_emika_panda'
    paths.update(p for p in asset.rglob('*') if p.is_file())
    prior_roots = [
        data / 'structured_evidence_v1/20261115T224529Z',
        data / 'simple_selector_controls_v1/20261116T125135Z',
    ]
    for previous in prior_roots:
        for name in ('protocol.json', 'base_config.json', 'FINDINGS_CN.md', 'README.md',
                     'provenance/protected_before.json', 'provenance/final_audit.json'):
            if (previous / name).is_file():
                paths.add(previous / name)
        for directory in ('reports', 'results', 'models'):
            parent = previous / directory
            if parent.exists():
                paths.update(p for p in parent.rglob('*') if p.is_file()
                             and p.suffix in ('.csv', '.json', '.pt', '.pth', '.npz', '.safetensors'))
    records = [{'path': str(p.resolve()), 'size_bytes': p.stat().st_size, 'sha256': file_hash(p)}
               for p in sorted(paths) if p.is_file()]
    value = {'coverage': 'All pre-existing Git-tracked files and pinned Panda assets; prior '
             'SED/simple-selector protocol, model, JSON/CSV results and preservation inventories. '
             'Historical raw stores are inventory references, not freshly rehashed in this check.',
             'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo).decode().strip(),
             'files': records, 'count': len(records)}
    write_json(root / 'checks/protected_before.json', value)
    return value


def check_protected(run_root):
    root = Path(run_root)
    before = json.loads((root / 'checks/protected_before.json').read_text())
    failures = []
    for row in before['files']:
        p = Path(row['path'])
        if not p.is_file():
            failures.append({'path': str(p), 'reason': 'missing'})
        elif file_hash(p) != row['sha256']:
            failures.append({'path': str(p), 'reason': 'hash_changed'})
    result = {'passed': not failures, 'checked_files': len(before['files']),
              'coverage': before['coverage'], 'failures': failures}
    write_json(root / 'checks/protected_after.json', result)
    return result


def legacy_regression(run_root):
    """Seven real old-controller executions, charged before every execution."""
    import numpy as np
    from cgx.env import System, generate_task
    from cgx.graphs import generate_graph
    from cgx.observation.actions import trace_ports

    root = Path(run_root)
    ledger = PhysicsLedger(root)
    graph = generate_graph(5, 12, 'chain', 2)
    task = generate_task(123)
    types = {'cache_stale': 'cache', 'queue_hold': 'queue', 'adapter_scale': 'adapter'}

    def case(family, severity=None):
        return {'fault_family': family,
                'fault_node': next((n['id'] for n in graph['nodes']
                                    if n['type'] == types.get(family)), None),
                'severity': severity,
                'observer_offsets_ms': {d: 3.1 + i for i, d in enumerate(graph['clock_domains'])},
                'streams': {'noise': 1234, 'task': 123}}

    def execute(system, name):
        token = ledger.reserve(10., 'legacy_regression:' + name, 'legacy_regression')
        try:
            result = system.run(trace_ports(graph), 100, debug=True)
        except BaseException:
            # Unknown progress remains conservatively charged at the reserved cap.
            ledger.complete(token, 10., 'failed_progress_unknown')
            raise
        ledger.complete(token, result.metrics['simulated_seconds'])
        return result

    baseline = execute(System(graph, case('healthy'), task), 'healthy')
    checks = [{'name': 'healthy_old_controller', 'passed': bool(baseline.metrics['task_pass']),
               'metrics': baseline.metrics, 'record_count': len(baseline.observation['records'])}]
    base_q = np.asarray(baseline.debug['physical_q'])
    for family, severity in [('cache_stale', 10), ('queue_hold', 10), ('adapter_scale', .15)]:
        private = case(family, severity)
        system = System(graph, private, task)
        faulty = execute(system, family)
        delta = float(np.max(np.abs(np.asarray(faulty.debug['physical_q']) - base_q)))
        system.repair(private['fault_node'])
        repaired = execute(system, family + ':repaired')
        equal = np.array_equal(np.asarray(repaired.debug['physical_q']), base_q)
        checks.append({'name': family + '_actual_effect_and_repair',
                       'passed': bool(delta > 1e-8 and equal and repaired.metrics['task_pass']),
                       'max_physical_joint_change_rad': delta,
                       'repaired_trajectory_exactly_matches_healthy': bool(equal),
                       'faulty_metrics': faulty.metrics, 'repaired_metrics': repaired.metrics})
    result = {'passed': all(x['passed'] for x in checks), 'checks': checks,
              'executed_old_controller_runs': 7, 'simulated_seconds': 70.,
              'not_new_payload_scientific_results': True}
    write_json(root / 'checks/legacy_regression.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--mode', choices=['snapshot', 'check', 'legacy'], required=True)
    args = parser.parse_args()
    operation = {'snapshot': protected_snapshot, 'check': check_protected, 'legacy': legacy_regression}[args.mode]
    result = operation(args.run_root)
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'checks')}, indent=2))
    if result.get('passed') is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
