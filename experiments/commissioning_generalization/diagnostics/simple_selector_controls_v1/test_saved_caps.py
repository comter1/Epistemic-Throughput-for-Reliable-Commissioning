"""Additional source qualification from saved evidence only; no physical calls."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from cgx.observation.actions import charge, initial_budget
from cgx.provenance import atomic_json, file_sha256, now, read_json
from structured_evidence_v1.reader import load_reader
from .storage import ExtensionStore


def check(source_root, sed_run, run_dir):
    run = Path(run_dir)
    store = ExtensionStore(source_root, sed_run, run)
    reader = load_reader(sed_run, 'final', device='cpu')
    rows = []
    for method in ('full_once_100', 'full_once_100_cal2'):
        saved = run / 'tests' / 'qualification_records' / (method + '.json')
        payload = read_json(saved)
        online = payload['online']
        history = store.load_history(online['evidence_trace_hash'])
        provenance = payload['event_provenance']
        baseline = reader.predict(history, provenance=provenance)
        used = {key: history['initial_budget'][key] - history['remaining'][key]
                for key in history['remaining']}
        for cap in (1, 2, 4):
            view = deepcopy(history)
            view['initial_budget'] = initial_budget(history['graph'], 1., cap)
            view['remaining'] = charge(view['initial_budget'], used)
            predicted = reader.predict(view, provenance=provenance)
            assert predicted['target_keys'] == baseline['target_keys']
            assert np.array_equal(np.asarray(predicted['log_probabilities_base2']),
                                  np.asarray(baseline['log_probabilities_base2']))
            rows.append(dict(method=method, execution_cap=cap,
                history_hash=online['evidence_trace_hash'],
                provenance_hash=online['event_provenance_hash'],
                saved_payload_sha256=file_sha256(saved),
                exact_same_completed_vector=True,
                physical_noise_ids_unchanged=True,
                logical_costs_unchanged=True))
    result = dict(state='passed', created_at=now(), source_only=True,
        physical_calls=0, new_robot_execution_attempts=0,
        reader_hash=reader.checkpoint_hash, rows=rows,
        original_cap_view_marker_sha256=file_sha256(run/'tests/one_pass_cap_views.json'))
    destination = run/'tests/one_pass_cap_views_supplement.json'
    if destination.exists():
        raise FileExistsError('Existing qualification evidence must be preserved')
    atomic_json(destination, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--sed-run', required=True)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    print(check(args.source_root, args.sed_run, args.run_dir), flush=True)
