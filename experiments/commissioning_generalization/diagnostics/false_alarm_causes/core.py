"""Read-only access and frozen inference for CGX-FA-CAUSE-V1.

Private labels stay in result metadata. Every model receives an ordinary public
history, never a case, Store or evaluation artifact.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import fcntl
import json
import os
import pickle
import time

import numpy as np
import torch

from cgx.data.store import Store
from cgx.models.features import FEATURE_VERSION, collate_features, tensorize_history
from cgx.provenance import atomic_json, file_sha256, hash_json, read_json
from cgx.training.engine import load_reader as _load_reader, load_policy as _load_policy


def inference_settings():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_reader(root, name='final', device='cpu'):
    model = _load_reader(root, name, device)
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    model.reader_name = name
    return model


def load_policy(root, method, seed, device='cpu'):
    model = _load_policy(root, method, seed, device)
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    return model


def load_artifact(root, case_id, method, seed, execution_cap=4):
    key = hash_json([method, seed, execution_cap, 1.0, case_id])
    path = Path(root) / 'evaluation/main/cases' / (key + '.json')
    item = read_json(path)
    if item['state'] != 'passed':
        raise ValueError(f'Incomplete original result: {path}')
    if hash_json({'row': item['row'], 'trace': item['trace']}) != item['result_hash']:
        raise ValueError(f'Original row/trace hash mismatch: {path}')
    row = item['row']
    if [row['method'], row['policy_seed'], row['execution_cap'], row['byte_fraction'], row['case_id']] != [method, seed, execution_cap, 1.0, case_id]:
        raise ValueError(f'Original result identity mismatch: {path}')
    return item


def timeout_stage(row):
    errors = row.get('errors', [])
    for phase in ('acquisition', 'diagnostic', 'selection', 'fixed_plan', 'initial'):
        if any(phase in str(e) for e in errors):
            return phase
    return 'none'


def prefix_descriptors(root, item):
    """Read compact original history references; no observations are regenerated."""
    store = Store(root)
    row = item['row']
    packed = store.history_references(row['evidence_trace_hash'])
    observed = [t for t in item['trace'] if 'evidence_hash' in t]
    if len(packed['observation_refs']) != len(observed) + 1:
        raise ValueError('Observation count does not match the actual acquired trace')
    if packed['observation_refs'][1:] != [t['evidence_hash'] for t in observed]:
        raise ValueError('Observation order differs from actual acquisition order')
    errors = row.get('errors', [])
    initial_failed = any(str(e).startswith(('initial_', 'diagnostic_warmup')) for e in errors)
    # An initial diagnostic failure aborts before any acquired-action trace.
    initial_failed |= not observed and any(str(e).startswith('diagnostic_') for e in errors)
    last_valid = -1 if initial_failed else 0
    descriptors = []
    remaining = dict(packed['initial_budget'])
    for i in range(len(packed['observation_refs'])):
        action = None if i == 0 else observed[i - 1]['action']
        updated = i == 0 and last_valid == 0
        if i:
            tr = observed[i - 1]
            # These values are checked against the actual acquired cost when the
            # prefix is materialized. They contain no unacquired observation.
            remaining = dict(tr['resources_after'])
            updated = bool(tr.get('diagnosis_updated', False)) and tr.get('stop_reason') != 'timeout_or_error'
            if updated:
                last_valid = i
        prefix = dict(packed, observation_refs=packed['observation_refs'][:i + 1],
                      remaining=dict(remaining), action_keys=[t['action']['action_key'] for t in observed[:i]])
        digest = hash_json(prefix)
        used = {k: packed['initial_budget'][k] - remaining[k] for k in remaining}
        original_at_prefix = observed[i-1]['prediction'] if i else None
        original_prediction_source = 'original_action_trace' if i else 'not_recorded'
        if i == 0 and not observed:
            original_at_prefix = row['prediction']
            original_prediction_source = 'original_initial_only_return'
        elif i == 0 and (not observed[0].get('diagnosis_updated', False) or observed[0].get('stop_reason') == 'timeout_or_error'):
            original_at_prefix = observed[0]['prediction']
            original_prediction_source = 'initial_prediction_retained_in_first_action_trace'
        descriptors.append((prefix, {
            'configuration_id': row['configuration_id'], 'case_id': row['case_id'],
            'method_key': row['method'], 'policy_seed': row['policy_seed'],
            'execution_cap': row['execution_cap'], 'decision_index': i,
            'executions_used': used['executions'], 'calibrations_used': used['calibrations'],
            'bytes_used': used['bytes'], 'decisions_used': used['decisions'],
            'action_key': action['action_key'] if action else 'initial',
            'action_kind': action['kind'] if action else 'initial',
            'ports': action['ports'] if action else [],
            'clock_pair': action.get('calibration_pair') if action else None,
            'evidence_prefix_hash': digest, 'source_final_history_hash': row['evidence_trace_hash'],
            'observation_hash': packed['observation_refs'][i],
            'diagnosis_updated_in_original': updated,
            'last_valid_prediction_prefix': last_valid,
            'timeout_stage': timeout_stage(row),
            'termination_reason': ('timeout' if row['timeout'] else
                next((t.get('stop_reason') for t in reversed(item['trace']) if t.get('stop_reason')), 'resource_cap')),
            'is_terminal': i == len(packed['observation_refs']) - 1,
            'original_source': 'original_hashed_observations',
            'true_target_key': row['true_target_key'],
            'original_predicted_target_key': row['predicted_target_key'],
            'original_final_log_probabilities': row['prediction']['log_probabilities_base2'],
            'original_target_keys': row['prediction']['target_keys'],
            'original_timeout': row['timeout'], 'original_abstained': row['abstained'],
            'original_trace_prediction_source': original_prediction_source,
            'original_trace_log_probabilities': None if original_at_prefix is None else original_at_prefix['log_probabilities_base2'],
            'original_trace_predicted_target_key': None if original_at_prefix is None else
                original_at_prefix['target_keys'][int(np.argmax(original_at_prefix['log_probabilities_base2']))],
        }))
    return descriptors


def materialize_prefix(root, packed):
    store = Store(root)
    history = {
        'graph': store._get('graphs', packed['graph_ref']),
        'observations': [store.load_observation(h) for h in packed['observation_refs']],
        'nominal_task': store._get('tasks', packed['task_ref']),
        'requirements': deepcopy(packed['requirements']),
        'initial_budget': dict(packed['initial_budget']),
        'remaining': dict(packed['remaining']), 'action_keys': list(packed['action_keys']),
    }
    expected = dict(history['initial_budget'])
    for obs in history['observations'][1:]:
        expected = {k: expected[k] - obs['actual_cost'][k] for k in expected}
    if expected != history['remaining'] or min(expected.values()) < 0:
        raise ValueError('Prefix resource history differs from actual observation costs')
    return history


def reconstruct_prefixes(root, item):
    descriptors = prefix_descriptors(root, item)
    final = materialize_prefix(root, descriptors[-1][0])
    return [(dict(final, observations=final['observations'][:i + 1],
                  remaining=dict(packed['remaining']), action_keys=list(packed['action_keys'])), meta)
            for i, (packed, meta) in enumerate(descriptors)]


def prediction_fields(prediction):
    keys = prediction['target_keys']
    lp = np.asarray(prediction['log_probabilities_base2'], dtype=np.float64)
    p = np.exp2(lp)
    healthy = keys.index('healthy')
    candidates = [i for i in range(len(keys)) if i != healthy]
    fault = max(candidates, key=lambda i: p[i])
    best = int(p.argmax())
    return {
        'target_keys': list(keys), 'full_log_probabilities': lp.tolist(),
        'predicted_target_key': keys[best], 'p_healthy': float(p[healthy]),
        'max_fault_probability': float(p[fault]), 'max_fault_target': keys[fault],
        'fault_minus_healthy': float(p[fault] - p[healthy]),
        'argmax_margin': float(np.sort(p)[-1] - np.sort(p)[-2]),
    }


def _sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def predict(model, history, ledger, detail='reader', cache_dir=None):
    """Frozen FP32 inference; optional cache is exact-history/model/device keyed."""
    device = next(model.parameters()).device
    cache_path = None
    if cache_dir is not None:
        key = hash_json([history, model.checkpoint_hash, FEATURE_VERSION, str(device), 'fp32'])
        cache_path = Path(cache_dir) / (key + '.json')
        if cache_path.exists():
            saved = read_json(cache_path)
            return saved['prediction'], dict(saved['timing'], cache_hit=True)
    ledger.reserve('feature_build', detail=detail)
    stamp = time.perf_counter()
    features = tensorize_history(history)
    batch = collate_features([features], device)
    _sync(device)
    pre = time.perf_counter() - stamp
    ledger.reserve('model_forward', detail=detail)
    stamp = time.perf_counter()
    with torch.inference_mode():
        lp = model(batch)[0, :len(features.targets)].cpu().tolist()
    _sync(device)
    timing = {'elapsed_preprocess_s': pre, 'elapsed_model_s': time.perf_counter() - stamp,
              'cache_hit': False, 'device': str(device), 'precision': 'fp32'}
    result = {'target_keys': list(features.targets), 'log_probabilities_base2': lp}
    if not np.isfinite(lp).all() or abs(np.exp2(lp).sum() - 1) > 1e-5:
        raise ValueError('Reader returned invalid normalized probabilities')
    if cache_path is not None:
        atomic_json(cache_path, {'prediction': result, 'timing': timing})
    return result, timing


def code_hash():
    base = Path(__file__).resolve().parents[2]
    return hash_json({str(p.relative_to(base)): file_sha256(p)
        for name in ('models/features.py', 'models/networks.py', 'evaluation/rollout.py', 'data/store.py')
        for p in [base / 'src/cgx' / name]})


def cached_features(root, run, packed, ledger):
    """One immutable original feature build per exact packed public history."""
    digest = hash_json(packed)
    destination = Path(run) / 'cache/features' / digest[:2] / (digest + '.pkl')
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_suffix('.lock')
    with lock.open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if destination.exists():
            return destination, True
        ledger.reserve('feature_build', detail='original_prefix_feature')
        stamp = time.perf_counter()
        history = materialize_prefix(root, packed)
        read_seconds = time.perf_counter() - stamp
        stamp = time.perf_counter()
        features = tensorize_history(history)
        preprocessing_seconds = time.perf_counter() - stamp
        record = {'features': features, 'elapsed_rawread_s': read_seconds,
                  'elapsed_preprocess_s': preprocessing_seconds, 'prefix_hash': digest,
                  'feature_version': FEATURE_VERSION, 'code_hash': code_hash()}
        temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
        with temporary.open('wb') as stream:
            pickle.dump(record, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)
    return destination, False
