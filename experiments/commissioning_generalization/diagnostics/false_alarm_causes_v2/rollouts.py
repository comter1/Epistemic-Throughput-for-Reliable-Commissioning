"""V2 frozen SOURCE/A0 rollouts: one GPU coordinator and CPU acquisition workers.

The evaluator owns cases. GPU calls receive only public numerical features. A
physical realization is cached only after the original policy chooses its action.
All acquired prefixes, including the terminal prefix, remain reconstructible.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import copy, deepcopy
from dataclasses import fields
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import socket
import time
import traceback

import numpy as np
import pandas as pd
import torch

from cgx.config import load_config
from cgx.data.store import Store
from cgx.env.system import System, generate_task
from cgx.evaluation.rollout import call_deadline, target_key
from cgx.models.features import FEATURE_VERSION, tensorize_history, collate_features, collate_actions
from cgx.observation.actions import action_catalog, initial_budget, feasible_actions, charge
from cgx.provenance import atomic_json, read_json, hash_json, file_sha256
from cgx.seeds import seed_for
from diagnostics.false_alarm_causes.core import inference_settings, load_reader, load_policy
from diagnostics.false_alarm_causes.interventions import dump_blob, load_blob, prediction_metrics
from .ledger import Ledger, TimedStage

METHODS = ('gnn_depth2', 'gnn_adaptive')
SEEDS = (11, 23, 37)
WATCHDOG_SECONDS = 240.0
_WORKER = None


def feature_code_hash():
    import cgx.models.features as module
    return file_sha256(Path(module.__file__))


def selector_features(base, prediction, remaining, initial, method):
    """Exactly the original tensorization's q/resource changes; no raw reread."""
    out = copy(base)
    out.nodes = base.nodes.copy()
    out.resources = base.resources.copy()
    probabilities = dict(zip(prediction['target_keys'], np.exp2(prediction['log_probabilities_base2'])))
    out.nodes[:, 27:29] = 0
    for key, owner in zip(out.targets[1:], out.target_nodes[1:]):
        out.nodes[owner, 27] = probabilities.get(key, 0.)
    out.nodes[:, 28] = probabilities.get('healthy', 0.)
    if method == 'gnn_depth2':
        decisions = min(2, remaining['decisions'])
        out.resources[3] = decisions / max(1, initial['decisions'])
        out.resources[6] = decisions / 10
    return out


def verify_feature_derivation(history, prediction, method, ledger):
    """Engineering check against the unchanged raw tensorizer, field by field."""
    ledger.reserve('feature_build', 2, detail='rollout_engineering')
    base = tensorize_history(history)
    state = dict(history)
    if method == 'gnn_depth2':
        state['remaining'] = {**history['remaining'], 'decisions': min(2, history['remaining']['decisions'])}
    direct = tensorize_history(state, prediction=prediction)
    derived = selector_features(base, prediction, history['remaining'], history['initial_budget'], method)
    for field in fields(base):
        a, b = getattr(direct, field.name), getattr(derived, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=field.name)
        elif field.name == 'sequences':
            assert len(a) == len(b)
            for aa, bb in zip(a, b):
                np.testing.assert_array_equal(aa, bb)
        else:
            assert a == b, field.name
    return {'method': method, 'fields': len(fields(base)), 'exact_array_equivalence': True}


class CPUWorker:
    def __init__(self, root, run):
        self.root, self.run = Path(root), Path(run)
        self.source, self.output = Store(root), Store(run)
        self.config, self.ledger = load_config(), Ledger(run)
        self.assets = self.root / 'assets/mujoco_menagerie/franka_emika_panda'
        self.code_hash = feature_code_hash()

    def prepare(self, history):
        digest = self.output.put_history(history)
        path = self.run / 'cache/features' / f'{digest}.pkl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                with path.open('rb') as handle:
                    payload = pickle.load(handle)
                assert payload['history_hash'] == digest
                assert payload['feature_version'] == FEATURE_VERSION
                assert payload['scientific_code_hash'] == self.code_hash
                self.ledger.reserve('cache_hit', detail='rollout_features')
            else:
                self.ledger.reserve('feature_build', detail='rollout_public_history')
                with call_deadline(WATCHDOG_SECONDS):
                    features = tensorize_history(history)
                payload = dict(features=features, history_hash=digest,
                               feature_version=FEATURE_VERSION, scientific_code_hash=self.code_hash)
                temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
                with temporary.open('wb') as handle:
                    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                temporary.replace(path)
        return dict(history_hash=digest, feature_cache_path=str(path), features=payload['features'],
                    remaining=history['remaining'], initial_budget=history['initial_budget'],
                    graph=history['graph'], observation_ref=self.output.put_observation(history['observations'][-1]),
                    observations_count=len(history['observations']),
                    record_counts_by_observation=[len(obs.get('records', ())) for obs in history['observations']],
                    total_record_count=sum(len(obs.get('records', ())) for obs in history['observations']),
                    sequence_count=len(payload['features'].sequences))

    def initial(self, case, phase):
        path = self.run / 'cache/initial' / f"{case['case_id']}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            reused = path.exists()
            if reused:
                self.ledger.reserve('cache_hit', detail=phase + '_initial')
                history = load_blob(path)
            else:
                graph = deepcopy(self.source.graph(case['configuration_id']))
                task = generate_task(case['streams']['task'])
                system = System(graph, case, task, self.config, self.assets)
                self.ledger.reserve('robot_execution', detail=phase + '_initial')
                with call_deadline(WATCHDOG_SECONDS):
                    result = system.run(ports=(), sample_hz=10, execution_index=0)
                self.ledger.success('robot_execution', detail=phase + '_initial')
                rng = np.random.default_rng(seed_for(self.config['seeds']['manifest'], 'evaluation_node_order', case['case_id']))
                rng.shuffle(graph['nodes'])
                budget = initial_budget(graph, 1.0, 4)
                history = dict(graph=graph, observations=[result.observation], remaining=dict(budget),
                               initial_budget=budget, action_keys=[], nominal_task=task,
                               requirements={k: self.config['system'][k] for k in
                                             ('position_tolerance_m', 'hold_seconds', 'joint_limit_tolerance_rad')})
                dump_blob(path, history)
        return {**self.prepare(history), 'initial_reused': reused}

    def acquire(self, case, history_hash, action, execution_index, burst_index, phase):
        history = self.output.load_history(history_hash)
        # An independent System.run resets plant/modules and keys all physical
        # noise by case streams plus ordinal, exactly like the original runner.
        system = System(self.source.graph(case['configuration_id']), case, history['nominal_task'], self.config, self.assets)
        key = hash_json([system.case, system.task, action, execution_index, burst_index, system.implementation])
        path = self.run / 'cache/acquired' / f'{key}.json.gz'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            reused = path.exists()
            if reused:
                observation = load_blob(path)
                self.ledger.reserve('cache_hit', detail=phase + '_acquired')
            else:
                if action['ports']:
                    self.ledger.reserve('robot_execution', detail=phase + '_acquired')
                if action['calibration_pair']:
                    self.ledger.reserve('calibration', detail=phase + '_acquired')
                with call_deadline(WATCHDOG_SECONDS):
                    result = system.acquire(action, execution_index, burst_index)
                if action['ports']:
                    self.ledger.success('robot_execution', detail=phase + '_acquired')
                if action['calibration_pair']:
                    self.ledger.success('calibration', detail=phase + '_acquired')
                observation = result.observation
                dump_blob(path, observation)
        history['remaining'] = charge(history['remaining'], observation['actual_cost'])
        history['observations'].append(observation)
        history['action_keys'].append(action['action_key'])
        return {**self.prepare(history), 'acquisition_reused': reused,
                'acquisition_identity': key, 'actual_cost': observation['actual_cost']}


def initialize_worker(root, run):
    global _WORKER
    torch.set_num_threads(1)
    _WORKER = CPUWorker(root, run)


def cpu_job(kind, args):
    wall, cpu = time.monotonic(), time.process_time()
    try:
        result = getattr(_WORKER, kind)(*args)
        result['state'] = 'passed'
    except Exception as exc:
        result = dict(state='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    result.update(worker_wall_seconds=time.monotonic() - wall, worker_cpu_seconds=time.process_time() - cpu)
    return result


class Coordinator:
    def __init__(self, root, run, device='cuda:1', batch_size=16):
        inference_settings()
        self.root, self.run, self.device = Path(root), Path(run), device
        self.ledger, self.config = Ledger(run), load_config()
        self.reader = load_reader(root, 'final', device)
        self.policies = {(m, s): load_policy(root, m, s, device) for m in METHODS for s in SEEDS}
        self.batch_size, self.predictions = batch_size, {}
        self.timings = defaultdict(float)

    def predict(self, states):
        unique = {}
        for state in states:
            digest = state['prepared']['history_hash']
            if digest not in self.predictions:
                path = self.run / 'cache/rollout_predictions' / f'{self.reader.checkpoint_hash}_{digest}.json'
                if path.exists():
                    self.predictions[digest] = read_json(path)['prediction']
                    self.ledger.reserve('cache_hit', detail='rollout_reader')
                else:
                    unique[digest] = state['prepared']['features']
        items = list(unique.items())
        for offset in range(0, len(items), self.batch_size):
            group = items[offset:offset + self.batch_size]
            stamp = time.monotonic()
            batch = collate_features([item[1] for item in group], self.device)
            self.ledger.reserve('model_forward', len(group), detail='rollout_final_reader')
            with call_deadline(WATCHDOG_SECONDS), torch.inference_mode():
                values = self.reader(batch).cpu().numpy()
            self.timings['reader_wall_seconds'] += time.monotonic() - stamp
            for (digest, features), value in zip(group, values):
                logp = value[:len(features.targets)]
                if not np.isfinite(logp).all() or abs(np.exp2(logp).sum() - 1) > 1e-5:
                    raise FloatingPointError('Invalid FINAL reader probabilities')
                prediction = dict(target_keys=list(features.targets), log_probabilities_base2=logp.tolist())
                self.predictions[digest] = prediction
                atomic_json(self.run / 'cache/rollout_predictions' / f'{self.reader.checkpoint_hash}_{digest}.json',
                            dict(history_hash=digest, reader_checkpoint_hash=self.reader.checkpoint_hash, prediction=prediction))
        for state in states:
            state['prediction'] = self.predictions[state['prepared']['history_hash']]

    def score(self, states):
        groups = defaultdict(list)
        for state in states:
            groups[(state['method'], state['seed'])].append(state)
        for identity, all_states in groups.items():
            model = self.policies[identity]
            for offset in range(0, len(all_states), self.batch_size):
                group = all_states[offset:offset + self.batch_size]
                features = [selector_features(s['prepared']['features'], s['prediction'], s['prepared']['remaining'],
                                              s['prepared']['initial_budget'], s['method']) for s in group]
                batch = collate_features(features, self.device)
                actions = collate_actions(batch, [s['actions'] for s in group], self.device)
                captured = {}
                before = model.action_head.register_forward_pre_hook(lambda _m, args: captured.update(inputs=args[0].detach()))
                after = model.action_head.register_forward_hook(lambda _m, _args, result: captured.update(raw=result.detach()))
                self.ledger.reserve('model_forward', len(group), detail='rollout_original_selector')
                stamp = time.monotonic()
                try:
                    with call_deadline(WATCHDOG_SECONDS), torch.inference_mode():
                        scores = model(batch, actions).cpu().numpy()
                    inputs = captured['inputs'].cpu().numpy()
                    raw = captured['raw'].squeeze(-1).cpu().numpy()
                finally:
                    before.remove()
                    after.remove()
                self.timings['selector_wall_seconds'] += time.monotonic() - stamp
                if not np.isfinite(scores).all() or not np.isfinite(inputs).all():
                    raise FloatingPointError('Invalid original selector output')
                cursor = 0
                for state in group:
                    n = len(state['actions'])
                    values = scores[cursor:cursor + n]
                    chosen = int(np.argmax(values))
                    action = state['actions'][chosen]
                    if action['kind'] != 'stop':
                        assert values[chosen] == raw[cursor + chosen]
                    state.update(scores=values.tolist(), chosen_index=chosen,
                                 chosen_head_input_final=inputs[cursor + chosen].astype(np.float32).tolist())
                    cursor += n

    def prefix(self, state):
        case, prepared = state['case'], state['prepared']
        policy = self.policies[state['method'], state['seed']]
        prefix = dict(prefix_ordinal=len(state['trace']), decision_index=len(state['trace']),
                      history_hash=prepared['history_hash'], feature_cache_path=prepared['feature_cache_path'],
                      prediction=state['prediction'], prediction_reader='final',
                      remaining=prepared['remaining'], initial_budget=prepared['initial_budget'],
                      executions_used=prepared['initial_budget']['executions'] - prepared['remaining']['executions'],
                      calibrations_used=prepared['initial_budget']['calibrations'] - prepared['remaining']['calibrations'],
                      observation_count=prepared['observations_count'], candidate_actions=state['actions'],
                      record_counts_by_observation=prepared['record_counts_by_observation'],
                      total_record_count=prepared['total_record_count'], sequence_count=prepared['sequence_count'],
                      source_split=case.get('source_split', case.get('split')),
                      source_fold=case.get('source_fold'),
                      oof_reader_name=f"crossfit_{case['source_fold']}" if case.get('source_fold') is not None else None,
                      policy_checkpoint_hash=policy.checkpoint_hash, reader_checkpoint_hash=self.reader.checkpoint_hash,
                      actionable=bool(state.get('scores')), terminal_reason=None)
        if state.get('scores'):
            chosen = state['chosen_index']
            prefix.update(original_scores=state['scores'], chosen_action_index=chosen,
                          chosen_action=state['actions'][chosen], chosen_action_key=state['actions'][chosen]['action_key'],
                          chosen_score=state['scores'][chosen], chosen_head_input_final=state['chosen_head_input_final'],
                          head_input_dimension=len(state['chosen_head_input_final']))
        else:
            prefix.update(original_scores=[0.] if len(state['actions']) == 1 else [],
                          chosen_action_index=None, chosen_action=None, chosen_action_key=None,
                          chosen_score=None, chosen_head_input_final=None)
        return prefix

    def finish(self, state, reason, error=None):
        case, phase = state['case'], state['phase']
        prediction = state.get('prediction')
        prepared = state.get('prepared')
        used = {k: prepared['initial_budget'][k] - prepared['remaining'][k] for k in prepared['remaining']} if prepared else {}
        if state['prefixes']:
            state['prefixes'][-1]['terminal_reason'] = reason
        row = dict(phase=phase, configuration_id=case['configuration_id'], case_id=case['case_id'],
                   method_key=state['method'], policy_seed=state['seed'], arm='SOURCE' if phase == 'source' else 'A0',
                   source_split=case.get('source_split', case.get('split')), source_fold=case.get('source_fold'),
                   oof_reader_name=f"crossfit_{case['source_fold']}" if case.get('source_fold') is not None else None,
                   true_target_key=target_key(case),
                   policy_checkpoint_hash=self.policies[state['method'], state['seed']].checkpoint_hash,
                   reader_checkpoint_hash=self.reader.checkpoint_hash, device=self.device,
                   execution_cap=4, byte_fraction=1., feature_version=FEATURE_VERSION,
                   watchdog_seconds_per_call=WATCHDOG_SECONDS, original_two_second_deadlines_disabled=True,
                   state='failed' if error else 'passed', error=error, termination_reason=reason,
                   prediction=prediction, evidence_trace_hash=prepared['history_hash'] if prepared else None,
                   executions_used=used.get('executions', 0), calibrations_used=used.get('calibrations', 0),
                   decisions_used=used.get('decisions', 0), bytes_used=used.get('bytes', 0),
                   worker_cpu_seconds=state['worker_cpu_seconds'], worker_wall_seconds=state['worker_wall_seconds'],
                   end_to_end_seconds=time.monotonic() - state['started'],
                   physical_wall_scope='shared batched execution; not an independently timed deployment')
        if prediction:
            row.update(prediction_metrics(prediction, target_key(case)))
        else:
            row.update(is_healthy=target_key(case) == 'healthy', healthy_false_alarm=None,
                       fault_miss=None, correct_joint=False, final_nll_bits=None)
        payload = dict(row=row, trace=state['trace'], prefixes=state['prefixes'],
                       initial_observation_ref=state.get('initial_observation_ref'),
                       trace_convention='trace contains acquired actions only; policy STOP remains on its unchanged prefix')
        payload['result_hash'] = hash_json(payload)
        dump_blob(state['path'], payload)
        return row

    def window(self, pool, jobs, phase):
        states = []
        futures = []
        for case, method, seed, path in jobs:
            state = dict(case=case, method=method, seed=seed, path=path, phase=phase, trace=[], prefixes=[],
                         worker_cpu_seconds=0., worker_wall_seconds=0., started=time.monotonic())
            states.append(state)
            futures.append(pool.submit(cpu_job, 'initial', (case, phase)))
        active = []
        rows = []
        for state, future in zip(states, futures):
            result = future.result()
            state['worker_cpu_seconds'] += result['worker_cpu_seconds']
            state['worker_wall_seconds'] += result['worker_wall_seconds']
            if result['state'] != 'passed':
                rows.append(self.finish(state, 'initial_failure', result['error']))
            else:
                state['prepared'] = result
                state['initial_observation_ref'] = result['observation_ref']
                state['catalog'] = action_catalog(result['graph'], self.config)
                active.append(state)
        while active:
            try:
                self.predict(active)
            except Exception as exc:
                for state in active:
                    rows.append(self.finish(state, 'reader_failure', f'{type(exc).__name__}: {exc}'))
                break
            selectable = []
            done = []
            for state in active:
                state.pop('scores', None)
                state['actions'] = feasible_actions(state['catalog'], state['prepared']['remaining'])
                if all(a['kind'] == 'stop' for a in state['actions']):
                    state['prefixes'].append(self.prefix(state))
                    rows.append(self.finish(state, 'resources_exhausted'))
                    done.append(state)
                else:
                    selectable.append(state)
            try:
                self.score(selectable)
            except Exception as exc:
                for state in selectable:
                    rows.append(self.finish(state, 'selector_failure', f'{type(exc).__name__}: {exc}'))
                break
            futures = []
            advancing = []
            for state in selectable:
                prefix = self.prefix(state)
                state['prefixes'].append(prefix)
                action = prefix['chosen_action']
                if action['kind'] == 'stop':
                    rows.append(self.finish(state, 'policy_stop'))
                    continue
                # The chosen a0 is fixed before consulting the realization cache.
                execution_index = prefix['executions_used'] + int(bool(action['ports']))
                burst_index = prefix['calibrations_used'] + int(bool(action['calibration_pair']))
                futures.append(pool.submit(cpu_job, 'acquire',
                    (state['case'], prefix['history_hash'], action, execution_index, burst_index, phase)))
                advancing.append(state)
            active = []
            for state, future in zip(advancing, futures):
                result = future.result()
                state['worker_cpu_seconds'] += result['worker_cpu_seconds']
                state['worker_wall_seconds'] += result['worker_wall_seconds']
                if result['state'] != 'passed':
                    # As in the original evaluator, failed instrument calls keep
                    # the previous q and charge their conservative public bound.
                    action = state['prefixes'][-1]['chosen_action']
                    state['prepared']['remaining'] = charge(state['prepared']['remaining'], action['cost_bound'])
                    rows.append(self.finish(state, 'acquisition_failure', result['error']))
                    continue
                prefix = state['prefixes'][-1]
                state['trace'].append(dict(decision_index=len(state['trace']) + 1,
                    action=prefix['chosen_action'], selected_score=prefix['chosen_score'],
                    history_before=prefix['history_hash'], history_after=result['history_hash'],
                    observation_ref=result['observation_ref'], acquisition_identity=result['acquisition_identity'],
                    observation_reused=result['acquisition_reused'], actual_cost=result['actual_cost'],
                    resources_before=prefix['remaining'], resources_after=result['remaining'],
                    prediction_before=state['prediction']))
                state['prepared'] = result
                active.append(state)
        return rows


def read_cases(path):
    value = read_json(path)
    if isinstance(value, list):
        return value
    return value.get('cases', value.get('selected_cases'))


def collect(run, phase, cases):
    directory = Path(run) / 'trajectories' / phase
    rows, prefixes, full_prefixes = [], [], []
    missing = []
    for case in cases:
        for method in METHODS:
            for seed in SEEDS:
                path = directory / 'cases' / f"{hash_json([case['case_id'], method, seed])}.json.gz"
                if not path.exists():
                    missing.append([case['case_id'], method, seed])
                    continue
                item = load_blob(path)
                assert item['result_hash'] == hash_json({k: v for k, v in item.items() if k != 'result_hash'})
                row = item['row']
                rows.append(row)
                for prefix in item['prefixes']:
                    identity = {k: row[k] for k in ('case_id', 'configuration_id', 'method_key', 'policy_seed',
                                                   'source_split', 'source_fold', 'oof_reader_name', 'true_target_key')}
                    full_prefixes.append({**prefix, **identity, 'trajectory_path': str(path)})
                    # Complex candidate lists live in the lossless trajectory.
                    scalar = {k: v for k, v in prefix.items() if not isinstance(v, (dict, list))}
                    prefixes.append({**scalar, **identity, 'trajectory_path': str(path)})
    directory.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_parquet(directory / 'outcomes.parquet', index=False)
    if prefixes:
        pd.DataFrame(prefixes).to_parquet(directory / 'prefixes.parquet', index=False)
        name = 'source' if phase == 'source' else 'fresh'
        pd.DataFrame(full_prefixes).to_parquet(Path(run) / f'{name}_prefixes.parquet', index=False)
    status = dict(state='passed' if not missing and all(r['state'] == 'passed' for r in rows) else 'incomplete',
                  trajectories_expected=len(cases) * 6, trajectories_present=len(rows),
                  failed_trajectories=sum(r['state'] != 'passed' for r in rows), prefixes=len(prefixes), missing=missing)
    atomic_json(directory / 'complete.json', status)
    return status


def run(root, run_dir, manifest, phase='source', device='cuda:1', workers=32, window=96,
        batch_size=16, case_limit=None):
    if workers > 32 or device != 'cuda:1':
        raise ValueError('V2 rollout allocation is at most 32 CPU workers and GPU cuda:1 only')
    run_dir = Path(run_dir)
    cases = read_cases(manifest)
    selected = cases if case_limit is None else cases[:case_limit]
    jobs = []
    for case in selected:
        for method in METHODS:
            for seed in SEEDS:
                path = run_dir / 'trajectories' / phase / 'cases' / f"{hash_json([case['case_id'], method, seed])}.json.gz"
                if not path.exists():
                    jobs.append((case, method, seed, path))
    plan = dict(phase=phase, case_manifest=str(manifest), case_manifest_sha256=file_sha256(manifest),
                expected_trajectories=len(cases) * 6, methods=METHODS, seeds=SEEDS,
                maximum_physical_executions=len(cases) * (1 + 6 * 4), maximum_calibrations=len(cases) * 6 * 2,
                device=device, cpu_workers=workers, window=window, gpu_batch_size=batch_size,
                source_file_sha256=file_sha256(__file__), scientific_feature_code_sha256=feature_code_hash(),
                original_two_second_deadlines_disabled=True, per_call_watchdog_seconds=WATCHDOG_SECONDS,
                cache_boundary='exact action realization reused only after original policy chooses a0')
    plan_path = run_dir / 'manifests' / f'{phase}_rollouts.json'
    if plan_path.exists() and read_json(plan_path) != json.loads(json.dumps(plan)):
        raise ValueError('Frozen rollout plan differs on resume')
    atomic_json(plan_path, plan)
    counts_before = Ledger(run_dir).snapshot()['counts']
    with TimedStage(run_dir, f'{phase}_rollouts'):
        coordinator = Coordinator(root, run_dir, device, batch_size)
        wall, cpu = time.monotonic(), time.process_time()
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                 initializer=initialize_worker, initargs=(str(root), str(run_dir))) as pool:
            for start in range(0, len(jobs), window):
                group = jobs[start:start + window]
                rows = coordinator.window(pool, group, phase)
                print(json.dumps(dict(phase=phase, completed_this_launch=min(start + window, len(jobs)),
                                      total_this_launch=len(jobs), failed=sum(r['state'] != 'passed' for r in rows),
                                      elapsed_seconds=time.monotonic() - wall)), flush=True)
        status = collect(run_dir, phase, cases)
        snapshot = Ledger(run_dir).snapshot()
        timing = dict(phase=phase, hostname=socket.gethostname(), device=device, workers=workers,
                      window=window, batch_size=batch_size, wall_seconds=time.monotonic() - wall,
                      coordinator_cpu_seconds=time.process_time() - cpu, model_timings=dict(coordinator.timings),
                      launched_trajectories=len(jobs), profile_subset=case_limit is not None,
                      count_deltas={k: snapshot['counts'][k] - counts_before.get(k, 0) for k in snapshot['counts']},
                      count_delta_scope='shared ledger; simultaneous independent stages may also increment counts',
                      resource_ledger=snapshot)
        atomic_json(run_dir / 'trajectories' / phase / f'timing_{time.time_ns()}.json', timing)
    print(json.dumps(status), flush=True)


def engineering(root, run_dir, device='cuda:1'):
    """Existing MAIN history only: exact feature derivation and batch validation."""
    inference_settings()
    root, run_dir = Path(root), Path(run_dir)
    ledger = Ledger(run_dir)
    artifact_path = next(iter(sorted((root / 'evaluation/main/cases').glob('*.json'))))
    artifact = read_json(artifact_path)
    history = Store(root).load_history(artifact['row']['evidence_trace_hash'])
    # Make the original terminal history actionable only for numerical engineering;
    # no physical action is chosen/acquired and no scientific rollout is written.
    history['remaining'] = dict(history['initial_budget'])
    reader = load_reader(root, 'final', device)
    with TimedStage(run_dir, 'rollout_engineering'):
        ledger.reserve('feature_build', detail='rollout_engineering')
        base = tensorize_history(history)
        single = collate_features([base], device)
        pair = collate_features([base, base], device)
        ledger.reserve('model_forward', 3, detail='rollout_engineering_reader')
        with torch.inference_mode():
            one_q, pair_q = reader(single).cpu().numpy(), reader(pair).cpu().numpy()
        np.testing.assert_allclose(pair_q[0], one_q[0], rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(pair_q[1], one_q[0], rtol=2e-5, atol=2e-5)
        q = dict(target_keys=base.targets, log_probabilities_base2=one_q[0, :len(base.targets)].tolist())
        checks = []
        actions = feasible_actions(action_catalog(history['graph']), history['remaining'])
        for method in METHODS:
            check = verify_feature_derivation(history, q, method, ledger)
            model = load_policy(root, method, 11, device)
            f = selector_features(base, q, history['remaining'], history['initial_budget'], method)
            b1, b2 = collate_features([f], device), collate_features([f, f], device)
            a1, a2 = collate_actions(b1, [actions], device), collate_actions(b2, [actions, actions], device)
            saved = {}
            pre = model.action_head.register_forward_pre_hook(lambda _m, args: saved.update(inputs=args[0].detach()))
            post = model.action_head.register_forward_hook(lambda _m, _a, output: saved.update(output=output.detach()))
            ledger.reserve('model_forward', 3, detail='rollout_engineering_selector')
            try:
                with torch.inference_mode():
                    s1 = model(b1, a1).cpu().numpy()
                    s2 = model(b2, a2).cpu().numpy()
                inputs, raw = saved['inputs'].cpu().numpy(), saved['output'].squeeze(-1).cpu().numpy()
            finally:
                pre.remove()
                post.remove()
            np.testing.assert_allclose(s2[:len(actions)], s1, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(s2[len(actions):], s1, rtol=2e-5, atol=2e-5)
            assert int(np.argmax(s1)) == int(np.argmax(s2[:len(actions)]))
            for i, action in enumerate(actions * 2):
                assert s2[i] == (0. if action['kind'] == 'stop' else raw[i])
            assert inputs.dtype == np.float32 and inputs.shape[1] == 281
            check.update(batch_matches_single=True, chosen_action_unchanged=True,
                         head_input_dimension=281, headhook_matches_original_score=True,
                         max_batch_single_error=float(np.max(np.abs(s2[:len(actions)] - s1))))
            checks.append(check)
        result = dict(state='passed', source='existing MAIN history; resources reset only for numerical test',
                      original_artifact=str(artifact_path), original_history_hash=artifact['row']['evidence_trace_hash'],
                      generated_physical_observations=0, reader_batch_single_max_error=float(np.max(np.abs(pair_q[0] - one_q[0]))),
                      checks=checks, ledger=ledger.snapshot())
        atomic_json(run_dir / 'tests/rollout_engineering.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--engineering-only', action='store_true')
    parser.add_argument('--phase', choices=('source', 'confirmation'), default='source')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--window', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--case-limit', type=int)
    args = parser.parse_args()
    if args.engineering_only:
        engineering(args.root, args.run_dir, args.device)
    else:
        if not args.manifest:
            parser.error('--manifest is required for scientific rollouts')
        run(args.root, args.run_dir, args.manifest, args.phase, args.device, args.workers, args.window,
            args.batch_size, args.case_limit)
