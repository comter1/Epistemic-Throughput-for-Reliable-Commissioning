"""Optional, presealed live reader-feedback arm and exact-prefix stopping.

R-live supplies the source-selected reader calculation to the original frozen
selector. RG-live stops on that actual route with the already selected head.
That head was calibrated under ORIGINAL FINAL-reader q; its numerical returns
are explicitly not assumed calibrated under this changed reader.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from cgx.data.store import Store
from cgx.models.features import FEATURE_VERSION
from cgx.provenance import atomic_json, read_json, hash_json, file_sha256
from cgx.evaluation.rollout import call_deadline
from diagnostics.false_alarm_causes.interventions import dump_blob, load_blob, prediction_metrics
from .hooks import FrozenReaderHooks, HOOK_VERSION
from .reader_candidates import predict_variant_state
from .heads import load_head
from .ledger import Ledger, TimedStage
from .rollouts import Coordinator, initialize_worker, METHODS, SEEDS, WATCHDOG_SECONDS, read_cases
from .confirmation import first_execution_prefix, predicted


class PhaseLedger:
    def __init__(self, ledger):
        self.ledger = ledger

    def reserve(self, kind, count=1, detail=''):
        return self.ledger.reserve(kind, count, detail='live_' + detail)


class LiveCoordinator(Coordinator):
    def __init__(self, root, run, variant, device='cuda:1', batch_size=16):
        super().__init__(root, run, device, batch_size)
        self.variant = variant
        self.store = Store(run)
        self.hooks = FrozenReaderHooks(self.reader, PhaseLedger(self.ledger), Path(run) / 'cache/reader_states')
        self.choices = read_json(Path(run) / 'head_selection.json')['methods']
        # Source selection and fresh fixed-route replay use the CPU FP32 MLP.
        # Keep that exact deployment backend here, including the zero threshold.
        self.gates = {(m, s): load_head(run, m, s, self.choices[m]['cell'], self.choices[m]['step'], 'cpu')[0]
                      for m in METHODS for s in SEEDS}
        self.head_examples = 0

    def predict(self, states):
        for state in states:
            prepared = state['prepared']
            provenance = [f"initial:{state['case']['case_id']}"] + [t['acquisition_identity'] for t in state['trace']]
            key = hash_json([HOOK_VERSION, FEATURE_VERSION, self.reader.checkpoint_hash,
                             self.variant, prepared['history_hash'], provenance])
            path = self.run / 'cache/live_reader_predictions' / f'{key}.json'
            if key not in self.predictions:
                if path.exists():
                    self.predictions[key] = read_json(path)['prediction']
                    self.ledger.reserve('cache_hit', detail='live_reader_prediction')
                else:
                    history = self.store.load_history(prepared['history_hash'])
                    stamp = time.monotonic()
                    with call_deadline(WATCHDOG_SECONDS):
                        reader_state = self.hooks.prepare(history, provenance,
                            features=prepared['features'], identity=prepared['history_hash'])
                        prediction = predict_variant_state(self.hooks, reader_state, self.variant)
                    self.timings['reader_variant_wall_seconds'] += time.monotonic() - stamp
                    self.predictions[key] = prediction
                    atomic_json(path, dict(prediction=prediction, variant=self.variant,
                        history_hash=prepared['history_hash'], provenance=provenance,
                        checkpoint_hash=self.reader.checkpoint_hash, feature_version=FEATURE_VERSION))
            state['prediction'] = self.predictions[key]

    def score(self, states):
        super().score(states)
        for method in METHODS:
            for seed in SEEDS:
                group = [s for s in states if s['method'] == method and s['seed'] == seed]
                continuing = [s for s in group if s['actions'][s['chosen_index']]['kind'] != 'stop']
                for state in group:
                    state['selected_gate_score'] = 0.
                if not continuing:
                    continue
                inputs = torch.as_tensor(np.asarray([s['chosen_head_input_final'] for s in continuing], np.float32),
                                         device='cpu')
                self.ledger.reserve('head_forward', detail='live_selected_gate')
                with torch.inference_mode():
                    values = self.gates[method, seed](inputs).squeeze(-1).cpu().numpy()
                if not np.isfinite(values).all():
                    raise FloatingPointError('Invalid source-selected live gate score')
                self.head_examples += len(continuing)
                for state, value in zip(continuing, values):
                    state['selected_gate_score'] = float(value)

    def prefix(self, state):
        prefix = super().prefix(state)
        prefix['prediction_reader'] = self.variant
        prefix['chosen_head_input_deployed'] = prefix.pop('chosen_head_input_final')
        prefix['deployed_q_reader'] = self.variant
        prefix['selected_gate_score'] = state.get('selected_gate_score') if prefix['actionable'] else None
        prefix['selected_gate_cell'] = self.choices[state['method']]['cell']
        prefix['selected_gate_step'] = self.choices[state['method']]['step']
        prefix['gate_calibration_q_reader'] = 'original_FINAL'
        prefix['gate_head_device'] = 'cpu'
        return prefix

    def finish(self, state, reason, error=None):
        row = super().finish(state, reason, error)
        item = load_blob(state['path'])
        row.update(arm='R-live', selected_reader_variant=self.variant, live_feedback=True,
                   gate_calibration_q_reader='original_FINAL',
                   gate_head_device='cpu',
                   changed_reader_gate_calibration_claimed=False)
        item['row'] = row
        item['result_hash'] = hash_json({k: v for k, v in item.items() if k != 'result_hash'})
        dump_blob(state['path'], item)
        return row


def gate_prefix(prefixes):
    for prefix in prefixes:
        action = prefix['chosen_action']
        if not prefix['actionable'] or action is None or action['kind'] == 'stop':
            return prefix
        score = prefix['selected_gate_score']
        if not np.isfinite(score):
            raise FloatingPointError('Invalid live gate score')
        if score <= 0:
            return prefix
    return prefixes[-1]


def summarize(run, cases, variant):
    rows, prefixes = [], []
    choices = read_json(Path(run) / 'head_selection.json')['methods']
    for case in cases:
        for method in METHODS:
            for seed in SEEDS:
                path = Path(run) / 'trajectories/live/cases' / f"{hash_json([case['case_id'], method, seed])}.json.gz"
                item = load_blob(path)
                assert item['result_hash'] == hash_json({k: v for k, v in item.items() if k != 'result_hash'})
                row = item['row']
                if row['state'] != 'passed':
                    rows.extend([dict(row, arm=arm, abstained=True) for arm in ('R-live', 'RG-live')])
                    continue
                actual = item['prefixes']
                assert len(actual) == len(item['trace']) + 1
                base_path = Path(run) / 'trajectories/confirmation/cases' / f"{hash_json([case['case_id'], method, seed])}.json.gz"
                a0 = load_blob(base_path)
                fixed_fault = prediction_metrics(a0['prefixes'][-1]['prediction'])['max_fault_target']
                first = first_execution_prefix(actual)
                correctness = [predicted(p['prediction']) == row['true_target_key'] for p in actual]
                selected = {'R-live': actual[-1], 'RG-live': gate_prefix(actual)}
                for arm, prefix in selected.items():
                    q = prefix['prediction']
                    result = {k: row[k] for k in ('configuration_id', 'case_id', 'method_key', 'policy_seed',
                              'state', 'true_target_key', 'is_healthy')}
                    result.update(arm=arm, selected_reader_variant=variant, **prediction_metrics(q, row['true_target_key']),
                        fault_family=case['fault_family'], retained_prefix_ordinal=prefix['prefix_ordinal'],
                        history_hash=prefix['history_hash'], selected_gate_score=prefix['selected_gate_score'],
                        gate_calibration_q_reader='original_FINAL', changed_reader_gate_calibration_claimed=False,
                        selected_gate_cell=choices[method]['cell'] if arm == 'RG-live' else None,
                        selected_gate_step=choices[method]['step'] if arm == 'RG-live' else None,
                        gate_selected_as_deployable_improvement=choices[method]['guardrails_passed'] if arm == 'RG-live' else None,
                        physical_wall_seconds=row['end_to_end_seconds'] if arm == 'R-live' else None,
                        physical_wall_scope='shared actual R-live acquisition window' if arm == 'R-live' else
                            'exact retained R-live prefix; logical cost only; never separately timed', abstained=False)
                    lp = dict(zip(q['target_keys'], q['log_probabilities_base2']))
                    first_reached = first['executions_used'] >= 1 and first['prefix_ordinal'] <= prefix['prefix_ordinal']
                    first_correct = predicted(first['prediction']) == row['true_target_key'] if first_reached else None
                    retained_correctness = correctness[:prefix['prefix_ordinal'] + 1]
                    first_wrong = next((i for i, correct in enumerate(retained_correctness) if not correct), None)
                    result.update(max_fault_log_odds=lp[result['max_fault_target']] - lp['healthy'],
                        fixed_fault_target=fixed_fault, fixed_fault_log_odds=lp[fixed_fault] - lp['healthy'],
                        initial_correct=correctness[0], first_execution_reached=first_reached,
                        first_execution_correct=first_correct, trajectory_prefix_diagnosis_reader=variant,
                        first_correct_final_wrong=bool(first_correct and not result['correct_joint']) if first_reached else None,
                        first_wrong_final_correct=bool(not first_correct and result['correct_joint']) if first_reached else None,
                        initially_wrong_still_wrong=bool(not correctness[0] and not result['correct_joint']),
                        first_wrong_prefix=first_wrong,
                        later_recovered=bool(first_wrong is not None and any(retained_correctness[first_wrong + 1:])),
                        watchdog_failed=False)
                    for resource in ('executions', 'calibrations', 'decisions', 'bytes'):
                        result[resource + '_used'] = prefix['initial_budget'][resource] - prefix['remaining'][resource]
                    rows.append(result)
                for prefix in actual:
                    prefixes.append({**prefix, **{k: row[k] for k in ('case_id', 'configuration_id', 'method_key', 'policy_seed')},
                                     'true_target_key': row['true_target_key']})
    frame = pd.DataFrame(rows)
    # Pure numerical audit, with no labels involved: compare windowed CPU gate
    # evaluation with the all-prefix CPU batching used by fixed-route replay.
    numeric = []
    for method in METHODS:
        for seed in SEEDS:
            usable = [p for p in prefixes if p['method_key'] == method and p['policy_seed'] == seed
                      and p['actionable'] and p['chosen_action']['kind'] != 'stop']
            if not usable:
                continue
            head, _ = load_head(run, method, seed, choices[method]['cell'], choices[method]['step'], 'cpu')
            inputs = torch.as_tensor(np.asarray([p['chosen_head_input_deployed'] for p in usable], np.float32))
            Ledger(run).reserve('head_forward', detail='live_gate_numerical_audit')
            with torch.inference_mode():
                recomputed = head(inputs).squeeze(-1).numpy()
            stored = np.asarray([p['selected_gate_score'] for p in usable], np.float32)
            mismatch = int(np.count_nonzero((recomputed > 0) != (stored > 0)))
            np.testing.assert_allclose(recomputed, stored, rtol=2e-5, atol=2e-5)
            assert mismatch == 0, 'Live gate batching changed a zero-threshold decision'
            numeric.append(dict(method_key=method, policy_seed=seed, prefixes=len(usable),
                windowed_device='cpu', replay_batch_device='cpu', dtype='float32', sign_disagreements=mismatch,
                maximum_absolute_score_error=float(np.max(np.abs(recomputed - stored))),
                minimum_absolute_stored_score=float(np.min(np.abs(stored)))))
    atomic_json(Path(run) / 'tests/live_gate_numerics.json', dict(state='passed',
                fixed_zero_threshold=True, label_free_engineering_only=True, full_model_forwards=0, checks=numeric))
    assert len(frame) == len(cases) * 6 * 2
    for _, group in frame.groupby('arm'):
        assert len(group) == 768 and group.case_id.nunique() == 128
    frame.to_csv(Path(run) / 'live_confirmation_per_case.csv', index=False)
    pd.DataFrame(prefixes).to_parquet(Path(run) / 'live_prefixes.parquet', index=False)
    atomic_json(Path(run) / 'trajectories/live/complete.json', dict(state='passed' if frame.state.eq('passed').all() else 'incomplete',
                trajectories=768, trajectories_expected=768, trajectories_present=768,
                failed_trajectories=int(frame[frame.arm.eq('R-live')].state.ne('passed').sum()),
                outcome_rows=len(frame), arms=['R-live', 'RG-live'],
                q_input_reader=variant, gate_calibrated_under='original_FINAL',
                changed_reader_gate_calibration_claimed=False, selected_gate_threshold=0.))


def run(root, run_dir, device='cuda:1', workers=32, window=96, batch_size=16):
    run_dir = Path(run_dir)
    authorization = read_json(run_dir / 'manifests/source_variant_selection.json')
    assert authorization['state'] == 'sealed_before_fresh_outcomes'
    assert authorization['optional_live']['enabled'] is True
    assert authorization['fresh_outcomes_exist_at_seal'] is False
    assert authorization['fresh_initial_acquisitions_exist_at_seal'] is False
    for name, digest in authorization['frozen_scientific_code'].items():
        assert file_sha256(Path(__file__).with_name(name)) == digest, name
    choice = read_json(run_dir / 'manifests/source_reader_selection.json')
    variant = choice['selected_variant']
    assert variant in ('R-P', 'R-C', 'R-PC')
    assert authorization['selected_reader_variant'] == authorization['optional_live']['reader_variant'] == variant
    assert authorization['frozen_files']['manifests/source_reader_selection.json'] == file_sha256(run_dir / 'manifests/source_reader_selection.json')
    assert authorization['frozen_files']['head_selection.json'] == file_sha256(run_dir / 'head_selection.json')
    for relative, digest in authorization['all_head_checkpoints'].items():
        assert file_sha256(run_dir / relative) == digest, relative
    manifest = run_dir / 'manifests/fresh_cases.json'
    assert authorization['frozen_files']['manifests/fresh_cases.json'] == file_sha256(manifest)
    assert device == 'cuda:1' and workers <= 32
    cases = read_cases(manifest)
    assert len(cases) == 128
    completed = read_json(run_dir / 'trajectories/confirmation/complete.json')
    assert completed['trajectories_present'] == 768 and completed['state'] == 'passed'
    assert all((run_dir / 'cache/initial' / f"{case['case_id']}.json.gz").exists() for case in cases)
    jobs = []
    for case in cases:
        for method in METHODS:
            for seed in SEEDS:
                path = run_dir / 'trajectories/live/cases' / f"{hash_json([case['case_id'], method, seed])}.json.gz"
                if not path.exists():
                    jobs.append((case, method, seed, path))
    plan = dict(phase='live', variant=variant, cases=len(cases), original_trajectories=768,
                maximum_additional_physical_executions=3072, initial_observations_reused_from_A0=True,
                RG_live_uses_exact_R_live_prefix=True, checkpoint_hashes=authorization,
                source_file_sha256=file_sha256(__file__), coordinator_code_sha256=file_sha256(Path(__file__).with_name('rollouts.py')),
                device=device, workers=workers, window=window, batch_size=batch_size,
                changed_reader_gate_calibration_claimed=False)
    plan_path = run_dir / 'manifests/live_rollouts.json'
    if plan_path.exists():
        assert read_json(plan_path) == plan
    atomic_json(plan_path, plan)
    with TimedStage(run_dir, 'live_rollouts'):
        coordinator = LiveCoordinator(root, run_dir, variant, device, batch_size)
        stamp, cpu = time.monotonic(), time.process_time()
        before = Ledger(run_dir).snapshot()['counts']
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                initializer=initialize_worker, initargs=(str(root), str(run_dir))) as pool:
            for start in range(0, len(jobs), window):
                rows = coordinator.window(pool, jobs[start:start + window], 'live')
                print(dict(stage='live', completed=min(start + window, len(jobs)), total=len(jobs),
                           failures=sum(r['state'] != 'passed' for r in rows), elapsed_seconds=time.monotonic() - stamp), flush=True)
        summarize(run_dir, cases, variant)
        after = Ledger(run_dir).snapshot()['counts']
        atomic_json(run_dir / 'trajectories/live/timing.json', dict(wall_seconds=time.monotonic() - stamp,
                    coordinator_cpu_seconds=time.process_time() - cpu, model_timings=dict(coordinator.timings),
                    head_forward_examples=coordinator.head_examples,
                    shared_ledger_count_deltas={k: after[k] - before.get(k, 0) for k in after}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--window', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    run(args.root, args.run_dir, args.device, args.workers, args.window, args.batch_size)
