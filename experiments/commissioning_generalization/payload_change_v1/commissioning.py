"""Evaluator harness for shared evidence, one update, and actual motion acceptance.

This module owns private cases. Estimation and selection receive only the public
configuration, acquired arrays and frozen public design bank, never this harness.
"""
from __future__ import annotations
import copy
from functools import lru_cache
from pathlib import Path
import time
import numpy as np
from .cases import MAIN_METHODS
from .storage import digest, write_json, PhysicsLedger
from .worker import read_json, context, noise_stream


@lru_cache(maxsize=2)
def frozen_inputs(root):
    root = Path(root)
    return tuple(read_json(root / 'configs' / name) for name in
                 ('estimator_parameters.json', 'design_bank.json', 'fixed_schedules.json', 'workload_spec.json'))


def result_path(root, split, method, cid, seed, threshold):
    base = Path(root) / 'runs' / split
    if split == 'validation':
        base /= f'threshold_{threshold:.2f}'
    return base / method / f'{cid}__{seed}.json'


def save_block(root, job, label, block):
    path = Path(root) / 'raw' / job['split'] / job['method'] / (
        f'{job["case_id"]}__{job["noise_seed"]}') / label
    return block.save(path)


def public_log(block):
    return {**block.public, 'execution_id': block.metadata['execution_id']}


def finish_candidate(job, case, plant, estimate, assessment, history, threshold, stop_reason,
                     selection_cpu_seconds, input_data, *, commissioning_aborted=False):
    from .physics import PayloadPlant
    from .trajectories import workload
    root = Path(job['run_root']); split = job['split']; cid = case['case_id']; method = job['method']
    parameters, bank, schedules, work_spec = input_data
    _, graph, controller = context(str(root), split)
    reference = method not in MAIN_METHODS
    actual_config = copy.deepcopy(assessment['candidate_payload_config'])
    speed = float(assessment['final_speed'])
    update = bool(assessment.get('update_config', False))
    setting_start = time.perf_counter()
    if update:
        plant.configure(actual_config)
    setting_seconds = time.perf_counter() - setting_start
    candidate_label = f'candidate_{threshold:.2f}'
    verification_seconds = 0.; verification_paths = None
    prediction = None
    if commissioning_aborted:
        verification_public = {'pass': False, 'hard_abort': True, 'not_executed': True,
                               'reason': 'commissioning_hard_abort_ends_online_motion'}
        verification_private = copy.deepcopy(verification_public)
    else:
        verification = plant.execute(workload(case['public_verification_seed'], speed),
            f'{split}/{cid}/{job["noise_seed"]}/{method}/{candidate_label}/verify', kind=split + '_verification')
        verification_paths = save_block(root, job, candidate_label + '/verification', verification)
        verification_public = verification.private['public_metrics']
        verification_private = verification.private['metrics']
        verification_seconds = verification.physical_seconds
        if estimate is not None:
            # Offline-only validation of the submitted estimate. Never ingested or used
            # to change settings, speed, stopping or the release decision.
            from .estimator import measurement_features
            features = measurement_features(public_log(verification), parameters)
            if len(features['y']):
                residual = (features['y'] - np.einsum('njp,p->nj', features['Y'], estimate['parameter_mean10'])
                            - np.asarray(estimate['persistent_bias_estimate']))
                prediction = {'rmse_Nm': float(np.sqrt(np.mean(residual ** 2))),
                              'per_joint_rmse_Nm': np.sqrt(np.mean(residual ** 2, axis=0)).tolist(),
                              'vectors': len(residual), 'held_out_from_parameter_fit': True,
                              'feedback_to_method': False}
    sufficient = bool(assessment['evidence_sufficient'])
    released = (sufficient and bool(verification_public['pass']) and not commissioning_aborted)
    status = ('pass_at_full_speed' if speed == 1. else 'pass_at_half_speed') if released else 'hold_unresolved'
    # Offline audit starts are independent declared HOME episodes, not free online resets.
    # A fixed case/seed initialization keeps persistent sensor bias identical across audit paths.
    # Path RNGs are independent sealed tasks. Private results never return to the method.
    audit_count = 4 if split == 'validation' else 8
    audits = []
    ledger = PhysicsLedger(root)
    for index, task_seed in enumerate(case['private_audit_seeds'][:audit_count]):
        audit_plant = PayloadPlant(case['true_payload'], actual_config,
            noise_stream(cid, job['noise_seed']), graph=graph, ledger=ledger, controller=controller)
        audit = audit_plant.execute(workload(task_seed, speed),
            f'{split}/{cid}/{job["noise_seed"]}/{method}/{candidate_label}/audit_{index}', kind=split + '_private_audit')
        paths = save_block(root, job, candidate_label + f'/audit_{index:02}', audit)
        audits.append({'task_index': index, 'task_seed': task_seed,
                       'physical_seconds': audit.physical_seconds, 'metrics': audit.private['metrics'],
                       'paths': paths, 'start_state': audit.metadata['start_state'],
                       'end_state': audit.metadata['end_state'], 'private_only': True})
    audit_passes = sum(bool(a['metrics']['pass']) for a in audits)
    test_seconds = sum(row['physical_seconds'] for row in history)
    fitting_seconds = float(estimate.get('fitting_cpu_seconds', 0)) if estimate else 0.
    raw_bytes = sum(row.get('bytes_uncompressed', 0) for row in history)
    plan = schedules.get(method, [])
    intended_seconds = 10. + sum(bank['actions'][a]['duration'] for a in plan) if method in MAIN_METHODS else 0.
    if method == 'adaptive_task_checks':
        intended_seconds = None  # No fixed full plan; executed seconds and the 60s cap are separate.
    return {
        'case_id': cid, 'family_id': case['family_id'], 'group': case['group'], 'split': split,
        'noise_seed': job['noise_seed'], 'method': method, 'input_signature': job['input_signature'],
        'threshold': float(threshold), 'reference_condition': reference,
        'privileged_true_parameters': method == 'true_payload_settings_reference',
        'initial_payload_config': case['configured_payload'], 'configured_payload': case['configured_payload'],
        'true_payload': case['true_payload'], 'old_payload': case['old_payload'],
        'public_config_hash': digest(case['configured_payload']), 'controller_hash': digest(controller),
        'estimator_hash': digest(parameters), 'workload_hash': digest(work_spec),
        'selected_test_ids': [row['test_id'] for row in history], 'history': copy.deepcopy(history),
        'final_estimate': estimate, 'assessment': assessment, 'stop_reason': stop_reason,
        'final_payload_config': actual_config, 'final_config': actual_config, 'final_speed': speed,
        'setting_updated': update, 'payload_update_count': int(update),
        'public_verification_result': verification_public, 'verification_private_metrics': verification_private,
        'verification_paths': verification_paths, 'released_status': status,
        'held_out_torque_prediction': prediction,
        'physical_test_seconds': test_seconds, 'verification_seconds': verification_seconds,
        'fit_cpu_seconds': fitting_seconds, 'selection_cpu_seconds': float(selection_cpu_seconds),
        'virtual_model_calculations': int(estimate.get('virtual_inverse_dynamics_calls', 0)) if estimate else 0,
        'virtual_model_calculations_note': 'Estimator public inverse-dynamics calls; zero hypothetical integrated rollouts; cached design-bank construction recorded separately.',
        'setting_operation_wall_seconds': setting_seconds,
        'serial_deployment_seconds': test_seconds + verification_seconds + fitting_seconds + selection_cpu_seconds + setting_seconds,
        'bytes_uncompressed_identification': raw_bytes,
        'saturation_seconds': sum(row['private_metrics'].get('saturation_seconds', 0) for row in history),
        'contact_samples': sum(row['private_metrics'].get('contact_samples', 0) for row in history),
        'joint_path_length_rad': sum(row['private_metrics'].get('joint_path_length_rad', 0) for row in history),
        'full_intended_identification_seconds': intended_seconds,
        'identification_budget_seconds': 60. if method in MAIN_METHODS else 0.,
        'private_audit_results': audits, 'audit_pass_count': audit_passes, 'audit_total': audit_count,
        'all_audits_pass': audit_passes == audit_count,
        'private_audit_physical_seconds': sum(a['physical_seconds'] for a in audits),
        'online_hard_abort': commissioning_aborted,
        'audit_start_policy': 'independent HOME episodes with common case/measurement-seed sensor initialization; disjoint private trajectory instances',
        'completed': True,
    }


def commissioning_job(job):
    from .physics import PayloadPlant
    from .trajectories import original_task, action_library
    from .estimator import PayloadEvidenceEstimator
    from .design import assess, choose_action
    root = Path(job['run_root']); split = job['split']; cid = job['case_id']; method = job['method']
    thresholds = list(job['thresholds']); seed = job['noise_seed']
    paths = {threshold: result_path(root, split, method, cid, seed, threshold) for threshold in thresholds}
    if all(path.exists() and read_json(path).get('input_signature') == job['input_signature'] for path in paths.values()):
        return {'status': 'cached', 'case_id': cid, 'method': method, 'seed': seed}
    thresholds = [threshold for threshold in thresholds
                  if not (paths[threshold].exists() and read_json(paths[threshold]).get('input_signature') == job['input_signature'])]
    existing_raw = root / 'raw' / split / method / f'{cid}__{seed}'
    if existing_raw.exists():
        # Retain interrupted/obsolete attempts. Existing completed threshold rows keep
        # their raw references; relocate paths in those rows before the retry.
        archive = root / 'failed_attempts' / split / method / f'{cid}__{seed}__{time.time_ns()}'
        archive.parent.mkdir(parents=True, exist_ok=True)
        existing_raw.rename(archive)
        for path in paths.values():
            if path.exists():
                text = path.read_text().replace(str(existing_raw), str(archive))
                path.write_text(text)
        write_json(archive / 'relocation.json', {'old_prefix': str(existing_raw), 'new_prefix': str(archive),
                   'reason': 'retained prior attempt before resuming unfinished unit',
                   'input_signature_for_new_attempt': job['input_signature']})
    started = time.perf_counter()
    cases, graph, controller = context(str(root), split); case = cases[cid]
    input_data = frozen_inputs(str(root)); parameters, bank, schedules, work_spec = input_data
    ledger = PhysicsLedger(root)
    plant = PayloadPlant(case['true_payload'], case['configured_payload'], noise_stream(cid, seed),
                         graph=graph, ledger=ledger, controller=controller)
    if method not in MAIN_METHODS:
        privileged = method == 'true_payload_settings_reference'
        config = case['true_payload'] if privileged else case['configured_payload']
        # Reference preserves full speed: no extra fitted predictor or outcome-selected slowdown.
        assessment = {'evidence_sufficient': True, 'final_speed': 1.,
                      'candidate_payload_config': config,
                      'update_config': privileged and digest(config) != digest(case['configured_payload']),
                      'reference': True, 'common_threshold': thresholds[0]}
        result = finish_candidate(job, case, plant, None, assessment, [], thresholds[0],
                                  'privileged_true_settings' if privileged else 'keep_existing_settings', 0., input_data)
        result['wall_seconds'] = time.perf_counter() - started
        write_json(paths[thresholds[0]], result)
        return {'status': 'completed', 'case_id': cid, 'method': method, 'seed': seed}
    estimator = PayloadEvidenceEstimator(parameters, geometry=case['configured_payload'])
    actions = {a['id']: a for a in action_library()}
    pending = set(thresholds); history = []; used = set(); selection_seconds = 0.
    next_action = original_task(case['task_seed'])
    reason = {'reason': 'shared_exact_original_task'}
    while pending:
        block = plant.execute(next_action, f'{split}/{cid}/{seed}/{method}/identification/{len(history)}',
                              kind=split + '_identification')
        raw = save_block(root, job, f'identification_{len(history):02}_{next_action["id"]}', block)
        estimate = estimator.ingest(public_log(block))
        # ingest's return may be a count/status; estimate() is the sole documented payload view.
        estimate = estimator.estimate()
        used.add(next_action['id'])
        history.append({'test_id': next_action['id'], 'physical_seconds': block.physical_seconds,
                        'estimate': estimate, 'selection_reason': reason,
                        'public_metrics': block.private['public_metrics'], 'private_metrics': block.private['metrics'],
                        'start_state': block.metadata['start_state'], 'end_state': block.metadata['end_state'],
                        'paths': raw, 'bytes_uncompressed': block.metadata['bytes_uncompressed'],
                        'wall_seconds': block.metadata['wall_seconds']})
        elapsed = sum(h['physical_seconds'] for h in history)
        aborted = bool(block.private['metrics']['hard_abort'])
        available = [a for a in actions if a not in used and elapsed + float(actions[a]['duration']) <= 60 + 1e-8]
        fixed_remaining = [a for a in schedules.get(method, []) if a not in used]
        exhausted = method == 'original_task_once' or (method != 'adaptive_task_checks' and not fixed_remaining) or not available
        outcomes_for_identical_candidates = {}
        for threshold in sorted(pending):
            assessment = assess(estimator, bank, case['configured_payload'], threshold)
            selection_seconds += assessment['assessment_cpu_seconds'] / len(thresholds)
            history[-1].setdefault('assessments_by_threshold', {})[str(threshold)] = assessment
            if not (assessment['stop'] or exhausted or aborted):
                continue
            stop_reason = ('hard_abort' if aborted else 'evidence_sufficient' if assessment['stop']
                           else 'original_task_only' if method == 'original_task_once'
                           else 'schedule_exhausted' if fixed_remaining == [] else 'identification_budget_exhausted')
            # Same prefix, candidate, speed and full cloned state: candidate outcomes may be
            # reused exactly across threshold labels. This never exposes a later prefix.
            prefix_identity = plant.cache_identity()
            cache_key = digest([prefix_identity, assessment['candidate_payload_config'], assessment['final_speed'], aborted])
            if cache_key not in outcomes_for_identical_candidates:
                result = finish_candidate(job, case, plant.clone(), estimate, assessment, history,
                                          threshold, stop_reason, selection_seconds, input_data,
                                          commissioning_aborted=aborted)
                result['candidate_initial_full_state_sha256'] = prefix_identity
                outcomes_for_identical_candidates[cache_key] = copy.deepcopy(result)
            else:
                result = copy.deepcopy(outcomes_for_identical_candidates[cache_key])
                result.update(threshold=threshold, assessment=assessment, stop_reason=stop_reason)
                result['serial_deployment_seconds'] += selection_seconds - result['selection_cpu_seconds']
                result['selection_cpu_seconds'] = selection_seconds
                sufficient = assessment['evidence_sufficient'] and result['public_verification_result']['pass'] and not aborted
                result['released_status'] = ('pass_at_full_speed' if result['final_speed'] == 1. else 'pass_at_half_speed') if sufficient else 'hold_unresolved'
                result['exact_candidate_outcome_reuse'] = {
                    'same_full_cloned_state': True, 'same_payload_config_and_speed': True,
                    'same_private_task_and_sensor_streams': True, 'same_input_code_signature': job['input_signature'],
                    'logical_cost_unchanged': True}
            result['wall_seconds'] = time.perf_counter() - started
            write_json(paths[threshold], result)
            pending.remove(threshold)
        if not pending:
            break
        if method == 'adaptive_task_checks':
            selected, reason = choose_action(estimator, available, bank, case['configured_payload'])
            selection_seconds += reason['selection_cpu_seconds']
        else:
            selected = fixed_remaining[0]
            reason = {'reason': 'frozen_source_designed_schedule', 'full_schedule': schedules[method]}
        next_action = actions[selected]
    return {'status': 'completed', 'case_id': cid, 'method': method, 'seed': seed,
            'wall_seconds': time.perf_counter() - started}
