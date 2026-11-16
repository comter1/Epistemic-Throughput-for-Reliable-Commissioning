"""Preregistered conditions, provenance, and outcome-independent seals."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from .cases import MAIN_METHODS, REFERENCES, SPLITS, generate_manifests, DATA_ROOT
from .storage import digest, file_hash, write_json

PACKAGE = Path(__file__).resolve().parent
EXPERIMENT = PACKAGE.parent
REPOSITORY = EXPERIMENT.parent.parent


def code_hashes():
    return {str(path.relative_to(EXPERIMENT)): file_hash(path)
            for path in sorted(PACKAGE.glob('*.py'))}


def draft(run_root):
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / 'REQUEST.md').exists():
        (root / 'REQUEST.md').write_text((PACKAGE / 'documents/REQUEST.md').read_text())
    generate_manifests(root)
    value = {
        'experiment_id': 'CGX-PAYLOAD-COMMISSIONING-V1', 'status': 'draft_before_source_qualification',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, text=True).strip(),
        'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPOSITORY, text=True).strip(),
        'source_request_sha256': file_hash(root / 'REQUEST.md'),
        'splits': {key: {'families': count[0], 'cases_per_family': count[1], 'noise_seeds': count[2]}
                   for key, count in SPLITS.items()},
        'main_methods': MAIN_METHODS, 'references': REFERENCES,
        'units': {'physics_dt_s': .002, 'control_hz': 100, 'logging_hz': 100, 'private_audit_hz': 500},
        'budgets': {'identification_s': 60, 'public_verification_s': 20, 'online_s': 80,
                    'payload_updates': 1, 'global_simulated_seconds': 1_000_000},
        'acceptance': {'ee_rms_m': .02, 'ee_peak_m': .05, 'final_error_m': .02,
                       'final_hold_s': .5, 'max_single_joint_continuous_saturation_s': .1,
                       'max_any_joint_saturation_fraction': .01, 'unexpected_contact': 0,
                       'joint_limit_violation': 0, 'hard_abort': 0},
        'sensor_simulation_assumptions': {'position_std_rad': 1e-4, 'velocity_std_rad_s': 1e-3,
            'torque_stationary_std_Nm': .05, 'torque_ar1_rho': .6, 'persistent_bias_std_Nm': .02,
            'torque_interface': 'simulated joint execution torque qfrc_actuator plus correlated error and persistent bias',
            'not_manufacturer_specifications': True},
        'stopping': {'threshold_grid': [.02, .05, .10], 'interval_probability': .99,
                     'residual_consistency_required': True, 'validation_seed': 7,
                     'validation_audits_per_candidate': 4, 'test_audits_per_candidate': 8,
                     'validation_order': ['false_release_rate_among_released',
                                          'correct_release_coverage', 'physical_seconds'],
                     'no_release_candidate_not_preferred_to_valid_release': True},
        'design': {'samples': 128, 'prospective_outcomes_per_action': 16,
                   'configuration_torque_impact_threshold': .05, 'speeds': [1., .5],
                   'additional_static_actions': 6, 'additional_dynamic_actions': 4,
                   'nominal_block_seconds': 8, 'L0_seconds': 10,
                   'fallback': 'workload_torque_variance_reduction_per_physical_second'},
        'compute': {'ssh_ports': [2379, 2982, 2404], 'cpu_quotas': [48, 96, 192],
                    'blas_threads_per_worker': 1, 'gpu_required': False, 'mujoco_version': '4.0.1'},
        'statistics': {'unit': 'physical case, averaging three paired measurement seeds first',
                       'bootstrap': 'family then paired cases', 'replicates': 2000,
                       'families_separate_direction_count': 8,
                       'private_audit_paths_are_not_independent_objects': True},
        'simulation_only': True,
    }
    write_json(root / 'protocol_draft.yaml', value)  # JSON is a strict YAML subset.
    write_json(root / 'public_private_field_map.json', {
        'policy_inputs': ['configured_payload', 'known_arm_tool_model', 'public_mount_geometry',
                          'public_work_requirements', 'acquired_sensor_records', 'candidate_action_specs'],
        'join_only_not_features': ['case_id', 'execution_id', 'measurement_event_id'],
        'evaluator_only': ['true_payload', 'old_internal_geometry', 'true_internal_geometry',
                           'group', 'family_id', 'private_audit_seeds', 'private_500Hz_physics'],
        'source_fit_only': ['source_true_payload_labels', 'source_public_measurements'],
        'forbidden_estimator_controller_inputs': ['plant_qacc', 'plant_qfrc_bias', 'plant_mass_matrix',
                                                'private_sensor_bias_draws', 'private_sensor_noise_draws'],
        'torque_measurement': 'qfrc_actuator with declared sensor errors; command logged separately',
        'deduplication': 'measurement_event_id equality only; equal numbers with new identities retained'})
    return value


def seal(run_root):
    root = Path(run_root)
    if (root / 'protocol_locked.yaml').exists():
        verify_seal(root)
        return json.loads((root / 'protocol_locked.yaml').read_text())
    value = json.loads((root / 'protocol_draft.yaml').read_text())
    selection = json.loads((root / 'source_validation/threshold_selection.json').read_text())
    required = ['configs/healthy_graph.json', 'configs/test_library.json', 'configs/fixed_schedules.json',
                'configs/workload_spec.json', 'configs/estimator_parameters.json', 'configs/controller.json',
                'configs/design_bank.json', 'configs/public_work_grid.json',
                'source_validation/threshold_selection.json',
                'checks/split_integrity.json', 'checks/physical_checks.json', 'checks/estimator_checks.json']
    required += [f'manifests/{split}_public.json' for split in SPLITS]
    required += [f'private/{split}.json' for split in SPLITS]
    required += [str(path) for path in sorted((DATA_ROOT / 'assets/mujoco_menagerie/franka_emika_panda').rglob('*')) if path.is_file()]
    required += [str(path.relative_to(root)) for path in sorted((root / 'resources').glob('environment_*.json'))]
    value.update({'status': 'frozen_before_test', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
                  'selected_stop_threshold': selection['selected_threshold'],
                  'input_hashes': {name: file_hash(root / name) for name in required},
                  'code_hashes': code_hashes(), 'threshold_selection_sha256': digest(selection)})
    write_json(root / 'protocol_locked.yaml', value)
    return value


def verify_seal(run_root):
    root = Path(run_root)
    value = json.loads((root / 'protocol_locked.yaml').read_text())
    changed = [name for name, expected in value['input_hashes'].items()
               if file_hash(root / name) != expected]
    current = code_hashes()
    # Reporting/package may be refined without changing any physical/inference decisions.
    inference = {k: v for k, v in value['code_hashes'].items()
                 if Path(k).name not in ('reporting.py', 'packaging.py', 'integrity.py')}
    changed += [name for name, expected in inference.items() if current.get(name) != expected]
    if changed:
        raise RuntimeError('sealed scientific inputs changed: ' + ', '.join(changed))
    return value
