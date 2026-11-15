# Required result/storage tables

All IDs below are evaluator/storage join keys, not automatic model features.

## transition table (training private store)

parent_case_id, configuration_id, source_fold, trajectory_id, step, history_hash,
action_key, acquired_blob_hash, next_history_hash, resources_before, charged_cost,
resources_after, terminal, stop_reason, true_target_key, reward_bits,
reward_reader_hash, available_action_catalog_hash, selection_seconds.

Public reader input is assembled from history_hash through a whitelisted acquired store.
The training label and reward columns are never serialized into that input. True target
is used to compute loss; this does not authorize adding it to neural inputs.

## per_case_results

protocol_hash, environment_hash, manifest_hash, suite, configuration_id, case_id,
method, policy_seed, checkpoint_hash, reader_hash, byte_fraction, execution_cap,
initial_nll_bits, final_nll_bits, score_improvement_bits, prior_nll_bits,
predicted_target_key, true_target_key, correct_joint, correct_mechanism,
bytes_used, executions_used, calibrations_used, decisions_used,
selection_seconds, inference_seconds, simulation_seconds, recorder_seconds,
end_to_end_seconds, timeout, abstained, visible_truncation, evidence_trace_hash.

## repair_results

All relevant case/run keys above, selected_repair, repair_node, changed_healthy_node,
repair_seconds, public_verification_count, public_pass, declared_completion,
audit_count, audit_pass_count, audit_all_pass, tracking_error, violation_count,
public_cost, external_audit_cost, independent_audit_seed_hash.

## job records

job_id, phase, method/seed, command_arguments, device_uuid, allocation,
input_hashes, output_hashes, state, started_at, finished_at, exit_code,
stdout_path, stderr_path, peak_gpu_memory_bytes, elapsed_seconds,
actual_updates, selected_checkpoint, retry_of, failure_reason.

Never leave missing jobs implicit. Missing/failed execution is not a zero result.
