"""Unchanged online limits/accounting with explicit simple-policy read schedules.

Adapted from the frozen SED rollout in this isolated namespace. Physics, action
legality, graph reindexing, fallback probabilities and provenance use the original
implementations. Random defers its diagnostic calls until collection ends; fixed
plans retain the original final-only schedule. Warm-up uses a policy clone.
"""
from copy import deepcopy
from pathlib import Path
import math
import time
import numpy as np
from cgx.observation.actions import action_catalog, initial_budget, feasible_actions, charge
from cgx.provenance import hash_json
from cgx.seeds import seed_for
from structured_evidence_v1.rollout import call_deadline, _synchronize, _recorder_seconds, target_key, normalized_prediction
from .cache import ExtensionCachedSystem

def rollout(graph, case, task, config, reader, policy, byte_fraction, execution_cap=4,
            implementation="A", asset_path=None, store=None, initial_observation=None, unrestricted=False,
            source_root=None, run_root=None, sed_run_root=None):
    from cgx.graphs import legal_targets
    started = time.monotonic()
    system = ExtensionCachedSystem(graph, case, task, config=config, asset_path=asset_path, implementation=implementation,
                          cache_root=Path(run_root)/"cache/physical",
                          readonly_cache_roots=[Path(sed_run_root)/"cache/physical"],attempt_root=run_root)
    # Permute model indices afresh, identically across methods; physical ports,
    # semantic IDs, clocks, and simulator schedule remain exactly as registered.
    graph=deepcopy(graph)
    rng=np.random.default_rng(seed_for(config["seeds"]["manifest"],"evaluation_node_order",case["case_id"]))
    rng.shuffle(graph["nodes"])
    keys, prior = legal_targets(graph)
    fallback = {"target_keys": list(keys), "log_probabilities_base2": np.log2(prior).tolist()}
    catalog = action_catalog(graph, config)
    real_budget = initial_budget(graph, byte_fraction, execution_cap)
    budget = initial_budget(graph, max(1.01, byte_fraction), execution_cap) if unrestricted else real_budget
    simulation_seconds = recorder_seconds = 0.0
    initial_error=None
    evaluator_initial_metrics=None
    if initial_observation is None:
        tick = time.monotonic()
        try:
            initial = system.run(ports=(), sample_hz=10, execution_index=0)
            initial_observation = initial.observation
            evaluator_initial_metrics=deepcopy(initial.metrics)
            recorder_seconds += _recorder_seconds(initial.metrics)
        except Exception as exc:
            initial_error=f"initial_execution_error:{type(exc).__name__}:{exc}"
            initial_observation={"action_key":"initial_failed","records":[],"calibration_bursts":[],"visible_status":"timeout",
                "actual_cost":{"executions":1,"bytes":0,"calibrations":0,"decisions":0}}
        simulation_seconds += time.monotonic() - tick
    history = {"graph": deepcopy(graph), "observations": [deepcopy(initial_observation)],
               "remaining": dict(budget), "initial_budget": dict(budget), "action_keys": [],
               "nominal_task": deepcopy(task), "requirements": {"position_tolerance_m": config["system"]["position_tolerance_m"],
                   "hold_seconds": config["system"]["hold_seconds"], "joint_limit_tolerance_rad": config["system"]["joint_limit_tolerance_rad"]}}
    observation_ids=[system.identity("initial",dict(ports=[],sample_hz=10,execution_index=0))]
    stamp=time.monotonic();observation_refs=[hash_json(initial_observation)]
    provenance_seconds=time.monotonic()-stamp
    def reader_provenance():
        return {"observation_ids":list(observation_ids),"observation_refs":list(observation_refs)}
    prediction = fallback
    inference_seconds = selection_seconds = 0.0
    timeout = False
    abstained = initial_error is not None
    errors = [initial_error] if initial_error else []
    earlier_termination=initial_error is not None
    diagnostic_termination_stage=None
    traces = []
    execution_index = burst_index = 0
    warmup_seconds=selection_warmup_seconds=0.
    reader_calls=reader_warmup_calls=selector_calls=selector_warmup_calls=plan_calls=0
    last_predicted_observation_count=0
    defer_final=getattr(policy,"diagnosis_schedule","after_each")=="final_only"
    if not abstained and not getattr(reader,"_cgx_eval_warmed",False):
        stamp=time.monotonic()
        try:
            with call_deadline(max(10.,config["resources"]["diagnosis_timeout_seconds_per_call"]*2)):
                reader_warmup_calls+=1
                reader.predict(history, provenance=reader_provenance()); _synchronize(reader)
            reader._cgx_eval_warmed=True
        except Exception as exc:
            abstained=True;timeout=isinstance(exc,TimeoutError)
            errors.append(f"diagnostic_warmup_error:{type(exc).__name__}:{exc}")
            earlier_termination=True;diagnostic_termination_stage="initial_warmup"
        warmup_seconds=time.monotonic()-stamp
    def predict(stage="initial"):
        nonlocal earlier_termination,diagnostic_termination_stage
        nonlocal prediction, inference_seconds, timeout, abstained,reader_calls,last_predicted_observation_count
        stamp = time.monotonic()
        reader_calls+=1
        try:
            with call_deadline(config["resources"]["diagnosis_timeout_seconds_per_call"]):
                candidate = normalized_prediction(reader.predict(history, provenance=reader_provenance()), keys, prior)
                _synchronize(reader)
            elapsed = time.monotonic() - stamp
            inference_seconds += elapsed
            if elapsed > config["resources"]["diagnosis_timeout_seconds_per_call"]:
                timeout = abstained = True
                errors.append("diagnostic_timeout")
                diagnostic_termination_stage=stage
                earlier_termination=earlier_termination or stage!="final"
            else:
                prediction = candidate
                last_predicted_observation_count=len(history["observations"])
        except Exception as exc:
            inference_seconds += time.monotonic() - stamp
            abstained = True
            timeout = timeout or isinstance(exc,TimeoutError)
            errors.append(f"diagnostic_error:{type(exc).__name__}:{exc}")
            diagnostic_termination_stage=stage
            earlier_termination=earlier_termination or stage!="final"
    if not abstained: predict()
    initial_prediction = deepcopy(prediction)
    plan = None
    plan_timeout=max(1,history["remaining"]["decisions"])*config["resources"]["selection_timeout_seconds_per_decision"]
    non_stop_feasible=any(a["kind"]!="stop" for a in feasible_actions(catalog,history["remaining"]))
    if not abstained and non_stop_feasible and not getattr(policy,"_cgx_eval_warmed",False):
        stamp=time.monotonic()
        try:
            with call_deadline(max(10.,plan_timeout if getattr(policy,"fixed_plan",False) else config["resources"]["selection_timeout_seconds_per_decision"]*2)):
                warm_policy=deepcopy(policy)
                selector_warmup_calls+=1
                if getattr(warm_policy,"fixed_plan",False): warm_policy.plan(deepcopy(history),deepcopy(catalog),deepcopy(prediction))
                else: warm_policy.choose(deepcopy(history),deepcopy(feasible_actions(catalog,history["remaining"])),deepcopy(prediction))
                _synchronize(warm_policy)
            policy._cgx_eval_warmed=True
        except Exception as exc:
            abstained=True;timeout=timeout or isinstance(exc,TimeoutError)
            errors.append(f"selection_warmup_error:{type(exc).__name__}:{exc}")
            earlier_termination=True
        selection_warmup_seconds=time.monotonic()-stamp
    if not abstained and non_stop_feasible and getattr(policy, "fixed_plan", False):
        stamp = time.monotonic()
        try:
            with call_deadline(plan_timeout):
                plan_calls+=1
                plan = policy.plan(deepcopy(history), deepcopy(catalog), deepcopy(prediction))
                _synchronize(policy)
                if not isinstance(plan,(list,tuple)): raise ValueError("Fixed planner must return a complete action sequence")
                plan=list(plan)
                planned_remaining=dict(history["remaining"])
                registered={a["action_key"]:a for a in catalog}
                for i,planned in enumerate(plan):
                    if not isinstance(planned,dict) or planned.get("action_key") not in registered or planned!=registered[planned["action_key"]]:
                        raise ValueError("Fixed plan contains an unknown or modified action")
                    if planned["kind"]=="stop" and i!=len(plan)-1:raise ValueError("Fixed plan continues after STOP")
                    planned_remaining=charge(planned_remaining,planned["cost_bound"])
        except Exception as exc:
            abstained = True
            timeout=timeout or isinstance(exc,TimeoutError)
            errors.append(f"fixed_plan_error:{type(exc).__name__}:{exc}")
            earlier_termination=True
        elapsed = time.monotonic() - stamp
        selection_seconds += elapsed
        if elapsed > plan_timeout:
            timeout = abstained = True
            errors.append("fixed_plan_timeout")
            earlier_termination=True
    while not abstained:
        actions = feasible_actions(catalog, history["remaining"])
        if all(a["kind"] == "stop" for a in actions):
            break
        stamp = time.monotonic()
        elapsed=None
        selector_calls+=1
        try:
            with call_deadline(config["resources"]["selection_timeout_seconds_per_decision"]):
                if plan is not None:
                    action = plan.pop(0) if plan else next(a for a in catalog if a["kind"] == "stop")
                else:
                    action = policy.choose(deepcopy(history), deepcopy(actions), deepcopy(prediction))
                    _synchronize(policy)
            elapsed = time.monotonic() - stamp
            selection_seconds += elapsed
            if elapsed > config["resources"]["selection_timeout_seconds_per_decision"]:
                timeout = abstained = True
                errors.append("selection_timeout")
                earlier_termination=True
                break
            legal = {a["action_key"]: a for a in actions}
            if action["action_key"] not in legal or action != legal[action["action_key"]]:
                raise ValueError("Policy returned an infeasible or modified action")
        except Exception as exc:
            if elapsed is None: selection_seconds+=time.monotonic()-stamp
            abstained = True
            timeout=timeout or isinstance(exc,TimeoutError)
            errors.append(f"selection_error:{type(exc).__name__}:{exc}")
            earlier_termination=True
            break
        if action["kind"] == "stop":
            traces.append({"action": action, "stop_reason": "policy_stop", "prediction": deepcopy(prediction)})
            break
        before = deepcopy(history["remaining"])
        execution_index += int(bool(action["ports"]))
        burst_index += int(action["calibration_pair"] is not None)
        stamp = time.monotonic()
        try:
            acquired = system.acquire(action, execution_index=execution_index, burst_index=burst_index)
            observation = acquired.observation if hasattr(acquired, "observation") else acquired
            if any(observation["actual_cost"][k]>action["cost_bound"][k] for k in before):
                raise ValueError("Acquired cost exceeds the declared bound")
            recorder_seconds += _recorder_seconds(getattr(acquired,"metrics",{}))
            history["remaining"] = charge(history["remaining"], observation["actual_cost"])
        except Exception as exc:
            # No fabricated evidence on an aborted instrument call. Charge its
            # entire declared bound conservatively and preserve the last q.
            history["remaining"]=charge(before,action["cost_bound"])
            abstained=True;timeout=timeout or isinstance(exc,TimeoutError)
            errors.append(f"acquisition_error:{type(exc).__name__}:{exc}")
            earlier_termination=True
            traces.append({"action":action,"resources_before":before,"resources_after":deepcopy(history["remaining"]),
                "stop_reason":"acquisition_error","cost_accounting":"conservative_declared_upper_bound","prediction":deepcopy(prediction)})
            simulation_seconds+=time.monotonic()-stamp
            break
        simulation_seconds += time.monotonic() - stamp
        history["observations"].append(observation)
        observation_ids.append(system.last_acquisition_id)
        stamp=time.monotonic();observation_refs.append(hash_json(observation))
        provenance_seconds+=time.monotonic()-stamp
        history["action_keys"].append(action["action_key"])
        # A committed plan receives no intermediate diagnosis/abstention feedback.
        # Its common final reader sees the entire acquired transcript only once.
        diagnosis_updated=(not defer_final) or (plan is not None and not any(a["kind"]!="stop" for a in plan))
        if diagnosis_updated:
            continuing=(plan is None and any(a["kind"]!="stop" for a in feasible_actions(catalog,history["remaining"])))
            predict("intermediate" if continuing else "final")
        traces.append({"action": action, "evidence_hash": observation_refs[-1], "prediction": deepcopy(prediction),
                       "diagnosis_updated":diagnosis_updated,"measurement_acquisition_id":observation_ids[-1],
                       "resources_before": before, "resources_after": deepcopy(history["remaining"]),
                       "stop_reason": "timeout_or_error" if abstained else None})
    if defer_final and not abstained and last_predicted_observation_count!=len(history["observations"]):
        predict("final")
        for entry in reversed(traces):
            if entry.get("evidence_hash"):
                entry.update(diagnosis_updated=True,diagnosis_timing="after_collection",prediction=deepcopy(prediction))
                break
    truth = target_key(case)
    if truth not in keys:
        raise ValueError(f"Private truth does not match public target vocabulary: {truth}")
    label = keys.index(truth)
    predicted = keys[int(np.argmax(prediction["log_probabilities_base2"]))]
    initial_nll = -initial_prediction["log_probabilities_base2"][label]
    final_nll = -prediction["log_probabilities_base2"][label]
    used = {key: budget[key] - history["remaining"][key] for key in budget}
    provenance={**reader_provenance(),"identity_origin":"structured_runtime_physical_operations_v1"}
    history_hash = store.put_history(history) if store else hash_json(history)
    provenance_hash=store.save_provenance(history_hash,provenance) if store else hash_json(provenance)
    mechanism = lambda key: "healthy" if key == "healthy" else key.rsplit(":", 1)[-1]
    row = {"case_id": case["case_id"], "configuration_id": case["configuration_id"], "suite": case["split"],
           "byte_fraction": byte_fraction, "execution_cap": execution_cap, "initial_nll_bits": initial_nll,
           "final_nll_bits": final_nll, "score_improvement_bits": initial_nll-final_nll,
           "prior_nll_bits": -math.log2(prior[label]), "target_cardinality": len(keys),
           "predicted_target_key": predicted, "true_target_key": truth,
           "correct_joint": predicted == truth, "correct_mechanism": mechanism(predicted) == mechanism(truth),
           "bytes_used": used["bytes"], "executions_used": used["executions"], "calibrations_used": used["calibrations"],
           "decisions_used": used["decisions"], "initial_bytes": initial_observation["actual_cost"]["bytes"],
           "selection_seconds": selection_seconds, "inference_seconds": inference_seconds,
           "simulation_seconds": simulation_seconds, "recorder_seconds":recorder_seconds,
           "warmup_seconds":warmup_seconds,"selection_warmup_seconds":selection_warmup_seconds,
           "end_to_end_seconds": time.monotonic()-started,
           "timeout": timeout, "abstained": abstained, "errors": errors,
           "visible_truncation": any(o["visible_status"] == "truncated" for o in history["observations"]),
           "evidence_trace_hash": history_hash, "feasible": all(used[k] <= real_budget[k] for k in used),
           "unrestricted_reference": unrestricted, "prediction": prediction,
           "event_provenance_hash":provenance_hash,"diagnostic_family":"structured_evidence_v1",
           "physical_cache_hits":system.cache_hits,"new_robot_execution_attempts":system.robot_attempts,
           "new_calibration_attempts":system.calibration_attempts,
           "physical_acquisition_measured_seconds":system.physical_seconds,
           "logical_acquisition_reference_seconds":system.physical_seconds,
           "actual_new_physical_seconds":system.actual_physical_seconds,
           "provenance_seconds":provenance_seconds,
           "resource_accounting_unavailable":False,
           "recorder_seconds_scope":"measured_reference_acquisition_including_cached_operations",
           "wall_time_scope":"actual_execution_with_exact_physical_cache"}
    # Evaluator-only impact strata are appended after every policy/reader call.
    # They never enter the acquired public history or select a test case.
    row.update(initial_task_pass=evaluator_initial_metrics.get("task_pass") if evaluator_initial_metrics else None,
               initial_hold_error_m=max(evaluator_initial_metrics["waypoint_hold_max_error_m"]) if evaluator_initial_metrics else None,
               component_count=len(graph["nodes"]))
    row.update(final_online_prediction_complete=last_predicted_observation_count==len(history["observations"]),
        reader_calls=reader_calls,reader_warmup_calls=reader_warmup_calls,selector_calls=selector_calls,
        selector_warmup_calls=selector_warmup_calls,planning_calls=plan_calls,
        diagnosis_schedule=getattr(policy,"diagnosis_schedule","after_each"),
        imported_physical_cache_hits=system.imported_cache_hits,extension_physical_cache_hits=system.extension_cache_hits,
        initial_prediction=initial_prediction,
        earlier_termination_affected_acquisition=earlier_termination,
        diagnostic_termination_stage=diagnostic_termination_stage,
        early_termination_definition="acquisition phase terminated before terminal reader; hypothetical abandoned choices not inferred")
    return row, traces, history, system


def complete_final_read(online,history,provenance,reader,watchdog_seconds=120):
    """No new acquisitions; a distinct reader object avoids warming online calls."""
    completed=deepcopy(online)
    completed.update(online_timeout=online['timeout'],online_abstained=online['abstained'],
        completion_watchdog_seconds=watchdog_seconds,completed_read_seconds=0.,
        completion_reused_probability=bool(online['final_online_prediction_complete']),
        completed_read_calls=0,actual_history_unchanged=True,abandoned_checks_not_recovered=True,
        completed_read_error=None,completion_status='reused_verified_online')
    if online['final_online_prediction_complete']:
        return completed
    started=time.monotonic();fingerprint=reader.parameter_fingerprint()
    try:
        with call_deadline(watchdog_seconds):
            candidate=reader.predict(history,provenance=provenance)
            keys=online['prediction']['target_keys']
            candidate=normalized_prediction(candidate,keys,None)
        if reader.parameter_fingerprint()!=fingerprint:raise RuntimeError('Frozen reader parameters changed')
        key=keys[int(np.argmax(candidate['log_probabilities_base2']))]
        truth=online['true_target_key'];label=keys.index(truth)
        nll=-candidate['log_probabilities_base2'][label]
        mechanism=lambda value:'healthy' if value=='healthy' else value.rsplit(':',1)[-1]
        completed.update(prediction=candidate,predicted_target_key=key,final_nll_bits=nll,
            score_improvement_bits=online['initial_nll_bits']-nll,correct_joint=key==truth,
            correct_mechanism=mechanism(key)==mechanism(truth),
            healthy_false_alarm=truth=='healthy' and key!='healthy',
            fault_miss=truth!='healthy' and key=='healthy',completion_status='completed_same_history')
    except Exception as exc:
        completed.update(completion_status='unresolved_kept_online',
            completed_read_error=f'{type(exc).__name__}:{exc}')
    completed.update(completed_read_seconds=time.monotonic()-started,completed_read_calls=1)
    return completed

