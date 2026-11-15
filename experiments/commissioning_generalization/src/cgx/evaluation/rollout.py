"""Evaluator owns truth; the policy receives only a copied whitelisted history."""
from copy import deepcopy
from pathlib import Path
from contextlib import contextmanager
import math
import signal
import threading
import time
import numpy as np
from cgx.observation.actions import action_catalog, initial_budget, feasible_actions, charge
from cgx.provenance import hash_json
from cgx.seeds import seed_for


class CallDeadlineExceeded(TimeoutError):
    pass


@contextmanager
def call_deadline(seconds):
    """POSIX main-thread watchdog; interrupted calls retain their case denominator."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Strict online deadlines require a process main thread")
    previous_handler=signal.getsignal(signal.SIGALRM)
    previous_timer=signal.getitimer(signal.ITIMER_REAL)
    started=time.monotonic()
    def expired(signum,frame):
        raise CallDeadlineExceeded("online call exceeded its wall-time limit")
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,max(float(seconds),1e-6))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous_handler)
        if previous_timer[0]>0:
            signal.setitimer(signal.ITIMER_REAL,max(1e-6,previous_timer[0]-(time.monotonic()-started)),previous_timer[1])


def _synchronize(model):
    model=getattr(model,"model",model)
    if hasattr(model,"parameters"):
        parameter=next(model.parameters(),None)
        if parameter is not None and parameter.is_cuda:
            import torch
            torch.cuda.synchronize(parameter.device)


def _recorder_seconds(metrics):
    if not isinstance(metrics,dict): return 0.
    return float(metrics.get("recorder_seconds",0.))+sum(_recorder_seconds(v) for v in metrics.values() if isinstance(v,dict))

def target_key(case):
    return "healthy" if case["fault_family"] == "healthy" else f"{case['fault_node']}:{case['fault_family']}"

def normalized_prediction(prediction, keys, prior):
    if not isinstance(prediction, dict) or tuple(prediction.get("target_keys", ())) != tuple(keys):
        raise ValueError("Reader output target mapping differs from public legal targets")
    logp = np.asarray(prediction["log_probabilities_base2"], dtype=float)
    if logp.shape != (len(keys),) or not np.isfinite(logp).all():
        raise FloatingPointError("Nonfinite diagnostic output")
    p = np.exp2(logp)
    if abs(p.sum() - 1) > 1e-5:
        raise ValueError("Diagnostic probability is not normalized")
    return {"target_keys": list(keys), "log_probabilities_base2": logp.tolist()}

def reader_history(reader,history):
    # The built-in tensor reader only reads public fields and constructs fresh
    # arrays. Copying120k public record dictionaries consumes much of a2s call
    # deadline without changing the reader's information. Keep defensive copies
    # for arbitrary/custom readers, including mutating failure-test fixtures.
    from cgx.models import DiagnosticReader
    return history if type(reader) is DiagnosticReader else deepcopy(history)

def rollout(graph, case, task, config, reader, policy, byte_fraction, execution_cap=4,
            implementation="A", asset_path=None, store=None, initial_observation=None, unrestricted=False):
    from cgx.env.system import System
    from cgx.graphs import legal_targets
    started = time.monotonic()
    system = System(graph, case, task, config=config, asset_path=asset_path, implementation=implementation)
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
    prediction = fallback
    inference_seconds = selection_seconds = 0.0
    timeout = False
    abstained = initial_error is not None
    errors = [initial_error] if initial_error else []
    traces = []
    execution_index = burst_index = 0
    warmup_seconds=selection_warmup_seconds=0.
    if not abstained and not getattr(reader,"_cgx_eval_warmed",False):
        stamp=time.monotonic()
        try:
            with call_deadline(max(10.,config["resources"]["diagnosis_timeout_seconds_per_call"]*2)):
                reader.predict(reader_history(reader,history)); _synchronize(reader)
            reader._cgx_eval_warmed=True
        except Exception as exc:
            abstained=True;timeout=isinstance(exc,TimeoutError)
            errors.append(f"diagnostic_warmup_error:{type(exc).__name__}:{exc}")
        warmup_seconds=time.monotonic()-stamp
    def predict():
        nonlocal prediction, inference_seconds, timeout, abstained
        stamp = time.monotonic()
        try:
            with call_deadline(config["resources"]["diagnosis_timeout_seconds_per_call"]):
                candidate = normalized_prediction(reader.predict(reader_history(reader,history)), keys, prior)
                _synchronize(reader)
            elapsed = time.monotonic() - stamp
            inference_seconds += elapsed
            if elapsed > config["resources"]["diagnosis_timeout_seconds_per_call"]:
                timeout = abstained = True
                errors.append("diagnostic_timeout")
            else:
                prediction = candidate
        except Exception as exc:
            inference_seconds += time.monotonic() - stamp
            abstained = True
            timeout = timeout or isinstance(exc,TimeoutError)
            errors.append(f"diagnostic_error:{type(exc).__name__}:{exc}")
    if not abstained: predict()
    initial_prediction = deepcopy(prediction)
    plan = None
    plan_timeout=max(1,history["remaining"]["decisions"])*config["resources"]["selection_timeout_seconds_per_decision"]
    non_stop_feasible=any(a["kind"]!="stop" for a in feasible_actions(catalog,history["remaining"]))
    if not abstained and non_stop_feasible and not getattr(policy,"_cgx_eval_warmed",False):
        stamp=time.monotonic()
        try:
            with call_deadline(max(10.,plan_timeout if getattr(policy,"fixed_plan",False) else config["resources"]["selection_timeout_seconds_per_decision"]*2)):
                if getattr(policy,"fixed_plan",False): policy.plan(deepcopy(history),deepcopy(catalog),deepcopy(prediction))
                else: policy.choose(deepcopy(history),deepcopy(feasible_actions(catalog,history["remaining"])),deepcopy(prediction))
                _synchronize(policy)
            policy._cgx_eval_warmed=True
        except Exception as exc:
            abstained=True;timeout=timeout or isinstance(exc,TimeoutError)
            errors.append(f"selection_warmup_error:{type(exc).__name__}:{exc}")
        selection_warmup_seconds=time.monotonic()-stamp
    if not abstained and non_stop_feasible and getattr(policy, "fixed_plan", False):
        stamp = time.monotonic()
        try:
            with call_deadline(plan_timeout):
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
        elapsed = time.monotonic() - stamp
        selection_seconds += elapsed
        if elapsed > plan_timeout:
            timeout = abstained = True
            errors.append("fixed_plan_timeout")
    while not abstained:
        actions = feasible_actions(catalog, history["remaining"])
        if all(a["kind"] == "stop" for a in actions):
            break
        stamp = time.monotonic()
        elapsed=None
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
                break
            legal = {a["action_key"]: a for a in actions}
            if action["action_key"] not in legal or action != legal[action["action_key"]]:
                raise ValueError("Policy returned an infeasible or modified action")
        except Exception as exc:
            if elapsed is None: selection_seconds+=time.monotonic()-stamp
            abstained = True
            timeout=timeout or isinstance(exc,TimeoutError)
            errors.append(f"selection_error:{type(exc).__name__}:{exc}")
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
            traces.append({"action":action,"resources_before":before,"resources_after":deepcopy(history["remaining"]),
                "stop_reason":"acquisition_error","cost_accounting":"conservative_declared_upper_bound","prediction":deepcopy(prediction)})
            simulation_seconds+=time.monotonic()-stamp
            break
        simulation_seconds += time.monotonic() - stamp
        history["observations"].append(observation)
        history["action_keys"].append(action["action_key"])
        # A committed plan receives no intermediate diagnosis/abstention feedback.
        # Its common final reader sees the entire acquired transcript only once.
        diagnosis_updated=plan is None or not any(a["kind"]!="stop" for a in plan)
        if diagnosis_updated: predict()
        traces.append({"action": action, "evidence_hash": hash_json(observation), "prediction": deepcopy(prediction),
                       "diagnosis_updated":diagnosis_updated,
                       "resources_before": before, "resources_after": deepcopy(history["remaining"]),
                       "stop_reason": "timeout_or_error" if abstained else None})
    truth = target_key(case)
    if truth not in keys:
        raise ValueError(f"Private truth does not match public target vocabulary: {truth}")
    label = keys.index(truth)
    predicted = keys[int(np.argmax(prediction["log_probabilities_base2"]))]
    initial_nll = -initial_prediction["log_probabilities_base2"][label]
    final_nll = -prediction["log_probabilities_base2"][label]
    used = {key: budget[key] - history["remaining"][key] for key in budget}
    history_hash = store.put_history(history) if store else hash_json(history)
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
           "unrestricted_reference": unrestricted, "prediction": prediction}
    # Evaluator-only impact strata are appended after every policy/reader call.
    # They never enter the acquired public history or select a test case.
    row.update(initial_task_pass=evaluator_initial_metrics.get("task_pass") if evaluator_initial_metrics else None,
               initial_hold_error_m=max(evaluator_initial_metrics["waypoint_hold_max_error_m"]) if evaluator_initial_metrics else None,
               component_count=len(graph["nodes"]))
    return row, traces, history, system
