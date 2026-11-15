"""Ordinary addressed repairs, two public tests, independent twenty-run audit."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from cgx.provenance import atomic_json, read_json, hash_json, file_sha256, verify_hashes
from cgx.seeds import seed_for
from cgx.graphs.manifests import read_jsonl,load_cases
from cgx.baselines import RuntimePolicy
from .matrix import manifest_maps, require_frozen
from .rollout import rollout

def verification(system, case, config, predicted, confidence, threshold, abstained=False, audit=True, store=None):
    """Repaired system persists; fresh task and disturbance streams replace diagnostic ones."""
    from cgx.env.system import generate_task
    result = {"selected_repair": predicted, "repair_node": None, "changed_healthy_node": False,
              "repair_seconds": 0., "public_verification_count": 0, "public_pass": False,
              "declared_completion": False, "audit_count": 0, "audit_pass_count": 0,
              "audit_all_pass": None, "public_cost": {"executions": 0, "bytes": 0, "wall_seconds": 0.},
              "external_audit_cost": {"executions": 0, "wall_seconds": 0.}, "audit_errors": [],
              "public_observation_hashes":[],"audit_observation_hashes":[],"execution_errors":[]}
    if abstained or threshold is None or confidence < threshold:
        result["abstained"] = True
        return result
    result["abstained"] = False
    if predicted != "healthy":
        node = predicted.rsplit(":", 1)[0]
        try: change = system.repair(node)
        except Exception as exc:
            result.update(abstained=True,repair_error=f"{type(exc).__name__}:{exc}")
            return result
        result.update(repair_node=node, repair_seconds=change["wall_seconds"],
                      changed_healthy_node=case["fault_family"] == "healthy" or node != case["fault_node"])
    original_task = deepcopy(system.task)
    original_streams = deepcopy(system.streams)
    original_targets = system.target_xyz.copy()
    def execute(kind, index):
        seed = seed_for(case["streams"][kind], index, "task")
        system.task = generate_task(seed)
        system.target_xyz = np.asarray([system._fk(q) for q in system.task["waypoints"]])
        system.streams = {name: seed_for(case["streams"][kind], index, name) for name in original_streams}
        return system.run(ports=(), sample_hz=10, execution_index=index)
    public = []
    try:
        for index in range(config["resources"]["verification_executions_reserved"]):
            result["public_verification_count"] += 1
            result["public_cost"]["executions"] += 1
            stamp=time.monotonic()
            try:
                acquired = execute("verification", index)
                public.append((acquired,deepcopy(system.task)))
                result["public_cost"]["bytes"] += acquired.observation["actual_cost"]["bytes"]
                result["public_observation_hashes"].append(store.put_observation(acquired.observation) if store else hash_json(acquired.observation))
            except Exception as exc:
                result["execution_errors"].append(f"verification:{index}:{type(exc).__name__}:{exc}")
            result["public_cost"]["wall_seconds"] += time.monotonic()-stamp
        # The external interface exposes measured requirement flags, never fault truth.
        from cgx.env.verification import public_verification_pass
        result["public_pass"] = len(public)==config["resources"]["verification_executions_reserved"] and all(public_verification_pass(x.observation,task,config) for x,task in public)
        result["declared_completion"] = result["public_pass"]
        if result["declared_completion"] and audit:
            for index in range(config["resources"]["audit_executions"]):
                result["audit_count"] += 1
                result["external_audit_cost"]["executions"] += 1
                stamp=time.monotonic()
                try:
                    acquired = execute("audit", index)
                    result["audit_pass_count"] += int(acquired.metrics["task_pass"])
                    result["audit_errors"].append(max(acquired.metrics["waypoint_hold_max_error_m"]))
                    result["audit_observation_hashes"].append(store.put_observation(acquired.observation) if store else hash_json(acquired.observation))
                except Exception as exc:
                    result["execution_errors"].append(f"audit:{index}:{type(exc).__name__}:{exc}")
                result["external_audit_cost"]["wall_seconds"] += time.monotonic()-stamp
            result["audit_all_pass"] = result["audit_pass_count"] == config["resources"]["audit_executions"]
        result["independent_audit_seed_hash"] = hash_json([case["streams"]["audit"], config["resources"]["audit_executions"]])
    finally:
        system.task, system.streams = original_task, original_streams
        system.target_xyz = original_targets
    return result

def evaluate_repairs(root, config, resume=False):
    from .parallel import evaluation_workers
    if evaluation_workers()>1 or config['protocol_id']=='CGX-2.0':
        from .parallel_repairs import evaluate_repairs_parallel
        return evaluate_repairs_parallel(root,config,resume)
    root = Path(root)
    frozen = require_frozen(root, config)
    if not (root/"evaluation/main/complete.json").exists():
        raise ValueError("Main frozen acquisition must be complete")
    thresholds = read_json(root/"models/thresholds.json")
    subset = read_json(root/"manifests/repair_subset.json")
    cases = {c["case_id"]: c for split in config["splits"] if split.startswith("test_") for c in load_cases(root,split)}
    rows,graphs,tasks = manifest_maps(root)
    from cgx.env.system import System
    from cgx.data.store import Store
    store=Store(root)
    inputs = {}
    for path in (root/"evaluation/main/cases").glob("*.json"):
        item = read_json(path)["row"]
        if item["byte_fraction"] == config["evaluation"]["repair_byte_fraction"] and item["method"] in config["evaluation"]["repair_methods"]:
            inputs[(item["case_id"],item["method"],item["policy_seed"])] = item
    count = 0;expected_keys=set()
    for case_id in [case_id for item in subset for case_id in item["case_ids"]]:
        case = cases[case_id]; cid = case["configuration_id"]
        for method in config["evaluation"]["repair_methods"]:
            seeds = config["evaluation"]["repair_training_seeds"] if method in config["methods"]["learned"] else [None]
            for seed in seeds:
                key = (case["case_id"],method,seed)
                digest=hash_json(key)
                if digest in expected_keys:raise ValueError("Duplicate preregistered repair cell")
                expected_keys.add(digest)
                path = root/"evaluation/repairs/cases"/f"{digest}.json"
                if resume and path.exists():
                    old = read_json(path)
                    if old.get("state") != "passed" or old.get("frozen_hash") != hash_json(frozen) or old.get("result_hash")!=hash_json(old["row"]):
                        raise ValueError("Repair artifact hash mismatch")
                    continue
                acquired = inputs[key]
                prediction = acquired["prediction"]
                confidence = float(np.exp2(max(prediction["log_probabilities_base2"])))
                threshold = thresholds["repair_confidence"].get(f"{method}:{seed}")
                system = System(graphs[cid],case,tasks[case["case_id"]],config,root/"assets/mujoco_menagerie/franka_emika_panda",rows[cid]["implementation"])
                result = verification(system,case,config,acquired["predicted_target_key"],confidence,threshold,acquired["abstained"],store=store)
                result.update(case_id=case["case_id"],configuration_id=cid,suite=case["split"],method=method,policy_seed=seed,
                              confidence=confidence,threshold=threshold,evidence_trace_hash=acquired["evidence_trace_hash"],
                              diagnostic_bytes=acquired["bytes_used"],diagnostic_executions=acquired["executions_used"],
                              total_deployed_executions=1+acquired["executions_used"]+result["public_verification_count"],
                              total_deployed_bytes=acquired["initial_bytes"]+acquired["bytes_used"]+result["public_cost"]["bytes"],
                              total_deployed_seconds=acquired["end_to_end_seconds"]+result["repair_seconds"]+result["public_cost"]["wall_seconds"],technical_smoke=False)
                atomic_json(path,{"state":"passed","frozen_hash":hash_json(frozen),"result_hash":hash_json(result),"row":result})
                count += 1
    verify_hashes(frozen["files"],root)
    from .matrix import verify_result_inventory
    complete=verify_result_inventory(root/"evaluation/repairs/cases",expected_keys)
    report={"state":"passed","rows":complete,"expected_rows":len(expected_keys),"completed_this_invocation":count}
    atomic_json(root/"evaluation/repairs/complete.json",report)
    return report
