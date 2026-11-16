"""Validation-only checkpoint scoring, sealed models, complete held-out matrix."""
from collections import defaultdict
from pathlib import Path
from copy import deepcopy
import json
import os
import time
import numpy as np
import torch
from cgx.config import EXPERIMENT
from cgx.provenance import atomic_json, append_jsonl, file_sha256, hash_json, read_json, require_protocol_lock, verify_hashes, now
from cgx.graphs.manifests import read_jsonl,load_cases
from cgx.baselines import RuntimePolicy
from .rollout import rollout,call_deadline,normalized_prediction,_synchronize

def manifest_maps(root):
    root = Path(root)
    rows = {r["configuration_id"]: r for r in read_jsonl(root / "manifests/split_manifest.jsonl")}
    graphs = {r["configuration_id"]: r["graph"] for r in read_jsonl(root / "manifests/public_graphs.jsonl")}
    tasks = {r["case_id"]: r["task"] for r in read_jsonl(root / "manifests/tasks.jsonl")}
    return rows, graphs, tasks

def validation_callback(root, config):
    """Full validation cases at preregistered primary budget; no test access."""
    rows, graphs, tasks = manifest_maps(root)
    cases = list(load_cases(root,"validation"))
    def evaluate(model, reader, method):
        was_training = model.training
        gradient_flags = [p.requires_grad for p in model.parameters()]
        deployed_method='cost_aware_adaptation' if config['protocol_id']=='CGX-2.0' and method=='gnn_myopic' else method
        policy = RuntimePolicy(deployed_method, model)
        policy.fixed_plan = method == "gnn_fixed_plan"
        by_config = defaultdict(list)
        try:
            from .parallel import evaluation_workers, rollout_pool, map_rollouts
            workers = evaluation_workers()
            began = time.monotonic()
            if workers > 1:
                jobs = ((graphs[c["configuration_id"]], c, tasks[c["case_id"]],
                         config["evaluation"]["primary_byte_fraction"],
                         config["evaluation"]["main_execution_cap"], "A", False) for c in cases)
                with rollout_pool(reader, model, deployed_method, config,
                        Path(root)/"assets/mujoco_menagerie/franka_emika_panda", workers=workers) as pool:
                    for result, _ in map_rollouts(pool, jobs):
                        by_config[result["configuration_id"]].append(result)
            else:
                for case in cases:
                    result, _, _, _ = rollout(graphs[case["configuration_id"]], case, tasks[case["case_id"]], config,
                        reader, policy, config["evaluation"]["primary_byte_fraction"], asset_path=Path(root)/"assets/mujoco_menagerie/franka_emika_panda")
                    by_config[case["configuration_id"]].append(result)
            mean = lambda field: float(np.mean([np.mean([r[field] for r in group]) for group in by_config.values()]))
            return {"mean_final_nll_bits": mean("final_nll_bits"), "mean_bytes_used": mean("bytes_used"),
                    "case_count": len(cases), "configurations": len(by_config), "byte_fraction": config["evaluation"]["primary_byte_fraction"],
                    "validation_only": True, "timeouts": sum(r["timeout"] for group in by_config.values() for r in group),
                    "elapsed_seconds":time.monotonic()-began,"rollout_workers":workers,
                    "inference_device":"cpu" if workers>1 else str(next(model.parameters()).device)}
        finally:
            model.train(was_training)
            for parameter, flag in zip(model.parameters(), gradient_flags):
                parameter.requires_grad_(flag)
    return evaluate

def freeze_models(root, config):
    root = Path(root)
    protocol = require_protocol_lock(root, config)
    selected = read_json(root / "models/selection.json")
    if selected.get("state") != "passed" or not selected.get("validation_only"):
        raise ValueError("All checkpoints must first be selected using validation only")
    verify_hashes(selected["files"], root)
    record = {"state": "passed", "frozen_at": now(), "protocol_hash": hash_json(protocol),
              "selection_hash": file_sha256(root / "models/selection.json"), "files": selected["files"],
              "preprocessing_hash": file_sha256(EXPERIMENT / "src/cgx/models/features.py"), "online_learning": False}
    destination = root / "models/frozen.json"
    if destination.exists():
        existing=read_json(destination)
        if any(existing[k]!=record[k] for k in ("files","protocol_hash","selection_hash","preprocessing_hash")):
            raise ValueError("Cannot replace frozen models after sealing")
        return existing
    atomic_json(destination, record)
    return record

def require_frozen(root, config):
    root = Path(root)
    protocol = require_protocol_lock(root, config)
    frozen = read_json(root / "models/frozen.json")
    if frozen["protocol_hash"] != hash_json(protocol):
        raise ValueError("Frozen model protocol mismatch")
    verify_hashes(frozen["files"], root)
    if frozen["preprocessing_hash"] != file_sha256(EXPERIMENT / "src/cgx/models/features.py"):
        raise ValueError("Preprocessing changed after freezing")
    append_jsonl(root/"test_access.jsonl",{"time":now(),"purpose":"verify_sealed_test_hashes_after_model_freeze","frozen_model_hash":hash_json(frozen)})
    require_protocol_lock(root,config,include_test=True)
    return frozen


def result_identity(row):
    return hash_json([row["method"],row["policy_seed"],row["execution_cap"],row["byte_fraction"],row["case_id"]])


def validate_result_artifact(item,key,frozen_hash):
    if item.get("state")!="passed" or item.get("frozen_hash")!=frozen_hash:
        raise ValueError("Result is incomplete or belongs to another frozen model set")
    if result_identity(item["row"])!=key: raise ValueError("Result identity differs from its requested matrix cell")
    if item.get("result_hash")!=hash_json({"row":item["row"],"trace":item["trace"]}): raise ValueError("Result row or trace content changed")


def verify_result_inventory(directory,expected_keys,allow_other_suites=False):
    actual={p.stem for p in Path(directory).glob("*.json")}
    missing=expected_keys-actual
    extra=actual-expected_keys
    if missing or (extra and not allow_other_suites):
        raise ValueError(f"Evaluation matrix incomplete/duplicated: {len(missing)} missing and {len(extra)} unexpected result files")
    return len(expected_keys)

def evaluate(root, config, suite="all", resume=False, sensitivity=False):
    from .parallel import evaluation_workers
    if evaluation_workers() > 1:
        from .parallel_matrix import evaluate_parallel
        return evaluate_parallel(root,config,suite,resume,sensitivity)
    from cgx.data.store import Store
    from cgx.training import load_reader, load_policy
    root = Path(root)
    frozen = require_frozen(root, config)
    rows, graphs, tasks = manifest_maps(root)
    suites = [s for s in config["splits"] if s.startswith("test_")] if suite == "all" else [suite]
    if any(s not in config["splits"] or not s.startswith("test_") for s in suites):
        raise ValueError("Unknown test suite")
    append_jsonl(root / "test_access.jsonl", {"time": now(), "purpose": "read_frozen_test_case_splits", "suites": suites, "frozen_model_hash": hash_json(frozen)})
    cases = [c for split in suites for c in load_cases(root,split)]
    for split in suites:
        selected=[c for c in cases if c["split"]==split]
        expected=config["splits"][split]["configurations"]*config["splits"][split]["cases_per_configuration"]
        if len(selected)!=expected or len({c["case_id"] for c in selected})!=expected:
            raise ValueError(f"Sealed test split {split} is missing cases or duplicates them")
    if sensitivity:
        subset = read_json(root / "manifests/round_sensitivity_subset.json")
        ids = {case_id for r in subset for case_id in r["case_ids"]}
        cases = [c for c in cases if c["case_id"] in ids]
    append_jsonl(root / "test_access.jsonl", {"time": now(), "purpose": "frozen_round_sensitivity" if sensitivity else "frozen_evaluation",
                                            "suites": suites, "frozen_model_hash": hash_json(frozen), "case_count": len(cases)})
    store = Store(root)
    reader = load_reader(root, name="final", device="cuda")
    from .design import main_methods,conditions,matrix_cells
    methods=main_methods(config)
    output = root / "evaluation" / ("round_sensitivity" if sensitivity else "main")
    output.mkdir(parents=True, exist_ok=True)
    environment_hash = file_sha256(root / "environment.json")
    manifest_hash = file_sha256(root / "manifests/hashes.json")
    count = 0
    expected_keys=set()
    for method, seed in methods:
        if method in config["methods"]["learned"] or method == "cost_aware_adaptation":
            base = "gnn_myopic" if method == "cost_aware_adaptation" else method
            model = load_policy(root, base, seed, device="cuda")
        else:
            model = None
        beta = read_json(root / "models/thresholds.json")["cost_aware_beta"].get(str(seed), .5) if method == "cost_aware_adaptation" else .5
        policy = RuntimePolicy(method, model, beta)
        policy.fixed_plan = method in ("gnn_fixed_plan", "full_recording_feasible", "full_recording_unrestricted_reference")
        for cap,fraction in conditions(config,method,sensitivity):
            if True:
                for case in cases:
                    key = hash_json([method, seed, cap, fraction, case["case_id"]])
                    if key in expected_keys: raise ValueError("Duplicate registered evaluation matrix cell")
                    expected_keys.add(key)
                    path = output / "cases" / f"{key}.json"
                    if resume and path.exists():
                        previous = read_json(path)
                        validate_result_artifact(previous,key,hash_json(frozen))
                        continue
                    cid = case["configuration_id"]
                    result, traces, _, _ = rollout(graphs[cid], case, tasks[case["case_id"]], config, reader, policy,
                        fraction, cap, rows[cid]["implementation"], root / "assets/mujoco_menagerie/franka_emika_panda", store,
                        unrestricted=method == "full_recording_unrestricted_reference")
                    result.update(method=method, policy_seed=seed, protocol_hash=frozen["protocol_hash"],
                                  environment_hash=environment_hash, manifest_hash=manifest_hash,
                                  checkpoint_hash=model.checkpoint_hash if model is not None else hash_json({"method":method,"rule_source":file_sha256(EXPERIMENT/"src/cgx/baselines/policies.py")}),
                                  reader_hash=reader.checkpoint_hash, technical_smoke=False)
                    payload={"row":result,"trace":traces}
                    atomic_json(path, {"state": "passed", "frozen_hash": hash_json(frozen),"result_hash":hash_json(payload), **payload})
                    count += 1
                    if count % 100 == 0:
                        print(json.dumps({"completed_this_invocation": count, "method": method, "seed": seed, "case": case["case_id"]}), flush=True)
    verify_hashes(frozen["files"], root)
    completed=verify_result_inventory(output/"cases",expected_keys,allow_other_suites=suite!="all")
    summary = {"state": "passed", "frozen_hash": hash_json(frozen), "suites": suites, "case_files": completed,
               "expected_for_requested_suites": len(cases)*len(matrix_cells(config,sensitivity))}
    atomic_json(output / ("complete.json" if suite == "all" else f"complete-{suite}.json"), summary)
    return summary

def reread(root, config):
    from .parallel import evaluation_workers
    if evaluation_workers() > 1:
        from .parallel_matrix import reread_parallel
        return reread_parallel(root,config)
    from cgx.data.store import Store
    from cgx.training import load_reader
    root = Path(root)
    frozen = require_frozen(root, config)
    store = Store(root)
    reader = load_reader(root, name="independent", device="cuda")
    inputs = sorted((root/"evaluation/main/cases").glob("*.json"))
    if not inputs or not (root/"evaluation/main/complete.json").exists():
        raise ValueError("Complete frozen acquisition is required before independent rereading")
    cache = {}; output=[]
    for path in inputs:
        row = read_json(path)["row"]
        digest = row["evidence_trace_hash"]
        if digest not in cache:
            history = store.load_history(digest)
            from cgx.graphs import legal_targets
            keys,prior=legal_targets(history["graph"])
            prediction={"target_keys":keys,"log_probabilities_base2":np.log2(prior).tolist()}
            error=None;warmup=0.;stamp=time.monotonic()
            try:
                if not getattr(reader,"_cgx_eval_warmed",False):
                    with call_deadline(10.): reader.predict(deepcopy(history));_synchronize(reader)
                    reader._cgx_eval_warmed=True;warmup=time.monotonic()-stamp
                stamp=time.monotonic()
                with call_deadline(config["resources"]["diagnosis_timeout_seconds_per_call"]):
                    candidate=normalized_prediction(reader.predict(history),keys,prior);_synchronize(reader)
                prediction=candidate
            except Exception as exc:
                error=f"{type(exc).__name__}:{exc}"
            cache[digest] = (prediction,error,time.monotonic()-stamp,warmup)
        p,error,seconds,warmup = cache[digest]
        index = p["target_keys"].index(row["true_target_key"])
        output.append({"source_result_hash": file_sha256(path), "evidence_trace_hash": digest,
                       "final_nll_bits": -p["log_probabilities_base2"][index], "method": row["method"],
                       "policy_seed": row["policy_seed"], "suite": row["suite"], "configuration_id": row["configuration_id"],
                       "case_id": row["case_id"], "byte_fraction": row["byte_fraction"],"execution_cap":row['execution_cap'], "reacquired_executions": 0,
                       "reader_hash":reader.checkpoint_hash,"abstained":error is not None,"error":error,"inference_seconds":seconds,"warmup_seconds":warmup})
    from cgx.graphs.manifests import write_jsonl
    write_jsonl(root/"evaluation/independent_reader.jsonl", output)
    verify_hashes(frozen["files"], root)
    report={"state": "passed", "rows": len(output), "unique_transcripts": len(cache), "reacquired_executions": 0,
        "output_hash":file_sha256(root/"evaluation/independent_reader.jsonl"),"reader_hash":reader.checkpoint_hash,"frozen_hash":hash_json(frozen)}
    atomic_json(root/"evaluation/independent_reader_complete.json",report)
    return report
