"""Validation-only scalar choices and auditable selected checkpoint inventory."""
from pathlib import Path
from collections import defaultdict
import numpy as np
from cgx.provenance import read_json, atomic_json, file_sha256, hash_json, require_protocol_lock
from cgx.graphs.manifests import read_jsonl,load_cases
from cgx.baselines import RuntimePolicy
from .matrix import manifest_maps
from .rollout import rollout
from .repairs import verification

def select_checkpoints(root,config,validation_only=True):
    if not validation_only: raise ValueError("Checkpoint selection must be validation-only")
    root=Path(root);lock=require_protocol_lock(root,config)
    paths=[root/"models/readers"/name for name in [*[f"crossfit_{i}" for i in range(config["training"]["crossfit_folds"])],"final","independent"]]
    paths += [root/"models/policies"/f"{method}_seed{seed}" for method in config["methods"]["learned"] for seed in config["seeds"]["training"]]
    inventory={}; files={}
    for path in paths:
        marker=read_json(path/"complete.json")
        if marker.get("state")!="passed" or marker.get("technical_smoke"):
            raise ValueError(f"Formal completed training required: {path}")
        validation=read_json(path/"validation.json")
        selected=min(validation,key=lambda r:(r["mean_final_nll_bits"],r.get("mean_bytes_used",0),r["iteration"]))
        filename=f"checkpoint_{selected['iteration']:07d}.pt"
        if filename!=marker["selected_checkpoint"] or file_sha256(path/filename)!=marker["selected_checkpoint_hash"]:
            raise ValueError("Selected checkpoint differs from validation rule or hash")
        expected=config["training"]["predictor_update_cap" if "/readers/" in str(path) else "policy_update_cap"]
        from cgx.training.stopping import validate_completion
        validate_completion(marker,validation,config['training'],expected)
        inventory[str(path.relative_to(root))]={"selected":filename,"validation":selected,"updates":marker["actual_updates"]}
        for p in [path/filename,path/"complete.json",path/"validation.json"]: files[str(p.relative_to(root))]=file_sha256(p)
    thresholds_path=root/"models/thresholds.json"
    if not thresholds_path.exists():
        select_thresholds(root,config)
    threshold=read_json(thresholds_path)
    if not threshold.get("validation_only") or threshold.get("state")!="passed": raise ValueError("Validation thresholds incomplete")
    files[str(thresholds_path.relative_to(root))]=file_sha256(thresholds_path)
    report={"state":"passed","validation_only":True,"protocol_hash":hash_json(lock),"inventory":inventory,"files":files}
    atomic_json(root/"models/selection.json",report)
    return report

def select_thresholds(root,config):
    from cgx.training import load_reader,load_policy
    root=Path(root);rows,graphs,tasks=manifest_maps(root)
    cases=list(load_cases(root,"validation"))
    reader=load_reader(root,"final",device="cuda");betas={};curves={};thresholds={}
    artifact=root/"models/thresholds_progress.json"
    progress=read_json(artifact) if artifact.exists() else {"cost_aware_beta":{},"repair_confidence":{},"curves":{}}
    for seed in config["seeds"]["training"]:
        if str(seed) in progress["cost_aware_beta"]: continue
        model=load_policy(root,"gnn_myopic",seed,device="cuda");metrics=[]
        for beta in config["methods"]["cost_aware_adaptation"]["beta_grid"]:
            policy=RuntimePolicy("cost_aware_adaptation",model,beta);groups=defaultdict(list)
            from .parallel import evaluation_workers,rollout_pool,map_rollouts
            if evaluation_workers()>1:
                jobs=((graphs[c['configuration_id']],c,tasks[c['case_id']],config['evaluation']['primary_byte_fraction'],config['evaluation']['main_execution_cap'],'A',False) for c in cases)
                with rollout_pool(reader,model,'cost_aware_adaptation',config,root/'assets/mujoco_menagerie/franka_emika_panda',beta=beta) as pool:
                    for result,_ in map_rollouts(pool,jobs):groups[result['configuration_id']].append(result)
            else:
                for case in cases:
                    result,_,_,_=rollout(graphs[case["configuration_id"]],case,tasks[case["case_id"]],config,reader,policy,config['evaluation']['primary_byte_fraction'],
                                        asset_path=root/"assets/mujoco_menagerie/franka_emika_panda")
                    groups[case["configuration_id"]].append(result)
            loss=float(np.mean([np.mean([r["final_nll_bits"] for r in g]) for g in groups.values()]))
            cost=float(np.mean([np.mean([r["bytes_used"] for r in g]) for g in groups.values()]))
            metrics.append({"beta":beta,"loss":loss,"bytes":cost})
        progress["cost_aware_beta"][str(seed)]=min(metrics,key=lambda x:(x["loss"],x["bytes"],x["beta"]))["beta"]
        progress["curves"][f"beta:{seed}"]=metrics;atomic_json(artifact,progress)
    for method in config["evaluation"]["repair_methods"]:
        from .design import seeded_method,trained_method
        seeds=config["evaluation"]["repair_training_seeds"] if seeded_method(method,config) else [None]
        for seed in seeds:
            key=f"{method}:{seed}"
            if key in progress["repair_confidence"]: continue
            model=load_policy(root,trained_method(method),seed,device="cuda") if seed is not None else None
            beta=progress['cost_aware_beta'].get(str(seed),.5) if method=='cost_aware_adaptation' else .5
            policy=RuntimePolicy(method,model,beta)
            predictions=[]
            # Test thresholds using the same acquired result; do not reacquire for each threshold.
            from .parallel import evaluation_workers,rollout_pool,map_rollouts
            if evaluation_workers()>1:
                jobs=((graphs[c['configuration_id']],c,tasks[c['case_id']],config['evaluation']['repair_byte_fraction'],config['evaluation']['main_execution_cap'],'A',False,'validation_repair') for c in cases)
                with rollout_pool(reader,model,method,config,root/'assets/mujoco_menagerie/franka_emika_panda',beta=beta) as pool:
                    predictions=[result['_verification'] for result,_ in map_rollouts(pool,jobs)]
            else:
                for case in cases:
                    result,_,_,system=rollout(graphs[case["configuration_id"]],case,tasks[case["case_id"]],config,reader,policy,config['evaluation']['repair_byte_fraction'],
                                             asset_path=root/"assets/mujoco_menagerie/franka_emika_panda")
                    confidence=float(np.exp2(max(result["prediction"]["log_probabilities_base2"])))
                    outcome=verification(system,case,config,result["predicted_target_key"],confidence,min(config["evaluation"]["repair_threshold_grid"]),result["abstained"],audit=True)
                    predictions.append({"confidence":confidence,"abstained":result["abstained"],"public_pass":outcome["public_pass"],"audit_all_pass":outcome["audit_all_pass"]})
            curve=[]
            for threshold in config["evaluation"]["repair_threshold_grid"]:
                accepted=[p for p in predictions if not p["abstained"] and p["confidence"]>=threshold and p["public_pass"]]
                success=sum(p["audit_all_pass"] is True for p in accepted)
                risk=(len(accepted)-success)/len(accepted) if accepted else None
                curve.append({"threshold":threshold,"coverage":len(accepted)/len(cases),"risk":risk,"successful_cases":success,"cases":len(cases)})
            eligible=[r for r in curve if r["risk"] is not None and r["risk"]<=config["evaluation"]["repair_risk_target_validation"]]
            chosen=min(eligible,key=lambda r:(-r["successful_cases"],r["risk"],r["threshold"]))["threshold"] if eligible else None
            progress["repair_confidence"][key]=chosen;progress["curves"][key]=curve;atomic_json(artifact,progress)
    from cgx.reporting.diagnostics import validation_quality_target
    progress["cost_to_quality"]=validation_quality_target(root)
    progress['cost_to_quality']['budget_axis']='execution_cap' if config['protocol_id']=='CGX-2.0' else 'byte_fraction'
    progress.update(state="passed",validation_only=True,validation_cases=len(cases),config_hash=hash_json(config))
    atomic_json(root/"models/thresholds.json",progress)
    return progress
