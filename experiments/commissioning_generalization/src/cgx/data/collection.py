"""Resumable source-only behavior collection with actual physical executions."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import time
import traceback
import numpy as np
from .store import Store, public_requirements, _atomic_gzip
from cgx.provenance import atomic_json, canonical_hash, file_hash, require_protocol_lock, now
from cgx.seeds import seed_for


def _reserve_execution(root,limit):
    """Durable pre-execution accounting includes interrupted or failed executions."""
    path=Path(root)/"private"/"training_execution_ledger.json"; path.parent.mkdir(parents=True,exist_ok=True)
    lock=path.with_suffix(".lock")
    with lock.open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        current=json.loads(path.read_text()) if path.exists() else {"attempted_executions":0}
        if current["attempted_executions"]>=limit: raise RuntimeError("Hard source physical execution cap reached")
        current["attempted_executions"]+=1
        atomic_json(path,current)


def _wire(root,wire):
    digest=hashlib.sha256(wire).hexdigest()
    path=Path(root)/"public"/"wire"/(digest+".bin.gz")
    if not path.exists(): _atomic_gzip(path,wire)
    return digest


def _fixed_diverse_plan(actions,budget,rng):
    from cgx.observation.actions import can_afford,charge
    remaining=deepcopy(budget); chosen=[]; seen=Counter()
    priorities={a["action_key"]:float(rng.random()) for a in actions}
    while True:
        legal=[a for a in actions if a["kind"]!="stop" and can_afford(remaining,a["cost_bound"])]
        if not legal: break
        # Prefer action families not yet planned, then independent random order.
        action=min(legal,key=lambda a:(seen[a["kind"]],sum(a["action_key"]==x["action_key"] for x in chosen),priorities[a["action_key"]]))
        chosen.append(action); seen[action["kind"]]+=1; remaining=charge(remaining,action["cost_bound"])
    return chosen


def collect_case(config,root,case,graph,task,configuration,resume=False,technical_smoke=False,forced_family=None):
    from cgx.env.system import System
    from cgx.observation.actions import action_catalog,initial_budget,can_afford,charge
    from cgx.baselines.policies import interface_rule
    store=Store(root); split=case["split"]; case_id=case["case_id"]
    suffix="" if forced_family is None else "-coverage-"+forced_family
    shard_id=case_id+suffix; marker=store.completion_path(split,shard_id)
    config_hash=canonical_hash(config)
    lock_path=Path(root)/"protocol_lock.json"
    lock_hash=file_hash(lock_path) if lock_path.exists() else None
    input_hash=canonical_hash([case,graph,task,configuration,config_hash,lock_hash,technical_smoke,forced_family])
    if resume and marker.exists():
        previous=json.loads(marker.read_text())
        if previous.get("state")=="passed" and previous.get("input_hash")==input_hash:
            if file_hash(store.transition_path(split,shard_id))!=previous["transition_file_hash"]: raise ValueError("Completed transition artifact changed")
            return previous
        if previous.get("input_hash")!=input_hash: raise ValueError("Resume inputs differ from prior case job")
    marker.parent.mkdir(parents=True,exist_ok=True)
    status={"state":"running","started_at":now(),"input_hash":input_hash,"configuration_hash":config_hash,"protocol_lock_hash":lock_hash,"case_id":case_id,"split":split,"technical_smoke":technical_smoke,"pid":os.getpid(),"raw_retention":"all acquired observations losslessly compressed; binary wires deterministic stratified 2% source cases"}
    atomic_json(marker,status); started=time.monotonic(); records=[]; counts=Counter(); executions=0
    try:
        if shutil.disk_usage(root).free < 10*1024**3: raise RuntimeError("Less than10GiB free; collection stopped before disk-full")
        system=System(graph,case,task,config=config,implementation=configuration.get("implementation","A"))
        if split=="train" and not technical_smoke: _reserve_execution(root,config["training"]["max_training_executions"])
        initial=system.run(execution_index=0); executions+=1
        actions=action_catalog(graph,config); catalog_hash=canonical_hash(actions)
        target="healthy" if case["fault_family"]=="healthy" else case["fault_node"]+":"+case["fault_family"]
        retain_wire=split!="train" or seed_for("raw_retention",case["configuration_id"],case_id)%100<2 or technical_smoke
        if retain_wire: _wire(root,initial.wire)
        fractions=config["training"]["main_byte_fractions_for_collection"]
        replicates=config["training"]["replicates_per_byte_fraction"]
        specs=[(f,r) for f in fractions for r in range(replicates)] if forced_family is None else [(1.,0)]
        independent_behavior_seen=False
        fixed_support_trajectories=0
        designated=len(specs)
        for trajectory_index,(fraction,replicate) in enumerate(specs):
            fixed_support=trajectory_index>=designated
            branch_label="fixed_plan_support" if fixed_support else forced_family
            behavior_seed=seed_for(config["seeds"]["behavior_collection"],case_id,fraction,replicate,forced_family)
            rng=np.random.default_rng(behavior_seed)
            cap=int(rng.choice(config["training"]["collection_execution_caps"],p=config["training"]["collection_execution_cap_probabilities"]))
            budget=initial_budget(graph,fraction,cap,config["resources"]["main_calibration_cap"])
            history={"graph":graph,"observations":[initial.observation],"remaining":deepcopy(budget),"initial_budget":budget,"action_keys":[],"nominal_task":task,"requirements":public_requirements(config)}
            modes=list(config["training"]["collection_policy_mix"])
            behavior=str(rng.choice(modes,p=[config["training"]["collection_policy_mix"][m] for m in modes]))
            if fixed_support:
                behavior="fixed_diverse_plan";fixed_support_trajectories+=1
            independent_behavior_seen=independent_behavior_seen or behavior in ("random_legal","fixed_diverse_plan")
            plan=_fixed_diverse_plan(actions,budget,rng) if behavior=="fixed_diverse_plan" else None
            tid=canonical_hash([case_id,fraction,replicate,forced_family]); step=0; run_index=0; burst_index=0
            while True:
                old_hash=store.put_history(history)
                legal=[a for a in actions if can_afford(history["remaining"],a["cost_bound"])]
                non_stop=[a for a in legal if a["kind"]!="stop"]
                stop=next(a for a in actions if a["kind"]=="stop")
                selection_started=time.monotonic()
                if forced_family and step==0:
                    family_actions=[a for a in non_stop if a["kind"]==forced_family]
                    if not family_actions: raise ValueError(f"Graph has no affordable {forced_family} action")
                    action=family_actions[int(rng.integers(len(family_actions)))]
                elif forced_family or not non_stop: action=stop
                elif plan is not None: action=plan[step] if step<len(plan) else stop
                elif behavior=="structure_rule": action=interface_rule(history,legal)
                else: action=non_stop[int(rng.integers(len(non_stop)))]
                selection_seconds=time.monotonic()-selection_started
                before=deepcopy(history["remaining"])
                if action["kind"]=="stop":
                    new_hash=old_hash; obs_hash=None; terminal=True; actual={k:0 for k in before}; stop_reason="stop"
                else:
                    if action["cost_bound"]["executions"]:
                        if split=="train" and not technical_smoke: _reserve_execution(root,config["training"]["max_training_executions"])
                        run_index+=1; executions+=1
                    # Disturbance streams are paired by case, trajectory, and execution;
                    # action selection never changes the RNG draws of another sensor.
                    branch_offset=(10000+64*config["observations"]["action_families"].index(forced_family)) if forced_family else 0
                    ex_index=1+trajectory_index*16+run_index+branch_offset
                    result=system.acquire(action,execution_index=ex_index,burst_index=trajectory_index*16+burst_index+branch_offset)
                    if action["cost_bound"]["calibrations"]: burst_index+=1
                    actual=result.observation["actual_cost"]
                    if any(actual[k]>action["cost_bound"][k] for k in actual): raise ValueError("Physical observation exceeds declared upper cost")
                    history={**history,"observations":history["observations"]+[result.observation],"remaining":charge(before,actual),"action_keys":history["action_keys"]+[action["action_key"]]}
                    new_hash=store.put_history(history); obs_hash=store.put_observation(result.observation)
                    if retain_wire: _wire(root,result.wire)
                    terminal=not any(a["kind"]!="stop" and can_afford(history["remaining"],a["cost_bound"]) for a in actions)
                    stop_reason="resources_exhausted" if terminal else None
                records.append(dict(parent_case_id=case_id,configuration_id=case["configuration_id"],source_fold=configuration.get("source_fold"),trajectory_id=tid,step=step,
                    history_hash=old_hash,action_key=action["action_key"],action=action,acquired_blob_hash=obs_hash,next_history_hash=new_hash,resources_before=before,
                    charged_cost=actual,resources_after=deepcopy(history["remaining"]),terminal=terminal,stop_reason=stop_reason,true_target_key=target,
                    reward_bits=None,reward_reader_hash=None,available_action_catalog_hash=catalog_hash,selection_seconds=selection_seconds,
                    behavior=behavior,behavior_seed=behavior_seed,byte_fraction=fraction,execution_cap=cap,technical_smoke=technical_smoke,split=split,coverage_branch=branch_label))
                counts[action["kind"]]+=1; step+=1
                if terminal: break
                if step>cap+config["resources"]["main_calibration_cap"]+1: raise RuntimeError("Decision loop exceeded cap")
            if trajectory_index==designated-1 and forced_family is None and split=="train" and not independent_behavior_seen:
                # Keep all eight registered trajectories. Add a shared training-only
                # support branch instead of conditioning a fixed plan on rule feedback.
                specs.append((config['training'].get('fixed_plan_support_byte_fraction',config['evaluation']['primary_byte_fraction']),replicates))
        digest=store.write_transitions(split,shard_id,records)
        status.update(state="passed",transition_file_hash=digest,transitions=len(records),trajectories=len(specs),fixed_plan_support_trajectories=fixed_support_trajectories,physical_executions=executions,action_counts=dict(counts),exit_code=0)
    except BaseException as exc:
        status.update(state="failed",exit_code=1,failure_reason=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        raise
    finally:
        status.update(finished_at=now(),elapsed_seconds=time.monotonic()-started,physical_executions=executions)
        atomic_json(marker,status)
    return status


def _worker(args): return collect_case(*args)


def collect(config,root,split="train",resume=False,workers=1,technical_smoke=False,max_cases=None):
    root=Path(root)
    if split not in ("train","validation"): raise ValueError("Behavior collection cannot access a test split")
    if not technical_smoke:
        require_protocol_lock(root,config)
        if max_cases is not None: raise ValueError("Formal collection cannot silently truncate cases")
    store=Store(root); configs={r["configuration_id"]:r for r in store.configurations(split)}
    cases=list(store.cases(split)); cases=cases[:max_cases] if max_cases is not None else cases
    jobs=[(config,str(root),c,store.graph(c["configuration_id"]),store.task(c["case_id"]),configs[c["configuration_id"]],resume,technical_smoke) for c in cases]
    results=[]; started=time.monotonic(); report=root/"reports"/("collection_"+split+".json")
    def update(result):
        results.append(result)
        counts=Counter()
        for r in results: counts.update(r.get("action_counts",{}))
        atomic_json(report,{"state":"running","completed_cases":len(results),"designated_cases":len(cases),"transitions":sum(r["transitions"] for r in results),"physical_executions":sum(r["physical_executions"] for r in results),"action_counts":dict(counts),"elapsed_seconds":time.monotonic()-started,"technical_smoke":technical_smoke})
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(_worker,j) for j in jobs]
            for future in as_completed(futures): update(future.result())
    else:
        for j in jobs: update(_worker(j))
    counts=Counter()
    for r in results: counts.update(r["action_counts"])
    # Predeclared training-only branch pass, same source access for every method.
    if split=="train" and not technical_smoke:
        minimum=config["training"]["action_sampling_coverage_min_per_type"]
        for family in config["observations"]["action_families"]:
            if family=="stop": continue
            needed=max(0,minimum-counts[family])
            for c in sorted(cases,key=lambda c:canonical_hash(["coverage",family,c["case_id"]]))[:needed]:
                result=collect_case(config,root,c,store.graph(c["configuration_id"]),store.task(c["case_id"]),configs[c["configuration_id"]],resume,False,family)
                update(result); counts.update(result["action_counts"])
    summary=json.loads(report.read_text()); summary.update(state="passed",completed_cases=len(cases),coverage_branches=len(results)-len(cases),fixed_plan_support_trajectories=sum(r.get("fixed_plan_support_trajectories",0) for r in results),action_counts=dict(counts),exit_code=0)
    ledger=root/"private/training_execution_ledger.json"
    if split=="train" and not technical_smoke and ledger.exists():
        summary["successful_case_executions"]=sum(r["physical_executions"] for r in results)
        summary['imported_prior_executions']=sum(r['physical_executions'] for r in results if r.get('imported_from'))
        summary['new_attempted_executions']=json.loads(ledger.read_text())["attempted_executions"]-json.loads(ledger.read_text()).get('imported_executions',0)
        summary["physical_executions"]=summary['new_attempted_executions']+summary['imported_prior_executions']
        summary["execution_accounting"]="durable pre-execution ledger, including failed/interrupted/retried attempts"
    atomic_json(report,summary)
    return summary


def validate_data(config,root,split="train",strict=True):
    store=Store(root); configs={r["configuration_id"]:r for r in store.configurations(split)}
    case_map={c["case_id"]:c for c in store.cases(split)}; seen=set(); trajectories=Counter(); counts=Counter(); transitions=0
    for row in store.transitions(split):
        case=case_map[row["parent_case_id"]]; cfg=configs[row["configuration_id"]]
        if case["configuration_id"]!=row["configuration_id"] or row["source_fold"]!=cfg["source_fold"] or row["split"]!=split: raise ValueError("Branch/fold/split leakage")
        expected_target="healthy" if case["fault_family"]=="healthy" else case["fault_node"]+":"+case["fault_family"]
        if row["true_target_key"]!=expected_target: raise ValueError("Stored training label differs from sealed case")
        if row.get("technical_smoke"): raise ValueError("Technical smoke leaked into formal dataset")
        if row["coverage_branch"] is None: trajectories[row["parent_case_id"],row["trajectory_id"]]+=1
        for k in row["resources_before"]:
            if row["resources_after"][k]!=row["resources_before"][k]-row["charged_cost"][k] or row["resources_after"][k]<0: raise ValueError("Budget accounting mismatch")
        # All public evidence hashes are verified on strict validation, including blobs.
        if strict:
            for h in (row["history_hash"],row["next_history_hash"]):
                seen.add(h)
        counts[row["action"]["kind"]]+=1; transitions+=1
    per_case=Counter(c for c,t in trajectories)
    expected=config["training"]["source_trajectories_per_case"]
    if set(per_case)!=set(case_map) or any(v!=expected for v in per_case.values()): raise ValueError("Missing or duplicated designated trajectories")
    if split=="train" and any(counts[k]<config["training"]["action_sampling_coverage_min_per_type"] for k in config["observations"]["action_families"] if k!="stop"): raise ValueError("Action family coverage below preregistered minimum")
    if strict:
        from .validation import verify_histories
        verify_histories(root,sorted(seen),split)
    report={"state":"passed","split":split,"cases":len(per_case),"trajectories":sum(per_case.values()),"transitions":transitions,"action_counts":dict(counts),"raw_retention":"all public JSON evidence losslessly retained; wire subset deterministic 2% source by case; all validation wire","strict":strict}
    atomic_json(Path(root)/"reports"/("data_validation_"+split+".json"),report)
    return report
