"""Real miniature pipeline for software qualification, excluded from research outputs.

Run after candidate formal manifests exist, before the formal protocol seal:
  python scripts/technical_smoke.py --data-root /external/cgx_data
Each invocation uses a fresh timestamped technical_smoke root and records actual work.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import argparse
import os
import time
import numpy as np
from cgx.config import load_config
from cgx.provenance import atomic_json,read_json,hash_json,file_sha256,source_hash,command_record
from cgx.config import EXPERIMENT
from cgx.graphs.manifests import read_jsonl,write_jsonl,load_cases
from cgx.data.collection import collect_case

def collect_one(arguments):
    config,root,case,graph,task,row=arguments
    return collect_case(config,root,case,graph,task,row,technical_smoke=True)

def run(root,updates=2,batch_size=2):
    root=Path(root);config=load_config();destination=root/"technical_smoke"/f"final_{time.time_ns()}"
    candidate_hash=source_hash(EXPERIMENT)
    destination.mkdir(parents=True);(destination/"assets").symlink_to(root/"assets",target_is_directory=True)
    rows=list(read_jsonl(root/"manifests/split_manifest.jsonl"))
    chosen=[next(r for r in rows if r["split"]=="train" and r["source_fold"]==fold) for fold in range(3)]
    chosen.append(next(r for r in rows if r["split"]=="validation"))
    ids={r["configuration_id"] for r in chosen};byid={r["configuration_id"]:r for r in chosen}
    graphs={r["configuration_id"]:r["graph"] for r in read_jsonl(root/"manifests/public_graphs.jsonl") if r["configuration_id"] in ids}
    allcases=[c for split in ("train","validation") for c in load_cases(root,split) if c["configuration_id"] in ids]
    cases=[next(c for c in allcases if c["configuration_id"]==row["configuration_id"] and c["fault_family"]==family) for row in chosen for family in config["targets"]["families"]]
    caseids={c["case_id"] for c in cases}
    tasks={r["case_id"]:r["task"] for r in read_jsonl(root/"manifests/tasks.jsonl") if r["case_id"] in caseids}
    write_jsonl(destination/"manifests/split_manifest.jsonl",chosen)
    write_jsonl(destination/"manifests/public_graphs.jsonl",[{"configuration_id":k,"graph":v} for k,v in graphs.items()])
    write_jsonl(destination/"manifests/tasks.jsonl",[{"case_id":k,"task":v} for k,v in tasks.items()])
    write_jsonl(destination/"private/private_cases.jsonl",cases)
    for split in ("train","validation"):
        write_jsonl(destination/"private/cases"/(split+".jsonl"),[c for c in cases if c["split"]==split])
    lock={"state":"passed","technical_smoke":True,"config_hash":hash_json(config),"source_hash":source_hash(EXPERIMENT),
          "files":{},"excluded_from_scientific_tables":True}
    atomic_json(destination/"protocol_lock.json",lock)
    os.environ["CGX_DATA_ROOT"]=str(destination)
    started=time.monotonic()
    with command_record(destination,["python","scripts/technical_smoke.py","--data-root",str(root),'--updates',str(updates),'--batch-size',str(batch_size)],config):
        args=[(config,destination,c,graphs[c["configuration_id"]],tasks[c["case_id"]],byid[c["configuration_id"]]) for c in cases]
        with ProcessPoolExecutor(max_workers=4) as pool: collection=list(pool.map(collect_one,args))
        from cgx.training import fit_readers,score_transitions,fit_policy,load_reader,load_policy,prepare_replay
        from cgx.evaluation.matrix import validation_callback
        from cgx.evaluation.rollout import rollout
        from cgx.evaluation.repairs import verification
        from cgx.baselines import RuntimePolicy
        from cgx.data.store import Store
        prepared=prepare_replay(destination,config,workers=4)
        readers=fit_readers(destination,config,technical_smoke=True,technical_updates=updates,technical_batch_size=batch_size)
        scored=score_transitions(destination,config,technical_smoke=True)
        policies={method:fit_policy(destination,config,method,11,technical_smoke=True,validation_fn=validation_callback(destination,config),technical_updates=updates,technical_batch_size=batch_size) for method in config["methods"]["learned"]}
        reader=load_reader(destination);independent=load_reader(destination,"independent");store=Store(destination)
        evaluation=[];repairs=[];independent_count=0
        for method in config["methods"]["learned"]+["interface_rule","full_recording_feasible","default_only"]:
            model=load_policy(destination,method,11) if method in config["methods"]["learned"] else None
            policy=RuntimePolicy(method,model)
            for case in [c for c in cases if c["split"]=="validation"]:
                for fraction in (0.,.25):
                    row,trace,history,system=rollout(graphs[case["configuration_id"]],case,tasks[case["case_id"]],config,reader,policy,fraction,store=store)
                    if row["abstained"]:raise RuntimeError(f"Technical rollout failed: {row['errors']}")
                    digest=store.put_history(history);prediction=independent.predict(store.load_history(digest))
                    assert digest==store.put_history(history) and np.isfinite(prediction["log_probabilities_base2"]).all()
                    independent_count+=1;row.update(method=method,technical_smoke=True);evaluation.append(row)
                    if method=="interface_rule" and fraction==.25:
                        # Threshold0 exercises ordinary repair/public verification/audit mechanics only.
                        outcome=verification(system,case,config,row["predicted_target_key"],1.,0.,audit=True)
                        repairs.append(outcome)
        zero=[r["final_nll_bits"] for r in evaluation if r["byte_fraction"]==0]
        for case in [c for c in cases if c["split"]=="validation"]:
            values=[r["final_nll_bits"] for r in evaluation if r["case_id"]==case["case_id"] and r["byte_fraction"]==0]
            assert max(values)-min(values)<1e-6
        write_jsonl(destination/"reports/technical_evaluation.jsonl",evaluation)
        if source_hash(EXPERIMENT)!=candidate_hash:
            raise RuntimeError('Source changed during technical qualification; preserve artifacts and rerun')
        report={"state":"passed","technical_smoke":True,"excluded_from_scientific_tables":True,
                "source_hash":source_hash(EXPERIMENT),"destination":str(destination),"config_hash":hash_json(config),
                "source_cases":12,"validation_cases":4,"collection_transitions":sum(x["transitions"] for x in collection),
                "collection_executions":sum(x["physical_executions"] for x in collection),"reader_runs":readers,"policy_runs":policies,
                "technical_updates_per_model":updates,"technical_batch_size":batch_size,
                "training_throughput":{str(p.parent.relative_to(destination/'models')):read_json(p)[-1] for p in (destination/'models').rglob('validation.json')},
                "scoring":scored,"prepared_features":prepared,"frozen_rollouts":len(evaluation),"independent_same_transcript_reads":independent_count,
                "repair_results":repairs,"elapsed_seconds":time.monotonic()-started,
                "artifacts":{str(p.relative_to(destination)):file_sha256(p) for p in (destination/"models").rglob("complete.json")}}
        atomic_json(destination/"reports/technical_smoke.json",report)
        atomic_json(root/"reports/technical_smoke.json",report)
        print({k:report[k] for k in ("state","destination","frozen_rollouts","collection_executions","elapsed_seconds")},flush=True)
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument('--updates',type=int,default=2);parser.add_argument('--batch-size',type=int,default=2)
    a=parser.parse_args();run(a.data_root,a.updates,a.batch_size)
