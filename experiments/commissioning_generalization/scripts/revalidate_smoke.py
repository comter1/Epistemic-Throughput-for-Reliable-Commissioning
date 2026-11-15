"""Revalidate final runtime with already executed, hash-checked technical checkpoints."""
from pathlib import Path
import argparse
import os
import time
import numpy as np
from cgx.config import EXPERIMENT,load_config
from cgx.provenance import read_json,atomic_json,command_record,source_hash,file_sha256
from cgx.graphs.manifests import load_cases
from cgx.data.store import Store
from cgx.training import load_reader,load_policy
from cgx.baselines import RuntimePolicy
from cgx.evaluation.rollout import rollout
from cgx.evaluation.repairs import verification

def main(root):
    root=Path(root);original=read_json(root/"reports/technical_smoke.json")
    if original.get("state")!="passed" or not original.get("technical_smoke"):raise ValueError("An executed technical pipeline is required")
    fixture_root=Path(original["destination"]);os.environ["CGX_DATA_ROOT"]=str(fixture_root)
    config=load_config();start_hash=source_hash(EXPERIMENT);started=time.monotonic()
    with command_record(root,["python","scripts/revalidate_smoke.py","--data-root",str(root)],config):
        store=Store(fixture_root);reader=load_reader(fixture_root);independent=load_reader(fixture_root,"independent")
        cases=list(load_cases(fixture_root,"validation"));results=[];repairs=[];reads=0
        for method in config["methods"]["learned"]+["interface_rule","full_recording_feasible","default_only"]:
            model=load_policy(fixture_root,method,11) if method in config["methods"]["learned"] else None
            policy=RuntimePolicy(method,model)
            for case in cases:
                for fraction in (0.,.25):
                    row,_,history,system=rollout(store.graph(case["configuration_id"]),case,store.task(case["case_id"]),config,reader,policy,fraction,store=store)
                    if row["abstained"]:raise RuntimeError(row["errors"])
                    digest=store.put_history(history);prediction=independent.predict(store.load_history(digest))
                    assert digest==store.put_history(history) and np.isfinite(prediction["log_probabilities_base2"]).all()
                    reads+=1;results.append(row)
                    if method=="interface_rule" and fraction==.25:
                        repairs.append(verification(system,case,config,row["predicted_target_key"],1.,0.,audit=True))
        for case in cases:
            nll=[r["final_nll_bits"] for r in results if r["case_id"]==case["case_id"] and r["byte_fraction"]==0]
            assert max(nll)-min(nll)<1e-6
        if start_hash!=source_hash(EXPERIMENT):raise RuntimeError("Source changed during final runtime verification; rerun")
        original_path=fixture_root/"reports/technical_smoke.json"
        updated={**original,"source_hash":start_hash,"runtime_revalidation":{
            "state":"passed","source_hash":start_hash,"frozen_rollouts":len(results),"same_transcript_reads":reads,
            "elapsed_seconds":time.monotonic()-started,"original_pipeline_report":str(original_path),
            "original_pipeline_report_hash":file_sha256(original_path),"repair_results":repairs,
            "checkpoint_reuse":"Previously executed technical weights; all selected checkpoint bytes verified by loaders"}}
        atomic_json(root/"reports/technical_smoke.json",updated)
        print({"state":"passed","frozen_rollouts":len(results),"same_transcript_reads":reads,"elapsed_seconds":time.monotonic()-started},flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data-root",required=True);main(p.parse_args().data_root)
