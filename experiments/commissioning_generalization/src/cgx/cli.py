"""Executable CGX command interface. Missing prerequisites are fatal, never success."""
from pathlib import Path
import argparse
import json
import os
import sys
from .config import load_config, data_root, default_workers, EXPERIMENT
from .provenance import command_record

COMMANDS=("doctor","legacy-check","qualify-system","make-manifests","validate-manifests","lock-protocol",
          "collect","validate-data","prepare-replay","fit-readers","score-transitions","fit-policy","build-job-manifest","run-queue",
          "select-checkpoints","freeze-models","evaluate","evaluate-round-sensitivity","reread","evaluate-repairs",
          "aggregate","report","export","acceptance","run-main","import-compatible-data")

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=COMMANDS)
    p.add_argument("--config",type=Path,default=EXPERIMENT/"configs/protocol.json")
    p.add_argument("--data-root",type=Path)
    p.add_argument("--repo",type=Path,default=EXPERIMENT.parents[1])
    p.add_argument("--report",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument('--source-root',type=Path)
    p.add_argument("--split",choices=("train","validation"),default="train")
    p.add_argument("--suite",default="all")
    p.add_argument("--phase",choices=("readers","policies"),default="policies")
    p.add_argument("--method")
    p.add_argument("--seed",type=int)
    p.add_argument("--reader",default="independent")
    p.add_argument("--only")
    p.add_argument("--jobs",type=Path)
    p.add_argument("--gpus",default="0,1")
    p.add_argument("--workers",type=int,default=default_workers())
    p.add_argument("--bootstrap",type=int,default=2000)
    for flag in ("resume","strict","source-fixtures","crossfit","final","independent","crossfit-only","validation-only",
                 "same-transcripts","preregistered-subset","paired","include-negative-results","technical-smoke"):
        p.add_argument("--"+flag,action="store_true")
    return p

def execute(a,root,config):
    c=a.command
    if c=="doctor":
        from .runtime.hardware import doctor
        return doctor(root,config)
    if c=="legacy-check":
        from .runtime.status import legacy_check
        return legacy_check(a.repo,root)
    if c=="qualify-system":
        from .env.qualify import qualify_system
        return qualify_system(config,root,a.report)
    if c=="make-manifests":
        from .graphs.manifests import make_manifests
        return make_manifests(config,root,workers=a.workers)
    if c=='import-compatible-data':
        from .data.reuse import import_cases
        if a.source_root is None:raise ValueError('--source-root required')
        return import_cases(a.source_root,root,config)
    if c=="validate-manifests":
        from .graphs.manifests import validate_manifests
        return validate_manifests(config,root,strict=a.strict)
    if c=="lock-protocol":
        from .runtime.locking import lock_protocol
        return lock_protocol(root,config)
    if c in ("collect","validate-data"):
        from .data.collection import collect,validate_data
        return collect(config,root,a.split,a.resume,a.workers,technical_smoke=a.technical_smoke) if c=="collect" else validate_data(config,root,a.split,a.strict)
    if c=="fit-readers":
        from .training import fit_readers
        return fit_readers(root,config,crossfit=True,final=True,independent=True,resume=a.resume,technical_smoke=a.technical_smoke,only=a.only)
    if c=="prepare-replay":
        from .training import prepare_replay
        return prepare_replay(root,config,workers=a.workers,resume=a.resume)
    if c=="score-transitions":
        from .training import score_transitions
        if not a.crossfit_only: raise ValueError("--crossfit-only is required")
        return score_transitions(root,config,crossfit_only=True,resume=a.resume,technical_smoke=a.technical_smoke)
    if c=="fit-policy":
        from .training import fit_policy
        from .evaluation.matrix import validation_callback
        if a.method not in config["methods"]["learned"] or a.seed not in config["seeds"]["training"]:
            raise ValueError("Registered --method and --seed required")
        return fit_policy(root,config,a.method,a.seed,resume=a.resume,technical_smoke=a.technical_smoke,validation_fn=validation_callback(root,config))
    if c=="build-job-manifest":
        from .runtime.queue import build_jobs
        return build_jobs(root,config,a.phase,a.out)
    if c=="run-queue":
        from .runtime.queue import run_queue
        if a.jobs is None: raise ValueError("--jobs required")
        return run_queue(root,a.jobs,a.gpus.split(","),a.resume)
    if c=="select-checkpoints":
        from .evaluation.selection import select_checkpoints
        if not a.validation_only: raise ValueError("--validation-only required")
        return select_checkpoints(root,config)
    if c=="freeze-models":
        from .evaluation.matrix import freeze_models
        return freeze_models(root,config)
    if c in ("evaluate","evaluate-round-sensitivity"):
        from .evaluation.matrix import evaluate
        return evaluate(root,config,a.suite,a.resume,c=="evaluate-round-sensitivity")
    if c=="reread":
        from .evaluation.matrix import reread
        if a.reader!="independent" or not a.same_transcripts: raise ValueError("--reader independent --same-transcripts required")
        return reread(root,config)
    if c=="evaluate-repairs":
        from .evaluation.repairs import evaluate_repairs
        if not a.preregistered_subset: raise ValueError("--preregistered-subset required")
        return evaluate_repairs(root,config,a.resume)
    if c=="aggregate":
        from .reporting.results import aggregate
        if not a.paired or a.bootstrap!=config["evaluation"]["bootstrap_repetitions"]: raise ValueError("Registered paired bootstrap required")
        return aggregate(root,config,a.bootstrap)
    if c=="report":
        from .runtime.status import report_status
        from .reporting.results import figures
        result=report_status(root,config)
        if (root/"results/per_seed.csv").exists(): result["figures"]=figures(root,config)
        return result
    if c=="export":
        from .reporting.results import export
        if a.out is None: raise ValueError("--out required")
        return export(root,a.out)
    if c=="acceptance":
        from .runtime.status import acceptance
        return acceptance(root)
    if c=="run-main":
        from .runtime.pipeline import run_main
        return run_main(root,config,a.workers,a.gpus,a.resume)
    raise RuntimeError("Command dispatcher incomplete")

def main(argv=None):
    a=parser().parse_args(argv)
    config=load_config(a.config);root=data_root(a.data_root)
    os.environ["CGX_DATA_ROOT"]=str(root)
    os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("MKL_NUM_THREADS","1");os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    with command_record(root,[sys.executable,"-m","cgx.cli",*(argv if argv is not None else sys.argv[1:])],config) as record:
        result=execute(a,root,config)
        record["result"]=result
        print(json.dumps(result,sort_keys=True,default=str),flush=True)

if __name__=="__main__": main()
