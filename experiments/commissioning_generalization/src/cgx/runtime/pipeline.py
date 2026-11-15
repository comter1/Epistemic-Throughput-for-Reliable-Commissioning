"""Sequential prerequisite-gated main run; survives the launching shell disconnecting."""
from pathlib import Path
import fcntl
import os
import subprocess
import sys
import time
from cgx.config import EXPERIMENT
from cgx.provenance import atomic_json, read_json, require_protocol_lock, hash_json, now, file_sha256

def run_main(root,config,workers=16,gpus="0,1",resume=False):
    try:
        return _run_main(root,config,workers,gpus,resume)
    except BaseException as exc:
        path=Path(root)/"main_run/status.json"
        if path.exists():
            status=read_json(path)
            # A second driver failing the lock must not overwrite the active owner's status.
            if status.get("pid")==os.getpid():
                status.update(state="failed",finished_at=now(),failure_reason=f"{type(exc).__name__}: {exc}")
                step=status.get("steps",{}).get(status.get("current_step"))
                if step and step.get("state")=="running":
                    step.update(state="failed",failure_reason=status["failure_reason"])
                atomic_json(path,status)
        raise

def _run_main(root,config,workers=16,gpus="0,1",resume=False):
    if config['protocol_id']=='CGX-2.0':
        os.environ.setdefault('CGX_EVAL_WORKERS','16')
        os.environ.setdefault('CGX_DATA_WORKERS',str(workers))
    root=Path(root);lock=require_protocol_lock(root,config)
    steps=[
        ("collect_train",["collect","--split","train","--workers",str(workers),"--resume"]),
        ("validate_train",["validate-data","--split","train","--strict"]),
        ("collect_validation",["collect","--split","validation","--workers",str(workers),"--resume"]),
        ("validate_validation",["validate-data","--split","validation","--strict"]),
        ("prepare_replay",["prepare-replay","--workers",str(workers),"--resume"]),
        ("readers_jobs",["build-job-manifest","--phase","readers","--out",str(root/"jobs/readers.jsonl")]),
        ("readers",["run-queue","--gpus",gpus,"--jobs",str(root/"jobs/readers.jsonl"),"--resume"]),
        ("crossfit_scores",["score-transitions","--crossfit-only","--resume"]),
        ("policy_jobs",["build-job-manifest","--phase","policies","--out",str(root/"jobs/policies.jsonl")]),
        ("policies",["run-queue","--gpus",gpus,"--jobs",str(root/"jobs/policies.jsonl"),"--resume"]),
        ("select",["select-checkpoints","--validation-only"]),
        ("freeze",["freeze-models"]),
        ("main_evaluation",["evaluate","--suite","all","--resume"]),
        ("round_sensitivity",["evaluate-round-sensitivity","--resume"]),
        ("independent_reader",["reread","--reader","independent","--same-transcripts"]),
        ("repairs",["evaluate-repairs","--preregistered-subset","--resume"]),
        ("aggregate",["aggregate","--paired","--bootstrap",str(config["evaluation"]["bootstrap_repetitions"])]),
        ("report",["report","--include-negative-results"]),
        ("export",["export","--out",str(root/"release/cgx_main_results.zip")]),
    ]
    if config['protocol_id']=='CGX-2.0':
        # Appendix horizon/clock/implementation experiments cannot block main results.
        steps=[step for step in steps if step[0]!='round_sensitivity']
    for _,args in steps:
        args += ['--config',str(root/'software/protocol.json')]
    directory=root/"main_run";directory.mkdir(parents=True,exist_ok=True)
    with (directory/"driver.lock").open("a") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status_path=directory/"status.json"
        status={"state":"running","pid":os.getpid(),"started_at":now(),"protocol_hash":hash_json(lock),
                "steps":{name:{"state":"not_started","command":[sys.executable,"-m","cgx.cli",*args,"--data-root",str(root)]} for name,args in steps}}
        if status_path.exists():
            previous=read_json(status_path)
            if not resume: raise FileExistsError("Main run exists; use --resume")
            if previous["protocol_hash"]!=status["protocol_hash"]: raise ValueError("Main run protocol changed")
            for prior_step in previous.get("steps",{}).values():
                if prior_step.get("state")=="running" and prior_step.get("pid"):
                    try: os.kill(prior_step["pid"],0)
                    except ProcessLookupError: pass
                    else: raise RuntimeError("A recorded phase process is still alive; refusing a duplicate main run")
            status["steps"]=previous["steps"]
        atomic_json(status_path,status)
        for name,args in steps:
            step=status["steps"][name]
            # Reinvoke completed stages too. Their strict resumable readers verify
            # underlying data/checkpoint/result artifacts, not just this driver flag.
            # This can repeat validation, but cannot bless a missing/corrupt output.
            require_protocol_lock(root,config)
            step.update(state="running",started_at=now(),log=str(directory/f"{name}.log"))
            status["current_step"]=name;atomic_json(status_path,status)
            from .status import report_status
            report_status(root,config)
            print(f"{now()} START {name}",flush=True)
            began=time.monotonic()
            env=dict(os.environ,CGX_DATA_ROOT=str(root),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONUNBUFFERED="1")
            with (directory/f"{name}.log").open("a") as handle:
                process=subprocess.Popen(step["command"],cwd=EXPERIMENT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                step["pid"]=process.pid;atomic_json(status_path,status)
                code=process.wait()
            step.update(state="passed" if code==0 else "failed",exit_code=code,finished_at=now(),elapsed_seconds=time.monotonic()-began,
                        log_hash=file_sha256(directory/f"{name}.log"))
            atomic_json(status_path,status)
            from .status import report_status
            report_status(root,config)
            if code:
                status.update(state="failed",finished_at=now());atomic_json(status_path,status)
                raise RuntimeError(f"Main phase {name} failed; later phases not run. See {step['log']}")
            print(f"{now()} PASSED {name}",flush=True)
        status.update(state="passed",finished_at=now());atomic_json(status_path,status)
        report_status(root,config)
        return status
