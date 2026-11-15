"""Completion evidence is derived from executed artifacts, never imports."""
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from cgx.config import EXPERIMENT
from cgx.provenance import atomic_json, read_json, file_sha256, now

def legacy_check(repo, root):
    root=Path(root); repo=Path(repo)
    destination=root/"reports/legacy_integrity.json"
    subprocess.run([sys.executable,str(EXPERIMENT/"tools/snapshot_legacy.py"),"--repo",str(repo),"--out",str(destination),"--compare",str(EXPERIMENT/"reports/legacy_before.json")],check=True)
    env=__import__('os').environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    path=root/"logs/legacy_tests.log";path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w") as handle:
        result=subprocess.run([sys.executable,"-m","unittest","-v","test_benchmark.py"],cwd=repo/"finite_delay_benchmark_complete",env=env,stdout=handle,stderr=subprocess.STDOUT)
    report=read_json(destination);report.update(state="passed" if result.returncode==0 else "failed",unit_tests_exit_code=result.returncode,unit_tests_log=str(path),unit_tests_log_hash=file_sha256(path))
    atomic_json(destination,report);atomic_json(EXPERIMENT/"reports/legacy_integrity.json",report)
    if result.returncode: raise RuntimeError("Legacy tests failed; protected code unchanged")
    return report

def acceptance(root,junit=None):
    root=Path(root)
    source=(EXPERIMENT/"docs/08_ACCEPTANCE_TESTS.md").read_text()
    identifiers=re.findall(r"^([ASODLE]\d{2}):",source,re.M)
    items={key:{"state":"not_started","evidence":[]} for key in identifiers}
    junit=Path(junit or root/"reports/pytest.xml")
    if junit.exists():
        tree=ET.parse(junit)
        for test in tree.iter("testcase"):
            ids=re.findall(r"[ASODLE]\d{2}",test.attrib.get("name",""))
            state="failed" if test.find("failure") is not None or test.find("error") is not None else "blocked" if test.find("skipped") is not None else "passed"
            for key in ids:
                if key in items:
                    old=items[key]["state"]
                    items[key]["state"]="failed" if old=="failed" or state=="failed" else state
                    items[key]["evidence"].append({"test":test.attrib["name"],"junit":str(junit),"sha256":file_sha256(junit)})
    legacy=root/"reports/legacy_integrity.json"
    if legacy.exists():
        record=read_json(legacy)
        for key in ("A01","A02"):
            items[key]={"state":record["state"],"evidence":[{"path":str(legacy),"sha256":file_sha256(legacy)}]}
        items["E01"]={"state":record["state"],"evidence":[{"path":str(legacy),"sha256":file_sha256(legacy),"scope":"unchanged restricted finite regression, including actual same-library DP unit fixture"}]}
    smoke=root/"reports/technical_smoke.json"
    if smoke.exists():
        executed=read_json(smoke)
        if executed.get("state")=="passed" and executed.get("technical_smoke"):
            if executed.get("scoring",{}).get("state")=="passed" and executed["scoring"].get("transitions",0)>0:
                items["L14"]={"state":"passed","evidence":[{"path":str(smoke),"sha256":file_sha256(smoke),"scope":"actual topology-excluded scoring with training-configuration assertion for every source transition"}]}
            if executed.get("independent_same_transcript_reads",0)>0:
                items["E09"]={"state":"passed","evidence":[{"path":str(smoke),"sha256":file_sha256(smoke),"scope":"actual independent neural rereading with unchanged acquired history hashes"}]}
    summary={"updated_at":now(),"items":items,"counts":{s:sum(v["state"]==s for v in items.values()) for s in ("not_started","implemented_not_run","passed","failed","blocked")},
             "state":"passed" if all(v["state"]=="passed" for v in items.values()) else "incomplete"}
    atomic_json(root/"reports/acceptance.json",summary);atomic_json(EXPERIMENT/"reports/acceptance.json",summary)
    return summary

def report_status(root,config):
    root=Path(root)
    phases={"P0":["environment.json","reports/legacy_integrity.json"],
            "P1":["reports/qualification.json","reports/source_healthy_catalog.json"],"P2":["protocol_lock.json"],
            "P3":["reports/data_validation_train.json","reports/data_validation_validation.json"],
            "P4":["models/selection.json"],"P5":["evaluation/main/complete.json","evaluation/round_sensitivity/complete.json","evaluation/independent_reader_complete.json"],
            "P6":["evaluation/repairs/complete.json","results/hashes.json","release/complete.json"]}
    if config['protocol_id']=='CGX-2.0':
        phases['P5'].remove('evaluation/round_sensitivity/complete.json')
    status={}
    for phase,relatives in phases.items():
        evidence=[];missing=[];failed=[]
        for relative in relatives:
            path=root/relative
            if not path.exists(): missing.append(str(path));continue
            obj=read_json(path);passed=obj.get("state")=="passed" or (relative=="results/hashes.json" and bool(obj))
            evidence.append({"path":str(path),"sha256":file_sha256(path)})
            if not passed: failed.append(str(path))
        status[phase]={"state":"failed" if failed else "implemented_not_run" if missing else "passed",
                       "evidence":evidence,"missing":missing,"failed":failed}
    live=root/"main_run/status.json"
    run=read_json(live) if live.exists() else None
    if run and run.get("state")=="running" and run.get("steps",{}).get(run.get("current_step"),{}).get("state")=="running":
        step=run.get("current_step","")
        phase=("P3" if step in ("collect_train","validate_train","collect_validation","validate_validation","prepare_replay")
               else "P4" if step in ("readers_jobs","readers","crossfit_scores","policy_jobs","policies","select")
               else "P5" if step in ("freeze","main_evaluation","round_sensitivity","independent_reader") else "P6")
        status[phase]["execution_state"]="running"
        status[phase]["current_step"]=step
    summary={"updated_at":now(),"protocol_id":config["protocol_id"],"data_root":str(root),"phases":status,
             "main_experiment_completed":all(v["state"]=="passed" for v in status.values()),"main_run":run}
    summary['optional_appendix_excluded_from_completion']=config['evaluation'].get('optional_appendix_suites',[])
    atomic_json(root/"reports/status.json",summary)
    lines=["# Actual execution status","",f"Updated: {summary['updated_at']}",f"Artifact root: `{root}`","", "| Phase | Actual state | Evidence |","|---|---|---|"]
    lines += [f"| {p} | {('running (completion not yet verified)' if v.get('execution_state')=='running' else v['state'])} | {'; '.join(x['path'] for x in v['evidence']) or '; '.join(v['missing'])} |" for p,v in status.items()]
    if run: lines += ["",f"Main queue: **{run['state']}**; current step: `{run.get('current_step')}`; PID: `{run.get('pid')}`."]
    lines += ["",f"Main experiment completed: **{summary['main_experiment_completed']}**.","", "Technical smoke results are excluded from scientific estimates. Negative/unfinished outcomes remain explicit."]
    (EXPERIMENT/"STATUS.md").write_text("\n".join(lines)+"\n")
    return summary
