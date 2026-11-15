"""Real serial/parallel CPU trajectory equivalence, using technical fixtures only."""
from pathlib import Path
import argparse
import time
import os
import numpy as np
from cgx.config import load_config,EXPERIMENT
from cgx.provenance import atomic_json,read_json,command_record,source_hash
from cgx.data.store import Store
from cgx.graphs.manifests import load_cases
from cgx.training import load_reader,load_policy
from cgx.baselines import RuntimePolicy
from cgx.evaluation.rollout import rollout
from cgx.evaluation.parallel import rollout_pool,map_rollouts


def main(base,out,workers):
    base=Path(base);out=Path(out);config=load_config()
    fixture=Path(read_json(base/'reports/technical_smoke.json')['destination'])
    os.environ['CGX_DATA_ROOT']=str(fixture)
    store=Store(fixture);cases=list(load_cases(fixture,'validation'))
    reader=load_reader(fixture,'final',device='cpu')
    reports=[];serial_seconds=parallel_seconds=0.;count=0
    with command_record(out,['python','scripts/qualify_parallel.py','--base',str(base),'--out',str(out),'--workers',str(workers)],config):
        for method in config['methods']['learned']+['interface_rule','full_recording_feasible','default_only']:
            model=load_policy(fixture,method,11,device='cpu') if method in config['methods']['learned'] else None
            policy=RuntimePolicy(method,model);jobs=[(store.graph(c['configuration_id']),c,store.task(c['case_id']),fraction,4,'A',False) for c in cases for fraction in (0.,.25)]
            start=time.monotonic();serial=[]
            for graph,case,task,fraction,cap,implementation,unrestricted in jobs:
                row,trace,_,_=rollout(graph,case,task,config,reader,policy,fraction,cap,implementation,fixture/'assets/mujoco_menagerie/franka_emika_panda')
                serial.append((row,trace))
            duration=time.monotonic()-start;serial_seconds+=duration;start=time.monotonic()
            with rollout_pool(reader,model,method,config,fixture/'assets/mujoco_menagerie/franka_emika_panda',workers=workers) as pool:
                parallel=list(map_rollouts(pool,jobs))
            pduration=time.monotonic()-start;parallel_seconds+=pduration
            for (a,at),(b,bt) in zip(serial,parallel,strict=True):
                assert not a['abstained'] and not b['abstained'],(method,a['errors'],b['errors'])
                for field in ('case_id','predicted_target_key','bytes_used','executions_used','calibrations_used','decisions_used','evidence_trace_hash','timeout'):
                    assert a[field]==b[field],(method,field,a[field],b[field])
                assert abs(a['final_nll_bits']-b['final_nll_bits'])<1e-6
                assert [t['action']['action_key'] for t in at]==[t['action']['action_key'] for t in bt]
                count+=1
            reports.append(dict(method=method,cases=len(jobs),serial_seconds=duration,parallel_seconds=pduration))
            print(reports[-1],flush=True)
        result=dict(state='passed',technical_smoke=True,excluded_from_scientific_tables=True,source_hash=source_hash(EXPERIMENT),
            serial_parallel_pairs=count,workers=workers,serial_seconds=serial_seconds,parallel_seconds=parallel_seconds,
            boundary='Includes new worker startup per method; tiny fixture is not a formal throughput estimate',methods=reports)
        atomic_json(out/'reports/parallel_rollouts.json',result);print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();main(a.base,a.out,a.workers)
