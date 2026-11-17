"""CPU workers; evaluator labels never enter the policy API."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from .protocol import Ledger, write_json, digest, scientific_hashes
from .cases import policy_view
from .physics import PayloadPlant
from .trajectories import b0, library, scaled, work
from .workstations import kinematic
from .prediction import PublicPredictionModel, Clock
from payload_change_v1.estimator import calibration_record

def context(root,case):
    root=Path(root)
    return (json.loads((root/'configs/workstations.json').read_text())[case['context']],
            json.loads((root/'configs/healthy_graph.json').read_text()))

def stream(case,seed,tag='main'):
    return int(digest(['V2',case['case_id'],int(seed),tag])[:15],16)

def new_plant(root,case,seed,configured=None,tag='main'):
    ctx,graph=context(root,case)
    return PayloadPlant(case['true_payload'],configured or case['configured_payload'],stream(case,seed,tag),ctx,
                        graph=graph,ledger=Ledger(root,'plant'))

def save_block(root,plant,action,path,kind):
    result=plant.execute(action,str(path),kind=kind)
    result.save(Path(root)/path)
    return result

def profile(root,case):
    ctx,_=context(root,case); p=new_plant(root,case,5,tag='profile')
    rows=[]; cpu=time.process_time(); wall=time.perf_counter()
    for i,action in enumerate([b0(),library()[0]]):
        b=save_block(root,p,action,Path('raw/profile')/case['case_id']/str(i),'profile')
        rows.append(dict(action=action,metrics=b.private['metrics'],seconds=b.physical_seconds,
                         wall_seconds=b.metadata['wall_seconds']))
    physical_cpu=time.process_time()-cpu; predictor=PublicPredictionModel(root,ctx,case['configured_payload'])
    pcpu=time.process_time()
    k=predictor.kernel(case['configured_payload'],work(ctx),clock=Clock(30))
    return dict(blocks=rows,physical_cpu_seconds=physical_cpu,plant_seconds=sum(r['seconds'] for r in rows),
                prediction_cpu_seconds=time.process_time()-pcpu,prediction_sim_seconds=float(k['duration']),
                wall_seconds=time.perf_counter()-wall,nominal_prediction_peak_m=float(np.max(np.linalg.norm(k['ee'],axis=1))))

def source(root,case):
    ctx,_=context(root,case); p=new_plant(root,case,5,tag='source')
    rng=np.random.default_rng(int(digest(['source_order',case['case_id']])[:8],16))
    actions=[b0()]+[scaled(library()[int(j)], [1.,.75,.5][i%3]) for i,j in enumerate(rng.permutation(10))]
    logs=[]; rows=[]
    for i,a in enumerate(actions):
        check=kinematic(case['configured_payload'],ctx,a)
        if not check['admissible']:
            rows.append(dict(index=i,action=a,rejected='public_kinematics',screen=check)); continue
        b=save_block(root,p,a,Path('raw/source')/case['case_id']/f'block_{i:02d}','source_acquisition')
        logs.append(b.public|{'execution_id':b.metadata['execution_id']})
        rows.append(dict(index=i,action=a,metrics=b.private['metrics'],seconds=b.physical_seconds,
                         path=str(Path('raw/source')/case['case_id']/f'block_{i:02d}'),screen=check))
        if b.private['metrics']['hard_abort']: break
    calibration=calibration_record(logs,case['true_payload'])
    refs=[]
    for speed in [1.,.5]:
        p=new_plant(root,case,5,configured=case['true_payload'],tag=f'qualification_{speed}')
        a=work(ctx,speed)
        b=save_block(root,p,a,Path('raw/source_qualification')/case['case_id']/str(speed),'source_true_configuration_qualification')
        refs.append(dict(speed=speed,metrics=b.private['metrics'],seconds=b.physical_seconds))
    return dict(case_id=case['case_id'],object_id=case['object_id'],object_index=case['object_index'],
                context=case['context'],blocks=rows,calibration=calibration,qualification=refs)

def run_job(root,job):
    root=Path(root); stage=job['stage']; key=job['key']
    out=root/'jobs'/stage/(key+'.json'); code=digest(scientific_hashes())
    if out.exists():
        previous=json.loads(out.read_text())
        if (previous.get('status')=='complete' and previous.get('input_hash')==digest(job)
                and (not job.get('require_code_hash') or previous.get('code_hash')==code)):
            return str(out)
    budget=Ledger(root,'cpu'); reservation=budget.reserve(600.,unit=key,kind=stage)
    start=time.process_time(); wall=time.perf_counter()
    try:
        if stage in ('profile','source'): result=globals()[stage](root,job['case'])
        elif stage=='fit_case':
            from .fitting import fit_case
            result=fit_case(root,job['case'])
        elif stage in ('validation','evaluation','reference'):
            from .evaluate import execute_job
            result=execute_job(root,job)
        elif stage=='branches':
            from .branch_checks import execute
            result=execute(root,job['case'])
        elif stage=='source_ablation':
            from .source_analysis import execute
            result=execute(root,job['case'])
        else: raise ValueError(stage)
        cpu=time.process_time()-start
        budget.complete(reservation,cpu)
        write_json(out,dict(status='complete',input_hash=digest(job),code_hash=code,result=result,
                            cpu_seconds=cpu,wall_seconds=time.perf_counter()-wall))
    except BaseException as e:
        cpu=time.process_time()-start
        budget.complete(reservation,cpu,'failed')
        write_json(root/'failures'/stage/(key+f'_{time.time_ns()}.json'),dict(status='failed',job=job,
                    error=str(e),traceback=traceback.format_exc(),cpu_seconds=cpu,code_hash=code))
        raise
    return str(out)

def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--jobs',required=True)
    ap.add_argument('--workers',type=int,required=True); a=ap.parse_args()
    jobs=json.loads(Path(a.jobs).read_text()); errors=[]
    import socket
    quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
    allocated=int(quota)//int(period) if quota!='max' else os.cpu_count()
    a.workers=min(a.workers,max(1,allocated))
    write_json(Path(a.root)/'resources'/(Path(a.jobs).stem+'_allocation.json'),dict(
        hostname=socket.gethostname(),allocated_cores=allocated,workers=a.workers,
        memory_limit_bytes=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
        memory_current_bytes=int(Path('/sys/fs/cgroup/memory.current').read_text()),
        blas_threads=int(os.environ.get('OPENBLAS_NUM_THREADS','1'))))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run_job,a.root,j):j for j in jobs}
        for i,f in enumerate(as_completed(futures)):
            try: f.result()
            except BaseException as e: errors.append(dict(job=futures[f],error=str(e))); print(traceback.format_exc(),flush=True)
            if (i+1)%10==0 or i+1==len(jobs): print(json.dumps(dict(completed=i+1,total=len(jobs),errors=len(errors))),flush=True)
    if errors: raise RuntimeError(f'{len(errors)} jobs failed; retained under failures/')

if __name__=='__main__': main()
