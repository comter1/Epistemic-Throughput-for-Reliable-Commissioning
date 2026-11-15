"""Execute every registered healthy training task before the formal protocol seal."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import time
from cgx.config import load_config
from cgx.provenance import atomic_json,command_record,hash_json,file_sha256
from cgx.graphs.manifests import load_cases,read_jsonl


def one(args):
    from cgx.env.system import System
    graph,case,task,config,asset=args
    result=System(graph,case,task,config,asset).run()
    return dict(case_id=case['case_id'],configuration_id=case['configuration_id'],**result.metrics)


def main(root,workers):
    root=Path(root);config=load_config();start=time.monotonic()
    graphs={r['configuration_id']:r['graph'] for r in read_jsonl(root/'manifests/public_graphs.jsonl')}
    cases=[c for c in load_cases(root,'train') if c['fault_family']=='healthy'];ids={c['case_id'] for c in cases}
    tasks={r['case_id']:r['task'] for r in read_jsonl(root/'manifests/tasks.jsonl') if r['case_id'] in ids}
    with command_record(root,['python','scripts/qualify_source.py','--data-root',str(root),'--workers',str(workers)],config):
        jobs=[(graphs[c['configuration_id']],c,tasks[c['case_id']],config,root/'assets/mujoco_menagerie/franka_emika_panda') for c in cases]
        with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(one,jobs,chunksize=4))
        report=dict(state='passed' if all(r['task_pass'] for r in results) else 'failed',technical_smoke=True,
            criterion='Every selected healthy source case reaches all three waypoints without limits/contact violations',
            executions=len(results),configuration_hash=hash_json(config),
            source_cases_hash=file_sha256(root/'private/cases/train.jsonl'),results=results,
            workers=workers,elapsed_seconds=time.monotonic()-start)
        atomic_json(root/'reports/source_healthy_catalog.json',report)
        if report['state']!='passed':raise RuntimeError('Healthy source catalog failed; do not filter cases')
        print({k:v for k,v in report.items() if k!='results'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();main(a.data_root,a.workers)
