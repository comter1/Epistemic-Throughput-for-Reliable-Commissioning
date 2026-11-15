"""Real V2 source collection and frozen CPU evaluation throughput, engineering only."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import os
import time
from cgx.config import load_config,EXPERIMENT
from cgx.provenance import atomic_json,read_json,command_record,source_hash,file_sha256
from cgx.graphs.manifests import load_cases,read_jsonl
from cgx.data.collection import collect_case


def collect_one(args):
    return collect_case(*args,technical_smoke=True)


def main(root,fixture,workers):
    root=Path(root);fixture=Path(fixture);config=load_config()
    destination=root/'technical_smoke'/f'capacity_{time.time_ns()}';destination.mkdir(parents=True)
    (destination/'assets').symlink_to(root/'assets',target_is_directory=True)
    os.environ['CGX_DATA_ROOT']=str(destination)
    rows={r['configuration_id']:r for r in read_jsonl(root/'manifests/split_manifest.jsonl')}
    graphs={r['configuration_id']:r['graph'] for r in read_jsonl(root/'manifests/public_graphs.jsonl')}
    tasks={r['case_id']:r['task'] for r in read_jsonl(root/'manifests/tasks.jsonl')}
    cases=list(load_cases(root,'train'))
    # One of each fault family from all32 source graphs; deterministic manifest order.
    selected=[next(c for c in cases if c['configuration_id']==cid and c['fault_family']==family)
        for cid,row in rows.items() if row['split']=='train' for family in config['targets']['families']]
    with command_record(root,['python','scripts/profile_v2.py','--data-root',str(root),'--fixture',str(fixture),'--workers',str(workers)],config):
        collection_cases=[c for i,c in enumerate(selected) if (i//4)%2==0];start=time.monotonic()
        args=[(config,destination,c,graphs[c['configuration_id']],tasks[c['case_id']],rows[c['configuration_id']]) for c in collection_cases]
        with ProcessPoolExecutor(max_workers=workers) as pool:collected=list(pool.map(collect_one,args,chunksize=1))
        seconds=time.monotonic()-start
        collection=dict(cases=len(collected),transitions=sum(r['transitions'] for r in collected),
            physical_executions=sum(r['physical_executions'] for r in collected),elapsed_seconds=seconds,
            cases_per_second=len(collected)/seconds,transitions_per_second=sum(r['transitions'] for r in collected)/seconds)
        print({'collection':collection},flush=True)
        from cgx.training import load_reader,load_policy
        from cgx.evaluation.parallel import rollout_pool,map_rollouts
        reader=load_reader(fixture,'final',device='cpu');evaluations=[]
        for method in config['methods']['main_learned_evaluation']+['full_recording_feasible']:
            model=None if method=='full_recording_feasible' else load_policy(fixture,'gnn_myopic' if method=='cost_aware_adaptation' else method,11,device='cpu')
            jobs=[(graphs[c['configuration_id']],c,tasks[c['case_id']],1.,4,'A',False) for c in selected]
            start=time.monotonic()
            with rollout_pool(reader,model,method,config,root/'assets/mujoco_menagerie/franka_emika_panda',workers=workers,store_root=destination) as pool:
                evaluated=[row for row,_ in map_rollouts(pool,jobs)]
            seconds=time.monotonic()-start
            entry=dict(method=method,cases=len(evaluated),elapsed_seconds=seconds,cases_per_second=len(evaluated)/seconds,
                extra_physical_executions=sum(r['executions_used'] for r in evaluated),abstentions=sum(r['abstained'] for r in evaluated),
                checkpoint_hash=model.checkpoint_hash if model is not None else None)
            atomic_json(destination/'reports'/f'{method}.json',dict(technical_smoke=True,rows=evaluated,measurement=entry))
            evaluations.append(entry);print(entry,flush=True)
        report=dict(state='passed',technical_smoke=True,excluded_from_scientific_tables=True,source_hash=source_hash(EXPERIMENT),
            workers=workers,destination=str(destination),fixture=str(fixture),collection=collection,evaluation=evaluations,
            reader_checkpoint_hash=reader.checkpoint_hash,manifest_hash=file_sha256(root/'manifests/hashes.json'),
            boundary='Source topologies only. Technical checkpoint weights, not a research result. Includes worker startup, actual acquisition, encoding/inference, lossless evidence storage. Future trained policies can choose different actions.')
        atomic_json(root/'reports/v2_throughput.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--fixture',required=True);p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();main(a.data_root,a.fixture,a.workers)
