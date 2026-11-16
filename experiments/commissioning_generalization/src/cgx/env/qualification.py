"""Source-only deterministic engineering fixtures, never diagnostic method selection."""
from __future__ import annotations
import json
import gzip
from pathlib import Path
import time
import numpy as np
from .system import System, generate_task
from ..graphs import generate_graph
from ..observation.actions import action_catalog

def healthy_case(graph, seed):
    rng=np.random.default_rng(seed)
    return {'fault_family':'healthy','fault_node':None,'severity':None,
            'observer_offsets_ms':{d:float(rng.uniform(-20,20)) for d in graph['clock_domains']},
            'streams':{'noise':seed,'task':seed+1}}

def qualify_system(config=None, task_count=16):
    from ..observation.protocol import require_supported
    config=require_supported(config)
    seed=config['seeds']['healthy_qualification']
    rows=[]
    for motif,n in [('chain',10),('single',12),('double',16)]:
        graph=generate_graph(seed,n,motif,2)
        for k in range(task_count):
            case=healthy_case(graph,seed+k)
            result=System(graph,case,generate_task(seed+100+k),config).run()
            rows.append({'fixture':motif,'task_index':k,**result.metrics})
    return {'technical_smoke':True,'status':'passed' if all(r['task_pass'] for r in rows) else 'failed',
            'criterion':'all 3 targets within 0.02 m for final 0.5 seconds; no limits/contact violations',
            'executions':len(rows),'results':rows}

def profile_executions(count=1000, config=None):
    """Representative recording workload; reports actual bytes and steady CPU time."""
    from ..observation.protocol import require_supported
    config=require_supported(config)
    graph=generate_graph(config['seeds']['healthy_qualification'],12,'single',2)
    case=healthy_case(graph,config['seeds']['healthy_qualification'])
    system=System(graph,case,generate_task(case['streams']['task']),config)
    actions=action_catalog(graph)
    kinds=['single_port','module_pair','link_pair','pair_plus_calibration','full_trace']
    selected=[next(a for a in actions if a['kind']==k and a['sample_hz']==r) for k in kinds for r in (10,50,100)]
    rows=[]
    for i in range(count):
        action=selected[i%len(selected)]
        result=system.acquire(action,execution_index=i,burst_index=i)
        storage_start=time.perf_counter()
        encoded=json.dumps(result.observation,sort_keys=True,separators=(',',':')).encode()
        compressed=gzip.compress(encoded,compresslevel=1,mtime=0)
        compressed_wire=gzip.compress(result.wire,compresslevel=1,mtime=0)
        storage_seconds=time.perf_counter()-storage_start
        rows.append({'index':i,'action_kind':action['kind'],'rate':action['sample_hz'],
                     'json_bytes':len(encoded),'gzip_json_bytes':len(compressed),
                     'gzip_wire_bytes':len(compressed_wire),'storage_seconds':storage_seconds,
                     'wire_bytes':len(result.wire),
                     'wall_seconds':sum(m.get('wall_seconds',0) for m in result.metrics.values()),
                     'record_count':len(result.observation['records'])})
    timings=np.array([r['wall_seconds'] for r in rows])
    return {'technical_smoke':True,'executions':count,'summary':{
        'mean_seconds':float(timings.mean()),'p50_seconds':float(np.quantile(timings,.5)),
        'p95_seconds':float(np.quantile(timings,.95)),
        'mean_wire_bytes':float(np.mean([r['wire_bytes'] for r in rows])),
        'mean_gzip_json_bytes':float(np.mean([r['gzip_json_bytes'] for r in rows])),
        'mean_gzip_wire_bytes':float(np.mean([r['gzip_wire_bytes'] for r in rows])),
        'mean_storage_seconds':float(np.mean([r['storage_seconds'] for r in rows])),
        'total_wire_bytes':sum(r['wire_bytes'] for r in rows)},'rows':rows}
