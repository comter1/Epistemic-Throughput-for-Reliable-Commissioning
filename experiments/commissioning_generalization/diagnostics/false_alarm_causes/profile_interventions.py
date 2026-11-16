"""Engineering-only parity and component timers; never select by outcome."""
import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import pandas as pd
from cgx.provenance import atomic_json, read_json
from cgx.observation.actions import action_catalog, feasible_actions
from .core import reconstruct_prefixes
from .interventions import Runner, METHODS


def run(root,run_dir):
    directory=Path(run_dir);case=read_json(directory/'manifests/discovery_cases.json')[0]
    cpu=Runner(root,run_dir,'cpu');gpu=Runner(root,run_dir,'cuda:2')
    rows=[];timings=[]
    for method in METHODS:
        item=cpu.artifact(case,method,11)
        prefixes=reconstruct_prefixes(root,item)
        for index in sorted(set((0,len(prefixes)-1))):
            history=prefixes[index][0]
            q,_=cpu.predict(history,'engineering:reader')
            actions=feasible_actions(action_catalog(history['graph'],cpu.config),history['remaining'])
            a=cpu.score(cpu.policy(method,11),history,actions,q,'engineering')
            hook_records=defaultdict(float);starts={};handles=[]
            model=gpu.policy(method,11)
            modules=[('GRU',model.backbone.evidence.gru),('candidate_action_head',model.action_head)]
            modules += [('graph_layer_'+str(i),layer) for i,layer in enumerate(model.backbone.layers)]
            def before(name):
                def hook(*args):gpu.sync();starts[name]=time.perf_counter()
                return hook
            def after(name):
                def hook(*args):gpu.sync();hook_records[name]+=time.perf_counter()-starts[name]
                return hook
            for name,module in modules:
                handles.extend((module.register_forward_pre_hook(before(name)),module.register_forward_hook(after(name))))
            b=gpu.score(model,history,actions,q,'engineering')
            for handle in handles:handle.remove()
            cpu.ledger.reserve('feature_build',detail='engineering:matching_microbenchmark')
            start=time.perf_counter();groups=[];matches=[]
            for observation in history['observations']:
                grouped=defaultdict(list);matched=defaultdict(list)
                for record in observation['records']:
                    grouped[record['port']].append(record);matched[record['message_token']].append(record)
                groups.append(grouped);matches.append(matched)
            matching=time.perf_counter()-start
            start=time.perf_counter();pairs=0
            for grouped,matched in zip(groups,matches):
                for port,records in grouped.items():
                    for record in records:
                        for peer in matched[record['message_token']]:
                            if peer['port']!=port:pairs+=1
            matching+=time.perf_counter()-start
            from cgx.models.features import _bin_sequence
            directions={n['id']+':'+p['name']:(-1 if p['direction']=='in' else 1) for n in history['graph']['nodes'] for p in n['ports']}
            start=time.perf_counter()
            for grouped,matched in zip(groups,matches):
                for port,records in grouped.items():_bin_sequence(records,matched,directions.get(port,1))
            binning=time.perf_counter()-start
            start=time.perf_counter();deepcopy(history);copy_seconds=time.perf_counter()-start
            rows.append({'method_key':method,'prefix':index,'actions':len(actions),'max_abs_score_difference':float(np.max(np.abs(a-b))),
                         'same_argmax':bool(a.argmax()==b.argmax()),'cpu_top_gap':float(np.sort(a)[-1]-np.sort(a)[-2]) if len(a)>1 else None})
            for stage,seconds in dict(hook_records,message_matching_microbenchmark=matching,time_sequence_binning=binning,history_copy=copy_seconds).items():
                timings.append({'method_key':method,'prefix':index,'stage':stage,'seconds':seconds,'device':'cuda:2' if stage in hook_records else 'cpu',
                                'scope':'engineering_microbenchmark','overlap_note':'binning includes peer-delta lookup; matching microbenchmark overlaps binning and is not additive'})
    history,_=gpu.initial(case,METHODS[0],11)
    action=next(a for a in action_catalog(history['graph'],gpu.config) if a['ports'] and not a['calibration_pair'])
    start=time.perf_counter();gpu.acquire(gpu.system(case,history),action,1,0,'engineering:physical')
    timings.append({'method_key':METHODS[0],'prefix':0,'stage':'actual_simulation_and_system_setup','seconds':time.perf_counter()-start,'device':'cpu','scope':'engineering_microbenchmark'})
    pd.DataFrame(rows).to_csv(directory/'policy_cpu_gpu_parity.csv',index=False)
    pd.DataFrame(timings).to_csv(directory/'intervention_runtime_stage_breakdown.csv',index=False)
    result={'state':'passed' if all(r['same_argmax'] and r['max_abs_score_difference']<1e-3 for r in rows) else 'failed',
            'rows':rows,'cpu_single_thread':True,'precision':'FP32','scientific_labels_not_used':True,
            'max_simulation_upper_bound_calls':15360,'profile_scope':'two policy architectures, two fixed positions; module timers, one real acquisition'}
    atomic_json(directory/'intervention_engineering_profile.json',result)
    print(result,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();run(args.root,args.run_dir)
