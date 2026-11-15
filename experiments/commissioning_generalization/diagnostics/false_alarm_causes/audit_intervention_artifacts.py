"""Independent artifact-level pairing, real-prefix, resource and prediction audit."""
import argparse
from pathlib import Path
import json
from cgx.data.store import Store
from cgx.provenance import read_json,hash_json,atomic_json
from .interventions import METHODS,SEEDS,load_blob


def run(run_dir,phase):
    run_dir=Path(run_dir);store=Store(run_dir)
    cases=read_json(run_dir/'manifests'/('fresh_cases.json' if phase=='confirmation' else 'discovery_cases.json'))
    arms=['A','S'] if phase=='confirmation' else ['A','B','C','D']
    failures=[];groups=rows=prefix_checks=0
    for case in cases:
        for method in METHODS:
            for seed in SEEDS:
                identity=[case['case_id'],method,seed]
                try:
                    items={arm:load_blob(run_dir/'interventions'/phase/'cases'/f'{hash_json(identity+[arm])}.json.gz') for arm in arms}
                    assert len({v['row']['checkpoint_hash'] for v in items.values()})==1,'paired policy differs'
                    assert len({v['row']['diagnostic_checkpoint_hash'] for v in items.values()})==1,'paired reader differs'
                    assert len({v['initial_observation_ref'] for v in items.values()})==1,'paired initial observation differs'
                    for arm,item in items.items():
                        row=item['row'];trace=item['trace'];packed=store.history_references(row['evidence_trace_hash'])
                        assert row['state']=='passed','failed row retained'
                        assert len(packed['observation_refs'])==len(trace)+1,'acquired observation count differs'
                        assert packed['observation_refs'][0]==item['initial_observation_ref'],'initial reference differs'
                        assert packed['observation_refs'][1:]==[t['observation_ref'] for t in trace],'actual prefix sequence differs'
                        assert packed['action_keys']==[t['action']['action_key'] for t in trace],'chosen action sequence differs'
                        expected=trace[-1]['resources_after'] if trace else packed['initial_budget']
                        assert packed['remaining']==expected,'remaining differs from actual acquired endpoint'
                        previous=packed['initial_budget']
                        for tr in trace:
                            assert tr['resources_before']==previous,'resource prefix is discontinuous'
                            assert all(0<=tr['resources_after'][k]<=previous[k] for k in previous),'resource cap exceeded'
                            previous=tr['resources_after']
                        assert row['prediction']==(trace[-1]['prediction'] if trace else row['initial_prediction']),'returned q is stale'
                        for name in ('executions','bytes','calibrations','decisions'):
                            assert row[name+'_used']==packed['initial_budget'][name]-packed['remaining'][name],'reported use differs'
                        rows+=1
                    if phase=='discovery':
                        if items['A']['trace'] and items['B']['trace']:
                            assert items['A']['trace'][0]['action']==items['B']['trace'][0]['action'],'A/B initial input gives different first action'
                        plan=read_json(run_dir/'manifests/calibration_plans'/f'{hash_json(identity)}.json')
                        n=len(plan['actions'])
                        for arm in ('C','D'):
                            assert [t['action'] for t in items[arm]['trace'][:n]]==plan['actions'],'prepaid calibration differs from sealed plan'
                        assert [t['observation_ref'] for t in items['C']['trace'][:n]]==[t['observation_ref'] for t in items['D']['trace'][:n]],'paired calibrations not identical'
                    else:
                        a=items['A'];s=items['S']
                        first=next((i+1 for i,t in enumerate(a['trace']) if t['action']['ports']),len(a['trace']))
                        assert s['trace']==a['trace'][:first],'S differs from the nonanticipatory actual first-execution prefix'
                        assert s['row']['end_to_end_seconds'] is None,'S standalone time fabricated'
                        assert not s['independently_resampled'],'S mislabeled independent'
                        assert s['row']['additional_model_forward_calls']==s['row']['additional_robot_execution_attempts']==0,'S introduces untracked work'
                        prefix_checks+=1
                    groups+=1
                except Exception as exc:failures.append({'identity':identity,'error':type(exc).__name__+':'+str(exc)})
    result={'phase':phase,'state':'passed' if not failures else 'failed','expected_groups':len(cases)*len(METHODS)*len(SEEDS),
            'groups_checked':groups,'rows_checked':rows,'S_actual_prefix_equality_checks':prefix_checks,'failures':failures,
            'uses_model_or_feature_calls':False,'history_content_address_and_observation_reference_order_checked':True,
            'raw_record_blobs_not_reloaded_in_this_metadata_audit':True}
    atomic_json(run_dir/(phase+'_artifact_audit.json'),result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--phase',choices=['discovery','confirmation'],required=True)
    args=parser.parse_args();run(args.run_dir,args.phase)
