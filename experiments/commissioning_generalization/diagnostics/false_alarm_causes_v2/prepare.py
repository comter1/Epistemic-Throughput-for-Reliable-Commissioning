"""Protect prior experiments and seal V2 samples without opening new outcomes."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json
import subprocess
import tarfile

import numpy as np
import pandas as pd
from cgx.config import EXPERIMENT,load_config
from cgx.env.system import generate_task
from cgx.graphs.manifests import read_jsonl
from cgx.provenance import atomic_json,read_json,file_sha256,hash_json
from cgx.seeds import seed_for
from .ledger import Ledger

FAMILIES=['healthy','cache_stale','queue_hold','adapter_scale']
METHODS=['gnn_adaptive','gnn_depth2'];SEEDS=[11,23,37]


def make_cases(config,graphs,selected,purpose,count,folds):
    output=[]
    for ci,cid in enumerate(selected):
        graph=graphs[cid]
        families=([f for f in FAMILIES for _ in range(count//4)] if purpose!='confirmation'
                  else ['healthy']*4+[FAMILIES[1+(ci+j)%3] for j in range(4)])
        occurrences={f:0 for f in FAMILIES}
        for j,family in enumerate(families):
            occurrence=occurrences[family];occurrences[family]+=1
            ident=hash_json(['CGX-FA-CAUSE-V2',671930,purpose,cid,family,j])
            streams={name:seed_for('CGX-FA-CAUSE-V2',ident,name) for name in ('task','plant','measurement','clock','fault')}
            # The original stream key schema is inserted by prepare below.
            ntype={'cache_stale':'cache','queue_hold':'queue','adapter_scale':'adapter'}.get(family)
            eligible=sorted((n['id'] for n in graph['nodes'] if n['type']==ntype),key=lambda x:hash_json([purpose,cid,family,x]))
            node=eligible[(ci+occurrence)%len(eligible)] if eligible else None
            if family=='adapter_scale':levels=[s*m for m in config['faults']['adapter_scale_magnitudes'] for s in config['faults']['adapter_scale_signs']]
            elif family=='cache_stale':levels=config['faults']['cache_stale_delay_ticks']
            elif family=='queue_hold':levels=config['faults']['queue_hold_delay_ticks']
            else:levels=[None]
            severity=levels[(ci+occurrence)%len(levels)]
            rng=np.random.default_rng(streams['clock'])
            offsets=dict(zip(graph['clock_domains'],map(float,rng.uniform(*config['sensing']['observer_offset_uniform_ms'],len(graph['clock_domains'])))))
            output.append(dict(case_id=ident,configuration_id=cid,split=('train' if purpose=='source_train' else 'validation' if purpose=='source_validation' else 'test_connections'),
                purpose=purpose,source_fold=folds.get(cid),fault_family=family,fault_node=node,severity=severity,
                observer_offsets_ms=offsets,streams=streams,example_only=False,investigation_only=True))
    return output


def prepare(root,v1,run):
    root,v1,run=Path(root),Path(v1),Path(run)
    if (run/'source_checks.json').exists():raise RuntimeError('V2 already sealed; refuse overwrite')
    for name in ('manifests','provenance','logs','commands','tests','figures'): (run/name).mkdir(parents=True,exist_ok=True)
    config=load_config();protected={};mismatches=[]
    old=read_json(v1/'INPUT_MANIFEST.json')
    for relative,expected in old['original_hashes'].items():
        path=root/relative;actual=file_sha256(path);protected[str(path)]=actual
        if actual!=expected:mismatches.append(str(path))
    v1_package=read_json(v1/'PACKAGE_MANIFEST.json')
    for item in v1_package['files']:
        path=v1/item['path'];actual=file_sha256(path);protected[str(path)]=actual
        if actual!=item['sha256']:mismatches.append(str(path))
    for path in (v1/'PACKAGE_MANIFEST.json',v1/'SHA256SUMS',root/'private/private_cases.jsonl',EXPERIMENT/'configs/protocol.json'):
        protected[str(path)]=file_sha256(path)
    source_before={}
    with tarfile.open(run/'provenance/workspace_before_v2.tar.gz','w:gz') as archive:
        for path in sorted(EXPERIMENT.rglob('*')):
            if not path.is_file() or any(x in path.parts for x in ('__pycache__','.pytest_cache','false_alarm_causes_v2')):continue
            relative=str(path.relative_to(EXPERIMENT));source_before[relative]=file_sha256(path);archive.add(path,arcname=relative)
    v1_code=[]
    for path in (v1/'software/diagnostics/false_alarm_causes').rglob('*.py'):
        relative=path.relative_to(v1/'software');current=EXPERIMENT/relative
        same=current.exists() and file_sha256(path)==file_sha256(current)
        v1_code.append(dict(path=str(relative),same_as_v1=same))
        if not same:mismatches.append(str(current))
    assert not mismatches,f'Changed prior files: {mismatches}'
    atomic_json(run/'provenance/protected_before.json',protected)
    atomic_json(run/'provenance/source_before.json',source_before)
    original_cases=list(read_jsonl(root/'private/private_cases.jsonl'))
    splits=list(read_jsonl(root/'manifests/split_manifest.jsonl'))
    graphs={x['configuration_id']:x['graph'] for x in read_jsonl(root/'manifests/public_graphs.jsonl')}
    train=sorted((x['configuration_id'] for x in splits if x['split']=='train'),key=lambda c:hash_json(['CGX-FA-CAUSE-V2:source-config',c]))[:16]
    validation=sorted(x['configuration_id'] for x in splits if x['split']=='validation')
    test=sorted(x['configuration_id'] for x in splits if x['split']=='test_connections')
    folds={x['configuration_id']:int(x['source_fold']) for x in splits if x['split']=='train'}
    folds.update({cid:int(hash_json(['CGX-FA-CAUSE-V2:validation-reader',cid])[:16],16)%3 for cid in validation})
    source=make_cases(config,graphs,train,'source_train',16,folds)+make_cases(config,graphs,validation,'source_validation',8,folds)
    fresh=make_cases(config,graphs,test,'confirmation',8,folds)
    stream_names=list(original_cases[0]['streams'])
    for case in source+fresh:
        case['streams']={name:seed_for('CGX-FA-CAUSE-V2',case['case_id'],name) for name in stream_names}
        rng=np.random.default_rng(case['streams']['clock']);g=graphs[case['configuration_id']]
        case['observer_offsets_ms']=dict(zip(g['clock_domains'],map(float,rng.uniform(*config['sensing']['observer_offset_uniform_ms'],len(g['clock_domains'])))))
    v1cases=read_json(v1/'manifests/fresh_cases.json')
    old_ids={c['case_id'] for c in original_cases+v1cases}
    assert len(source)==320 and len(fresh)==128 and not old_ids&{c['case_id'] for c in source+fresh}
    old_tasks={c['streams']['task'] for c in original_cases+v1cases}
    assert not old_tasks&{c['streams']['task'] for c in source+fresh}
    assert len({c['case_id'] for c in source+fresh})==448
    # Check OOF ownership from real frozen checkpoint metadata, no inference.
    from cgx.training.engine import load_reader
    reader_configs={}
    for name in ('crossfit_0','crossfit_1','crossfit_2','final'):
        reader=load_reader(root,name,'cpu');reader_configs[name]=list(reader.training_configurations)
    for case in source:
        assert case['configuration_id'] not in reader_configs[f"crossfit_{case['source_fold']}"]
        case['final_reader_seen_configuration']=case['configuration_id'] in reader_configs['final']
        case['all_readers_seen_case']=False
    atomic_json(run/'manifests/source_cases.json',source)
    atomic_json(run/'manifests/source_tasks.json',{c['case_id']:generate_task(c['streams']['task']) for c in source})
    atomic_json(run/'manifests/source_configurations.json',dict(train=train,validation=validation,reader_assignments=folds,
        training_configuration_selection='16 smallest domain-separated manifest hashes; independent of outcomes'))
    atomic_json(run/'manifests/fresh_cases.json',fresh)
    atomic_json(run/'manifests/fresh_tasks.json',{c['case_id']:generate_task(c['streams']['task']) for c in fresh})
    atomic_json(run/'manifests/fresh_confirmation_sealed.json',dict(cases=fresh,cases_sha256=file_sha256(run/'manifests/fresh_cases.json'),
        tasks_sha256=file_sha256(run/'manifests/fresh_tasks.json'),generation_seed=671930,case_count=128,
        healthy_count=64,faulty_count=64,new_case_and_task_streams_verified=True,
        topologies='same original16test_connections, already examined inV1',outcomes_opened=False,
        live_feedback_policy='Optional only if substantive R* qualifies and remaining worst-case budget permits; decision sealed before fresh outcomes',
        variants_selection_pending=True))
    jobs=[dict(case_id=c['case_id'],configuration_id=c['configuration_id'],method_key=m,policy_seed=s,
               source_artifact=str(v1/'interventions/confirmation/cases'/f"{hash_json([c['case_id'],m,s,'A'])}.json.gz")) for c in v1cases for m in METHODS for s in SEEDS]
    atomic_json(run/'manifests/existing_case_reader_probes.json',dict(v1_run=str(v1),cases=v1cases,jobs=jobs,
        population='all128V1freshcases, bothprimarymethods, all3seeds, everyactualprefix',selection_independent_of_outcome=True))
    variants=dict(task='CGX-FA-CAUSE-V2',sealed_at=datetime.now(timezone.utc).isoformat(),methods=METHODS,seeds=SEEDS,
        resource_limits=Ledger.LIMITS,compute=dict(host_port=2404,gpu_indices=[0,1],cpu_workers_up_to=64,local_role='coordination/code only'),
        reader_groups=dict(P='local andexternal GRU pooledvectors',C=[23,24,25,31],E='matched-time/calibration edges includingpresence'),
        reader_candidates=['original','R-P','R-C','R-PC'],reader_selection='one shared rule across bothmethods and3seeds; sourcevalidation equalconfig meanjointNLL',
        guardrails=dict(healthy_false_alarm_increase_max=0.,fault_miss_increase_max=.02,fault_joint_accuracy_decrease_max=.02,tie_nll_bits=.005),
        reader_ties='single modification first, then measuredprocessingcost, then R-P before R-C; includeoriginal in losscomparison',
        head_cells=dict(T00=dict(q='OOF',label='OOF'),T01=dict(q='FINAL',label='OOF'),T10=dict(q='OOF',label='FINAL'),T11=dict(q='FINAL',label='FINAL')),
        head_training=dict(fits=24,updates=2000,batch_size=128,optimizer='AdamW',learning_rate=1e-4,weight_decay=1e-4,
            loss='Huber delta1',snapshots=[0,500,1000,1500,2000],sampling='uniform configuration,then case,then actionableprefix',matched_batch_seed=671931),
        head_selection='permethod commoncell+checkpoint across3seeds; guardrails then equalconfig finaljointNLL; within0.005bits fewerexecutions,earliercheckpoint,lexicographiccell',
        no_qualified_head='T11 at2000, labeled notselectedasdeployableimprovement',gate_threshold=0.,
        bootstrap=dict(seed=671932,repetitions=2000,unit='caseafterseedaverage; pairedconfigthen cases,stratifiedhealthy/fault'),
        significance='effectsizes and confidenceintervals; no significance claims or p-valueselection',
        optional_noise_branches='not scheduled; requiredstages takepriority',
        new_features='diagnostic hooks / score-head corrections only; original sourceandweights immutable')
    atomic_json(run/'manifests/variants.json',variants)
    atomic_json(run/'PROTOCOL_FA_CAUSE_V2.json',variants)
    required=['FINDINGS_CN.md','training_audit.md','training_reward_checks.json','input_path_checks.md','input_probe_outcome_summary.csv',
        'replay_fixed_summary.csv','fresh_case_confirmation.csv','confirmation_per_seed.csv','confirmation_per_configuration.csv','confirmation_arm_summary.csv',
        'action_value_summary.csv','q_input_action_checks.csv']
    assert all((v1/x).exists() for x in required)
    rows=pd.read_parquet(v1/'fresh_intervention_outcomes.parquet');facts=[]
    expected={('gnn_adaptive','A'):(62,39,105),('gnn_adaptive','S'):(21,27,126),('gnn_depth2','A'):(47,28,117),('gnn_depth2','S'):(21,27,126)}
    for (method,arm),(fa,miss,joint) in expected.items():
        d=rows[(rows.method_key==method)&(rows.arm==arm)];h=d[d.is_healthy];f=d[~d.is_healthy]
        assert (len(h),len(f),int(h.healthy_false_alarm.sum()),int(f.fault_miss.sum()),int(f.correct_joint.sum()))==(192,192,fa,miss,joint)
        facts.append(dict(method=method,arm=arm,healthy_denominator=192,fault_denominator=192,false_alarms=fa,misses=miss,joint_correct=joint))
    checks=dict(state='passed',original_data_root=str(root),v1_run=str(v1),repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=EXPERIMENT,text=True).strip(),
        v1_file_hashes_verified=len(v1_package['files']),original_and_v1_protected_files=len(protected),
        source_files_before=len(source_before),v1_code_comparison=v1_code,v1_counts_reproduced=facts,
        v1_required_artifacts=[dict(path=x,sha256=file_sha256(v1/x)) for x in required],
        source_cases=320,confirmation_cases=128,old_case_collision_count=0,old_task_seed_collision_count=0,
        actual_source_configs=train,validation_configs=validation,source_OOF_exclusion_verified=True,
        active_config_sha256=file_sha256(EXPERIMENT/'configs/protocol.json'),request_sha256=file_sha256(run/'REQUEST.md'),
        manifest_hashes={str(p.relative_to(run)):file_sha256(p) for p in (run/'manifests').glob('*.json')})
    atomic_json(run/'source_checks.json',checks)
    Ledger(run).reserve('cache_hit',0,'initialization')
    print(json.dumps({k:checks[k] for k in ('state','v1_file_hashes_verified','source_cases','confirmation_cases','source_OOF_exclusion_verified')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--v1',required=True);p.add_argument('--run',required=True)
    a=p.parse_args();prepare(a.root,a.v1,a.run)
