"""Import actually completed, byte-identical source cases into the V2 sealed corpus."""
from pathlib import Path
import os
from cgx.provenance import read_json,atomic_json,hash_json,file_sha256,require_protocol_lock,verify_hashes,now
from cgx.graphs.manifests import load_cases,read_jsonl
from .store import Store


def import_cases(source,destination,config):
    source=Path(source);destination=Path(destination);require_protocol_lock(destination,config)
    old=read_json(source/'software/protocol.json')
    old_lock=read_json(source/'protocol_lock.json')
    if old_lock.get('state')!='passed' or old_lock.get('technical_smoke') or old_lock['config_hash']!=hash_json(old):
        raise ValueError('Original formal protocol provenance is invalid')
    verify_hashes(old_lock['files'],source)
    for key in ('system','faults','sensing','observations'):
        if old[key]!=config[key]:raise ValueError(f'Incompatible data-generating protocol: {key}')
    collection_keys=('source_trajectories_per_case','main_byte_fractions_for_collection','replicates_per_byte_fraction',
        'collection_execution_caps','collection_execution_cap_probabilities','collection_policy_mix')
    if any(old['training'][k]!=config['training'][k] for k in collection_keys):raise ValueError('Acquisition-history collection changed')
    old_support=old['training'].get('fixed_plan_support_byte_fraction',old['evaluation']['primary_byte_fraction'])
    new_support=config['training'].get('fixed_plan_support_byte_fraction',config['evaluation']['primary_byte_fraction'])
    if old_support!=new_support:raise ValueError('Fixed-plan support collection changed')
    for key in ('case_generation','behavior_collection','crossfit_assignment','manifest'):
        if old['seeds'][key]!=config['seeds'][key]:raise ValueError('Source RNG contract changed')
    original=Store(source);store=Store(destination)
    configs={r['configuration_id']:r for r in store.configurations()}
    old_configs={r['configuration_id']:r for r in original.configurations()};imported=[];counts={}
    for split in ('train','validation'):
        old_cases={c['case_id']:c for c in load_cases(source,split)};total=0
        for case in load_cases(destination,split):
            cid=case['case_id'];path=original.completion_path(split,cid)
            if not path.exists():continue
            prior=read_json(path)
            if prior.get('state')!='passed' or prior.get('technical_smoke'):continue
            if old_cases.get(cid)!=case:raise ValueError('Case truth/seeds changed; cannot reuse its observations')
            graph=store.graph(case['configuration_id']);task=store.task(cid);configuration=configs[case['configuration_id']]
            if graph!=original.graph(case['configuration_id']) or task!=original.task(cid):raise ValueError('Graph/task changed')
            original_input=hash_json([case,graph,task,old_configs[case['configuration_id']],hash_json(old),file_sha256(source/'protocol_lock.json'),False,None])
            if prior.get('input_hash')!=original_input:raise ValueError('Original shard input identity does not match its sealed protocol')
            shard=original.transition_path(split,cid)
            if file_sha256(shard)!=prior['transition_file_hash']:raise ValueError('Original case shard corrupt')
            target=store.transition_path(split,cid);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if file_sha256(target)!=prior['transition_file_hash']:raise ValueError('Destination differs from imported evidence')
            else:os.link(shard,target)
            lock_hash=file_sha256(destination/'protocol_lock.json');config_hash=hash_json(config)
            result={**prior,'input_hash':hash_json([case,graph,task,configuration,config_hash,lock_hash,False,None]),
                'configuration_hash':config_hash,'protocol_lock_hash':lock_hash,
                'imported_from':str(path),'source_marker_hash':file_sha256(path),'imported_at':now(),
                'source_data_configuration_hash':prior['configuration_hash'],'executed_in_this_run':False}
            marker=store.completion_path(split,cid)
            if marker.exists():
                if read_json(marker)['input_hash']!=result['input_hash']:raise ValueError('Changed reuse identity')
            else:atomic_json(marker,result)
            imported.append({'split':split,'case_id':cid,'source_marker_hash':file_sha256(path),'transition_hash':prior['transition_file_hash'],
                'transitions':prior['transitions'],'previously_executed_runs':prior['physical_executions']});total+=1
        counts[split]=total
    report={'state':'passed','imported_at':now(),'source_root':str(source),'source_protocol_hash':file_sha256(source/'protocol_lock.json'),
        'destination_protocol_hash':file_sha256(destination/'protocol_lock.json'),'cases_by_split':counts,
        'selection':'Pre-registered topology and balanced case manifest; no observed outcome filtering',
        'fresh_executions_in_import':0,'imports':imported}
    prior_executions=sum(r['previously_executed_runs'] for r in imported if r['split']=='train')
    ledger=destination/'private/training_execution_ledger.json'
    if not ledger.exists():atomic_json(ledger,{'attempted_executions':prior_executions,'imported_executions':prior_executions})
    elif read_json(ledger).get('imported_executions',0)!=prior_executions:raise ValueError('Reused execution accounting changed')
    atomic_json(destination/'reports/reused_data.json',report);return report
