"""Import fixed SED predictions and measure recorded-port coverage without inference."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from cgx.config import load_config
from cgx.data.store import Store
from cgx.evaluation.design import main_methods,conditions
from cgx.provenance import atomic_json,file_sha256,hash_json,read_json,now
from structured_evidence_v1.runtime_audit import cell_key

PORT_FIELD=re.compile(rb'"port"\s*:\s*("(?:\\.|[^"\\])*")')


def _profile_cell(job):
    path,key,frozen_hash=job;item=read_json(path);row=item['row'];trace=item['trace']
    assert item['state']=='passed' and item['frozen_hash']==frozen_hash
    assert item['result_hash']==hash_json(dict(row=row,trace=trace)) and cell_key(row)==key
    acquired=[step for step in trace if step.get('evidence_hash') and step['action']['kind']!='stop']
    actions=[step['action'] for step in acquired];families=Counter(a['kind'] for a in actions)
    return dict(cell_key=key,case_id=row['case_id'],configuration_id=row['configuration_id'],
        method=row['method'],execution_cap=row['execution_cap'],policy_seed=row['policy_seed'],
        full_trace_used=any(a['kind']=='full_trace' for a in actions),
        full_trace_action_count=families.get('full_trace',0),action_family_counts=dict(families),
        sample_rates=[a['sample_hz'] for a in actions if a['ports']],
        chosen_action_keys=[a['action_key'] for a in actions],
        requested_ports=sorted({p for a in actions for p in a['ports']}),
        observation_refs=[step['evidence_hash'] for step in acquired],
        imported_trace_sha256=file_sha256(path),saved_result_hash=item['result_hash'])


def _observation_ports(row):
    path=Path(row['resolved_path']);blob=path.read_bytes()
    assert len(blob)==row['compressed_bytes']
    assert hashlib.sha256(blob).hexdigest()==row['compressed_sha256'],str(path)
    # These are already schema-validated observation objects. Only record objects
    # have a port field; opaque tokens/action keys and numeric payloads cannot
    # contain another JSON port member. Decode unique quoted strings, not values.
    payload=gzip.decompress(blob)
    ports=sorted(json.loads(s) for s in set(PORT_FIELD.findall(payload)))
    return dict(observation_ref=row['content_address'],acquired_ports=ports,
        compressed_sha256=row['compressed_sha256'],resolved_path=str(path),compressed_bytes=len(blob))


def import_rows(source_root,sed_run,run_dir,workers=64):
    source,sed,run=map(Path,(source_root,sed_run,run_dir));out=run/'evaluation'
    marker=out/'import_complete.json'
    if marker.exists():
        saved=read_json(marker)
        for name,digest in saved['output_hashes'].items():assert file_sha256(out/name)==digest
        return saved
    inventory=read_json(run/'source_inventory.json');assert inventory['state']=='passed'
    reader_hash=inventory['reader_hash'];config=load_config();store=Store(source)
    old=sed/'evaluation/main/per_case_results.parquet';adjusted=sed/'runtime_audit/full_matrix_adjusted.parquet'
    for name in ('evaluation/main/per_case_results.parquet','runtime_audit/full_matrix_adjusted.parquet'):
        assert file_sha256(sed/name)==inventory['inputs'][name]['sha256'],name
    official=pd.read_parquet(old);completed=pd.read_parquet(adjusted)
    assert len(official)==len(completed)==147200
    assert set(official.reader_hash)==set(completed.reader_hash)=={reader_hash}
    expected={hash_json([method,seed,cap,fraction,case['case_id']])
        for split in ('test_seen','test_connections','test_size') for case in store.cases(split)
        for method,seed in main_methods(config) for cap,fraction in conditions(config,method)}
    for frame in (official,completed):
        frame['cell_key']=[cell_key(r) for r in frame.to_dict('records')]
        assert len(set(frame.cell_key))==147200 and set(frame.cell_key)==expected
        assert int((frame.suite=='test_connections').sum())==73600
    assert set(official.case_id)==set(completed.case_id) and official.case_id.nunique()==3200
    frozen=read_json(sed/'models/frozen.json');fh=hash_json(frozen)
    for (method,seed),part in official.groupby(['method','policy_seed'],dropna=False):
        if method=='full_recording_feasible':continue
        trained='gnn_myopic' if method=='cost_aware_adaptation' else method
        record=frozen['inventory'][f'{trained}_seed{int(seed)}']
        expected_hash=frozen['files'][f"models/policies/{trained}_seed{int(seed)}/{record['checkpoint']}"]
        assert set(part.checkpoint_hash)=={expected_hash}
    profile_path=out/'imported_action_profile.parquet';started=time.monotonic()
    if profile_path.exists():
        profiles=pd.read_parquet(profile_path).to_dict('records')
        assert {p['cell_key'] for p in profiles}==expected
    else:
        jobs=[(str(sed/'evaluation/main/cases'/f'{key}.json'),key,fh) for key in sorted(expected)]
        with ProcessPoolExecutor(max_workers=min(workers,32),mp_context=mp.get_context('spawn')) as pool:
            profiles=list(pool.map(_profile_cell,jobs,chunksize=64))
        pd.DataFrame(profiles).to_parquet(profile_path,index=False)
    needed={str(ref) for profile in profiles for ref in profile['observation_refs']}
    index={}
    with gzip.open(sed/'evidence_export/new_main/blobs.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line)
            if row['kind']=='observations' and row['content_address'] in needed:index[row['content_address']]=row
    assert set(index)==needed,'Existing evidence export cannot resolve imported acquired observations'
    port_path=out/'imported_observation_ports.parquet'
    if port_path.exists():
        port_rows=pd.read_parquet(port_path).to_dict('records');assert {r['observation_ref'] for r in port_rows}==needed
    else:
        # Independently compare the field-only extraction to complete JSON on a
        # hash-selected set before using the optimization on the full raw store.
        checks=[]
        for key in sorted(needed,key=lambda k:hash_json(['port-extraction-check',k]))[:32]:
            item=_observation_ports(index[key])
            with gzip.open(index[key]['resolved_path'],'rt') as f:parsed=json.load(f)
            exact=sorted({record['port'] for record in parsed['records']})
            assert item['acquired_ports']==exact
            checks.append(dict(observation_ref=key,exact_ports=exact,compressed_sha256=item['compressed_sha256']))
        atomic_json(run/'tests/imported_port_extraction.json',dict(state='passed',cases=len(checks),checks=checks,
            physical_acquisitions=0,reader_calls=0,actual_record_fields_not_requested_ports=True))
        port_rows=[]
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
            for i,item in enumerate(pool.map(_observation_ports,index.values(),chunksize=32),1):
                port_rows.append(item)
                if i%4096==0:
                    atomic_json(out/'import_progress.json',dict(state='running',stage='actual_recorded_ports',completed=i,expected=len(index),elapsed_seconds=time.monotonic()-started))
        pd.DataFrame(port_rows).to_parquet(port_path,index=False)
    ports_by_ref={r['observation_ref']:set(r['acquired_ports']) for r in port_rows}
    graph_ports={c['configuration_id']:{f"{n['id']}:{p['name']}" for n in store.graph(c['configuration_id'])['nodes'] for p in n['ports']}
        for split in ('test_seen','test_connections','test_size') for c in store.configurations(split)}
    for p in profiles:
        actual=set().union(*(ports_by_ref[ref] for ref in p['observation_refs'])) if len(p['observation_refs']) else set()
        actual&=graph_ports[p['configuration_id']]
        p.update(acquired_ports=sorted(actual),distinct_port_coverage=len(actual),
            distinct_port_coverage_fraction=len(actual)/len(graph_ports[p['configuration_id']]),
            requested_port_coverage=len(p['requested_ports']),coverage_definition='distinct internal ports actually present in acquired additional observation records')
    profile=pd.DataFrame(profiles);profile.to_parquet(profile_path,index=False)
    join_columns=['cell_key','full_trace_used','full_trace_action_count','action_family_counts','sample_rates',
        'chosen_action_keys','acquired_ports','distinct_port_coverage','distinct_port_coverage_fraction','requested_port_coverage','coverage_definition']
    for frame,label,name in [(official,'online','imported_learned_rows.parquet'),(completed,'completed','imported_completed_read.parquet')]:
        enriched=frame.merge(profile[join_columns],on='cell_key',how='left',validate='one_to_one')
        assert len(enriched)==147200 and enriched.distinct_port_coverage.notna().all()
        enriched['result_mode']=label;enriched['imported_previous_run']=True
        enriched['physical_work_in_this_extension']=0;enriched['inference_work_in_this_extension']=0
        enriched['historical_runtime_not_new_speed_measurement']=True
        enriched['source_run']=str(sed);enriched['source_per_case_file']=str(old if label=='online' else adjusted)
        enriched.to_parquet(out/name,index=False)
    hashes={name:file_sha256(out/name) for name in ['imported_learned_rows.parquet','imported_completed_read.parquet',
        'imported_action_profile.parquet','imported_observation_ports.parquet']}
    record=dict(state='passed',completed_at=now(),rows=147200,primary_rows=73600,physical_cases=3200,
        reader_hash=reader_hash,all_planned_keys_verified=True,official_input_sha256=file_sha256(old),
        completed_input_sha256=file_sha256(adjusted),input_runtime_audit_sha256=file_sha256(sed/'runtime_audit/complete.json'),
        unique_acquired_observation_blobs=len(port_rows),raw_blob_checksums_verified=True,
        actual_recorded_port_coverage=True,source_parameter_and_checkpoint_hashes_verified=True,
        physical_acquisitions=0,reader_calls=0,selector_calls=0,training_steps=0,
        imported_action_profiles_are_historical=True,wall_seconds=time.monotonic()-started,workers=workers,output_hashes=hashes)
    atomic_json(marker,record);return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--sed-run',required=True)
    p.add_argument('--run-dir',required=True);p.add_argument('--workers',type=int,default=64);a=p.parse_args()
    print(json.dumps(import_rows(a.source_root,a.sed_run,a.run_dir,a.workers)),flush=True)
