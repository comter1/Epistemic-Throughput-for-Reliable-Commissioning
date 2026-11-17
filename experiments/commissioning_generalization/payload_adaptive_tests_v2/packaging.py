"""Independent physical-summary audit and a <500 MB review archive."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
from .protocol import HERE,EXP,write_json,digest,file_hash,verify_protected,Ledger

def independent_audit(root):
    from payload_change_v1.physics import Dynamics
    from .trajectories import reference
    from .physics import _metric
    root=Path(root); paths=sorted((root/'jobs/evaluation').glob('*.json')); records=[json.loads(p.read_text()) for p in paths]
    assert len(records)==3456
    counts=dict(runs=0,blocks=0,audits=0);max_decision=0.; budget_fail=[]
    for wrapper in records:
        row=wrapper['result']; counts['runs']+=1
        physical=sum(b['physical_seconds'] for b in row['blocks'])
        if abs(physical-row['physical_seconds'])>1e-8 or row['identification_seconds']>60+1e-8 or row['verification_seconds']>20+1e-8:
            budget_fail.append(row['case_id'])
        decisions=[s.get('decision_cpu_seconds',0.) for s in row['steps']]
        initial=row['compute']['prediction_planning']-sum(decisions)
        max_decision=max(max_decision,initial,*decisions)
        assert sum(a['metrics']['pass'] for a in row['audits'])==row['audit_pass_count']
        assert row['false_release']==bool(row['release'] and row['audit_pass_count']<8)
        assert row['normal_correct']==bool(row['status']=='approved_normal' and row['audit_pass_count']==8)
        if row['release']: assert len(row['audits'])==8
        for b in row['blocks']+row['audits']:
            meta=json.loads((root/b['path']/'metadata.json').read_text())
            assert meta['private_metrics']==b['metrics']
            counts['blocks']+=1
        counts['audits']+=len(row['audits'])
    assert not budget_fail
    assert max_decision<=10.+1e-3, max_decision
    # Independently rebuild Cartesian errors from the saved 500 Hz joint path;
    # do not merely trust the saved scalar error or pass flag.
    selected=sorted(records,key=lambda r:digest(['independent-audit',r['result']['case_id'],r['result']['method'],r['result']['noise_seed']]))[:24]
    kin=Dynamics(None); lower,upper=kin.joint_range.T; samples=[]
    for wrapper in selected:
        row=wrapper['result']
        for b in row['blocks']+row['audits'][:1]:
            path=root/b['path']; meta=json.loads((path/'metadata.json').read_text())
            with np.load(path/'private.npz') as z: a={k:z[k] for k in z.files}
            error=[];start=np.asarray(meta['measured_start_reference']);dt=a['time']-meta['start_state']['time']
            for q,t in zip(a['q'],dt):
                qr,_,_=reference(meta['action'],min(float(t),meta['action']['duration']),start)
                error.append(np.linalg.norm(kin.fk(q)-kin.fk(qr)))
            error=np.asarray(error)
            discrepancy=float(np.max(abs(error-a['error'])))
            assert discrepancy<1e-8,discrepancy
            recomputed=_metric(a['q'],a['dq'],error,a['saturation'],a['contacts'],lower,upper,meta['physical_seconds'],meta['private_metrics']['hard_abort'])
            assert recomputed['pass']==meta['private_metrics']['pass']
            for name in ('rms_error_m','peak_error_m','final_hold_peak_m','saturation_fraction','max_continuous_saturation_seconds'):
                assert abs(recomputed[name]-meta['private_metrics'][name])<1e-8,(name,recomputed[name],meta['private_metrics'][name])
            samples.append(dict(path=str(path),private_sha256=file_hash(path/'private.npz'),public_sha256=file_hash(path/'public.npz'),
                samples=len(error),cartesian_recompute_max_difference_m=discrepancy,pass_matches=True))
    result=dict(passed=True,counts=counts,maximum_decision_cpu_seconds=max_decision,physical_budget_errors=budget_fail,
                independent_sampled_runs=24,independent_blocks=len(samples),independent_500hz_states=sum(s['samples'] for s in samples),samples=samples)
    write_json(root/'checks/independent_results_audit.json',result);verify_protected(root)
    return result

def index_raw(root):
    root=Path(root)
    files=sorted(p for p in root.rglob('*') if p.is_file() and (p.suffix=='.npz' or p.name=='metadata.json'))
    index=root/'raw_artifact_index.jsonl.gz';total=0
    def entry(path):
        return dict(path=str(path),relative_path=str(path.relative_to(root)),bytes=path.stat().st_size,
                    sha256=file_hash(path),kind='public_prediction_cache' if '/cache/' in str(path) else 'raw_sensor_or_physics_artifact',
                    this_index_entry_contains_raw_data=False)
    with gzip.open(index,'wt') as out,ThreadPoolExecutor(max_workers=8) as pool:
        for row in pool.map(entry,files):
            total+=row['bytes'];out.write(json.dumps(row,separators=(',',':'))+'\n')
    result=dict(files=len(files),bytes=total,index_sha256=file_hash(index),
                index_is_not_raw_data=True,missing_failed_tail_arrays='See engineering_attempt_01/ACCOUNTING_FIX.json and SECOND_ATTEMPT.json; exceptions after execution prevented saving those tail arrays.')
    write_json(root/'raw_artifact_inventory.json',result)
    return result

def package(root):
    root=Path(root);cpu=Ledger(root,'cpu');tok=cpu.reserve(1800,unit='independent_audit_and_packaging',kind='reporting_packaging');start=time.process_time()
    audit=independent_audit(root)
    inventory=index_raw(root)
    readme='''# Analysis archive / 分析包

Start with FINDINGS_CN.md, constraint_and_false_release_counts.csv and
paired_method_comparisons.csv. Formal test results contain 3456 method runs and
576 reference conditions; validation includes all 2304 registered combinations.
Every tested run has its full JSON under jobs/. Source branch and score analyses
are separate from the held-out primary comparison.

The archive includes complete summaries, code, frozen configurations/manifests,
provenance, ledgers, figures with data and standalone TikZ/PGFPlots sources, and
selected genuine raw traces. raw_artifact_index.jsonl.gz is a reference index,
NOT the complete sensor data. Each external path has exact size and SHA256.
The external run store retains large 100 Hz public and 500 Hz private arrays.
Engineering attempts and known unsaved failed-tail arrays are explicitly documented.

Earlier source/validation engineering revisions are not part of the formal test
matrix. Their failed/completed attempts are retained and charged. Archived source
relative paths resolve within their engineering revision directory first; archived
validation paths remain under the original run root. Protocol/scientific code and
robot assets are frozen before held-out evaluation.
'''
    (root/'ARCHIVE_README.md').write_text(readme)
    snapshots={str(p.relative_to(root)):file_hash(p) for p in root.rglob('*.json')
               if 'cache' not in p.parts and p.name!='file_hashes.json'}
    write_json(root/'checks/json_artifact_hashes.json',snapshots)
    representatives=[]
    for row in audit['samples'][:24]: representatives.append(Path(row['path']))
    # Include static, dynamic, and each speed among the actual source raw traces.
    source_paths=sorted((root/'raw/source').glob('*/block_*/metadata.json'))
    seen=set()
    for p in source_paths:
        meta=json.loads(p.read_text());key=(meta['action']['kind'],meta['action']['speed'])
        if key not in seen: seen.add(key);representatives.append(p.parent)
        if len(representatives)>=32: break
    for example in json.loads((root/'real_timeline_examples.json').read_text()):
        for row in example['runs'].values():
            representatives.extend(root/b['path'] for b in row['blocks'])
    for example in json.loads((root/'concrete_measurement_choice_examples.json').read_text()):
        representatives.append(root/'raw/branches'/example['case_id']/'prefix')
    representatives=list(dict.fromkeys(representatives))
    write_json(root/'representative_traces/manifest.json',[dict(
        raw_directory=str(p),archive_directory=str(Path('payload_adaptive_tests_v2')/p.relative_to(root)),
        files={f.name:file_hash(f) for f in p.iterdir() if f.is_file()}) for p in representatives])
    representative_files={p for directory in representatives for p in directory.iterdir() if p.is_file()}
    destination=root.parent/(f'Payload_Adaptive_Tests_V2_{root.name}_GPT_Pro.zip')
    selected=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix in ('.lock','.tmp') or '__pycache__' in p.parts: continue
        rel=p.relative_to(root)
        if 'cache' in rel.parts: continue
        if p.suffix=='.npz' and p not in representative_files: continue
        # Scalar metadata is retained for every raw block; the large arrays remain indexed.
        selected.append((p,Path('payload_adaptive_tests_v2')/rel))
    for namespace in ('payload_adaptive_tests_v2','payload_change_v1','src/cgx'):
        for p in (EXP/namespace).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.md','.json','.txt'):
                selected.append((p,Path('payload_adaptive_tests_v2/code')/p.relative_to(EXP)))
    for name in ('requirements.lock.txt','requirements-spec.txt','AGENTS.md'):
        p=EXP/name
        if p.exists(): selected.append((p,Path('payload_adaptive_tests_v2/code')/name))
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p,name in selected:
            archive.write(p,str(name),compress_type=zipfile.ZIP_STORED if p.suffix=='.npz' else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(destination) as archive:
        bad=archive.testzip()
        if bad: raise RuntimeError('ZIP CRC failure: '+bad)
        assert sum(n.startswith('payload_adaptive_tests_v2/jobs/evaluation/') and n.endswith('.json') for n in archive.namelist())==3456
        assert sum(n.startswith('payload_adaptive_tests_v2/jobs/reference/') and n.endswith('.json') for n in archive.namelist())==576
    size=destination.stat().st_size
    if size>=500_000_000: raise RuntimeError(f'Archive is {size} bytes, over the 500 MB limit; do not publish')
    cpu.complete(tok,time.process_time()-start)
    result=dict(path=str(destination),bytes=size,decimal_MB=size/1e6,sha256=file_hash(destination),
                archive_entries=len(selected),representative_raw_blocks=len(representatives),raw_inventory=inventory,crc_pass=True)
    write_json(root/'DELIVERY.json',result)
    return result
