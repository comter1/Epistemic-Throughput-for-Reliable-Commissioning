"""Verify frozen inputs and export a compact, auditable analysis ZIP."""
from __future__ import annotations

import argparse
import difflib
import gzip
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

from cgx.config import EXPERIMENT
from cgx.data.store import Store
from cgx.provenance import atomic_json, file_sha256, read_json
from .core import load_artifact


REQUIRED = ['INPUT_MANIFEST.json','PROTOCOL_FA_CAUSE_V1.yaml','WORKLOAD_AND_STATUS.md',
    'baseline_recheck.csv','baseline_integrity.md','input_path_checks.md','prefix_predictions.parquet',
    'trajectory_transitions.csv','budget_path_divergence.csv','runtime_stage_breakdown.csv',
    'fixed_transcript_reread.csv','calibration_same_records.csv','intervention_outcomes.parquet',
    'paired_effects.csv','action_value_checks.csv','fresh_case_confirmation.csv','FINDINGS_CN.md',
    'REPRODUCE.md','RESOURCE_USAGE.md','README_CN.md']


def verify(root, run):
    original=read_json(run/'INPUT_MANIFEST.json');checks=[]
    for relative,expected in original['original_hashes'].items():
        actual=file_sha256(root/relative)
        checks.append(dict(path=str(root/relative),expected=expected,actual=actual,unchanged=actual==expected))
    for relative,expected in original['sample_manifest_hashes'].items():
        actual=file_sha256(run/relative)
        checks.append(dict(path=str(run/relative),expected=expected,actual=actual,unchanged=actual==expected))
    actual=file_sha256(run/'PROTOCOL_FA_CAUSE_V1.yaml')
    checks.append(dict(path='PROTOCOL_FA_CAUSE_V1.yaml',expected=original['protocol_sha256'],actual=actual,
                       unchanged=actual==original['protocol_sha256']))
    with tarfile.open(run/'provenance/workspace_before_analysis.tar.gz','r:gz') as archive:
        for member in archive:
            if not member.isfile():continue
            current=EXPERIMENT/member.name
            content=archive.extractfile(member).read()
            checks.append(dict(path=str(current),unchanged=current.exists() and current.read_bytes()==content,
                               comparison='bytewise to pre-investigation workspace snapshot'))
    result=dict(state='passed' if all(x['unchanged'] for x in checks) else 'failed',
                checked_files=len(checks),changed=[x for x in checks if not x['unchanged']],checks=checks)
    atomic_json(run/'provenance/final_integrity_verification.json',result)
    assert result['state']=='passed', 'Protected input or prior workspace content changed'
    return result


def source_bundle(root,run):
    destination=run/'software';destination.mkdir(exist_ok=True)
    source=EXPERIMENT/'diagnostics/false_alarm_causes'
    changes=[]
    for path in sorted(source.rglob('*.py')):
        relative=path.relative_to(EXPERIMENT)
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        changes.extend(difflib.unified_diff([],path.read_text().splitlines(keepends=True),fromfile='/dev/null',tofile=str(relative)))
    (run/'diagnostic_code_additions.diff').write_text(''.join(changes))
    shutil.copy2(EXPERIMENT/'diagnostics/__init__.py',destination/'diagnostics/__init__.py')
    for relative in ['software/source.tar.gz','software/source_files.json','software/protocol.json',
                     'models/selection.json','models/frozen.json','models/thresholds.json']:
        target=destination/'original_frozen_reference'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/relative,target)


def examples(root,run):
    store=Store(root)
    for index,story in enumerate(read_json(run/'six_case_stories.json'),1):
        base=story['identity'];directory=run/'case_examples'/f'{index:02d}_{story["category"]}'
        directory.mkdir(parents=True,exist_ok=True)
        item=load_artifact(root,base['case_id'],base['method_key'],int(base['policy_seed']))
        public=store.load_history(item['row']['evidence_trace_hash'])
        with gzip.open(directory/'public_acquired_history.json.gz','wt') as handle:json.dump(public,handle)
        atomic_json(directory/'original_evaluator_result.json',item)
        atomic_json(directory/'reread_story.json',story)


def package(root,run):
    missing=[name for name in REQUIRED if not (run/name).is_file()]
    assert not missing,f'Missing required deliverables: {missing}'
    for relative in ['baseline_complete.json','replay_complete.json','interventions/discovery/complete.json',
                     'interventions/confirmation/complete.json','calibration_same_records_complete.json',
                     'q_input_action_checks_complete.json','action_value_checks_complete.json',
                     'discovery_artifact_audit.json','confirmation_artifact_audit.json']:
        assert read_json(run/relative)['state']=='passed', f'Incomplete stage: {relative}'
    counts=read_json(run/'resource_ledger.json')['counts']
    assert counts['robot_execution']<=18000 and counts['model_forward']+counts['feature_build']<=120000
    verify(root,run);source_bundle(root,run);examples(root,run)
    included_directories={'figures','manifests','provenance','logs','software','case_examples',
                          'interventions','same_records','action_branches','q_input_checks'}
    def include(path):
        relative=path.relative_to(run)
        return (len(relative.parts)==1 or relative.parts[0] in included_directories) and not any(
            part=='__pycache__' or part.endswith(('.lock','.tmp','.pyc')) for part in relative.parts)
    files=[p for p in sorted(run.rglob('*')) if p.is_file() and include(p) and p.name not in ('SHA256SUMS','PACKAGE_MANIFEST.json')]
    manifests=[dict(path=str(p.relative_to(run)),bytes=p.stat().st_size,sha256=file_sha256(p)) for p in files]
    atomic_json(run/'PACKAGE_MANIFEST.json',dict(files=manifests,omitted=[
        'cache/: reproducible intermediate tensor/features/prediction caches',
        'public/: complete new raw-observation store; six selected original histories included in case_examples/',
        'budget_scores/: shards consolidated losslessly in initial_budget_action_scores.parquet',
        'checkpoint tensors, MuJoCo assets, full original observation store: retained at the original data root; hashes in INPUT_MANIFEST.json'],
        completion='See stage completion JSONs, FINDINGS_CN.md and final_integrity_verification.json; no patched model or original result is included'))
    files.append(run/'PACKAGE_MANIFEST.json')
    (run/'SHA256SUMS').write_text(''.join(f'{file_sha256(p)}  {p.relative_to(run)}\n' for p in files))
    files.append(run/'SHA256SUMS')
    destination=run.parent/f'CGX_FA_CAUSE_V1_{run.name}.zip'
    temporary=destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for path in files:archive.write(path,arcname='CGX_FA_CAUSE_V1/'+str(path.relative_to(run)))
    with zipfile.ZipFile(temporary) as archive:assert archive.testzip() is None
    temporary.replace(destination)
    receipt=dict(path=str(destination),bytes=destination.stat().st_size,sha256=file_sha256(destination),files=len(files))
    destination.with_suffix('.zip.sha256').write_text(receipt['sha256']+'  '+destination.name+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only:print(json.dumps(verify(a.root,a.run_dir),indent=2))
    else:package(a.root,a.run_dir)
