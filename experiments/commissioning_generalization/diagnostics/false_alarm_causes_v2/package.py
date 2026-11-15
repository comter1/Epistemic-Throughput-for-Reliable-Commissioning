"""Build an auditable analysis ZIP; large original assets remain hash-addressed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import difflib
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile
from cgx.config import EXPERIMENT
from cgx.provenance import atomic_json, file_sha256, read_json

REQUIRED=['README.md','FINDINGS_CN.md','EXECUTIVE_COUNTS_CN.md','METHOD_NAMES.md','source_checks.json',
    'feature_path_inventory.md','stop_gradient_audit.json','reader_factorial_per_case.csv',
    'reader_factorial_summary.csv','reader_real_prefix_effects.csv','reader_candidate_validation.csv',
    'continuing_value_source_per_prefix.csv','continuing_value_calibration.csv',
    'correction_2x2_source_validation.csv','correction_2x2_fresh_replay.csv','confirmation_per_case.csv',
    'confirmation_per_configuration.csv','confirmation_per_seed.csv','confirmation_counts_and_rates.csv',
    'paired_effects.csv','remaining_errors.csv','RESOURCE_USAGE.md','case_examples/selection.json',
    'manifests/source_cases.json','manifests/existing_case_reader_probes.json',
    'manifests/source_variant_selection.json','manifests/fresh_confirmation_sealed.json','manifests/variants.json',
    'tests/final_audit.json','tests/confirmation_statistics.json']


def include_copy(source,destination):
    if source.is_dir():
        for path in source.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.lock','.tmp','.pyc'):continue
            relative=path.relative_to(source);out=destination/relative;out.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,out)
    else:destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)


def run(run_dir):
    run=Path(run_dir)
    missing=[p for p in REQUIRED if not (run/p).is_file()]
    assert not missing, f'Required evidence missing: {missing}'
    assert read_json(run/'tests/final_audit.json')['state']=='passed'
    assert read_json(run/'tests/confirmation_statistics.json')['state']=='passed'
    bundle=run/'CGX_FA_CAUSE_V2'
    if bundle.exists():
        raise RuntimeError('Existing bundle is preserved; choose explicit rebuild after inspection')
    bundle.mkdir()
    # Root-level tables and compact evidence are portable analysis artifacts.
    for path in sorted(run.iterdir()):
        if path.is_file() and (path.suffix in ('.csv','.json','.md','.parquet','.jsonl','.log') or path.name.endswith('.json.gz')):
            include_copy(path,bundle/path.name)
    for directory in ('manifests','provenance','figures','case_examples','commands','tests','logs','head_data','head_checkpoints','trajectories','live_dispatch'):
        if (run/directory).exists():include_copy(run/directory,bundle/directory)
    code=bundle/'code';code.mkdir()
    for directory in ('src','diagnostics','configs','docs','tests','schemas','scripts'):
        if (EXPERIMENT/directory).exists():include_copy(EXPERIMENT/directory,code/directory)
    for name in ('AGENTS.md','pyproject.toml','requirements.txt','requirements.lock','README.md'):
        if (EXPERIMENT/name).exists():include_copy(EXPERIMENT/name,code/name)
    patch=[]
    for path in sorted((EXPERIMENT/'diagnostics/false_alarm_causes_v2').glob('*.py')):
        relative=path.relative_to(EXPERIMENT)
        patch.extend(difflib.unified_diff([],path.read_text().splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+str(relative)))
    patch_path=bundle/'code/changes_v2.patch';patch_path.write_text(''.join(patch))
    with tempfile.TemporaryDirectory(prefix='cgx_v2_patch_') as temp:
        check=subprocess.run(['git','apply','--check',str(patch_path)],cwd=temp,text=True,capture_output=True)
        assert check.returncode==0,check.stderr
    atomic_json(bundle/'tests/executable_patch.json',dict(state='passed',command='git apply --check code/changes_v2.patch',
        context='empty temporary working directory; additions under diagnostics/false_alarm_causes_v2',
        original_source_unchanged=read_json(run/'provenance/protection_after.json')['state']=='passed'))
    protected=read_json(run/'provenance/protected_before.json')
    original_root=Path(read_json(run/'source_checks.json')['original_data_root'])
    original_assets=[dict(path=path,sha256=sha,bytes=Path(path).stat().st_size,included=False)
                     for path,sha in protected.items() if str(original_root/'models') in path]
    # Every observation/history is content addressed; hashes/locations permit
    # checking raw provenance without inflating the GPT analysis ZIP with caches.
    retained_paths=[]
    for dirname in ('public','cache'):
        directory=run/dirname
        if not directory.exists():continue
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix not in ('.lock','.tmp'):
                retained_paths.append(path)
    def retained_asset(path):
        return dict(path=str(path),relative_to_run=str(path.relative_to(run)),
            sha256=file_sha256(path),bytes=path.stat().st_size,included=False)
    # Read-only hashing on the remote CPU node; deterministic map order.
    with ThreadPoolExecutor(max_workers=16) as pool:
        retained=list(pool.map(retained_asset,retained_paths))
    atomic_json(bundle/'LARGE_ASSETS.json',dict(original_model_assets=original_assets,
        retained_run_data=retained,run_directory=str(run),original_data_root=str(original_root),
        note='Original checkpoints and raw observation/feature caches stay on server; newly trained small heads and result trajectories are included.'))
    items=[]
    for path in sorted(bundle.rglob('*')):
        if path.is_file():items.append(dict(path=str(path.relative_to(bundle)),sha256=file_sha256(path),bytes=path.stat().st_size))
    atomic_json(bundle/'PACKAGE_MANIFEST.json',dict(task='CGX-FA-CAUSE-V2',files=items,required_files=REQUIRED,
        required_files_present=True,archive_root='CGX_FA_CAUSE_V2',source_run=str(run)))
    (bundle/'SHA256SUMS').write_text(''.join(f"{i['sha256']}  {i['path']}\n" for i in items)+
        f"{file_sha256(bundle/'PACKAGE_MANIFEST.json')}  PACKAGE_MANIFEST.json\n")
    output=run.parent/f'CGX_FA_CAUSE_V2_{run.name}.zip'
    if output.exists():raise RuntimeError(f'Archive already exists: {output}')
    temporary=output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():archive.write(path,arcname=str(Path('CGX_FA_CAUSE_V2')/path.relative_to(bundle)))
    temporary.replace(output)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for item in items:
            actual=hashlib.sha256(archive.read('CGX_FA_CAUSE_V2/'+item['path'])).hexdigest()
            assert actual==item['sha256'],item['path']
    result=dict(state='passed',path=str(output),bytes=output.stat().st_size,sha256=file_sha256(output),
        archived_files=len(items)+2,zip_crc_and_all_member_sha256_verified=True)
    atomic_json(run/'package_result.json',result)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);run(p.parse_args().run_dir)
