"""Bounded analysis delivery with complete row coverage and raw-store provenance."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .storage import file_hash, write_json

MAX_BYTES = 500_000_000
RAW_DIRS = {'raw', 'telemetry', 'measurements', 'cache', 'physics_cache', 'raw_observations',
            'private_trajectories', 'physical_states', 'sensor_records'}
BINARY_RAW_SUFFIXES = {'.npz', '.npy', '.bin', '.mjb', '.h5', '.hdf5', '.parquet'}


def _atomic_gzip_jsonl(path, records):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with gzip.open(tmp, 'wt', encoding='utf-8', compresslevel=6) as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n')
    os.replace(tmp, path)


def index_artifacts(run_root):
    """Hash every retained artifact, including failed executions and external raw."""
    root = Path(run_root)
    index = root / 'raw_record_index.jsonl.gz'
    previous={}
    if index.exists():
        with gzip.open(index,'rt') as stream:
            previous={r['relative_path']:r for r in (json.loads(line) for line in stream)}
    candidates=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path == index or path.name.endswith('.tmp'):
            continue
        relative = path.relative_to(root)
        if any(part in ('code', '.git', '__pycache__') for part in relative.parts):
            continue
        if path.name in ('delivery.json', 'delivery_file_index.json', 'checksums.sha256'):
            continue
        raw = bool(set(relative.parts) & RAW_DIRS or path.suffix in BINARY_RAW_SUFFIXES)
        if raw:
            candidates.append(path)
    def hash_one(path):
        relative=str(path.relative_to(root));stat=path.stat();old=previous.get(relative)
        if old and old.get('mtime_ns')==stat.st_mtime_ns and old['size_bytes']==stat.st_size:
            return old
        return {'relative_path':relative,'absolute_path':str(path.resolve()),
                'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':file_hash(path),
                'retained_on_server':True,'success_filter_applied':False}
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(hash_one,candidates))
    raw_count=len(records);raw_bytes=sum(r['size_bytes'] for r in records)
    categories=defaultdict(lambda:{'files':0,'bytes':0})
    for row in records:
        category=Path(row['relative_path']).parts[0]
        categories[category]['files']+=1;categories[category]['bytes']+=row['size_bytes']
    _atomic_gzip_jsonl(index, records)
    return {'raw_files':raw_count,'raw_bytes':raw_bytes,'index':str(index),'index_sha256':file_hash(index),
            'categories':dict(categories),'hash_threads':4,'hash_once_reused_for_package_file_index':True}


def _README(root):
    path=root/'README.md'
    if path.exists():
        return
    path.write_text('''# CGX-PAYLOAD-COMMISSIONING-V1 analysis delivery

Start with FINDINGS_CN.md, IMPLEMENTATION_AUDIT.md, CONTROLLER_DECISION.md,
protocol_locked.yaml, and results/summary.csv. Per-run records are under runs/test;
results/per_case.csv preserves every run, including failures and unresolved cases.
The true-payload reference is privileged and excluded from ordinary method ranking.

The methods share the same frozen estimator, controller, library, update rule and
acceptance requirements. Each method has 192 physical cases and three noise seeds,
not 576 independent objects. The eight private audit trajectories per run are
validation tasks, not eight extra objects. Physical seconds include motion/return,
and remain separate from simulation wall time and estimator/decision CPU time.

Run from experiments/commissioning_generalization with the supplied environment:
PYTHONPATH=.:src python -m payload_change_v1.run --stage all --run-root ABSOLUTE_ROOT
Stage hashes and completion markers govern resumption. Test outcomes must never
be used to fit priors, noise, controller gains, or stopping thresholds.

Raw sensor and 500 Hz private evaluation trajectories remain at the absolute
paths in raw_record_index.jsonl.gz, each with byte count and SHA256. The index
includes unsuccessful executions; no successful-only filtering is used. This
analysis archive stays below 500,000,000 bytes. Exact package exclusions are in
delivery_file_index.json. Figures are vector PDFs plus standalone PGF/TikZ sources
and CSV data. Any uncompiled LaTeX source is labeled in figures/rendering.json.
''')


def package(run_root, destination=None, allow_incomplete=False):
    root=Path(run_root).resolve()
    if not allow_incomplete:
        from .protocol import verify_seal
        verify_seal(root)
    status_path=root/'results/report_status.json'
    if not status_path.exists():
        raise RuntimeError('Run the reporting stage before packaging.')
    status=json.loads(status_path.read_text())
    if not status.get('completed_matrix') and not allow_incomplete:
        raise RuntimeError('Cannot label an incomplete formal matrix as a completed delivery.')
    _README(root)
    code_dir=root/'code/payload_change_v1'
    code_dir.mkdir(parents=True,exist_ok=True)
    package_source=Path(__file__).parent
    for source in package_source.rglob('*'):
        if source.is_file() and source.suffix in ('.py','.md','.yaml','.json') and '__pycache__' not in source.parts:
            target_source=code_dir/source.relative_to(package_source)
            target_source.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target_source)
    raw_summary=index_artifacts(root)
    with gzip.open(root/'raw_record_index.jsonl.gz','rt') as stream:
        raw_index={r['relative_path']:r for r in (json.loads(line) for line in stream)}
    included,omitted=[],[]
    required_files=('README.md','FINDINGS_CN.md','IMPLEMENTATION_AUDIT.md','CONTROLLER_DECISION.md',
                    'CODEX_RUN_LOG.md','resource_usage.json','commands.txt',
                    'protocol_locked.yaml','public_private_field_map.json','results/per_case.csv','results/summary.csv',
                    'checks/final_result_audit.json','provenance/git_commit.json')
    missing=[name for name in required_files if not (root/name).is_file()]
    if missing and not allow_incomplete:
        raise RuntimeError('Missing required delivery files: '+', '.join(missing))
    for path in sorted(root.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(root)
        if path.name.endswith(('.tmp','.lock','.building','.zip')) or '__pycache__' in relative.parts:continue
        if path.name in ('checksums.sha256','delivery_file_index.json','delivery.json'):continue
        reason=None
        if str(relative)=='stages/package.json':
            reason='Active package-stage marker finishes only after ZIP creation; final marker and archive SHA256/CRC receipt remain outside this self-containing archive.'
        elif set(relative.parts)&RAW_DIRS or path.suffix in BINARY_RAW_SUFFIXES:
            reason='Full raw store retained externally; SHA256 is in raw_record_index.jsonl.gz.'
        elif path.stat().st_size>25_000_000 and relative.parts[0] not in ('results','runs'):
            reason='Large ancillary artifact retained externally to meet the analysis archive size limit.'
        cached=raw_index.get(str(relative))
        entry={'relative_path':str(relative),'size_bytes':path.stat().st_size,
               'sha256':cached['sha256'] if cached else file_hash(path)}
        if reason:
            omitted.append(dict(entry,absolute_path=str(path),reason=reason))
        else:
            included.append(entry)
    write_json(root/'delivery_file_index.json',{'included':included,'retained_external':omitted,
               'all_test_run_summaries_included':True,'raw_index':raw_summary,'missing_required_files':missing})
    included.append({'relative_path':'delivery_file_index.json','size_bytes':(root/'delivery_file_index.json').stat().st_size,
                     'sha256':file_hash(root/'delivery_file_index.json')})
    (root/'checksums.sha256').write_text(''.join(row['sha256']+'  '+row['relative_path']+'\n' for row in included))
    target=Path(destination) if destination else root.parent/('Payload_Change_Commissioning_'+root.name+'_GPT_Pro.zip')
    temporary=target.with_name(target.name+'.building')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for row in included:
            archive.write(root/row['relative_path'],arcname='payload_change_v1/'+row['relative_path'])
        archive.write(root/'checksums.sha256',arcname='payload_change_v1/checksums.sha256')
    actual=temporary.stat().st_size
    if actual>=MAX_BYTES:
        oversize=target.with_name(target.stem+'.oversize.zip')
        os.replace(temporary,oversize)
        raise RuntimeError(f'Archive {actual} bytes exceeds the {MAX_BYTES}-byte limit; preserved at {oversize}.')
    with zipfile.ZipFile(temporary) as archive:
        bad=archive.testzip()
        if bad:raise RuntimeError('Archive CRC failure: '+bad)
        test_files=[n for n in archive.namelist() if '/runs/test/' in n and n.endswith('.json')]
        if status.get('completed_matrix') and len(test_files)!=3456:
            raise RuntimeError(f'Archive contains {len(test_files)} test records, expected3456.')
    os.replace(temporary,target)
    delivery={'archive':str(target),'size_bytes':actual,'limit_bytes':MAX_BYTES,'sha256':file_hash(target),
              'crc_passed':True,'formal_matrix_complete':bool(status.get('completed_matrix')),
              'files':len(included)+1,'raw_store':raw_summary,'created_at_unix':time.time()}
    write_json(root/'delivery.json',delivery)
    return delivery


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',required=True);parser.add_argument('--destination')
    parser.add_argument('--allow-incomplete',action='store_true');args=parser.parse_args()
    print(json.dumps(package(args.run_root,args.destination,args.allow_incomplete),indent=2))


if __name__=='__main__':main()
