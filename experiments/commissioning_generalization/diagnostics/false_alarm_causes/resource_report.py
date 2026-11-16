"""Conservative resource accounting: operation attempts, measured CPU, separate clocks.

No estimate is silently promoted to measurement. GPU synchronized call wall time
includes host work/waits and is not active kernel time. Parent process CPU excludes
child process CPU; separately measured workers are added once per unique artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
    temporary.replace(path)


def collect_environment(run_dir):
    """Metadata only; no GPU utilization/process list, credentials, addresses or UUIDs."""
    script=r'''
import csv,io,json,os,platform,subprocess,sys
from pathlib import Path
from importlib import metadata
from datetime import datetime,timezone
cpuinfo=Path('/proc/cpuinfo').read_text()
model=next((x.split(':',1)[1].strip() for x in cpuinfo.splitlines() if x.startswith('model name')),None)
memory={x.split(':',1)[0]:x.split(':',1)[1].strip() for x in Path('/proc/meminfo').read_text().splitlines()}
quota=Path('/sys/fs/cgroup/cpu.max')
q=quota.read_text().split() if quota.exists() else []
quota_cpus=float(q[0])/float(q[1]) if q and q[0]!='max' else None
result=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
gpus=[dict(index=i,name=r[0].strip(),memory_total_mib=int(r[1].strip()),driver_version=r[2].strip()) for i,r in enumerate(csv.reader(io.StringIO(result.stdout)))]
versions={}
for name in ['torch','numpy','pandas','pyarrow','mujoco','networkx','pytest']:
 try:versions[name]=metadata.version(name)
 except metadata.PackageNotFoundError:versions[name]=None
import torch
print(json.dumps(dict(collected_at=datetime.now(timezone.utc).isoformat(),python=sys.version.split()[0],os=platform.system(),kernel=platform.release(),cpu_model=model,logical_cpu_count=os.cpu_count(),affinity_cpu_count=len(os.sched_getaffinity(0)),cgroup_cpu_quota=quota_cpus,memory_total_kib=int(memory['MemTotal'].split()[0]),gpus=gpus,pytorch_cuda_build=torch.version.cuda,packages=versions)))
'''
    nodes=[]
    for port in (2379,2982,2404):
        result=subprocess.run(['ssh','-p',str(port),'-o','BatchMode=yes','-o','ConnectTimeout=15',
                               'leiyo@ssh.cloud.sdu.dk','/home/leiyo/work/cgx_env/bin/python -'],
                              input=script,text=True,capture_output=True,timeout=60)
        if result.returncode:
            nodes.append(dict(ssh_port=port,state='unavailable',error='Metadata probe failed; no process or credential output retained.'))
        else:nodes.append(dict(ssh_port=port,state='passed',**json.loads(result.stdout)))
    report=dict(collected_at=datetime.now(timezone.utc).isoformat(),nodes=nodes,
                total_confirmed_gpus=sum(len(n.get('gpus',[])) for n in nodes),
                scope='Read-only GPU/CPU/software inventory; no user process information or GPU UUIDs collected.')
    write_json(Path(run_dir)/'environment.json',report)
    return report


def sum_csv(path,field,filter_key=None,filter_value=None):
    if not Path(path).exists():return None
    total=0.;n=0
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            if filter_key and row.get(filter_key)!=filter_value:continue
            value=row.get(field)
            if value not in (None,''):
                total+=float(value);n+=1
    return dict(seconds=total,rows=n)


def runtime_breakdown(run_dir,components,gpu_calls,task_walls):
    """Preserve distinct granularities, including genuine original CPU totals."""
    run_dir=Path(run_dir);rows=[]
    boundaries={
        'selection_seconds':'Original accumulated selector/plan-call wall including preparation/copy; no internal GRU/graph split.',
        'inference_seconds':'Original accumulated diagnosis-call wall including preprocessing/model/synchronization; no internal split.',
        'simulation_seconds':'Original acquisition/initial-execution wall; includes recorder and may include calibration; not pure physics.',
        'recorder_seconds':'Original recorder component nested inside simulation_seconds; do not add it again.',
        'warmup_seconds':'Original diagnosis warmup wall, with its own outer deadline.',
        'selection_warmup_seconds':'Original selector/plan warmup wall.',
        'end_to_end_seconds':'Original whole-episode wall; contains nested components and cannot be summed with them.'}
    original=run_dir/'original_cpu_runtime_components.csv'
    if not original.exists():
        root=Path(read_json(run_dir/'INPUT_MANIFEST.json')['data_root'])
        cases=read_json(run_dir/'manifests/discovery_cases.json');source_rows=[]
        for case in cases:
            for method in ('gnn_depth2','gnn_adaptive'):
                for seed in (11,23,37):
                    key=hashlib.sha256(json.dumps([method,seed,4,1.0,case['case_id']],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                    artifact=root/'evaluation/main/cases'/f'{key}.json'
                    item=read_json(artifact);row=item['row']
                    for field,boundary in boundaries.items():
                        source_rows.append(dict(source='original_evaluation/main/cases/'+key+'.json',
                            scope='original_discovery_64_cases_2_methods_3_seeds',case_id=case['case_id'],
                            configuration_id=case['configuration_id'],method_key=method,policy_seed=seed,
                            stage=field,seconds=row.get(field),device='original_cpu_single_thread_reference',
                            timing_boundary=boundary,overlap_note='Different clocks/granularities are not additive.',
                            original_timeout=row['timeout'],result_hash=item['result_hash'],
                            measured=field in row))
        with original.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(source_rows[0]));writer.writeheader();writer.writerows(source_rows)
    with original.open() as handle:rows.extend(dict(row) for row in csv.DictReader(handle))
    for filename in ('input_runtime_stages.csv','intervention_runtime_stage_breakdown.csv'):
        path=run_dir/filename
        if not path.exists():continue
        with path.open() as handle:
            for row in csv.DictReader(handle):
                rows.append(dict(row,source=filename,scope=row.get('scope') or 'input_reading_probe_component',
                                 timing_boundary='New remote component timing; GPU hooks use synchronized wall, not active kernels.',
                                 overlap_note=row.get('overlap_note') or 'Nested in feature/model/whole-job timings; do not add across levels.',measured=True))
    path=run_dir/'cpu_gpu_parity.csv'
    if path.exists():
        with path.open() as handle:
            for row in csv.DictReader(handle):
                for field,device in [('cpu_seconds','cpu_single_thread'),('gpu_seconds','cuda'),('rawread_seconds','cpu'),('preprocessing_seconds','cpu')]:
                    rows.append(dict(source=path.name,scope='fixed_engineering_parity_subset',reader=row['reader'],prefix_hash=row['prefix_hash'],
                                     stage=field,seconds=row[field],device=device,measured=True,
                                     timing_boundary='New parity-run timing; cpu/gpu call includes collation and model; rawread/preprocess are cached feature-source measurements.',
                                     overlap_note='Shared prefix rawread/preprocess repeated across reader rows; do not sum those duplicates.'))
    for bucket,items in [('aggregate_component',components),('synchronized_model_call_wall',gpu_calls),('gpu_task_wall',task_walls)]:
        for item in items:
            rows.append(dict(item,scope=item.get('scope',bucket),stage=item.get('stage',bucket),
                             timing_boundary=item.get('boundary','See source'),measured=True,
                             overlap_note='Aggregated or nested measurements; overlaps detailed rows and must not be summed across sources.'))
    rows.append(dict(source='measurement_limitations',scope='original_cpu_component_breakdown',stage='message_matching_GRU_graph_action_head_split',
                     seconds=None,device='original_cpu_single_thread_reference',measured=False,
                     timing_boundary='Not captured in original evaluation artifacts. New remote microbenchmarks cannot reconstruct old CPU substage times.',
                     overlap_note='Explicitly unmeasured; no synthetic allocation of original total time.'))
    fields=sorted({k for row in rows for k in row})
    with (run_dir/'runtime_stage_breakdown.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return len(rows)


def report(run_dir,final=False):
    run_dir=Path(run_dir);ledger=read_json(run_dir/'resource_ledger.json')
    counts=ledger['counts'];combined=counts['model_forward']+counts['feature_build']
    cpu=[];unmeasured=[];gpu_calls=[];task_walls=[];components=[]
    path=run_dir/'stage_timings.jsonl'
    stages=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    by_stage=defaultdict(lambda:dict(seconds=0.,calls=0,wall_seconds=0.))
    for row in stages:
        entry=by_stage[row['stage']];entry['seconds']+=row['process_cpu_seconds'];entry['wall_seconds']+=row['wall_seconds'];entry['calls']+=1
    for name,values in by_stage.items():
        cpu.append(dict(source='stage_timings.jsonl',scope=name,seconds=values['seconds'],records=values['calls'],
                        boundary='Current process CPU only; children excluded; stage records are sequential within each emitting process.'))
        if name.startswith(('input_checks','training_audit','replay_profile','replay_batch-parity','initial_budget_scores')):
            task_walls.append(dict(source='stage_timings.jsonl',scope=name,seconds=values['wall_seconds'],
                                   boundary='Whole stage wall while using GPU; includes CPU/I/O/waits; device concurrency not allocated.'))
    for name in ('final','independent'):
        path=run_dir/f'replay_{name}_status.json'
        if not path.exists():continue
        row=read_json(path)
        cpu.append(dict(source=path.name,scope=f'replay_{name}_feature_workers',seconds=row['worker_cpu_seconds'],
                        boundary='Child-worker process CPU; disjoint from replay parent stage CPU.'))
        gpu_calls.append(dict(source=path.name,scope=f'replay_{name}',device=row['device'],seconds=row['gpu_model_seconds'],
                              boundary='Sum of synchronized model-call wall time, CPU copy included; not active GPU kernel time.'))
        task_walls.append(dict(source=path.name,scope=f'replay_{name}',device=row['device'],seconds=row['elapsed_seconds'],
                               boundary='One replay task elapsed wall; not additive with parent TimedStage wall.'))
    for phase in ('discovery','confirmation'):
        paths=sorted((run_dir/'interventions'/phase/'jobs').glob('*.json'))
        jobs=[read_json(p) for p in paths]
        measured=[j for j in jobs if 'cpu_seconds' in j]
        if measured:
            cpu.append(dict(source=f'interventions/{phase}/jobs/*.json',scope=f'{phase}_worker_jobs',
                            seconds=sum(j['cpu_seconds'] for j in measured),records=len(measured),
                            boundary='Unique worker-job CPU, includes all arms within job; excludes model init and shared-plan prepare phase.'))
            task_walls.append(dict(source=f'interventions/{phase}/jobs/*.json',scope=f'{phase}_worker_job_wall_sum',
                                   seconds=sum(j['wall_seconds'] for j in measured),records=len(measured),
                                   boundary='Sum of concurrent worker-job elapsed seconds; neither experiment elapsed wall nor GPU allocation hours.'))
            names={k for j in measured for k in j.get('timing',{})}
            for name in sorted(names):
                total=sum(j.get('timing',{}).get(name,0.) for j in measured)
                record=dict(source=f'interventions/{phase}/jobs/*.json',scope=phase,stage=name,seconds=total,
                            boundary='Worker stage wall; nested stages overlap and are not summed as total runtime.')
                components.append(record)
                if name=='selector_model':gpu_calls.append(dict(record,boundary='Synchronized selector call wall; host+device+wait, not GPU kernel active.'))
            if len(measured)!=len(jobs):unmeasured.append(f'{phase}: {len(jobs)-len(measured)} worker job markers lack process CPU timing.')
    for dirname in ('same_records','q_input_checks','action_branches'):
        paths=sorted((run_dir/dirname).glob('*.json.gz'))
        values=[]
        for path in paths:
            with gzip.open(path,'rt') as handle:row=json.load(handle)
            if 'cpu_seconds' in row:values.append(float(row['cpu_seconds']))
        if values:cpu.append(dict(source=dirname+'/*.json.gz',scope=dirname+'_workers',seconds=sum(values),records=len(values),
                                  boundary='Unique worker artifacts only; not also counted from per-case/per-variant tables.'))
        if len(paths)>len(values):unmeasured.append(f'{dirname}: {len(paths)-len(values)} stored jobs have no process CPU timing.')
    probes=sum_csv(run_dir/'feature_interventions.csv','elapsed_model_s')
    if probes:gpu_calls.append(dict(source='feature_interventions.csv',scope='input_reading_probes',**probes,
                                   boundary='One row per actual probe; synchronized reader call wall including CPU transfer. Metadata/relabel/batch forwards not timed here.'))
    for filename in ('input_runtime_stages.csv','intervention_runtime_stage_breakdown.csv'):
        path=run_dir/filename
        if not path.exists():continue
        grouped=defaultdict(float);n=Counter()
        with path.open() as handle:
            for row in csv.DictReader(handle):
                key=(row.get('stage'),row.get('device'));grouped[key]+=float(row['seconds']);n[key]+=1
        for (stage,device),seconds in grouped.items():
            components.append(dict(source=filename,stage=stage,device=device,seconds=seconds,records=n[(stage,device)],
                                   boundary='Component/microbenchmark wall; subset of model/feature timings and possibly overlapping; not additive totals.'))
    # Reader timing from worker jobs is whole diagnosis (hashing/preprocessing/model),
    # so it deliberately is not labeled GPU model time.
    unmeasured.extend([
        'GPU active kernel time and CUDA-event intervals were not measured; synchronized wall is not substituted.',
        'GPU device allocation time and GPU utilization integration were not measured. Concurrent worker wall sums cannot recover them.',
        'Original existing training/evaluation compute is excluded; this report covers this investigation only.',
        'Pool worker CPU for initial_budget_scores is not emitted by that worker; parent-only stage CPU does not include it.',
        'Model loading, shared calibration-plan preparation, uninstrumented helper/test setup, retries whose job marker was replaced, and any still-running work can be absent from measured CPU totals.',
        'Ledger reservations count attempted operations including failures; cached reuse charges logical arm resources but does not incur another physical operation.',
        'Confirmation S shares A physical prefixes and existing q. A worker CPU is counted once; jobs_with_shared_prefix copies and S standalone timing are not counted as separate simulated jobs.',
    ])
    complete={}
    for name in ('replay_final_status.json','replay_independent_status.json','training_audit_status.json',
                 'interventions/discovery/complete.json','interventions/confirmation/complete.json',
                 'calibration_same_records_complete.json','q_input_action_checks_complete.json','action_value_checks_complete.json',
                 'initial_budget_scores_status.json'):
        path=run_dir/name;complete[name]=read_json(path).get('state','unknown') if path.exists() else 'not_yet_written'
    complete['input_checks.json']='passed' if (run_dir/'input_checks.json').exists() and not read_json(run_dir/'input_checks.json')['n_failed'] else 'missing_or_failed'
    complete['additional_input_checks.json']='passed' if (run_dir/'additional_input_checks.json').exists() and not read_json(run_dir/'additional_input_checks.json')['n_failed'] else 'missing_or_failed'
    for phase in ('discovery','confirmation'):
        name=f'intervention_independent_review_{phase}.json';path=run_dir/name
        review=read_json(path) if path.exists() else {}
        complete[name]='passed' if review and review['missing_rows']==0 and not review['failure_examples'] else 'not_complete_or_not_reviewed'
    cpu_total=sum(x['seconds'] for x in cpu)
    environment=read_json(run_dir/'environment.json') if (run_dir/'environment.json').exists() else None
    runtime_rows=runtime_breakdown(run_dir,components,gpu_calls,task_walls)
    result=dict(generated_at=datetime.now(timezone.utc).isoformat(),requested_final=final,
                state='final_snapshot' if final and all(v in ('passed','completed') for v in complete.values()) else 'partial_snapshot',
                operation_attempt_counts=counts,combined_model_forward_and_feature_build=combined,
                limits=ledger['limits'],limits_satisfied=counts['robot_execution']<=18000 and combined<=120000,
                operations_by_detail=ledger.get('by_detail',{}),
                measured_cpu_process_seconds_sum=cpu_total,measured_cpu_core_hours_sum=cpu_total/3600,
                cpu_scope='Measured partial sum/lower bound of investigation CPU; unmeasured portions explicitly listed.',
                cpu_components=cpu,gpu_synchronized_call_wall_components=gpu_calls,gpu_task_wall_components=task_walls,
                gpu_active_kernel_seconds=None,gpu_cuda_event_seconds=None,gpu_device_allocation_hours=None,
                stage_wall_components=components,unmeasured_or_nonadditive_notes=unmeasured,
                runtime_stage_breakdown_csv='runtime_stage_breakdown.csv',runtime_stage_breakdown_rows=runtime_rows,
                completion_status=complete,total_confirmed_hardware_gpus=environment.get('total_confirmed_gpus') if environment else None)
    write_json(run_dir/'resource_usage.json',result)
    lines=['# 本次调查资源用量', '',
           f"快照状态：{result['state']}，UTC {result['generated_at']}。仅统计本次新增调查，原训练/主实验成本不在这里重复计入。",'',
           '| 项目 | 实际账本次数 | 上限 |','|---|---:|---:|',
           f"| 机器人执行尝试 | {counts['robot_execution']} | 18,000 |",
           f"| 校准尝试 | {counts['calibration']} | 单臂按原预算；增补另列 |",
           f"| 模型 forward（批内按history计） | {counts['model_forward']} | 与feature合计120,000 |",
           f"| feature build | {counts['feature_build']} | 与forward合计120,000 |",
           f'| forward + feature | {combined} | 120,000 |','',
           f"可测进程CPU累计 {cpu_total:.3f} 秒 = {cpu_total/3600:.6f} 核时。这是有计时来源的部分总和/下界，包含独立worker CPU一次，未把worker wall或父stage wall当CPU再加一次。",'',
           '| CPU来源 | 核秒 |','|---|---:|']
    lines.extend(f"| {x['scope']} | {x['seconds']:.3f} |" for x in cpu)
    lines+=['','GPU同步模型调用wall、任务wall和部件microbenchmark分别存入resource_usage.json，互不相加。同步wall包含CPU调度/拷贝/等待；没有测GPU kernel active、CUDA event区间或GPU分配积分，不能称为GPU有效计算时间。','',
            'runtime_stage_breakdown.csv合并原CPU整体/调用级时间、新远端部件microbenchmark、replay与input profile；每行保留source、计时边界和重叠说明。原CPU没有记录matching/GRU/graph内部细分，明确标为未测。', '',
            f"environment.json只含硬件/软件清单，确认GPU数量：{result['total_confirmed_hardware_gpus']}；不含用户进程、GPU UUID或凭据。",'',
            '计量边界与未测项：','']
    lines.extend('- '+x for x in unmeasured)
    lines+=['','各阶段完成状态：','']
    lines.extend('- '+k+': '+v for k,v in complete.items())
    (run_dir/'RESOURCE_USAGE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('state','operation_attempt_counts','measured_cpu_core_hours_sum','limits_satisfied','total_confirmed_hardware_gpus')}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True)
    parser.add_argument('--collect-environment',action='store_true');parser.add_argument('--final',action='store_true')
    args=parser.parse_args()
    if args.collect_environment:collect_environment(args.run_dir)
    report(args.run_dir,args.final)
