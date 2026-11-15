"""User-authorized 48-hour first report, followed by the original full protocol.

The original data protocol/source seal is immutable. Collection runs under its
archived source; both registered analysis profiles share that exact corpus.
"""
from pathlib import Path
from copy import deepcopy
import fcntl
import json
import os
import subprocess
import sys
import time
import shutil

from cgx.config import EXPERIMENT,load_config
from cgx.provenance import atomic_json,read_json,file_sha256,hash_json,verify_hashes,now,require_protocol_lock


def primary_config(original):
    config=deepcopy(original)
    config['amendment_id']='CGX-1.0-P48-1'
    config['delivery_scope']='All systems and cases; seed11; primary byte fraction; shortened equal training. Full-protocol confirmation remains outstanding.'
    config['seeds']['training']=[11]
    config['training']['predictor_update_cap']=10000
    config['training']['policy_update_cap']=20000
    config['resources']['byte_fractions']=[config['evaluation']['primary_byte_fraction']]
    config['evaluation']['repair_methods']=[]
    config['evaluation']['repair_training_seeds']=[]
    config['evaluation']['report_all_budgets']=False
    return config


def _link(path,target):
    if path.is_symlink():
        if path.resolve()!=target.resolve():raise ValueError(f'Unexpected shared corpus link: {path}')
    elif path.exists():raise FileExistsError(path)
    else:path.parent.mkdir(parents=True,exist_ok=True);path.symlink_to(target,target_is_directory=True)


def prepare_profile(base,name,config,qualification):
    from .locking import lock_protocol
    from .forecast import resource_forecast
    base=Path(base);root=base/'runs'/name;qualification=Path(qualification)
    if (root/'protocol_lock.json').exists():
        require_protocol_lock(root,config);return root
    root.mkdir(parents=True,exist_ok=True)
    for directory in ('assets','manifests','private','public'):_link(root/directory,base/directory)
    _link(root/'jobs/collection',base/'jobs/collection')
    if name=='full_protocol':_link(root/'encoded',base/'runs/primary_48h/encoded')
    (root/'reports').mkdir(exist_ok=True)
    shutil.copy2(base/'environment.json',root/'environment.json')
    for filename in ('qualification.json','manifest_validation.json','legacy_integrity.json','source_healthy_catalog.json','execution_profile.json'):
        shutil.copy2(base/'reports'/filename,root/'reports'/filename)
    for filename in ('technical_smoke.json','pytest.xml','parallel_rollouts.json'):
        shutil.copy2(qualification/'reports'/filename,root/'reports'/filename)
    # The parent data manifest is the same on every profile. Training/profile
    # changes never regenerate a graph, target, task, fault, or observation.
    original=load_config(EXPERIMENT/'configs/protocol.json')
    expected=primary_config(original) if name=='primary_48h' else original
    if config!=expected:raise ValueError('Unregistered staged experiment profile')
    for key in ('splits','targets','system','faults','sensing','observations'):
        if config[key]!=original[key]:raise ValueError('The staged profile changed the data-generating protocol')
    inherited=dict(state='passed',source_root=str(base),source_protocol_lock_hash=file_sha256(base/'protocol_lock.json'),
        manifest_hash=file_sha256(base/'manifests/hashes.json'),data_configuration_hash=hash_json(original),profile_configuration_hash=hash_json(config),
        source_code_archive_hash=file_sha256(base/'software/source.tar.gz'),all_training_and_test_cases_preserved=True,
        authorization='User explicitly chose one-seed fixed shorter training and primary-budget first report on 2026-11-13; full remainder follows.')
    atomic_json(root/'reports/corpus_inheritance.json',inherited)
    resource_forecast(root,config,base/'technical_smoke/pipeline/reports/learning_profile.json')
    lock_protocol(root,config)
    return root


def _alive(pid):
    try:
        if Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z':return False
        os.kill(pid,0);return True
    except (ProcessLookupError,FileNotFoundError):return False


def run_staged(base,config,workers=16,gpus='0,1',resume=False):
    os.environ.setdefault('CGX_EVAL_WORKERS','16')
    os.environ.setdefault('CGX_DATA_WORKERS',str(workers))
    os.environ.setdefault('CGX_REMOTE_WORKERS_FILE',str(Path(base)/'reports/remote_workers.json'))
    base=Path(base);directory=base/'staged_run';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'driver.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run(base,config,workers,gpus,resume,directory)


def _run(base,config,workers,gpus,resume,directory):
    qualification=base/'replanning/qualification'
    state={'state':'running','started_at':now(),'pid':os.getpid(),'current_step':'wait_for_common_collection',
           'delivery_target_utc':'2026-11-15T21:00:00+00:00','steps':{},'profiles':{},'full_protocol_completed':False,
           'primary_report_completed':False,'gpus':gpus,'cpu_rollout_workers':os.environ.get('CGX_EVAL_WORKERS','16')}
    status=directory/'status.json'
    if status.exists():
        if not resume:raise FileExistsError(status)
        prior=read_json(status)
        for step in prior.get('steps',{}).values():
            if step.get('state')=='running' and _alive(step.get('pid',0)):raise RuntimeError('Previous staged subprocess still alive')
        state['steps']=prior.get('steps',{});state['started_at']=prior.get('started_at',state['started_at'])
    def update():
        atomic_json(status,state)
        text=['# Actual staged execution status','',f"Updated: {now()}",f"Current stage: {state['current_step']}",
              f"Orchestrator: {state['state']} (PID {state['pid']})",'',
              f"48-hour primary report completed: **{state['primary_report_completed']}**.",
              f"Original full protocol completed: **{state['full_protocol_completed']}**.",'',
              'The first report uses all systems/cases, five learned methods, seed11, 10k reader / 20k policy updates, and byte fraction0.25.',
              'Other seeds/budgets and repair/audit are a separately sealed subsequent stage. No unexecuted item is passed.',
              '',f'Live detailed status: `{status}`']
        (EXPERIMENT/'STATUS.md').write_text('\n'.join(text)+'\n')
    def command(label,args,root,cfg=None,old_source=False):
        state['current_step']=label
        cmd=[sys.executable,'-m','cgx.cli',*args,'--data-root',str(root),'--config',str(cfg or root/'software/protocol.json')]
        log=directory/(label.replace('/','_')+'.log')
        env=dict(os.environ,CGX_DATA_ROOT=str(root),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
        if old_source:env['PYTHONPATH']=str(base/'replanning/original_main/sealed_source/src')
        else:env['PYTHONPATH']=str(EXPERIMENT/'src')
        item=dict(state='running',command=cmd,started_at=now(),log=str(log));state['steps'][label]=item;update()
        with log.open('a') as stream:
            process=subprocess.Popen(cmd,env=env,cwd=EXPERIMENT,stdout=stream,stderr=subprocess.STDOUT)
            item['pid']=process.pid;update();code=process.wait()
        item.update(state='passed' if code==0 else 'failed',exit_code=code,finished_at=now(),log_hash=file_sha256(log));update()
        if code:raise RuntimeError(f'Stage {label} failed: {log}')
    try:
        # Both profiles are sealed before any learned test outcome is available.
        primary=prepare_profile(base,'primary_48h',primary_config(config),qualification)
        full=prepare_profile(base,'full_protocol',config,qualification)
        state['profiles']={name:{'root':str(root),'config_hash':read_json(root/'protocol_lock.json')['config_hash']} for name,root in [('primary_48h',primary),('full_protocol',full)]}
        update()
        train_pid=read_json(base/'reports/replanning.json')['collection_pid']
        val_pid=read_json(base/'reports/validation_collection_launch.json')['pid']
        while any(_alive(pid) for pid in (train_pid,val_pid)):
            state['collection']={split:read_json(base/'reports'/f'collection_{split}.json') for split in ('train','validation') if (base/'reports'/f'collection_{split}.json').exists()}
            update();time.sleep(30)
        for split in ('train','validation'):
            report=base/'reports'/f'collection_{split}.json'
            if not report.exists() or read_json(report).get('state')!='passed':
                command('resume_collection_'+split,['collect','--split',split,'--workers',str(workers),'--resume'],base,
                        base/'replanning/original_main/sealed_source/configs/protocol.json',True)
        for split in ('train','validation'):
            command('validate_'+split,['validate-data','--split',split,'--strict'],primary)
        for name,root in [('primary_48h',primary),('full_protocol',full)]:
            if name=='full_protocol':
                for split in ('train','validation'):shutil.copy2(primary/'reports'/f'data_validation_{split}.json',root/'reports'/f'data_validation_{split}.json')
            command(name+'/prepare',['prepare-replay','--workers',str(workers),'--resume'],root)
            for phase in ('readers','policies'):
                if phase=='policies':command(name+'/crossfit',['score-transitions','--crossfit-only','--resume'],root)
                jobs=root/'jobs'/f'{phase}.jsonl'
                command(name+'/'+phase+'_jobs',['build-job-manifest','--phase',phase,'--out',str(jobs)],root)
                command(name+'/'+phase,['run-queue','--gpus',gpus,'--jobs',str(jobs),'--resume'],root)
            command(name+'/select',['select-checkpoints','--validation-only'],root)
            command(name+'/freeze',['freeze-models'],root)
            command(name+'/main',['evaluate','--suite','all','--resume'],root)
            command(name+'/independent',['reread','--reader','independent','--same-transcripts'],root)
            if name=='full_protocol':
                command(name+'/sensitivity',['evaluate-round-sensitivity','--resume'],root)
                command(name+'/repairs',['evaluate-repairs','--preregistered-subset','--resume'],root)
            command(name+'/aggregate',['aggregate','--paired','--bootstrap','2000'],root)
            command(name+'/report',['report','--include-negative-results'],root)
            command(name+'/export',['export','--out',str(root/'release'/f'{name}_results.zip')],root)
            scope={'state':'passed','completed_at':now(),'profile':name,'scope':read_json(root/'software/protocol.json').get('delivery_scope','Full CGX-1.0'),
                   'full_protocol_completed':name=='full_protocol','seed_variability_established':name=='full_protocol',
                   'result_hash_manifest':file_sha256(root/'results/hashes.json')}
            atomic_json(root/'reports/delivery_complete.json',scope)
            state['primary_report_completed' if name=='primary_48h' else 'full_protocol_completed']=True;update()
        state.update(state='passed',finished_at=now());update();return state
    except BaseException as exc:
        state.update(state='failed',failure_reason=f'{type(exc).__name__}: {exc}',finished_at=now());update();raise
