"""Actual isolated subprocess fixtures with explicit artificial markers; no scientific training is simulated as done."""
from pathlib import Path
import json
import os
import sys
import pytest
from cgx.provenance import atomic_json,hash_json,file_sha256,read_json
from cgx.runtime import queue

@pytest.fixture
def queue_fixture(tmp_path,monkeypatch):
    # Isolated protocol seal mock; actual input/marker/checkpoint verification remains active.
    lock={'state':'passed','config_hash':'fixture-config','technical_smoke':True}
    monkeypatch.setattr(queue,'require_protocol_lock',lambda root,config=None:lock)
    queries=[]
    def discover(command,**kwargs):
        queries.append(command)
        if '--query-gpu=index,uuid' in command:return '0, GPU-fixture-0\n1, GPU-fixture-1\n'
        return ''
    monkeypatch.setattr(queue.subprocess,'check_output',discover)
    atomic_json(tmp_path/'software/protocol.json',{'compute':{'gpu_hours_initial_soft_cap':320},'training':{'predictor_update_cap':2,'policy_update_cap':2},'seeds':{'training':[11]}})
    feature_manifest=tmp_path/'training/feature_hashes.jsonl';feature_manifest.parent.mkdir(parents=True)
    feature_manifest.write_text('fixture-only\n')
    atomic_json(tmp_path/'training/features_complete.json',{'state':'passed','identity':{
        'protocol_lock_hash':hash_json(lock),'replays':{'train':'fixture-train','validation':'fixture-validation'}},
        'manifest_sha256':file_sha256(feature_manifest)})
    expected,prerequisites=queue._phase_inputs(tmp_path,lock,'reader_training')
    def make(name,mode='success',exit_code=0):
        directory=tmp_path/'models/readers'/name
        marker=directory/'complete.json'
        identity={**expected,'reader':name,'seed':11,'excluded_fold':None}
        # Child creates real files and exits; these bytes are fixture artifacts, not weights.
        script="""
from pathlib import Path
import hashlib,json,os,sys,time
p=Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True)
identity=json.loads(sys.argv[2]);mode=sys.argv[3]
(p/'device.txt').write_text(os.environ['CUDA_VISIBLE_DEVICES'])
(p/'checkpoint.pt').write_bytes(b'explicit-runtime-test-fixture-not-trained-weights')
(p/'validation.json').write_text('[{"fixture":true,"actual_updates":2,"iteration":2,"mean_final_nll_bits":1.0}]')
marker={'state':'passed','technical_smoke':False,'actual_updates':2,'identity':identity,'selected_checkpoint':'checkpoint.pt','selected_checkpoint_hash':hashlib.sha256((p/'checkpoint.pt').read_bytes()).hexdigest()}
if mode=='wrong_identity':marker['identity']['protocol_lock_hash']='foreign'
if mode=='wrong_updates':marker['actual_updates']=3
if mode=='smoke':marker['technical_smoke']=True
if mode=='missing_checkpoint':(p/'checkpoint.pt').unlink()
if mode=='malformed':(p/'complete.json').write_text('{')
elif mode!='missing_marker':(p/'complete.json').write_text(json.dumps(marker))
time.sleep(.1)
sys.exit(int(sys.argv[4]))
"""
        return {'job_id':name,'phase':'reader_training','reader':name,'state':'not_started',
                'input_hashes':{'protocol_lock':hash_json(lock)},'expected_identity':identity,
                'prerequisite_files':prerequisites,'maximum_updates':2,'completion_marker':str(marker),
                'command':[sys.executable,'-c',script,str(directory),json.dumps(identity),mode,str(exit_code)]}
    def manifest(jobs):
        path=tmp_path/'jobs_fixture.jsonl';path.write_text(''.join(json.dumps(j)+'\n' for j in jobs));return path
    return tmp_path,make,manifest,queries


def test_A06_E11_two_gpu_queue_and_corrupt_checkpoint_resume(queue_fixture,monkeypatch):
    root,make,manifest,queries=queue_fixture
    jobs=[make('fixture_a'),make('fixture_b')];path=manifest(jobs)
    result=queue.run_queue(root,path,gpus=(0,1))
    assert result['launched']==2
    devices=set()
    for job in jobs:
        directory=Path(job['completion_marker']).parent;devices.add((directory/'device.txt').read_text())
        status=read_json(root/'jobs'/job['job_id']/'status.json')
        assert status['state']=='passed' and status['actual_updates']==2
        assert len(status['output_hashes'])==3
        assert status['exit_code']==0
    assert devices=={'GPU-fixture-0','GPU-fixture-1'}
    # Completed artifact verification succeeds even with discovery unavailable.
    monkeypatch.setattr(queue,'gpu_uuids',lambda _:pytest.fail('Resume must not acquire GPUs'))
    assert queue.run_queue(root,path,resume=True)['launched']==0
    (Path(jobs[0]['completion_marker']).parent/'checkpoint.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checkpoint missing or changed'):
        queue.run_queue(root,path,resume=True)


@pytest.mark.parametrize('mode,exit_code',[('success',7),('missing_marker',0),('wrong_identity',0),('wrong_updates',0),('smoke',0),('missing_checkpoint',0),('malformed',0)])
def test_A06_failed_subprocess_or_invalid_marker_never_success(queue_fixture,mode,exit_code):
    root,make,manifest,_=queue_fixture
    job=make('fixture_failure',mode,exit_code)
    with pytest.raises(RuntimeError,match='Failed jobs retained'):
        queue.run_queue(root,manifest([job]),gpus=(0,))
    status=read_json(root/'jobs/fixture_failure/status.json')
    assert status['state']=='failed' and status['exit_code']==exit_code
    assert status['output_hashes']=={} and status['failure_reason']
    assert status['finished_at'] and status['elapsed_seconds']>0


def test_A06_spawn_failure_and_retry_evidence(queue_fixture):
    root,make,manifest,_=queue_fixture
    job=make('fixture_spawn_failure');job['command']=['/definitely/absent/cgx-fixture-executable']
    path=manifest([job])
    for resume in (False,True):
        with pytest.raises(RuntimeError,match='Failed jobs retained'):
            queue.run_queue(root,path,gpus=(0,),resume=resume)
    status=read_json(root/'jobs/fixture_spawn_failure/status.json')
    assert status['state']=='failed' and status['exit_code'] is None
    assert len(status['attempts'])==1 and status['attempts'][0]['state']=='failed'
    assert status['cumulative_elapsed_seconds']>=status['elapsed_seconds']


def test_A06_validation_change_and_prerequisite_change_rejected(queue_fixture):
    root,make,manifest,_=queue_fixture
    job=make('fixture_validation');path=manifest([job]);queue.run_queue(root,path,gpus=(0,))
    validation=Path(job['completion_marker']).parent/'validation.json';validation.write_text('[]')
    with pytest.raises(ValueError,match='completion/update history mismatch|Completed job outputs changed'):
        queue.run_queue(root,path,resume=True)
    (root/'training/feature_hashes.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='feature manifest changed'):
        queue.run_queue(root,path,resume=True)


def test_gpu_discovery_rejects_unknown_and_duplicate(queue_fixture):
    with pytest.raises(ValueError,match='Unknown physical GPU'):queue.gpu_uuids(['9'])
    with pytest.raises(ValueError,match='distinct'):queue.gpu_uuids(['0','GPU-fixture-0'])

def test_A05_registered_update_cap_and_real_manifest_inputs(queue_fixture):
    from cgx.config import load_config
    root,make,manifest,_=queue_fixture
    job=make('fixture_cap');job['maximum_updates']=1
    with pytest.raises(ValueError,match='update cap differs'):
        queue.run_queue(root,manifest([job]),gpus=(0,))
    config=load_config();config['training']['predictor_update_cap']=0;config['training']['policy_update_cap']=0
    reader_jobs=queue.build_jobs(root,config,phase='readers')
    rows=[json.loads(line) for line in Path(reader_jobs['path']).read_text().splitlines()]
    assert len(rows)==5 and rows[-1]['expected_identity']['seed']==1020
    assert rows[1]['expected_identity']['excluded_fold']==1
    final=make('final');queue.run_queue(root,manifest([final]),gpus=(0,))
    score_db=root/'training/scores.sqlite';score_db.write_bytes(b'explicit-fixture-not-a-training-database')
    atomic_json(root/'training/scores_complete.json',{'state':'passed','identity':{
        'protocol_lock_hash':final['expected_identity']['protocol_lock_hash'],'replay_hash':'fixture-train'},
        'output_hash':file_sha256(score_db)})
    policy_jobs=queue.build_jobs(root,config,phase='policies')
    rows=[json.loads(line) for line in Path(policy_jobs['path']).read_text().splitlines()]
    assert len(rows)==15 and rows[0]['expected_identity']['reward_hash']==file_sha256(score_db)
    assert rows[0]['expected_identity']['reader_hash']==read_json(Path(final['completion_marker']))['selected_checkpoint_hash']
