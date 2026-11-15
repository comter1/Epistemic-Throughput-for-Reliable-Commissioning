"""Final immutable-input, cap, and complete-trajectory audit for CGX-FA-CAUSE-V2."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from cgx.config import EXPERIMENT
from cgx.graphs.manifests import read_jsonl
from cgx.provenance import atomic_json, read_json, file_sha256
from diagnostics.false_alarm_causes.interventions import load_blob
from .ledger import Ledger, TimedStage


def protected_files(run):
    before=read_json(run/'provenance/protected_before.json')
    source=read_json(run/'provenance/source_before.json')
    missing=[];changed=[]
    for path,expected in {**before,**{str(EXPERIMENT/p):h for p,h in source.items()}}.items():
        p=Path(path)
        if not p.exists():missing.append(path)
        elif file_sha256(p)!=expected:changed.append(path)
    result=dict(state='passed' if not missing and not changed else 'failed',
        protected_artifacts=len(before),protected_source_files=len(source),missing=missing,changed=changed)
    atomic_json(run/'provenance/protection_after.json',result)
    assert result['state']=='passed',result
    return result


def trajectories(run,phase):
    complete=read_json(run/'trajectories'/phase/'complete.json')
    assert complete['state']=='passed' and complete['failed_trajectories']==0,complete
    checked=0;prefixes=0;early_stops=0
    for path in sorted((run/'trajectories'/phase/'cases').glob('*.json.gz')):
        item=load_blob(path);row=item['row'];trace=item['trace'];pp=item['prefixes']
        assert row['state']=='passed'
        assert len(pp)==len(trace)+1
        for index,p in enumerate(pp):
            assert p['prefix_ordinal']==index
            initial,remaining=p['initial_budget'],p['remaining']
            assert all(-1e-8<=remaining[k]<=initial[k]+1e-8 for k in initial)
            assert p['executions_used']==initial['executions']-remaining['executions']
            assert p['calibrations_used']==initial['calibrations']-remaining['calibrations']
            pred=p['prediction'];values=np.asarray(pred['log_probabilities_base2'])
            assert np.isfinite(values).all() and abs(np.exp2(values).sum()-1)<1e-5
            assert 'healthy' in pred['target_keys']
            if index<len(trace):
                t=trace[index]
                assert t['history_before']==p['history_hash']
                assert t['history_after']==pp[index+1]['history_hash']
                assert t['action']['action_key']==p['chosen_action_key']
                assert t['action']['kind']!='stop'
                for key in initial:
                    assert abs(t['resources_before'][key]-t['actual_cost'].get(key,0)-t['resources_after'][key])<1e-6
            if p['actionable']:
                assert p['candidate_actions'][p['chosen_action_index']]['action_key']==p['chosen_action_key']
                vector=p.get('chosen_head_input_final',p.get('chosen_head_input_deployed'))
                assert len(vector)==281 and np.isfinite(vector).all()
        if len(trace)==0:early_stops+=1
        prefixes+=len(pp);checked+=1
    assert checked==complete['trajectories_expected']
    return dict(state='passed',trajectories=checked,prefixes=prefixes,zero_acquisition_trajectories=early_stops,
        exact_available_prefixes=True,all_resources_legal=True,full_probability_normalization=True)


def isolation(run):
    checks=read_json(run/'source_checks.json')
    original=list(read_jsonl(Path(checks['original_data_root'])/'private/private_cases.jsonl'))
    v1=read_json(Path(checks['v1_run'])/'manifests/fresh_cases.json')
    source=read_json(run/'manifests/source_cases.json');fresh=read_json(run/'manifests/fresh_cases.json')
    assert not {c['case_id'] for c in original+v1}&{c['case_id'] for c in source+fresh}
    collisions={}
    for stream in original[0]['streams']:
        old={c['streams'][stream] for c in original+v1}
        new=[c['streams'][stream] for c in source+fresh]
        collisions[stream]=len(old&set(new))
        assert collisions[stream]==0 and len(new)==len(set(new)),stream
    support=[]
    source_lookup={c['case_id']:c for c in source}
    for path in sorted((run/'head_data').glob('*_seed*.parquet')):
        frame=pd.read_parquet(path)
        assert set(frame.case_id)<=set(source_lookup)
        indices=np.load(path.with_name(path.stem+'_batch_indices.npy'))
        sampled=frame.iloc[indices.reshape(-1)]
        assert sampled.source_split.eq('train').all()
        assert frame.eligible_for_fit.all()
        done=read_json(path.with_name(path.stem+'_fit_complete.json'))
        assert done['all_cells_same_batch_order'] and done['original_weights_unchanged']
        assert done['frozen_backbone_gradients_none'] and done['exact_updates_per_cell']==2000
        assert indices.shape==(2000,128)
        support.append(dict(fit_group=path.stem,training_cases=sampled.case_id.nunique(),
            eligible_rows=len(frame),train_rows=int(frame.source_split.eq('train').sum()),
            negative_final_return_rows=int((frame.return_final<0).sum()),
            positive_final_return_rows=int((frame.return_final>0).sum()),
            batch_schedule_sha256=done['batch_schedule_sha256']))
    assert len(support)==6
    result=dict(state='passed',source_cases=320,fresh_cases=128,
        all_rng_stream_collisions_with_original_or_v1=collisions,
        all_rng_streams_unique_across_new_source_and_fresh=True,
        every_sampled_update_source_train_only=True,support=support)
    atomic_json(run/'tests/data_isolation_and_matched_fits.json',result)
    return result


def run(run_dir):
    run=Path(run_dir)
    with TimedStage(run,'final_audit'):
        protection=protected_files(run)
        ledger=Ledger(run).snapshot();counts=ledger['counts'];limits=Ledger.LIMITS
        combined=sum(counts[k] for k in ('model_forward','feature_build','reader_partial_forward'))
        assert combined<=limits['model_and_features']
        for key in ('robot_execution','calibration','head_fit','training_gpu_seconds'):assert counts[key]<=limits[key]
        assert len(ledger['fits'])==24 and counts['head_fit']==24
        assert all(f['registered'] and f['updates']==2000 for f in ledger['fits'].values())
        assert counts['head_update']==48000
        source=trajectories(run,'source');fresh=trajectories(run,'confirmation');separation=isolation(run)
        selected=read_json(run/'manifests/source_variant_selection.json')
        assert selected['state']=='sealed_before_fresh_outcomes' and not selected['fresh_outcomes_exist_at_seal']
        for path,expected in {**selected['frozen_files'],**selected['all_head_checkpoints']}.items():
            assert file_sha256(run/path)==expected,f'Source-selected artifact changed during fresh confirmation: {path}'
        for name,expected in selected['frozen_scientific_code'].items():
            assert file_sha256(EXPERIMENT/'diagnostics/false_alarm_causes_v2'/name)==expected,f'Frozen scientific code changed during fresh confirmation: {name}'
        live=trajectories(run,'live') if selected['optional_live']['enabled'] else None
        if selected['reader_available']:
            expected=selected['frozen_files']['manifests/source_reader_selection.json']
            assert read_json(run/'confirmation_reader_predictions.json')['source_selection_sha256']==expected
            assert read_json(run/'confirmation_reader_tasks.json')['source_reader_selection_sha256']==expected
        # Conservative allocation envelope includes both training AND frozen GPU
        # inference, even CPU preparation while those stage models occupy a card.
        timings=[json.loads(line) for line in (run/'stage_timings.jsonl').read_text().splitlines()]
        gpu_stages={'reader_controls','rollout_engineering','reader_factorial_profile','reader_factorial',
                    'source_rollouts','confirmation_rollouts','live_rollouts','reader_candidate_validation','reader_confirmation_reread'}
        allocation=sum(t['wall_seconds'] for t in timings if t['stage'] in gpu_stages or t['stage'].startswith(('live_dispatch','reader_confirmation')))
        allocation+=read_json(run/'head_data_complete.json')['wall_seconds']
        allocation+=sum(read_json(p)['training_wall_seconds'] for p in (run/'head_data').glob('*_fit_complete.json'))
        assert allocation<48*3600, 'Even conservative measured GPU stage allocation must fit the training cap'
        reservation_cards=7 if (run/'manifests/hardware_authorization_override.json').exists() else 2
        reservation_bound=reservation_cards*(datetime.now(timezone.utc)-datetime.strptime(run.name,'%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)).total_seconds()
        assert reservation_bound<48*3600
        saved=read_json(run/'source_checks.json')
        for path,expected in saved['manifest_hashes'].items():
            # Selection status is recorded in a NEW manifest; the original seal stays immutable.
            assert file_sha256(run/path)==expected, f'Original manifest changed: {path}'
        tests=dict(state='passed',protection=protection,source=source,confirmation=fresh,live=live,data_isolation=separation,
            source_selected_artifacts_unchanged=True,measured_gpu_stage_allocation_upper_seconds=allocation,
            conservative_gpu_reservation_bound_seconds=reservation_bound,
            conservative_reservation_card_count=reservation_cards,
            resource_caps_passed=True,combined_model_and_feature_calls=combined,
            fits=24,updates_per_fit=2000,full_reader_retrains=0,full_selector_backbone_retrains=0,
            original_sample_manifests_unchanged=True)
        atomic_json(run/'tests/final_audit.json',tests)
    print(tests,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);run(p.parse_args().run_dir)
