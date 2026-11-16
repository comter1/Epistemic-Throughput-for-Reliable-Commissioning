"""Read-only, posthoc footprint of modeled evidence in erroneous diagnoses.

A missing modeled channel is not a causal label for selector failure: this SED
does not model external motion, and the original actions came from policies
trained with another reader. Presence of a channel is not identifiability or a
guarantee that noisy observations should yield the correct answer. The legacy
``within_clock_mixed_packet_groups`` counter counts packet-by-clock strata.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import pandas as pd
from cgx.provenance import atomic_json,file_sha256,hash_json,now,read_json
from .gates import verify_readers
from .reader import load_reader
from .replay import provenance_for_history,prediction_metrics
from .storage import OverlayStore

VERSION='sed-modeled-channel-footprint-v1'
DESIGN=dict(version=VERSION,analysis_type='posthoc descriptive evidence footprint; no model selection',
    initial_phase='All wrong Structured Evidence Diagnoser main replay contexts, with all correct and wrong main rows retained as denominators',
    work_unit='Unique physical history/provenance job ID; merge back to every original method/seed/budget context',
    gain_channel='At least one acquired matched-packet local adapter input/output gain factor after public preparation',
    timing_mean_channel='At least one packet-epoch group with both upstream and downstream fault-path indicators',
    process_variance_only='No timing mean channel, positive learned family process variance, and a downstream observation in a packet group with at least two observations',
    label_boundary='Prepare public history and all candidate footprints first; evaluator truth only selects its footprint afterwards',
    missing_channel_caveat='Absence is relative to this fixed SED, which ignores external motion and missingness as fault evidence',
    channel_presence_caveat='A channel can be weak, noisy, clock-confounded or misspecified; presence does not establish identifiability',
    policy_caveat='Old histories were selected by policies trained with another reader; these categories cannot uniquely blame the selector or diagnoser',
    no_causal_error_labels=True,no_model_or_policy_changes=True,physical_acquisitions=0,oracle_reruns=0,optimizer_steps=0)
MANIFEST='manifests/reader_error_footprint_posthoc.json'
_STATE=None


def declare(run_dir):
    run=Path(run_dir);verify_readers(run)
    path=run/MANIFEST
    value=dict(state='posthoc_analysis_declared',declared_at=now(),design=DESIGN,
        reader_frozen_sha256=file_sha256(run/'models/readers/frozen.json'),
        fixed_replay_complete_sha256=file_sha256(run/'replay/complete.json'))
    if path.exists():
        old=read_json(path)
        for key in ('design','reader_frozen_sha256','fixed_replay_complete_sha256'):assert old[key]==value[key]
        return old
    atomic_json(path,value);return value


def public_footprint(prepared,prediction,parameters):
    """All candidates are described without a true/predicted fault argument."""
    p=prepared;meta=p['meta'];n=len(p['times']);ng=p['group_count'];k=len(meta['clocks'])
    group_n=np.bincount(p['groups'],minlength=ng)
    group_clock_n=np.bincount(p['groups']*k+p['clocks'],minlength=ng*k).reshape(ng,k)
    clocks_per_group=(group_clock_n>0).sum(axis=1)
    channels={}
    for target in meta['targets']:
        if target=='healthy':
            channels[target]=dict(applicable=False,kind='healthy_has_no_fault_site')
            continue
        node,family=target.split(':')
        if family=='adapter_scale':
            count=len(p['gain'].get(node,[]))
            channels[target]=dict(applicable=True,kind='adapter_gain',matched_gain_packet_factors=count,
                direct_mean_channel=count>0,process_variance_only=False,any_modeled_channel=count>0)
            continue
        index=meta['delay_nodes'].index(node);v=p['delay_path'][:,index]
        down=np.bincount(p['groups'],weights=v,minlength=ng)
        mixed=(down>0)&(down<group_n)
        per_clock_down=np.bincount(p['groups']*k+p['clocks'],weights=v,minlength=ng*k).reshape(ng,k)
        clock_free=(per_clock_down>0)&(per_clock_down<group_clock_n)
        extra=float(parameters['timing_process_variance'].get(family,0.))
        variance_groups=(down>0)&(group_n>1)
        mean=bool(mixed.any());process=bool(extra>0 and variance_groups.any())
        channels[target]=dict(applicable=True,kind='timing_delay',mixed_upstream_downstream_packet_groups=int(mixed.sum()),
            within_clock_mixed_packet_groups=int(clock_free.sum()),downstream_timestamp_events=int(v.sum()),
            upstream_timestamp_events=int(n-v.sum()),downstream_only_packet_groups=int(((down==group_n)&(down>0)).sum()),
            variance_affected_packet_groups=int(variance_groups.sum()),learned_process_variance=extra,
            direct_mean_channel=mean,process_variance_channel=process,
            process_variance_only=bool(process and not mean),any_modeled_channel=bool(mean or process))
    diagnostics=prediction['structured_diagnostics'];cov=np.asarray(diagnostics['clock_covariance'])
    relative=np.diag(cov)[:,None]+np.diag(cov)[None,:]-2*cov
    pair_values=relative[np.triu_indices(k,1)]
    raw_internal=p['unique_events']-p['external_events'];calibrations=len(p['calibration'])
    return dict(unique_measurement_events=p['unique_events'],internal_record_events=raw_internal,
        modeled_internal_timestamp_events=n,external_record_events=p['external_events'],missing_events=p['missing_events'],
        no_internal_measurements=raw_internal==0,no_modeled_internal_measurements=n==0,
        external_only=raw_internal==0 and p['external_events']>0 and calibrations==0,
        external_plus_calibration_only=raw_internal==0 and p['external_events']>0 and calibrations>0,
        timing_packet_groups=ng,timing_contrasts=n-ng,cross_clock_packet_groups=int((clocks_per_group>1).sum()),
        observed_clock_groups=int(len(np.unique(p['clocks']))),public_clock_groups=k,
        calibration_exchanges=calibrations,calibrated_clock_pairs=len({tuple(sorted((int(a),int(b)))) for a,b,_ in p['calibration']}),
        matched_gain_packet_factors=sum(len(values) for values in p['gain'].values()),
        clock_prior_variance=parameters['clock_prior_variance'],
        mean_relative_clock_posterior_variance=float(pair_values.mean()) if len(pair_values) else None,
        target_channels_json=json.dumps(channels,sort_keys=True),
        relative_log_likelihoods_bits_json=json.dumps(dict(zip(meta['targets'],
            (np.asarray(diagnostics['relative_log_likelihoods'])/np.log(2.)).tolist())),sort_keys=True),
        recomputed_target_keys_json=json.dumps(prediction['target_keys']),
        recomputed_log_probabilities_json=json.dumps(prediction['log_probabilities_base2']))


def _initialize(source_root,run_dir,cohort):
    global _STATE
    _STATE=(OverlayStore(source_root,run_dir),load_reader(run_dir,'final',device='cpu'),Path(run_dir),cohort)


def _job(job):
    store,reader,run,cohort=_STATE;directory=run/'error_audit'/cohort
    path=directory/'jobs'/f"{job['job_id']}.json"
    if path.exists():
        saved=read_json(path);assert saved['input_identity']==job['input_identity']
        assert saved['record_sha256']==hash_json(saved['record'])
        return saved['record']
    result=dict(job_id=job['job_id'],history_hash=job.get('history_hash'),case_id=job['case_id'],audit_state='unreplayable')
    started=time.monotonic();fingerprint=reader.parameter_fingerprint()
    try:
        if not job.get('history_hash'):raise ValueError('Saved final public history unavailable')
        history=store.load_history(job['history_hash']);packed=store.history_references(job['history_hash'])
        if cohort=='old_histories':
            provenance=provenance_for_history(history,job['case_id'])
            assert hash_json(provenance)==job['provenance_hash']
        else:
            if not job.get('provenance_hash'):raise ValueError('Saved explicit acquisition provenance unavailable')
            provenance=store.load_provenance(job['history_hash'],job['provenance_hash'])
        if 'observation_refs' in provenance:assert provenance['observation_refs']==packed['observation_refs']
        provenance={**provenance,'observation_refs':packed['observation_refs']}
        prepared=reader.prepare(history,provenance)
        prediction=reader.predict_prepared(prepared,include_latents=True)
        # Every candidate footprint is prepared before any truth label selects
        # a site. The worker job contains no true fault or injected severity.
        result.update(public_footprint(prepared,prediction,reader.parameters_dict))
        if cohort=='old_histories':
            keys=json.loads(job['target_keys']);lp=np.asarray(json.loads(job['full_log_probabilities']))
            assert keys==prediction['target_keys']
            error=float(np.max(np.abs(np.exp2(lp)-np.exp2(prediction['log_probabilities_base2']))))
            if error>1e-6:raise ValueError('Frozen reader disagrees with sealed original-history replay')
            result['sealed_replay_max_probability_error']=error
        result.update(audit_state='passed',provenance_hash=job['provenance_hash'],reader_hash=reader.checkpoint_hash,
            all_candidate_footprints_prepared_before_truth=True)
    except Exception as exc:
        result.update(audit_state='unreplayable',audit_error=f'{type(exc).__name__}:{exc}')
    if reader.parameter_fingerprint()!=fingerprint:raise RuntimeError('Frozen reader parameters changed')
    result['wall_seconds']=time.monotonic()-started
    atomic_json(path,dict(input_identity=job['input_identity'],record_sha256=hash_json(result),record=result))
    return result


def _contexts(source_root,run,cohort):
    if cohort=='old_histories':
        complete=read_json(run/'replay/complete.json');path=run/'replay/per_case_predictions.parquet'
        assert complete['state']=='passed' and file_sha256(path)==complete['per_case_predictions_sha256']
        frame=pd.read_parquet(path,filters=[('cohort','==','main'),('reader','==','structured')])
    else:
        from .evaluation import require_frozen
        from .reporting import normalize_main
        require_frozen(run);complete=read_json(run/'evaluation/main/complete.json')
        assert complete['state']=='passed' and complete['rows']==147200
        path=run/'evaluation/main/per_case_results.parquet'
        frame=normalize_main(pd.read_parquet(path),'official')
        frame['provenance_hash']=frame.event_provenance_hash
        frame['job_id']=[hash_json([str(run),r['history_hash'],r['provenance_hash'],r['case_id']]) for r in frame.to_dict('records')]
    assert len(frame)==147200 and frame.case_id.nunique()==3200
    assert not frame.duplicated(['suite','method','policy_seed','execution_cap','case_id']).any()
    frame['footprint_context_cohort']=cohort
    return frame,path


def context_footprint(context,footprint):
    keep=['job_id','history_hash','case_id','configuration_id','suite','fault_family','method','policy_seed',
        'execution_cap','true_target_key','predicted_target_key','correct','nll_bits','p_healthy']
    result={k:context.get(k) for k in keep}
    errors=context.get('errors',[])
    errors=list(errors) if isinstance(errors,(list,tuple,np.ndarray)) else []
    from .runtime_audit import diagnostic_timeout_stages
    result.update(footprint_context_cohort=context.get('footprint_context_cohort'),
        untimed_correct=None,untimed_predicted_target_key=None,untimed_nll_bits=None,
        official_untimed_max_probability_difference=None,
        official_timeout=bool(context.get('timeout',False)),official_abstained=bool(context.get('abstained',False)),
        official_errors_json=json.dumps(errors),official_diagnostic_timeout=bool(diagnostic_timeout_stages(errors)),
        official_selection_timeout=any(e=='selection_timeout' or
            (e.startswith(('selection_error:','selection_warmup_error:','fixed_plan_error:')) and
             e.split(':',2)[1] in ('TimeoutError','CallDeadlineExceeded')) or e=='fixed_plan_timeout' for e in errors))
    result.update({k:v for k,v in footprint.items() if k not in ('target_channels_json',
        'relative_log_likelihoods_bits_json','recomputed_target_keys_json','recomputed_log_probabilities_json')})
    if footprint['audit_state']!='passed':
        result['footprint_category']='unreplayable';return result
    channels=json.loads(footprint['target_channels_json']);ll=json.loads(footprint['relative_log_likelihoods_bits_json'])
    truth=context['true_target_key'];predicted=context['predicted_target_key'];channel=channels[truth]
    untimed=dict(target_keys=json.loads(footprint['recomputed_target_keys_json']),
        log_probabilities_base2=json.loads(footprint['recomputed_log_probabilities_json']))
    metrics=prediction_metrics(untimed,truth)
    if json.loads(context['target_keys'])!=untimed['target_keys']:raise ValueError('Audit target mapping changed')
    difference=float(np.max(np.abs(np.exp2(json.loads(context['full_log_probabilities']))-
        np.exp2(untimed['log_probabilities_base2']))))
    result.update(untimed_correct=bool(metrics['correct']),untimed_predicted_target_key=metrics['predicted_target_key'],
        untimed_nll_bits=metrics['nll_bits'],official_untimed_max_probability_difference=difference,
        footprint_category_indexes_official_wrong_context=True)
    result['true_vs_healthy_log_likelihood_bits']=ll[truth]
    result['predicted_vs_healthy_log_likelihood_bits']=ll[predicted]
    result['true_target_channel_json']=json.dumps(channel,sort_keys=True)
    result['predicted_target_channel_json']=json.dumps(channels[predicted],sort_keys=True)
    result['true_fault_channel_applicable']=truth!='healthy'
    if truth=='healthy':
        category='healthy_false_alarm'
        result['predicted_fault_process_variance_only']=channels[predicted].get('process_variance_only',False)
    elif channel['direct_mean_channel']:category='direct_modeled_channel_present_but_wrong'
    elif channel['process_variance_only']:category='process_variance_only_channel_present_but_wrong'
    else:category='structural_absence_of_true_fault_channel'
    result.update(footprint_category=category,
        true_fault_any_modeled_channel=channel.get('any_modeled_channel'),
        true_fault_direct_mean_channel=channel.get('direct_mean_channel'),
        true_fault_process_variance_only=channel.get('process_variance_only'),
        true_fault_gain_packet_factors=channel.get('matched_gain_packet_factors'),
        true_fault_mixed_timing_groups=channel.get('mixed_upstream_downstream_packet_groups'),
        true_fault_within_clock_mixed_groups=channel.get('within_clock_mixed_packet_groups'))
    return result


def summarize(contexts,errors,keys):
    output=[]
    for key,part in contexts.groupby(keys,dropna=False,sort=True):
        key=key if isinstance(key,tuple) else (key,)
        mask=np.ones(len(errors),dtype=bool)
        for column,value in zip(keys,key):mask &= errors[column].isna().to_numpy() if pd.isna(value) else (errors[column]==value).to_numpy()
        selected=errors[mask];fault_errors=selected[selected.true_target_key!='healthy']
        counts=selected.footprint_category.value_counts()
        result=dict(zip(keys,key));result.update(all_rows=len(part),all_correct_rows=int(part.correct.sum()),
            all_wrong_rows=int((part.correct==0).sum()),physical_cases=part.case_id.nunique(),wrong_physical_cases=selected.case_id.nunique(),
            fault_error_rows=len(fault_errors),healthy_false_alarm_rows=int((selected.true_target_key=='healthy').sum()),
            replayable_wrong_rows=int((selected.audit_state=='passed').sum()),unreplayable_wrong_rows=int((selected.audit_state!='passed').sum()))
        assert result['all_wrong_rows']==len(selected)
        for category in ('structural_absence_of_true_fault_channel','direct_modeled_channel_present_but_wrong',
                         'process_variance_only_channel_present_but_wrong','healthy_false_alarm','unreplayable'):
            count=int(counts.get(category,0));result[category+'_rows']=count
            result[category+'_fraction_all_rows']=count/len(part)
            result[category+'_fraction_wrong_rows']=count/len(selected) if len(selected) else None
        for field in ('no_internal_measurements','no_modeled_internal_measurements','external_only','external_plus_calibration_only'):
            result[field+'_wrong_rows']=int(selected[field].fillna(False).sum()) if field in selected else 0
        for field in ('official_timeout','official_abstained','official_diagnostic_timeout','official_selection_timeout','untimed_correct'):
            if field in selected:result[field+'_wrong_rows']=int(selected[field].fillna(False).sum())
        if 'untimed_correct' in selected:
            for field in ('official_diagnostic_timeout','official_selection_timeout'):
                result[field+'_and_untimed_correct_rows']=int((selected[field].fillna(False)&selected.untimed_correct.fillna(False)).sum())
        for field in ('true_vs_healthy_log_likelihood_bits','true_fault_gain_packet_factors','true_fault_mixed_timing_groups',
                      'true_fault_within_clock_mixed_groups','timing_contrasts','cross_clock_packet_groups','calibration_exchanges'):
            if field in fault_errors:result[field+'_mean_fault_error']=fault_errors[field].mean()
        output.append(result)
    return pd.DataFrame(output)


def run_audit(source_root,run_dir,workers=24,cohort='old_histories'):
    run=Path(run_dir);declare(run);verify_readers(run)
    frame,input_path=_contexts(source_root,run,cohort);wrong=frame[frame.correct==0].copy()
    directory=run/'error_audit'/cohort;directory.mkdir(parents=True,exist_ok=True)
    inputs=dict(input_parquet_sha256=file_sha256(input_path),reader_frozen_sha256=file_sha256(run/'models/readers/frozen.json'),
        manifest_sha256=file_sha256(run/MANIFEST),implementation_sha256=file_sha256(__file__),cohort=cohort)
    identity=hash_json(inputs);marker=directory/'complete.json'
    if marker.exists():
        old=read_json(marker)
        if old['input_identity']!=identity:
            # The completed old-history analysis is immutable. Its executed
            # implementation is retained when later engineering adds fields
            # for the new-main deadline comparison or an empty-error cohort.
            prior_inputs=read_json(directory/'selection.json')['inputs']
            assert hash_json(prior_inputs)==old['input_identity']
            for key in inputs:
                if key!='implementation_sha256':assert prior_inputs[key]==inputs[key],key
            assert cohort=='old_histories'
            assert file_sha256(run/'provenance/error_audit_executed_old_histories.py')==prior_inputs['implementation_sha256']
        for name,digest in old['output_hashes'].items():assert file_sha256(directory/name)==digest
        return old
    selected=wrong.drop_duplicates('job_id').sort_values('job_id')
    jobs=[dict(job_id=r['job_id'],history_hash=r['history_hash'] if pd.notna(r['history_hash']) else None,
        case_id=r['case_id'],provenance_hash=r['provenance_hash'] if pd.notna(r['provenance_hash']) else None,
        target_keys=r['target_keys'],full_log_probabilities=r['full_log_probabilities'],input_identity=identity)
        for r in selected.to_dict('records')]
    atomic_json(directory/'selection.json',dict(state='posthoc_selection',inputs=inputs,input_identity=identity,
        all_contexts=len(frame),correct_contexts=int(frame.correct.sum()),wrong_contexts=len(wrong),unique_wrong_histories=len(jobs),
        rule='All wrong frozen SED diagnoses; no subsampling or filtering on evidence footprint',job_ids=[j['job_id'] for j in jobs]))
    started=time.monotonic();footprints=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
            initializer=_initialize,initargs=(str(source_root),str(run),cohort)) as pool:
        for result in pool.map(_job,jobs,chunksize=1):
            footprints.append(result)
            if len(footprints)%500==0:
                progress=dict(stage='error_footprint',cohort=cohort,completed=len(footprints),expected=len(jobs),wall_seconds=time.monotonic()-started)
                atomic_json(directory/'progress.json',progress);print(json.dumps(progress),flush=True)
    by_id={r['job_id']:r for r in footprints}
    assert len(by_id)==len(jobs)
    detail_rows=[context_footprint(r,by_id[r['job_id']]) for r in wrong.to_dict('records')]
    empty_columns=['job_id','history_hash','case_id','configuration_id','suite','fault_family','method','policy_seed',
        'execution_cap','true_target_key','predicted_target_key','correct','nll_bits','p_healthy','audit_state','footprint_category',
        'official_timeout','official_abstained','official_diagnostic_timeout','official_selection_timeout','untimed_correct']
    details=pd.DataFrame(detail_rows,columns=None if detail_rows else empty_columns)
    pd.DataFrame(footprints,columns=None if footprints else ['job_id','history_hash','case_id','audit_state']).to_parquet(directory/'unique_history_footprints.parquet',index=False)
    details.to_parquet(directory/'per_error_context.parquet',index=False)
    keys=['suite','method','execution_cap']
    summary=summarize(frame,details,keys);family=summarize(frame,details,keys+['fault_family'])
    summary.to_csv(directory/'summary.csv',index=False);family.to_csv(directory/'family_summary.csv',index=False)
    primary=pd.concat([summary.assign(fault_family='all'),family],ignore_index=True)
    primary=primary[(primary.suite=='test_connections')&(primary.execution_cap==4)]
    primary.to_csv(directory/'primary_b4_summary.csv',index=False)
    verify_readers(run);assert file_sha256(input_path)==inputs['input_parquet_sha256']
    files=['selection.json','unique_history_footprints.parquet','per_error_context.parquet','summary.csv','family_summary.csv','primary_b4_summary.csv']
    result=dict(state='passed',completed_at=now(),version=VERSION,cohort=cohort,input_identity=identity,
        all_contexts=len(frame),correct_contexts=int(frame.correct.sum()),wrong_contexts=len(wrong),
        unique_wrong_histories=len(jobs),unreplayable_unique_histories=sum(r['audit_state']!='passed' for r in footprints),
        unreplayable_wrong_contexts=int((details.audit_state!='passed').sum()),all_wrong_contexts_retained=True,
        official_wrong_contexts_correct_if_reread=int(details.untimed_correct.fillna(False).sum()),
        official_wrong_contexts_with_diagnostic_timeout=int(details.official_diagnostic_timeout.fillna(False).sum()),
        official_wrong_contexts_with_selection_timeout=int(details.official_selection_timeout.fillna(False).sum()),
        footprint_categories_index_official_error_subset=True,
        correct_and_wrong_full_denominators_preserved=True,truth_used_only_after_public_preparation=True,
        structural_categories_not_causal_error_labels=True,old_policy_trained_with_another_reader=cohort=='old_histories',
        external_motion_not_modeled_by_SED=True,models_and_policies_unchanged=True,
        physical_acquisitions=0,oracle_reruns=0,optimizer_steps=0,workers=workers,wall_seconds=time.monotonic()-started,
        output_hashes={name:file_sha256(directory/name) for name in files})
    atomic_json(marker,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',required=True);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--workers',type=int,default=24);parser.add_argument('--cohort',choices=['old_histories','new_main'],default='old_histories')
    parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    result=declare(args.run_dir) if args.declare_only else run_audit(args.source_root,args.run_dir,args.workers,args.cohort)
    print(json.dumps(result,sort_keys=True),flush=True)
