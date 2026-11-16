"""Identity and shared-error audits on saved, physically acquired histories.

Selection uses configuration/family identities, never outcomes. A/B alter raw
storage only after immutable measurement IDs are attached. C uses the full old
diagnostic replay and never synthesizes a physical execution. D explicitly
distinguishes an uncalibrated marginal shared-error floor from an identifiable
conditional clock posterior, which can legitimately contract.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
import pandas as pd
from cgx.data.store import Store
from cgx.provenance import atomic_json, file_sha256, hash_json, read_json
from .replay import materialize_packed, provenance_for_history, write_gzip, read_gzip

TOLERANCE=1e-6
DUPLICATES=(0,1,2,4,8,16,32)


def select(run_dir):
    run=Path(run_dir);path=run/'invariance/selection.json'
    if path.exists():return read_json(path)
    frame=pd.read_parquet(run/'replay/contexts.parquet')
    eligible=frame[(frame.cohort=='main') & (frame.method=='gnn_adaptive') &
        (frame.policy_seed==11) & (frame.execution_cap==4)]
    selected=eligible.sort_values(['configuration_id','fault_family','case_id']).groupby(
        ['configuration_id','fault_family'],sort=True).head(1)
    if len(selected)!=128 or selected.configuration_id.nunique()!=32:
        raise ValueError('Invariance sampling must cover 32 original configurations and four families')
    fields=['job_id','history_hash','case_id','configuration_id','suite','fault_family','true_target_key']
    result=dict(state='sealed',selection_rule='Lexicographically first case per configuration/family; GNN adaptive seed11 B4',
        outcome_independent_selection=True,cases=selected[fields].to_dict('records'),
        tolerance=TOLERANCE,duplicate_copies=list(DUPLICATES),
        contexts_sha256=file_sha256(run/'replay/contexts.parquet'))
    atomic_json(path,result);return result


def probability_error(a,b):
    if list(a['target_keys'])!=list(b['target_keys']):raise AssertionError('Invariance target ordering')
    return float(np.max(np.abs(np.exp2(a['log_probabilities_base2'])-np.exp2(b['log_probabilities_base2']))))


def split_raw_blocks(history, pieces=7):
    """IDs must already have been annotated before this packaging operation."""
    out=deepcopy(history);out['observations']=[]
    for obs in history['observations']:
        if not obs.get('execution_id'):raise ValueError('Raw split requires stable ingress identity')
        indices=np.array_split(np.arange(len(obs.get('records',[]))),pieces)
        for i,index in enumerate(indices):
            block=deepcopy(obs);block['records']=[deepcopy(obs['records'][int(j)]) for j in index]
            block['calibration_bursts']=deepcopy(obs.get('calibration_bursts',[])) if i==0 else []
            # Status belongs to the acquisition; repeating its stable ID dedups.
            out['observations'].append(block)
    return out


def merge_raw_blocks(history):
    out=deepcopy(history);out['observations']=[];by_id={}
    for obs in history['observations']:
        key=obs['execution_id']
        if key not in by_id:
            by_id[key]=deepcopy(obs);by_id[key]['records']=[];by_id[key]['calibration_bursts']=[]
            out['observations'].append(by_id[key])
        by_id[key]['records'].extend(deepcopy(obs.get('records',[])))
        by_id[key]['calibration_bursts'].extend(deepcopy(obs.get('calibration_bursts',[])))
    return out


def _audit_case(arguments):
    source_root,run_dir,case=arguments
    from .reader import load_reader
    from .events import annotate_history,prepare_history,evidence_identity
    reader=load_reader(run_dir);run=Path(run_dir);store=Store(source_root)
    history=store.load_history(case['history_hash']);provenance=provenance_for_history(history,case['case_id'])
    annotated=annotate_history(history,provenance);evidence=prepare_history(annotated)
    baseline=reader.predict_evidence(evidence);base_identity=evidence_identity(evidence)
    meta={k:case[k] for k in ('job_id','history_hash','case_id','configuration_id','suite','fault_family')}
    rows=[]
    candidates=[(oi,ri) for oi,o in enumerate(annotated['observations']) for ri,r in enumerate(o.get('records',[]))
                if r['port']!='external:out']
    if not candidates:raise AssertionError('Real robot history contains no internal acquired event')
    oi,ri=candidates[0];original=deepcopy(annotated['observations'][oi]['records'][ri])
    for copies in DUPLICATES:
        changed=deepcopy(annotated)
        changed['observations'][oi]['records'].extend([deepcopy(original) for _ in range(copies)])
        ee=prepare_history(changed);pred=reader.predict_evidence(ee)
        if evidence_identity(ee)!=base_identity:raise AssertionError('Duplicate storage changed unique evidence')
        error=probability_error(baseline,pred)
        if error>TOLERANCE:raise AssertionError('Duplicate storage changed posterior')
        rows.append(dict(meta,test='A_duplicate_storage',copies=copies,max_probability_error=error,
            p_healthy=float(np.exp2(pred['log_probabilities_base2'][pred['target_keys'].index('healthy')])),
            unique_events=len(ee['events']),physical_new_events=0,measurement_event_id=original['measurement_event_id']))
    split=split_raw_blocks(annotated)
    directory=run/'invariance/packaging'/case['job_id'];directory.mkdir(parents=True,exist_ok=True)
    # Real gzip files are written/read; JSON roundtrip cannot hide file grouping bugs.
    packed=[]
    for i,block in enumerate(split['observations']):
        path=directory/f'{i:03d}.json.gz';write_gzip(path,block);packed.append(read_gzip(path))
    split['observations']=packed
    for name,transformed in [('split_files',split),('merge_files',merge_raw_blocks(split))]:
        ee=prepare_history(transformed);pred=reader.predict_evidence(ee)
        error=probability_error(baseline,pred)
        if evidence_identity(ee)!=base_identity or error>TOLERANCE:raise AssertionError('Packaging changed evidence/posterior')
        rows.append(dict(meta,test='B_'+name,copies=0,max_probability_error=error,
            unique_events=len(ee['events']),physical_new_events=0,stored_blocks=len(transformed['observations'])))
    action=deepcopy(annotated);action['action_keys']=['action_metadata_only_debug']*17
    action['remaining']={k:0 for k in action['remaining']}
    error=probability_error(baseline,reader.predict(action))
    if error>TOLERANCE:raise AssertionError('Action metadata alone changed posterior')
    rows.append(dict(meta,test='action_only',max_probability_error=error,physical_new_events=0))
    # Identity check, not a physical new-execution experiment: equal numeric
    # values with a distinct ID remain separate events at the ingress boundary.
    equal=deepcopy(evidence);new=deepcopy(equal['events'][0]);new['measurement_event_id']=hash_json(['identity_debug',new['measurement_event_id']])
    new['message_event_id']=new['measurement_event_id'];equal['events'].append(new)
    from .events import canonical_evidence
    equal=canonical_evidence(equal)
    if len(equal['events'])!=len(evidence['events'])+1:raise AssertionError('Numeric equality deduplicated a new ID')
    rows.append(dict(meta,test='identity_equal_values_distinct_id',unique_events=len(equal['events']),
        identity_debug_only=True,physical_new_events=0))
    curve=reader.shared_error_curve(evidence)
    shared=[]
    if curve['state']=='passed':
        a,z=curve['clock_groups'];ports=set(curve['ports']);groups={}
        for record in evidence['events']:
            if record['port_id'] in ports and not record['missing']:
                groups.setdefault((record['execution_id'],record['message_token']),{}).setdefault(record['port_id'],record)
        complete=[g for _,g in sorted(groups.items()) if set(g)==ports]
        for row in curve['curves']:
            n=row['n'];limited=dict(evidence,events=[r for group in complete[:n] for r in group.values()],calibrations=[],statuses=[])
            pred=reader.predict_evidence(limited);diag=pred['structured_diagnostics']
            ci=diag['clock_group_ids'].index(a);cj=diag['clock_group_ids'].index(z)
            covariance=np.asarray(diag['clock_covariance'])
            variance=float(covariance[ci,ci]+covariance[cj,cj]-2*covariance[ci,cj])
            if not np.isfinite(variance) or variance < -1e-10:raise AssertionError('Invalid shared posterior covariance')
            expected=row['shared_clock_variance']+row['independent_variance']
            if not np.isclose(expected,row['correct_marginal_average_variance']):raise AssertionError('Shared variance incorrectly eliminated')
            shared.append(dict(meta,**row,clock_a=a,clock_z=z,port_a=curve['ports'][0],port_z=curve['ports'][1],
                posterior_relative_clock_variance=max(0.,variance),calibration_removed_for_control=True,
                evidence_condition='Only this actual acquired cross-clock port pair; no calibration; fault state marginalized',
                uncertainty_floor_scope='Marginal uncalibrated transit mean under learned healthy timing noise',
                available_unique_message_contrasts=curve['available_unique_message_contrasts']))
    output=dict(state='passed',case=case,reader_checkpoint_hash=reader.checkpoint_hash,
        provenance_hash=hash_json(provenance),base_evidence_identity=base_identity,
        transformations=rows,shared_clock=shared,shared_clock_state=curve['state'])
    write_gzip(run/'invariance/cases'/f"{case['job_id']}.json.gz",output)
    return len(rows),len(shared)


def transformations(source_root,run_dir,workers=16):
    run=Path(run_dir);selection=select(run);gate=read_json(run/'manifests/reader_replay_authorized.json')
    if gate['state']!='passed':raise ValueError('Outcome replay is not authorized')
    from .reader import load_reader
    reader=load_reader(run)
    if reader.checkpoint_hash!=gate['reader_checkpoint_hash']:raise ValueError('Reader selection changed')
    pending=[c for c in selection['cases'] if not (run/'invariance/cases'/f"{c['job_id']}.json.gz").exists()]
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        list(pool.map(_audit_case,[(source_root,str(run),c) for c in pending]))
    parts=[read_gzip(run/'invariance/cases'/f"{c['job_id']}.json.gz") for c in selection['cases']]
    if any(p['reader_checkpoint_hash']!=reader.checkpoint_hash for p in parts):raise ValueError('Reader changed on resume')
    pd.DataFrame([r for p in parts for r in p['transformations']]).to_csv(run/'invariance/transformations.csv',index=False)
    shared=pd.DataFrame([r for p in parts for r in p['shared_clock']]);shared.to_csv(run/'invariance/shared_clock.csv',index=False)
    if shared.empty:raise AssertionError('Test D did not exercise a real shared-clock pair')
    result=dict(state='passed',physical_cases=len(parts),max_probability_error=max(
        r.get('max_probability_error',0.) for p in parts for r in p['transformations']),
        shared_clock_physical_cases=shared.case_id.nunique(),shared_clock_rows=len(shared),
        reader_checkpoint_hash=reader.checkpoint_hash,wall_seconds=time.monotonic()-start,
        physical_new_acquisitions=0,selection_sha256=file_sha256(run/'invariance/selection.json'))
    atomic_json(run/'invariance/transformations_complete.json',result);return result


def finish(run_dir):
    run=Path(run_dir);trans=read_json(run/'invariance/transformations_complete.json')
    replay=read_json(run/'replay/predictions_complete.json')
    if trans['state']!='passed' or replay['state']!='passed':raise ValueError('Transformation or replay coverage incomplete')
    frame=pd.read_parquet(run/'replay/per_case_predictions.parquet')
    frame=frame[frame.cohort.isin(['fa_v1','fa_v2'])].sort_values(['cohort','source_trajectory','reader','prefix_ordinal'])
    paired=[]
    for (cohort,path,reader),part in frame.groupby(['cohort','source_trajectory','reader']):
        records=part.to_dict('records')
        for before,after in zip(records,records[1:]):
            if after['prefix_ordinal']!=before['prefix_ordinal']+1:raise ValueError('Diagnostic prefixes are not consecutive')
            de=after['executions_used']-before['executions_used']
            if de==0:continue
            if de!=1:raise ValueError('Prefix skipped an actual physical execution')
            paired.append(dict(cohort=cohort,source_trajectory=path,reader=reader,case_id=after['case_id'],
                configuration_id=after['configuration_id'],fault_family=after['fault_family'],method=after['method'],
                policy_seed=after['policy_seed'],prefix_ordinal=after['prefix_ordinal'],
                previous_history_hash=before['history_hash'],new_history_hash=after['history_hash'],
                first_additional_execution=after['executions_used']==1,
                true_label_nll_before=before['nll_bits'],true_label_nll_after=after['nll_bits'],
                nll_change_bits=after['nll_bits']-before['nll_bits'],
                p_healthy_before=before['p_healthy'],p_healthy_after=after['p_healthy'],
                correct_before=before['correct'],correct_after=after['correct'],
                genuine_new_physical_execution=True))
    pairs=pd.DataFrame(paired);pairs.to_parquet(run/'invariance/new_execution_pairs.parquet',index=False)
    summaries=[]
    for (cohort,reader,family),part in pairs.groupby(['cohort','reader','fault_family']):
        case_seed=part.groupby(['configuration_id','case_id','method','policy_seed']).nll_change_bits.mean()
        case=case_seed.groupby(['configuration_id','case_id']).mean()
        config=case.groupby('configuration_id').mean()
        summaries.append(dict(cohort=cohort,reader=reader,fault_family=family,
            execution_pairs=len(part),physical_cases=part.case_id.nunique(),configurations=part.configuration_id.nunique(),
            nll_change_bits_acquisition_weighted=part.nll_change_bits.mean(),
            nll_change_bits_case_balanced=case.mean(),nll_change_bits_configuration_balanced=config.mean(),
            improved_execution_pairs=int((part.nll_change_bits<0).sum()),
            worsened_execution_pairs=int((part.nll_change_bits>0).sum())))
    pd.DataFrame(summaries).to_csv(run/'invariance/new_execution_summary.csv',index=False)
    result=dict(state='passed',tests=['A_duplicate_storage','B_file_block_packaging','C_real_new_execution','D_shared_clock'],
        test_A_B_cases=trans['physical_cases'],test_A_B_max_probability_error=trans['max_probability_error'],
        test_C_structured_pairs=int((pairs.reader=='structured').sum()),
        test_C_note='NLL improvement is measured, not a prerequisite for the identity/covariance checks to pass',
        test_D_cases=trans['shared_clock_physical_cases'],
        test_D_scope='Learned uncalibrated cross-clock marginal variance retains shared prior floor; actual marginalized posterior covariance reported separately',
        physical_new_acquisitions=0,source_is_saved_physical_executions=True,
        transformations_complete_sha256=file_sha256(run/'invariance/transformations_complete.json'),
        predictions_complete_sha256=file_sha256(run/'replay/predictions_complete.json'))
    atomic_json(run/'invariance/complete.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('select');p.add_argument('--run-dir',required=True)
    p=sub.add_parser('transformations');p.add_argument('--source-root',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--workers',type=int,default=16)
    p=sub.add_parser('finish');p.add_argument('--run-dir',required=True)
    args=vars(parser.parse_args());command=args.pop('command');print(globals()[command](**args),flush=True)
