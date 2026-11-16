"""Predeclared, post-main sensitivity to the reader's wall-time deadline.

Only saved final evidence is reread. Official predictions, chosen actions,
physical evidence, budgets, deadlines and all model parameters remain unchanged.
An untimed final answer cannot recover actions abandoned after an earlier timeout.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import pandas as pd

from cgx.config import load_config
from cgx.evaluation.design import main_methods, conditions
from cgx.provenance import atomic_json, file_sha256, hash_json, now, read_json
from .gates import verify_readers
from .reader import load_reader
from .replay import EXPECTED_MAIN_ROWS, prediction_metrics
from .storage import OverlayStore

VERSION = 'sed-reader-deadline-sensitivity-v1'
DESIGN = dict(version=VERSION,
    trigger='Only after the complete official main matrix has been finalized',
    selection_errors=['diagnostic_timeout', 'diagnostic_error:TimeoutError:',
        'diagnostic_error:CallDeadlineExceeded:', 'diagnostic_warmup_error:TimeoutError:',
        'diagnostic_warmup_error:CallDeadlineExceeded:'],
    selection_scope='All and only official rows with a diagnostic deadline error; no selection or acquisition-only timeouts',
    treatment='Reread exact saved final public history and explicit saved event provenance with the frozen final SED, CPU, no call deadline',
    unchanged=['official main results', 'chosen actions', 'acquired evidence', 'calibrations',
        'model parameters', 'deployed deadlines', 'full matrix denominators'],
    unreplayable='Keep an explicit failed audit record and retain its official final prediction in adjusted summaries',
    adjusted_scope='Replace only successfully reread eligible final predictions; preserve all other official predictions',
    limitation='Cannot recover future actions abandoned after an earlier diagnostic timeout; not an unrestricted-policy counterfactual and not official main performance',
    labels='Official truth used only for evaluator metrics after prediction; never passed into the reader',
    physical_acquisitions=0, calibration_acquisitions=0, optimizer_steps=0,
    rationale='SOURCE-only deferred cold-call profile measured a call exceeding the unchanged 2-second limit')
DECLARATION = 'manifests/reader_deadline_audit_predeclared.json'
IMPLEMENTATION = 'manifests/reader_deadline_audit_implementation.json'
CELL_FIELDS = ['suite','method','policy_seed','execution_cap','byte_fraction','case_id','configuration_id']
_WORKER = None


def _plain(value):
    if isinstance(value, dict):return {k:_plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):return [_plain(v) for v in value]
    if isinstance(value, np.generic):return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):return None
    return value


def cell_key(row):
    seed=row.get('policy_seed')
    seed=None if seed is None or pd.isna(seed) else int(seed)
    return hash_json([row['method'],seed,int(row['execution_cap']),float(row['byte_fraction']),row['case_id']])


def diagnostic_timeout_stages(errors):
    """Do not confuse reader, policy, instrument and generic worker failures."""
    stages=set()
    for error in errors if errors is not None else ():
        if error=='diagnostic_timeout':stages.add('diagnosis')
        for prefix,stage in (('diagnostic_error:','diagnosis'),('diagnostic_warmup_error:','diagnostic_warmup')):
            if error.startswith(prefix) and error[len(prefix):].split(':',1)[0] in ('TimeoutError','CallDeadlineExceeded'):
                stages.add(stage)
    return sorted(stages)


def declare(run_dir):
    run=Path(run_dir);verify_readers(run)
    path=run/DECLARATION
    value=dict(state='predeclared_before_new_main_outcomes',created_at=now(),design=DESIGN,
        reader_frozen_sha256=file_sha256(run/'models/readers/frozen.json'),
        source_performance_profile_sha256=file_sha256(run/'tests/reader_deferred_call_profile.json'))
    if path.exists():
        previous=read_json(path)
        for key in ('design','reader_frozen_sha256','source_performance_profile_sha256'):
            assert previous[key]==value[key],key
        return previous
    if (run/'evaluation/main/complete.json').exists() or any((run/'evaluation/main/cases').glob('*.json')):
        raise RuntimeError('The audit selection must be predeclared before main outcomes')
    atomic_json(path,value);return value


def _check_declaration(run):
    value=read_json(run/DECLARATION)
    assert value['design']==DESIGN
    assert value['reader_frozen_sha256']==file_sha256(run/'models/readers/frozen.json')
    assert value['source_performance_profile_sha256']==file_sha256(run/'tests/reader_deferred_call_profile.json')
    implementation=read_json(run/IMPLEMENTATION)
    assert implementation['state']=='passed'
    assert implementation['implementation_sha256']==file_sha256(__file__)
    assert implementation['declaration_sha256']==file_sha256(run/DECLARATION)
    return value


def _initialize(source_root,run_dir):
    global _WORKER
    run=Path(run_dir)
    _WORKER=(OverlayStore(source_root,run),load_reader(run,'final',device='cpu'),run)


def _reread(row,store,reader):
    """Reader receives only public history/provenance, never the evaluator row."""
    stages=diagnostic_timeout_stages(row.get('errors',[]))
    result={k:_plain(row.get(k)) for k in CELL_FIELDS}
    result.update(cell_key=cell_key(row),true_target_key=row['true_target_key'],
        official_prediction=_plain(row['prediction']),untimed_prediction=None,
        history_hash=row.get('evidence_trace_hash'),event_provenance_hash=row.get('event_provenance_hash'),
        diagnostic_timeout_stages=stages,official_errors=_plain(row.get('errors',[])),
        official_timeout=bool(row.get('timeout')),official_abstained=bool(row.get('abstained')),
        audit_state='unreplayable',audit_error=None,reader_hash=reader.checkpoint_hash,
        actions_and_evidence_fixed=True,abandoned_actions_not_recovered=True,
        physical_acquisitions=0,calibration_acquisitions=0)
    result.update({'official_'+k:v for k,v in prediction_metrics(row['prediction'],row['true_target_key']).items()})
    started=time.monotonic();fingerprint=reader.parameter_fingerprint();stage='load_saved_history'
    try:
        if not result['history_hash'] or not result['event_provenance_hash']:
            raise ValueError('Exact saved history and explicit event provenance are required')
        history=store.load_history(result['history_hash'])
        packed=store.history_references(result['history_hash'])
        stage='load_saved_provenance'
        provenance=store.load_provenance(result['history_hash'],result['event_provenance_hash'])
        if len(provenance['observation_ids'])!=len(history['observations']):
            raise ValueError('Saved provenance does not align with the final history')
        if 'observation_refs' in provenance and provenance['observation_refs']!=packed['observation_refs']:
            raise ValueError('Provenance observation references contradict the saved history')
        provenance={**provenance,'observation_refs':packed['observation_refs']}
        stage='untimed_reader_call';stamp=time.monotonic()
        prediction=reader.predict(history,provenance=provenance,include_latents=False)
        result['untimed_reader_seconds']=time.monotonic()-stamp
        if list(prediction['target_keys'])!=list(row['prediction']['target_keys']):
            raise ValueError('Untimed reader changed the public target ordering')
        metrics=prediction_metrics(prediction,row['true_target_key'])
        if reader.parameter_fingerprint()!=fingerprint:raise RuntimeError('Frozen reader parameters changed')
        result.update(audit_state='replayed',untimed_prediction=_plain(prediction),
            saved_raw_records=sum(len(o['records']) for o in history['observations']),
            saved_observations=len(history['observations']),parameters_unchanged=True)
        result.update({'untimed_'+k:v for k,v in metrics.items()})
    except Exception as exc:
        result.update(audit_state='unreplayable',untimed_prediction=None,
            audit_error=f'{stage}:{type(exc).__name__}:{exc}',
            parameters_unchanged=reader.parameter_fingerprint()==fingerprint)
    result['audit_wall_seconds']=time.monotonic()-started
    return result


def _audit_job(job):
    row,input_identity=job;store,reader,run=_WORKER
    path=run/'runtime_audit/cases'/f'{cell_key(row)}.json'
    if path.exists():
        saved=read_json(path)
        assert saved['input_identity']==input_identity
        assert saved['record_sha256']==hash_json(saved['record'])
        return saved['record']
    result=_reread(row,store,reader)
    atomic_json(path,dict(input_identity=input_identity,record_sha256=hash_json(result),record=result))
    return result


def _verify_cell(argument):
    run,frozen_hash,row=argument;key=cell_key(row)
    path=Path(run)/'evaluation/main/cases'/f'{key}.json';item=read_json(path)
    assert item['state']=='passed' and item['frozen_hash']==frozen_hash
    assert item['result_hash']==hash_json({'row':item['row'],'trace':item['trace']})
    assert key==cell_key(item['row'])
    for name,value in item['row'].items():
        if _plain(row.get(name))!=value:raise ValueError(f'Official parquet/cell disagreement: {key}:{name}')
    return item['row'],dict(cell_key=key,sha256=file_sha256(path),result_hash=item['result_hash'])


def _write_summaries(rows,audited,directory):
    from .reporting import normalize_main,summary_table
    by_key={r['cell_key']:r for r in audited}
    adjusted=[]
    for row in rows:
        item=dict(row);audit=by_key.get(cell_key(row))
        item.update(reader_deadline_audit_eligible=audit is not None,
            reader_deadline_audit_state=audit['audit_state'] if audit else 'not_selected',
            reader_deadline_audit_replaced=bool(audit and audit['audit_state']=='replayed'))
        if item['reader_deadline_audit_replaced']:
            prediction=audit['untimed_prediction'];metrics=prediction_metrics(prediction,row['true_target_key'])
            predicted=metrics['predicted_target_key']
            item.update(prediction=prediction,predicted_target_key=predicted,
                correct_joint=bool(metrics['correct']),correct_mechanism=predicted.rsplit(':',1)[-1]==row['true_target_key'].rsplit(':',1)[-1],
                final_nll_bits=metrics['nll_bits'],score_improvement_bits=row['initial_nll_bits']-metrics['nll_bits'],
                healthy_false_alarm=bool(metrics['healthy_false_alarm']),fault_miss=bool(metrics['fault_miss']))
        # Original timeout/abstention and resources remain as provenance. This
        # sensitivity does not claim the deployment succeeded without a limit.
        adjusted.append(item)
    frame=pd.DataFrame(adjusted);frame.to_parquet(directory/'full_matrix_adjusted.parquet',index=False)
    official=normalize_main(pd.DataFrame(rows),'official')
    sensitivity=normalize_main(frame,'reader_deadline_sensitivity')
    assert len(official)==len(sensitivity)==EXPECTED_MAIN_ROWS
    joined=pd.concat([official,sensitivity],ignore_index=True)
    keys=['version','suite','method','execution_cap']
    for filename,groups in [('full_matrix_summary.csv',keys),('family_summary.csv',keys+['fault_family']),
                            ('seed_summary.csv',keys+['policy_seed'])]:
        summary_table(joined,groups).to_csv(directory/filename,index=False)


def run_audit(source_root,run_dir,workers=32):
    from .evaluation import require_frozen
    run=Path(run_dir);directory=run/'runtime_audit';directory.mkdir(parents=True,exist_ok=True)
    _check_declaration(run);verify_readers(run);frozen=require_frozen(run)
    frozen_hash=hash_json(frozen);complete=read_json(run/'evaluation/main/complete.json')
    assert complete['state']=='passed' and complete['rows']==complete['expected']==EXPECTED_MAIN_ROWS
    assert complete['frozen_hash']==frozen_hash and complete['all_planned_denominators_retained']
    inputs={name:file_sha256(run/name) for name in ('evaluation/main/complete.json',
        'evaluation/main/per_case_results.parquet','models/frozen.json','models/readers/frozen.json',DECLARATION,IMPLEMENTATION)}
    input_identity=hash_json(inputs)
    marker=directory/'complete.json'
    if marker.exists():
        old=read_json(marker);assert old['input_identity']==input_identity
        for name,digest in old['output_hashes'].items():assert file_sha256(directory/name)==digest,name
        return old
    input_path=directory/'inputs.json'
    if input_path.exists():assert read_json(input_path)['hashes']==inputs
    else:atomic_json(input_path,dict(state='sealed',created_at=now(),hashes=inputs,input_identity=input_identity))
    frame=pd.read_parquet(run/'evaluation/main/per_case_results.parquet')
    assert len(frame)==EXPECTED_MAIN_ROWS and frame.case_id.nunique()==3200
    table_rows=frame.to_dict('records');keys={cell_key(r) for r in table_rows}
    assert len(keys)==len(table_rows)
    config=load_config();store=OverlayStore(source_root,run)
    cases=[c for split in config['splits'] if split.startswith('test_') for c in store.cases(split)]
    expected={hash_json([method,seed,cap,fraction,case['case_id']]) for method,seed in main_methods(config)
        for cap,fraction in conditions(config,method) for case in cases}
    assert keys==expected=={p.stem for p in (run/'evaluation/main/cases').glob('*.json')}
    rows=[];hashes=[];started=time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row,digest in pool.map(_verify_cell,((str(run),frozen_hash,r) for r in table_rows)):
            rows.append(row);hashes.append(digest)
    (directory/'official_cell_hashes.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in hashes))
    selected=[r for r in rows if diagnostic_timeout_stages(r.get('errors',[]))]
    atomic_json(directory/'selection.json',dict(state='selected_by_predeclared_error_rule',
        selected_cells=len(selected),cell_keys=[cell_key(r) for r in selected],input_identity=input_identity,
        full_matrix_rows=len(rows),labels_or_probability_values_used_for_selection=False))
    audited=[]
    if selected:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
                initializer=_initialize,initargs=(str(source_root),str(run))) as pool:
            for result in pool.map(_audit_job,((row,input_identity) for row in selected),chunksize=1):
                audited.append(result)
                if len(audited)%100==0:print(json.dumps(dict(stage='reader_deadline_audit',completed=len(audited),selected=len(selected))),flush=True)
    columns=CELL_FIELDS+['cell_key','true_target_key','official_prediction','untimed_prediction','history_hash',
        'event_provenance_hash','diagnostic_timeout_stages','audit_state','audit_error']
    pd.DataFrame(audited,columns=None if audited else columns).to_parquet(directory/'official_vs_untimed.parquet',index=False)
    _write_summaries(rows,audited,directory)
    verify_readers(run);require_frozen(run)
    for name,digest in inputs.items():assert file_sha256(run/name)==digest,name
    replayed=sum(r['audit_state']=='replayed' for r in audited)
    outputs=['official_vs_untimed.parquet','full_matrix_adjusted.parquet','full_matrix_summary.csv',
        'family_summary.csv','seed_summary.csv','official_cell_hashes.jsonl','selection.json','inputs.json']
    result=dict(state='passed',completed_at=now(),version=VERSION,input_identity=input_identity,
        full_matrix_rows=len(rows),adjusted_matrix_rows=len(rows),physical_cases=3200,
        selected_cells=len(selected),replayed_cells=replayed,unreplayable_cells=len(audited)-replayed,
        unchanged_official_prediction_cells=len(rows)-replayed,full_matrix_denominator_unchanged=True,
        selected_unreplayable_retained=True,official_results_unchanged=True,chosen_actions_and_evidence_fixed=True,
        only_reader_deadline_removed=True,abandoned_actions_not_recovered=True,not_official_main=True,
        physical_acquisitions=0,calibration_acquisitions=0,optimizer_steps=0,
        reader_hash=frozen['reader_hash'],declaration_sha256=inputs[DECLARATION],
        implementation_sha256=file_sha256(__file__),wall_seconds=time.monotonic()-started,
        output_hashes={name:file_sha256(directory/name) for name in outputs})
    atomic_json(marker,result);return result


def engineering_test(source_root,run_dir):
    """No held-out outcomes: saved SOURCE evidence and explicit error fixtures."""
    import tempfile
    from .selectors import source_history_provenance
    from .selectors import ReadOnlyReplay
    run=Path(run_dir);declare(run)
    assert diagnostic_timeout_stages(['diagnostic_timeout'])==['diagnosis']
    assert diagnostic_timeout_stages(['diagnostic_error:CallDeadlineExceeded:limit'])==['diagnosis']
    assert diagnostic_timeout_stages(['diagnostic_warmup_error:TimeoutError:limit'])==['diagnostic_warmup']
    assert diagnostic_timeout_stages(['selection_timeout','acquisition_error:TimeoutError:limit','diagnostic_error:ValueError:bad'])==[]
    replay=ReadOnlyReplay(source_root,run,'train');cid=replay.case_rows[0][0]
    rows=replay.case_transitions(cid);source_row=rows[0];ref=source_row['next_history_hash']
    history=replay.history(ref);provenance=source_history_provenance(rows)[ref]
    provenance={**provenance,'observation_refs':replay.store.history_references(ref)['observation_refs']}
    reader=load_reader(run,'final',device='cpu')
    before=reader.parameter_fingerprint();prediction=reader.predict(history,provenance,include_latents=False)
    with tempfile.TemporaryDirectory(prefix='sed-runtime-audit-engineering-') as temp:
        store=OverlayStore(source_root,temp);ph=store.save_provenance(ref,provenance)
        fixture=dict(suite='train_engineering_only',method='gnn_adaptive',policy_seed=11,execution_cap=4,byte_fraction=1.,
            case_id=cid,configuration_id=source_row['configuration_id'],true_target_key=source_row['true_target_key'],
            prediction=prediction,evidence_trace_hash=ref,event_provenance_hash=ph,errors=['diagnostic_error:CallDeadlineExceeded:fixture'],
            timeout=True,abstained=True)
        result=_reread(fixture,store,reader)
        assert result['audit_state']=='replayed' and result['untimed_prediction']==prediction
        missing=_reread({**fixture,'event_provenance_hash':None},store,reader)
        assert missing['audit_state']=='unreplayable' and missing['untimed_prediction'] is None
        assert missing['official_prediction']==prediction
        # Exercise the actual nested Arrow artifact schema using SOURCE data.
        path=Path(temp)/'audit.parquet';pd.DataFrame([result,missing]).to_parquet(path,index=False)
        check=pd.read_parquet(path);assert len(check)==2 and set(check.audit_state)=={'replayed','unreplayable'}
    assert reader.parameter_fingerprint()==before
    output=dict(state='passed',source_only=True,source_history_hash=ref,heldout_outcomes_read=False,
        checks=['diagnostic_timeout_selection','actual_deadline_exception_subtype','policy_and_acquisition_timeout_exclusion',
                'saved_source_history_and_explicit_provenance_reread','unreplayable_retains_official_prediction','nested_parquet_roundtrip'],
        source_predictions_unchanged=True,parameters_unchanged=True,physical_acquisitions=0,optimizer_steps=0,
        implementation_sha256=file_sha256(__file__),declaration_sha256=file_sha256(run/DECLARATION))
    atomic_json(run/'tests/reader_deadline_audit_engineering.json',output)
    seal={**output,'created_at':now(),'engineering_test_sha256':file_sha256(run/'tests/reader_deadline_audit_engineering.json')}
    path=run/IMPLEMENTATION
    if path.exists():
        prior=read_json(path)
        for key in ('implementation_sha256','declaration_sha256'):assert prior[key]==seal[key]
    else:atomic_json(path,seal)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',required=True);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--workers',type=int,default=32)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--declare-only',action='store_true');mode.add_argument('--engineering-test',action='store_true')
    args=parser.parse_args()
    if args.workers<1:raise ValueError('workers must be positive')
    if args.declare_only:result=declare(args.run_dir)
    elif args.engineering_test:result=engineering_test(args.source_root,args.run_dir)
    else:result=run_audit(args.source_root,args.run_dir,args.workers)
    print(json.dumps(result,sort_keys=True),flush=True)
