"""Re-read all preregistered original trajectory prefixes with both frozen readers."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import torch

from cgx.models.features import FEATURE_VERSION, collate_features
from cgx.provenance import atomic_json, hash_json, read_json
from .core import (cached_features, code_hash, inference_settings, load_artifact,
                   load_reader, prediction_fields, prefix_descriptors)
from .ledger import Ledger, TimedStage

METHODS = ('gnn_depth2', 'gnn_adaptive', 'transformer_adaptive', 'gnn_fixed_plan')
SEEDS = (11, 23, 37)


def _atomic_frame(frame, destination):
    temporary=destination.with_name(destination.name+f'.{os.getpid()}.tmp')
    if destination.suffix=='.parquet':frame.to_parquet(temporary,index=False)
    else:frame.to_csv(temporary,index=False)
    temporary.replace(destination)


def write_timing_notes(run):
    atomic_json(run/'replay_timing_semantics.json',{
        'elapsed_preprocess_s':'Reference elapsed time of the one shared tensorize_history feature build for this evidence prefix; repeated on all matching trajectory and reader rows, not incurred once per row.',
        'elapsed_rawread_s':'Reference elapsed time of verified raw observation materialization for the same one shared feature build; do not sum over duplicate prefixes or both readers.',
        'elapsed_model_s':'Actual GPU batch forward wall divided equally among histories in the batch; count unique (reader,evidence_prefix_hash) pairs.',
        'elapsed_collation_s':'Actual batch collation and device transfer wall divided equally among histories in the batch.',
        'resource_total_source':'replay_final_status.json and replay_independent_status.json worker CPU plus disjoint parent process CPU in stage_timings.jsonl; original CPU latency is not GPU batch latency.',
        'feature_cache_identity':'Sealed run + full public packed-history hash + recorded feature version and scientific source hash; both reader checkpoints separately verified and recorded.',
    })


def inventory(root, run):
    cases = read_json(run / 'manifests/replay_cases.json')
    if isinstance(cases, dict):
        cases = cases.get('cases', cases.get('selected_cases'))
    jobs, descriptors, failures = [], {}, []
    for case in cases:
        for method in METHODS:
            for seed in SEEDS:
                try:
                    item = load_artifact(root, case['case_id'], method, seed)
                    for packed, meta in prefix_descriptors(root, item):
                        digest = meta['evidence_prefix_hash']
                        descriptors.setdefault(digest, packed)
                        jobs.append(meta)
                except Exception as exc:
                    failures.append(dict(case_id=case['case_id'], method_key=method,
                        policy_seed=seed, error=f'{type(exc).__name__}: {exc}'))
    atomic_json(run / 'manifests/replay_prefixes.json', descriptors)
    pd.DataFrame(jobs).to_parquet(run / 'replay_prefix_inventory.parquet', index=False)
    atomic_json(run / 'replay_inventory_failures.json', failures)
    report = {'state': 'passed' if not failures else 'partial', 'cases': len(cases),
        'expected_trajectories': len(cases) * len(METHODS) * len(SEEDS),
        'actual_trajectories': len({(x['case_id'], x['method_key'], x['policy_seed']) for x in jobs}),
        'prefix_rows': len(jobs), 'unique_prefixes': len(descriptors),
        'maximum_new_forward_plus_features': 3 * len(descriptors), 'failed_trajectories': len(failures)}
    atomic_json(run / 'replay_inventory.json', report)
    print(json.dumps(report), flush=True)


def _prepare(job):
    from cgx.evaluation.rollout import call_deadline
    root, run, digest, packed = job
    ledger = Ledger(run)
    cpu = time.process_time()
    try:
        with call_deadline(300):
            path, cache = cached_features(root, run, packed, ledger)
        return {'digest': digest, 'path': str(path), 'cache_hit': cache,
                'process_cpu_s': time.process_time() - cpu, 'state': 'passed'}
    except Exception as exc:
        return {'digest': digest, 'state': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                'process_cpu_s': time.process_time() - cpu}


def _read_feature(path):
    with Path(path).open('rb') as handle:
        return pickle.load(handle)


def _bounded_prepare(pool, jobs, capacity):
    """Bound outstanding work without requiring Python 3.14 map buffersize."""
    inputs = iter(jobs)
    futures = deque()
    for _ in range(capacity):
        job = next(inputs, None)
        if job is None:
            break
        futures.append(pool.submit(_prepare, job))
    while futures:
        yield futures.popleft().result()
        job = next(inputs, None)
        if job is not None:
            futures.append(pool.submit(_prepare, job))


def _batch_predict(reader, records, ledger, run, reader_name, device):
    from cgx.evaluation.rollout import call_deadline
    batch_features = [record['features'] for record in records]
    ledger.reserve('model_forward', count=len(records), detail=f'replay:{reader_name}')
    started = time.perf_counter()
    batch = collate_features(batch_features, device)
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    collation = time.perf_counter() - started
    started = time.perf_counter()
    with call_deadline(300), torch.inference_mode():
        probabilities = reader(batch).cpu().numpy()
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    output = Path(run) / 'cache/predictions' / reader_name
    output.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(records):
        prediction = {'target_keys': list(batch_features[i].targets),
            'log_probabilities_base2': probabilities[i, :len(batch_features[i].targets)].tolist()}
        if not np.isfinite(prediction['log_probabilities_base2']).all() or abs(np.exp2(prediction['log_probabilities_base2']).sum()-1)>1e-5:
            raise ValueError('Batched reader probabilities invalid')
        fields = prediction_fields(prediction)
        fields.update(diagnostic_reader='main_gnn' if reader_name == 'final' else 'independent_transformer',
            checkpoint_hash=reader.checkpoint_hash, code_hash=record['code_hash'],
            feature_version=FEATURE_VERSION, prediction_prefix_hash=record['prefix_hash'],
            elapsed_rawread_s=record['elapsed_rawread_s'], elapsed_preprocess_s=record['elapsed_preprocess_s'],
            elapsed_model_s=elapsed / len(records), elapsed_collation_s=collation / len(records),
            device=device, precision='fp32', batch_size=len(records), state='passed',
            timing_boundary='batch GPU wall time allocated equally; not original CPU online latency')
        atomic_json(output / (record['prefix_hash'] + '.json'), fields)
    return elapsed, collation


def profile(root, run):
    """Deterministic small prefixes, report parity and timing, not outcome selection."""
    inference_settings()
    descriptors = read_json(run / 'manifests/replay_prefixes.json')
    chosen = sorted(descriptors, key=lambda d: hash_json(['engineering_profile', d]))[:4]
    ledger = Ledger(run)
    records = []
    for digest in chosen:
        path, _ = cached_features(root, run, descriptors[digest], ledger)
        records.append(_read_feature(path))
    comparisons = []
    for name in ('final', 'independent'):
        cpu = load_reader(root, name, 'cpu')
        gpu = load_reader(root, name, 'cuda:0')
        for record in records:
            f = record['features']
            outputs, timings = {}, {}
            for device, model in (('cpu', cpu), ('cuda:0', gpu)):
                ledger.reserve('model_forward', detail='replay_profile')
                started = time.perf_counter()
                with torch.inference_mode():
                    lp = model(collate_features([f], device))[0, :len(f.targets)].cpu().numpy()
                outputs[device] = lp
                timings[device] = time.perf_counter() - started
            pp, qq = np.exp2(outputs['cpu']), np.exp2(outputs['cuda:0'])
            comparisons.append(dict(reader=name, prefix_hash=record['prefix_hash'],
                maximum_probability_difference=float(np.max(np.abs(pp - qq))),
                maximum_log_probability_difference=float(np.max(np.abs(outputs['cpu'] - outputs['cuda:0']))),
                argmax_changed=bool(pp.argmax() != qq.argmax()),
                cpu_argmax_margin=float(np.sort(pp)[-1] - np.sort(pp)[-2]),
                cpu_seconds=timings['cpu'], gpu_seconds=timings['cuda:0'],
                rawread_seconds=record['elapsed_rawread_s'], preprocessing_seconds=record['elapsed_preprocess_s']))
        del cpu, gpu
        torch.cuda.empty_cache()
    pd.DataFrame(comparisons).to_csv(run / 'cpu_gpu_parity.csv', index=False)
    report = {'state': 'passed', 'sample_rule': 'first four prefix hashes after fixed engineering hash ordering',
        'outcomes_not_used_for_sampling': True, 'prefixes': len(chosen), 'comparisons': len(comparisons),
        'max_probability_difference': max(x['maximum_probability_difference'] for x in comparisons),
        'argmax_changes': sum(x['argmax_changed'] for x in comparisons),
        'mean_feature_seconds': float(np.mean([x['preprocessing_seconds'] for x in comparisons])),
        'estimated_feature_cpu_hours': len(descriptors) * float(np.mean([x['preprocessing_seconds'] for x in comparisons])) / 3600,
        'unique_prefixes': len(descriptors), 'new_work_upper_bound': 3 * len(descriptors) + 16}
    atomic_json(run / 'replay_profile.json', report)
    print(json.dumps(report), flush=True)


def batch_parity(root, run, device):
    """Compare CPU singleton outputs with a heterogeneous GPU batch."""
    from cgx.evaluation.rollout import call_deadline
    inference_settings()
    descriptors = read_json(run / 'manifests/replay_prefixes.json')
    ledger = Ledger(run)
    records, graph_refs = [], set()
    for digest in sorted(descriptors, key=lambda d: hash_json(['engineering_profile', d])):
        packed = descriptors[digest]
        if packed['graph_ref'] in graph_refs:
            continue
        path, _ = cached_features(root, run, packed, ledger)
        records.append(_read_feature(path)); graph_refs.add(packed['graph_ref'])
        if len(records) == 4:
            break
    rows = []
    for name in ('final', 'independent'):
        cpu, gpu = load_reader(root, name, 'cpu'), load_reader(root, name, device)
        ledger.reserve('model_forward', count=len(records), detail='replay_batch_parity')
        with call_deadline(300), torch.inference_mode():
            batched = gpu(collate_features([r['features'] for r in records], device)).cpu().numpy()
        for i, record in enumerate(records):
            feature = record['features']
            ledger.reserve('model_forward', detail='replay_batch_parity')
            with call_deadline(300), torch.inference_mode():
                singleton = cpu(collate_features([feature], 'cpu'))[0,:len(feature.targets)].numpy()
            p, q = np.exp2(singleton), np.exp2(batched[i,:len(feature.targets)])
            rows.append(dict(reader=name,prefix_hash=record['prefix_hash'],nodes=len(feature.nodes),
                targets=len(feature.targets),sequences=len(feature.sequences),
                maximum_probability_difference=float(np.max(np.abs(p-q))),
                argmax_changed=bool(p.argmax()!=q.argmax()),
                cpu_argmax_margin=float(np.sort(p)[-1]-np.sort(p)[-2]),batch_size=len(records)))
        del cpu,gpu
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(run/'cpu_gpu_batch_parity.csv',index=False)
    if any(r['argmax_changed'] for r in rows):
        raise ValueError('Heterogeneous batch changed sampled argmax')
    print(json.dumps({'batch_parity_samples':len(rows),'max_probability_difference':max(r['maximum_probability_difference'] for r in rows),
                      'different_graphs':len(graph_refs),'node_counts':sorted({r['nodes'] for r in rows}),
                      'target_counts':sorted({r['targets'] for r in rows})}),flush=True)


def numerical_boundaries(root, run):
    """CPU reference for every unique GPU prefix within the observed parity envelope."""
    from cgx.evaluation.rollout import call_deadline
    inference_settings()
    parity=pd.read_csv(run/'cpu_gpu_batch_parity.csv')
    threshold=2*float(parity.maximum_probability_difference.max())
    original_report=run/'original_cpu_gpu_prefix_parity.json'
    if original_report.exists():
        threshold=max(threshold,2*read_json(original_report)['maximum_probability_difference'])
    frame=pd.read_parquet(run/'prefix_predictions.parquet')
    unique=frame.drop_duplicates(['diagnostic_reader','evidence_prefix_hash'])
    ledger=Ledger(run);checks=[]
    previous_path=run/'replay_numerical_boundary_checks.json'
    previous={(x['reader'],x['prefix_hash']):x for x in read_json(previous_path)} if previous_path.exists() else {}
    for reader_name,label in [('final','main_gnn'),('independent','independent_transformer')]:
        subset=unique[(unique.diagnostic_reader==label)&(unique.argmax_margin<=threshold)]
        if subset.empty:continue
        cpu=load_reader(root,reader_name,'cpu')
        for _,row in subset.iterrows():
            digest=row.evidence_prefix_hash
            if (label,digest) in previous:
                entry=dict(previous[(label,digest)])
                entry.setdefault('first_check_threshold',entry['threshold']);entry['threshold']=threshold
                checks.append(entry)
                continue
            features=_read_feature(run/'cache/features'/digest[:2]/(digest+'.pkl'))['features']
            ledger.reserve('model_forward',detail='replay_numerical_boundary')
            started=time.perf_counter()
            with call_deadline(300),torch.inference_mode():
                lp=cpu(collate_features([features],'cpu'))[0,:len(features.targets)].tolist()
            prediction={'target_keys':list(features.targets),'log_probabilities_base2':lp}
            fields=prediction_fields(prediction)
            affected=frame[(frame.diagnostic_reader==label)&(frame.evidence_prefix_hash==digest)]
            checks.append(dict(reader=label,prefix_hash=digest,threshold=threshold,
                gpu_argmax=row.predicted_target_key,cpu_argmax=fields['predicted_target_key'],
                argmax_changed=row.predicted_target_key!=fields['predicted_target_key'],
                gpu_margin=float(row.argmax_margin),cpu_margin=fields['argmax_margin'],
                maximum_probability_difference=float(np.max(np.abs(np.exp2(lp)-np.exp2(row.full_log_probabilities)))),
                cpu_prediction=prediction,cpu_elapsed_seconds=time.perf_counter()-started,
                affected_prefix_rows=len(affected),affected_terminal_rows=int(affected.is_terminal.sum())))
        del cpu
    atomic_json(run/'replay_numerical_boundary_checks.json',checks)
    summary={'threshold':threshold,'threshold_rule':'twice largest observed probability deviation in heterogeneous batch parity or matching original CPU prefixes',
        'unique_prefixes_checked':len(checks),'argmax_changes':sum(c['argmax_changed'] for c in checks),
        'minimum_argmax_margin':{name:float(g.argmax_margin.min()) for name,g in unique.groupby('diagnostic_reader')},
        'gpu_predictions_preserved':True}
    atomic_json(run/'replay_numerical_integrity.json',summary)
    print(json.dumps(summary),flush=True)


def original_parity(run):
    """Compare existing original CPU trace predictions to their exact reread prefix."""
    frame=pd.read_parquet(run/'prefix_predictions.parquet')
    frame=frame[(frame.diagnostic_reader=='main_gnn') & frame.diagnosis_updated_in_original]
    comparisons=[]
    for row in frame.to_dict('records'):
        old=row['original_trace_log_probabilities']
        if old is None:continue
        new=row['full_log_probabilities']
        comparisons.append({**{k:row[k] for k in ('case_id','method_key','policy_seed','decision_index','evidence_prefix_hash')},
            'maximum_probability_difference':float(np.max(np.abs(np.exp2(old)-np.exp2(new)))),
            'maximum_log_probability_difference':float(np.max(np.abs(np.asarray(old)-np.asarray(new)))),
            'argmax_changed':row['original_trace_predicted_target_key']!=row['predicted_target_key']})
    result=pd.DataFrame(comparisons)
    _atomic_frame(result,run/'original_cpu_gpu_prefix_parity.csv')
    summary={'compared_recorded_valid_prefix_rows':len(result),'argmax_changes':int(result.argmax_changed.sum()),
        'maximum_probability_difference':float(result.maximum_probability_difference.max()),
        'maximum_log_probability_difference':float(result.maximum_log_probability_difference.max()),
        'only_original_recorded_predictions_at_matching_valid_prefixes':True,
        'new_model_or_feature_calls':0}
    atomic_json(run/'original_cpu_gpu_prefix_parity.json',summary)
    print(json.dumps(summary),flush=True)


def run_reader(root, run, reader_name, device, workers=24, batch_size=8):
    inference_settings()
    ledger = Ledger(run)
    descriptors = read_json(run / 'manifests/replay_prefixes.json')
    reader = load_reader(root, reader_name, device)
    expected_code_hash = code_hash()
    prediction_dir = run / 'cache/predictions' / reader_name
    prediction_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for digest, packed in sorted(descriptors.items()):
        destination = prediction_dir / (digest + '.json')
        if destination.exists():
            existing = read_json(destination)
            if (existing['checkpoint_hash'],existing['feature_version'],existing['code_hash'],existing['prediction_prefix_hash']) != (
                    reader.checkpoint_hash,FEATURE_VERSION,expected_code_hash,digest):
                raise ValueError('Cached prediction identity mismatch')
        else:
            pending.append((str(root),str(run),digest,packed))
    done = len(descriptors) - len(pending)
    report = {'reader': reader_name, 'device': device, 'total': len(descriptors), 'resumed_completed': done,
        'workers': workers, 'batch_size': batch_size, 'failed': [], 'worker_cpu_seconds': 0.0,
        'gpu_model_seconds': 0.0, 'collation_seconds': 0.0, 'state': 'running'}
    started = time.monotonic()
    buffer = []
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'))
    try:
        iterator = _bounded_prepare(pool, pending, workers * 2)
        for result in iterator:
            report['worker_cpu_seconds'] += result['process_cpu_s']
            if result['state'] != 'passed':
                report['failed'].append(result)
                continue
            record = _read_feature(result['path'])
            if (record['prefix_hash'],record['feature_version'],record['code_hash']) != (
                    result['digest'],FEATURE_VERSION,expected_code_hash):
                raise ValueError('Cached feature identity mismatch')
            buffer.append(record)
            tokens = sum(sum(len(s) for s in record['features'].sequences) for record in buffer)
            if len(buffer) >= batch_size or tokens >= 120000:
                elapsed, collation = _batch_predict(reader, buffer, ledger, run, reader_name, device)
                report['gpu_model_seconds'] += elapsed; report['collation_seconds'] += collation
                done += len(buffer); buffer = []
                report.update(completed=done, elapsed_seconds=time.monotonic() - started)
                atomic_json(run / f'replay_{reader_name}_status.json', report)
                if done % 200 < batch_size:
                    print(json.dumps({'reader': reader_name, 'completed': done, 'total': len(descriptors),
                                      'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
        if buffer:
            elapsed, collation = _batch_predict(reader, buffer, ledger, run, reader_name, device)
            report['gpu_model_seconds'] += elapsed; report['collation_seconds'] += collation
            done += len(buffer)
        report.update(state='passed' if not report['failed'] else 'partial', completed=done,
                      elapsed_seconds=time.monotonic() - started)
        atomic_json(run / f'replay_{reader_name}_status.json', report)
    except BaseException as exc:
        report.update(state='failed', error=f'{type(exc).__name__}: {exc}', completed=done,
                      elapsed_seconds=time.monotonic() - started)
        atomic_json(run / f'replay_{reader_name}_status.json', report)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    print(json.dumps(report), flush=True)


def aggregate(run):
    inventory = pd.read_parquet(run / 'replay_prefix_inventory.parquet')
    records, missing = [], []
    for name in ('final', 'independent'):
        cached = {}
        for row in inventory.to_dict('records'):
            digest = row['evidence_prefix_hash']
            path = run / 'cache/predictions' / name / (digest + '.json')
            if not path.exists():
                missing.append({**{k: row[k] for k in ('case_id', 'method_key', 'policy_seed', 'decision_index')},
                    'reader': name, 'evidence_prefix_hash': digest})
                continue
            if digest not in cached:
                cached[digest] = read_json(path)
            item = {**row, **cached[digest]}
            item['is_healthy'] = row['true_target_key'] == 'healthy'
            item['false_alarm'] = item['is_healthy'] and item['predicted_target_key'] != 'healthy'
            item['missed_fault'] = not item['is_healthy'] and item['predicted_target_key'] == 'healthy'
            item['correct_joint'] = item['predicted_target_key'] == row['true_target_key']
            item['nll_bits'] = -item['full_log_probabilities'][item['target_keys'].index(row['true_target_key'])]
            records.append(item)
    frame = pd.DataFrame(records)
    _atomic_frame(frame,run / 'prefix_predictions.parquet')
    atomic_json(run / 'prefix_missing.json', missing)
    keys = ['configuration_id', 'case_id', 'method_key', 'policy_seed', 'diagnostic_reader']
    transitions, finals, stories, action_changes = [], [], [], []
    for identity, group in frame.groupby(keys, dropna=False):
        group = group.sort_values('decision_index')
        base = dict(zip(keys, identity))
        base['policy_seed'] = int(base['policy_seed'])
        switches = group.predicted_target_key.ne('healthy').to_numpy()
        h2f = int(np.sum((~switches[:-1]) & switches[1:]))
        f2h = int(np.sum(switches[:-1] & (~switches[1:])))
        end = group.iloc[-1]
        transitions.append(dict(base, is_healthy=bool(end.is_healthy), prefixes=len(group),
            healthy_to_fault=h2f, fault_to_healthy=f2h, ever_fault=bool(switches.any()),
            final_fault=bool(switches[-1]), temporary_fault_recovered=bool(switches.any() and not switches[-1]),
            initial_fault=bool(switches[0]), final_is_actual_terminal=bool(end.is_terminal)))
        prefix_rows=group.to_dict('records')
        for previous,current in zip(prefix_rows[:-1],prefix_rows[1:]):
            action_changes.append(dict(base, decision_index=current['decision_index'],
                is_healthy=bool(end.is_healthy), action_kind=current['action_kind'],
                added_execution=current['executions_used']-previous['executions_used'],
                added_calibration=current['calibrations_used']-previous['calibrations_used'],
                added_bytes=current['bytes_used']-previous['bytes_used'],
                delta_p_healthy=current['p_healthy']-previous['p_healthy'],
                delta_fault_minus_healthy=current['fault_minus_healthy']-previous['fault_minus_healthy'],
                previous_prediction=previous['predicted_target_key'], current_prediction=current['predicted_target_key'],
                healthy_to_fault=previous['predicted_target_key']=='healthy' and current['predicted_target_key']!='healthy',
                fault_to_healthy=previous['predicted_target_key']!='healthy' and current['predicted_target_key']=='healthy',
                prefix_hash=current['evidence_prefix_hash']))
        original = end.original_predicted_target_key
        finals.append(dict(base, true_target_key=end.true_target_key, original_predicted_target_key=original,
            reread_predicted_target_key=end.predicted_target_key, original_false_alarm=bool(end.is_healthy and original!='healthy'),
            reread_false_alarm=bool(end.false_alarm), original_missed_fault=bool(not end.is_healthy and original=='healthy'),
            reread_missed_fault=bool(end.missed_fault), nll_bits=float(end.nll_bits),
            original_nll_bits=-end.original_final_log_probabilities[list(end.original_target_keys).index(end.true_target_key)],
            q_healthy=float(end.p_healthy), original_timeout=bool(end.original_timeout),
            actual_observation_prefix=int(end.decision_index), original_prediction_prefix=int(end.last_valid_prediction_prefix),
            unread_observations=max(0,int(end.decision_index-end.last_valid_prediction_prefix)),
            final_is_actual_terminal=bool(end.is_terminal), source_final_history_hash=end.source_final_history_hash))
        if end.is_healthy and end.diagnostic_reader=='main_gnn':
            category = 'healthy_to_fault' if (h2f>0 and switches[-1]) else (
                'always_healthy' if not switches.any() else ('fault_then_recovered' if switches.any() and not switches[-1] else None))
            if category:
                stories.append((end.case_id,category,base,
                    group[['decision_index','executions_used','calibrations_used','bytes_used','action_kind',
                           'p_healthy','max_fault_probability','max_fault_target','predicted_target_key']].to_dict('records')))
    _atomic_frame(pd.DataFrame(transitions),run / 'trajectory_transitions.csv')
    _atomic_frame(pd.DataFrame(finals),run / 'fixed_transcript_reread.csv')
    _atomic_frame(pd.DataFrame(action_changes),run / 'prefix_action_changes.csv')
    chosen=[];globally_selected=set()
    for category in ('healthy_to_fault','always_healthy','fault_then_recovered'):
        seen_cases=set()
        for d,c,b,p in sorted((s for s in stories if s[1]==category),
                              key=lambda s:(s[0],s[2]['method_key'],s[2]['policy_seed'])):
            if d in globally_selected:
                continue
            seen_cases.add(d)
            globally_selected.add(d)
            chosen.append(dict(hash=d,category=c,identity=b,prefixes=p,
                               selection_rule='fixed category order healthy_to_fault, always_healthy, fault_then_recovered; within each, two smallest as-yet-unselected case hashes globally; ties by method key then seed'))
            if len(seen_cases)==2:
                break
    atomic_json(run / 'six_case_stories.json', chosen)
    atomic_json(run / 'replay_complete.json', {'state':'passed' if not missing else 'partial',
        'prefix_prediction_rows':len(frame),'final_trajectory_reader_rows':len(finals),
        'missing_prefix_reader_rows':len(missing),'stories':len(chosen),
        'actual_terminal_denominator_preserved':True})
    write_timing_notes(run)
    print(json.dumps({'prefix_rows':len(frame),'final_rows':len(finals),'missing':len(missing)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['inventory','profile','batch-parity','numerical-boundaries','original-parity','run','aggregate'])
    parser.add_argument('--root',type=Path,default=Path('/work/Users/leiyo/cgx_data_v2'))
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reader',choices=['final','independent'],default='final')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--workers',type=int,default=24)
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args()
    with TimedStage(args.run,'replay_'+args.stage+'_'+args.reader):
        if args.stage=='inventory':inventory(args.root,args.run)
        elif args.stage=='profile':profile(args.root,args.run)
        elif args.stage=='batch-parity':batch_parity(args.root,args.run,args.device)
        elif args.stage=='numerical-boundaries':numerical_boundaries(args.root,args.run)
        elif args.stage=='original-parity':original_parity(args.run)
        elif args.stage=='run':run_reader(args.root,args.run,args.reader,args.device,args.workers,args.batch_size)
        else:aggregate(args.run)


if __name__=='__main__':main()
