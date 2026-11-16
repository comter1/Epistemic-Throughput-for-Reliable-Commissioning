"""Source-only likelihood fitting; existing collected physical measurements only.

No simulator, hidden clock offsets or injected severities are read. The original
replay is opened immutable/read-only. All new artifacts live in the new run.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import numpy as np
from cgx.data.store import Store
from cgx.provenance import atomic_json, hash_json, file_sha256, now
from .events import prepare_history, EVENT_VERSION
from .likelihood import prepare_evidence, fit_mixture, LIKELIHOOD_VERSION

MODEL_SPEC = dict(version=LIKELIHOOD_VERSION,
    event_identity='opaque actual acquisition + port + event occurrence; preserve IDs through repackaging',
    numeric_equality_deduplication=False,
    timing='t_event=U_message+C_clock+D_fault*public_downstream_path+independent_residual',
    message_epoch='exact flat-prior integration by weighted centering, preserving contrast covariance',
    clocks='one persistent Gaussian latent per public clock group across every acquisition',
    delay='one case-persistent scalar with learned three-component Gaussian mixture per fault family',
    gain='one case-persistent scalar with learned six-component Gaussian mixture, shared adapter parameters',
    signal='nominal public transfer relation for acquired same-message adapter input/output packets',
    repeated_payload='same message values share a packet-local signal factor; no independent physical payload invented',
    residual_approximation='conditional residuals independent after U,C,D; learned delay-family process dispersion covers within-message age variation; extra shared downstream age residuals are not separately latent',
    external_motion='retained with identity; F-independent likelihood cancels, external-only initial history returns legal prior',
    missing_timeout='retained explicitly; no observed numeric value means uninformative likelihood',
    acquisition_count_features=False,action_frequency_features=False,
    learned_noise='TRAIN healthy timestamp residuals, public calibration contrasts, nominal transfer rounding residuals',
    learned_clock_prior='TRAIN public per-case calibration-offset means with independent-noise correction',
    forbidden_training_fields=['observer_offsets_ms','severity','debug','physical_q','true_time'],
    allowed_supervision='TRAIN true_target_key only; original topology crossfit exclusions retained',
    gaussian_mixture_em_iterations=100,source_validation_hyperparameter_grid=None,
    variance_floors=dict(timestamp_seconds_squared=1e-12,calibration_seconds_squared=1e-12,
        clock_seconds_squared=1e-12,gain_squared=1e-16),
    probability_floor=1e-8,full_reader_backbone_training=False)

_STORE = _DB = _RUN = None


def _initialize(root,run):
    global _STORE,_DB,_RUN
    _STORE=Store(root);_RUN=Path(run)
    path=Path(root)/'training/replay_train.sqlite'
    _DB=sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)


def _clock_centered_quadratic(p,fault_node=None):
    n=len(p['times']);k=len(p['meta']['clocks'])
    if not n:return np.zeros(3),0,0.
    groups=p['groups']*k+p['clocks'];_,g=np.unique(groups,return_inverse=True)
    count=np.bincount(g);t=p['times'];mean=np.bincount(g,weights=t)/count
    centered=t-mean[g];df=n-len(count)
    if fault_node in p['meta']['delay_nodes']:
        v=p['delay_path'][:,p['meta']['delay_nodes'].index(fault_node)]
        vc=v-np.bincount(g,weights=v)[g]/count[g]
        return np.array([centered@centered,centered@vc,vc@vc]),df,float(v.sum())
    return np.array([centered@centered,0.,0.]),df,0.


def extract_case(task):
    case_id,fold,configuration=task;output=_RUN/'reader_training/cases'/f'{case_id}.json'
    if output.exists():
        saved=json.loads(output.read_text())
        if saved['extractor_sha256']!=file_sha256(Path(__file__)):raise RuntimeError('Reader extractor changed; explicit new cache required')
        return dict(case_id=case_id,reused=True,wall_seconds=saved['wall_seconds'],raw_records=saved['raw_records'])
    started=time.perf_counter()
    rows=[json.loads(r[0]) for r in _DB.execute('SELECT data FROM transitions WHERE case_id=? ORDER BY idx',(case_id,))]
    if not rows:raise ValueError('Source case has no transitions')
    if any(r['split']!='train' or r['configuration_id']!=configuration or r['source_fold']!=fold for r in rows):raise ValueError('Source split/fold inconsistency')
    label=rows[0]['true_target_key']
    if any(r['true_target_key']!=label for r in rows):raise ValueError('Source label inconsistency')
    node=None if label=='healthy' else label.split(':')[0]
    family='healthy' if label=='healthy' else label.split(':')[1]
    initial=_STORE.history_references(rows[0]['history_hash'])['observation_refs'][0]
    observations=[(hash_json(['source_initial',case_id]),initial)]
    for row in rows:
        if row.get('acquired_blob_hash'):
            observations.append((hash_json(['source_acquisition',row['trajectory_id'],row['step']]),row['acquired_blob_hash']))
    identities=[x[0] for x in observations]
    if len(identities)!=len(set(identities)):raise ValueError('Duplicate physical source-operation identity')
    graph=_STORE.graph(configuration)
    raw_records=0;calibration_events=0;timing_events=0;missing=0;quadratic=np.zeros(3);df=0;downstream=0.
    healthy_gain=[];fault_gain=[];local_delay=[];calibration=defaultdict(list)
    read_seconds=extract_seconds=0.
    for identity,ref in observations:
        began=time.perf_counter();observation=_STORE.load_observation(ref);read_seconds+=time.perf_counter()-began
        began=time.perf_counter()
        evidence=prepare_history(dict(graph=graph,observations=[observation]),{'observation_ids':[identity]})
        p=prepare_evidence(evidence)
        extract_seconds+=time.perf_counter()-began
        raw_records+=p['unique_events'];calibration_events+=p['unique_calibrations'];timing_events+=len(p['times']);missing+=p['missing_events']
        q,d,v=_clock_centered_quadratic(p,node);quadratic+=q;df+=d;downstream+=v
        if node is not None:local_delay.extend(p['local_delay'].get(node,[]))
        for owner,values in p['gain'].items():
            (fault_gain if owner==node and family=='adapter_scale' else healthy_gain).extend(values)
        for a,b,offset in p['calibration']:
            if a>b:a,b,offset=b,a,-offset
            calibration[int(a),int(b)].append(float(offset))
    def summary(values):
        x=np.asarray(values,dtype=float)
        return dict(n=len(x),sum=float(x.sum()),sum_squares=float(x@x),
            mean=float(x.mean()) if len(x) else None,variance=float(x.var(ddof=1)) if len(x)>1 else None,
            median=float(np.median(x)) if len(x) else None)
    item=dict(state='passed',case_id=case_id,source_fold=fold,configuration_id=configuration,
        true_target_key=label,fault_family=family,fault_node=node,
        physical_observations=len(observations),acquisition_identity_manifest_sha256=hash_json(observations),
        raw_records=raw_records,calibration_events=calibration_events,timing_events=timing_events,missing_events=missing,
        clock_centered_quadratic=quadratic.tolist(),clock_centered_df=df,downstream_events=downstream,
        healthy_gain=summary(healthy_gain),fault_gain=summary(fault_gain),local_fault_delay=summary(local_delay),
        calibration_pairs=[dict(pair=list(pair),**summary(values)) for pair,values in sorted(calibration.items())],
        read_seconds=read_seconds,extract_seconds=extract_seconds,wall_seconds=time.perf_counter()-started,
        extractor_sha256=file_sha256(Path(__file__)),event_version=EVENT_VERSION,
        model_spec_sha256=hash_json(MODEL_SPEC),hidden_clock_offsets_read=False,hidden_fault_severity_read=False)
    atomic_json(output,item)
    return dict(case_id=case_id,reused=False,wall_seconds=item['wall_seconds'],raw_records=raw_records,
        read_seconds=read_seconds,extract_seconds=extract_seconds,physical_observations=len(observations))


def source_tasks(root):
    path=Path(root)/'training/replay_train.sqlite'
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        return db.execute('SELECT case_id,fold,configuration FROM cases ORDER BY case_id').fetchall()


def extract(root,run,workers,shard=0,shards=1,profile=False):
    run=Path(run);started=time.perf_counter();tasks=source_tasks(root)
    assert len(tasks)==4096
    spec_path=run/'manifests/structured_reader_model_spec.json'
    if spec_path.exists():assert json.loads(spec_path.read_text())==MODEL_SPEC
    else:atomic_json(spec_path,MODEL_SPEC)
    selected=tasks[shard::shards]
    if profile:selected=selected[:1]
    results=[]
    if workers==1:
        _initialize(root,run)
        for task in selected:results.append(extract_case(task))
    else:
        with ProcessPoolExecutor(max_workers=workers,initializer=_initialize,initargs=(root,str(run))) as pool:
            futures=[pool.submit(extract_case,t) for t in selected]
            for f in as_completed(futures):
                results.append(f.result())
                if len(results)%64==0:
                    print(json.dumps(dict(stage='reader_source_extraction',shard=shard,completed=len(results),expected=len(selected),
                        wall_seconds=time.perf_counter()-started)),flush=True)
    report=dict(state='passed',source_split='train',case_count=len(results),full_source_cases=4096,
        shard=shard,shards=shards,workers=workers,profile=profile,results=results,
        wall_seconds=time.perf_counter()-started,model_spec_sha256=hash_json(MODEL_SPEC))
    atomic_json(run/'reader_training'/('profile.json' if profile else f'extraction_shard_{shard}.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
    if profile:print(json.dumps(results[0]),flush=True)
    return report


def fit(run,name):
    run=Path(run);started=time.perf_counter();tasks=source_tasks(json.loads((run/'source_checks.json').read_text())['original_data_root'])
    cases=[json.loads((run/'reader_training/cases'/f'{t[0]}.json').read_text()) for t in tasks]
    excluded=int(name.split('_')[1]) if name.startswith('crossfit_') else None
    used=[c for c in cases if c['source_fold']!=excluded]
    if name!='final' and excluded not in (0,1,2):raise ValueError('Reader name')
    healthy=[c for c in used if c['fault_family']=='healthy']
    timestamp_variance=max(sum(c['clock_centered_quadratic'][0] for c in healthy)/sum(c['clock_centered_df'] for c in healthy),1e-12)
    cal_sse=sum(p['sum_squares']-p['sum']**2/p['n'] for c in used for p in c['calibration_pairs'] if p['n']>1)
    cal_df=sum(p['n']-1 for c in used for p in c['calibration_pairs'] if p['n']>1)
    if cal_df==0:raise ValueError('No public TRAIN calibration residual support')
    calibration_variance=max(cal_sse/cal_df,1e-12)
    clock_samples=[max(0.,p['mean']**2-calibration_variance/p['n'])/2 for c in used for p in c['calibration_pairs']]
    clock_variance=max(float(np.mean(clock_samples)),1e-12)
    gain_n=sum(c['healthy_gain']['n'] for c in used);gain_sum=sum(c['healthy_gain']['sum'] for c in used)
    gain_squares=sum(c['healthy_gain']['sum_squares'] for c in used)
    gain_mean=gain_sum/gain_n;gain_variance=max((gain_squares-gain_sum**2/gain_n)/(gain_n-1),1e-16)
    delay_mixtures={};process={};support={}
    for family in ('cache_stale','queue_hold'):
        rows=[c for c in used if c['fault_family']==family and c['local_fault_delay']['n']>0]
        values=[c['local_fault_delay']['median'] for c in rows]
        sampling_floor=max(float(np.median([2*timestamp_variance/c['local_fault_delay']['n'] for c in rows])),1e-12)
        delay_mixtures[family]=fit_mixture(values,3,sampling_floor)
        residual=sum(max(0.,c['clock_centered_quadratic'][0]-2*c['local_fault_delay']['median']*c['clock_centered_quadratic'][1]
            +c['local_fault_delay']['median']**2*c['clock_centered_quadratic'][2]) for c in rows)
        dof=sum(c['clock_centered_df'] for c in rows);affected=sum(c['downstream_events'] for c in rows)
        process[family]=max(0.,(residual-timestamp_variance*dof)/max(1.,affected))
        support[family]=dict(cases_with_local_pair=len(rows),total_family_cases=sum(c['fault_family']==family for c in used))
    gain_rows=[c for c in used if c['fault_family']=='adapter_scale' and c['fault_gain']['n']>0]
    gain_floor=max(float(np.median([gain_variance/c['fault_gain']['n'] for c in gain_rows])),1e-16)
    gain_mixture=fit_mixture([c['fault_gain']['median'] for c in gain_rows],6,gain_floor)
    support['adapter_scale']=dict(cases_with_local_pair=len(gain_rows),total_family_cases=sum(c['fault_family']=='adapter_scale' for c in used))
    parameters=dict(version=LIKELIHOOD_VERSION,probability_floor=1e-8,
        timestamp_noise_variance=timestamp_variance,calibration_noise_variance=calibration_variance,
        clock_prior_variance=clock_variance,healthy_gain_mean=gain_mean,gain_noise_variance=gain_variance,
        timing_process_variance=process,delay_mixtures=delay_mixtures,gain_mixture=gain_mixture,
        model_spec_sha256=hash_json(MODEL_SPEC))
    payload=dict(parameters=parameters,model_spec=MODEL_SPEC,name=name,excluded_fold=excluded,
        training_configurations=sorted({c['configuration_id'] for c in used}),
        training_cases=[c['case_id'] for c in used],support=support,
        training_feature_hashes={c['case_id']:file_sha256(run/'reader_training/cases'/f"{c['case_id']}.json") for c in used},
        training_data_only=True,validation_selected_hyperparameters=False,hidden_offsets_or_severities_used=False)
    directory=run/'models/readers'/name;path=directory/'model.json'
    if path.exists():raise RuntimeError('Existing model is immutable; no silent refit')
    atomic_json(path,payload)
    marker=dict(state='passed',name=name,selected_checkpoint='model.json',selected_checkpoint_hash=file_sha256(path),
        training_cases=len(used),training_configurations=payload['training_configurations'],excluded_fold=excluded,
        support=support,wall_seconds=time.perf_counter()-started,model_spec_sha256=hash_json(MODEL_SPEC),
        complete_at=now(),reader_backbone_updates=0,maximum_likelihood_fit=True)
    atomic_json(directory/'complete.json',marker);print(json.dumps(marker),flush=True)
    return marker


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','fit']);p.add_argument('--root');p.add_argument('--run-dir',required=True)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--profile',action='store_true');p.add_argument('--name',default='final')
    a=p.parse_args()
    if a.stage=='extract':extract(a.root,a.run_dir,a.workers,a.shard,a.shards,a.profile)
    else:fit(a.run_dir,a.name)


if __name__=='__main__':main()
