"""Paired configuration/case bootstrap and scientific exports."""
from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
from cgx.provenance import atomic_json, file_sha256, read_json

def paired_bootstrap(frame, first, second, repetitions=2000, seed=671917):
    """Same resampled configurations/cases for both methods; seeds averaged separately."""
    keys=["configuration_id","case_id"]
    reduced=frame.groupby(keys+["method"],dropna=False)["final_nll_bits"].mean().unstack("method")
    if first not in reduced or second not in reduced or reduced[[first,second]].isna().any().any():
        raise ValueError("Paired methods do not cover exactly the same cases")
    # Positive difference means first has larger (worse) NLL; never flip sign by result.
    difference=reduced[first]-reduced[second]
    systems=[v.to_numpy() for _,v in difference.groupby(level="configuration_id")]
    if not systems: raise ValueError("No paired systems")
    rng=np.random.default_rng(seed); draws=np.empty(repetitions)
    for i in range(repetitions):
        chosen=rng.integers(len(systems),size=len(systems))
        draws[i]=np.mean([np.mean(systems[j][rng.integers(len(systems[j]),size=len(systems[j]))]) for j in chosen])
    estimate=float(np.mean([v.mean() for v in systems]))
    # Centered bootstrap null distribution, two-sided, finite Monte Carlo correction.
    pvalue=float((1+np.count_nonzero(np.abs(draws-estimate)>=abs(estimate)))/(repetitions+1))
    return {"first":first,"second":second,"difference_nll_bits":estimate,
            "ci_low":float(np.quantile(draws,.025)),"ci_high":float(np.quantile(draws,.975)),
            "p_value_centered_bootstrap":pvalue,"bootstrap_unit":"configuration_then_case",
            "configurations":len(systems),"cases":len(difference),"repetitions":repetitions}

def holm(pvalues):
    order=np.argsort(pvalues); adjusted=np.empty(len(order)); running=0.
    for rank,index in enumerate(order):
        running=max(running,(len(order)-rank)*pvalues[index]); adjusted[index]=min(1.,running)
    return adjusted.tolist()

def validate_result_matrix(frame, cases, config):
    """Reject duplicate/missing cells even when a total row count happens to match."""
    frame=frame.copy()
    frame['_seed']=frame.policy_seed.map(lambda v:'none' if pd.isna(v) else str(int(v)))
    keys=['method','_seed','byte_fraction','execution_cap','case_id']
    if frame.duplicated(keys).any():raise ValueError('Duplicated evaluation matrix cell')
    registered={c['case_id']:(c['split'],c['configuration_id']) for c in cases if c['split'].startswith('test_')}
    from cgx.evaluation.design import matrix_cells
    expected={(m,'none' if seed is None else str(seed),float(f),cap) for m,seed,cap,f in matrix_cells(config)}
    found=set()
    for key,g in frame.groupby(keys[:-1],dropna=False):
        found.add(key)
        if set(g.case_id)!=set(registered):raise ValueError(f'Missing/unregistered cases for matrix cell {key}')
        for row in g[['case_id','suite','configuration_id']].itertuples(index=False):
            if registered[row.case_id]!=(row.suite,row.configuration_id):raise ValueError('Case/suite/configuration join mismatch')
    if found!=expected:raise ValueError(f'Method/seed/budget matrix differs: missing={len(expected-found)}, extra={len(found-expected)}')
    return True


def write_result_hashes(root):
    root=Path(root);output=root/'results'
    hashes={str(p.relative_to(root)):file_sha256(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='hashes.json'}
    atomic_json(output/'hashes.json',hashes)
    return hashes


def resource_tables(root):
    """Observed job/marker resources only; absent measurements remain missing."""
    root=Path(root);output=root/'results';models=[]
    for kind in ('readers','policies'):
        for path in sorted((root/'models'/kind).glob('*/complete.json')):
            row=read_json(path);name=path.parent.name
            job=root/'jobs'/('reader_'+name if kind=='readers' else name)/'status.json'
            status=read_json(job) if job.exists() else {}
            models.append(dict(kind=kind,run=name,state=row.get('state'),technical_smoke=row.get('technical_smoke'),
                actual_updates=row.get('actual_updates'),parameters=row.get('parameter_count'),
                trainer_elapsed_seconds=row.get('elapsed_seconds'),job_allocation_seconds=status.get('elapsed_seconds'),
                peak_gpu_memory_bytes=status.get('peak_gpu_memory_bytes',row.get('peak_gpu_memory_bytes')),
                selected_checkpoint_hash=row.get('selected_checkpoint_hash'),marker_sha256=file_sha256(path),
                allocation_boundary='job wall time includes validation; trainer marker is current resumed invocation'))
    pd.DataFrame(models).to_csv(output/'model_training_resources.csv',index=False)
    collection=[]
    for path in sorted((root/'jobs'/'collection').rglob('*.json')):
        row=read_json(path)
        if 'physical_executions' not in row:continue
        collection.append(dict(split=path.parent.name,case_shard=path.stem,state=row.get('state'),
            physical_executions=row.get('physical_executions'),transitions=row.get('transitions'),
            worker_elapsed_seconds=row.get('elapsed_seconds'),technical_smoke=row.get('technical_smoke'),marker_sha256=file_sha256(path)))
    pd.DataFrame(collection).to_csv(output/'source_collection_resources.csv',index=False)
    stages=[]
    for path in [root/'training/features_complete.json',root/'reports/resource_forecast.json',root/'reports/gpu_allocation_usage.json']:
        if path.exists():stages.append(dict(artifact=str(path.relative_to(root)),sha256=file_sha256(path),actual_record=read_json(path)))
    atomic_json(output/'resource_stage_evidence.json',stages)
    return models,collection


def aggregate(root,config,repetitions=2000):
    root=Path(root)
    completion=read_json(root/"evaluation/main/complete.json")
    if completion["state"]!="passed": raise ValueError("Main evaluation incomplete")
    files=sorted((root/"evaluation/main/cases").glob("*.json"))
    frame=pd.DataFrame([read_json(p)["row"] for p in files])
    if frame.empty or frame.technical_smoke.any(): raise ValueError("Empty or contaminated scientific result table")
    expected=completion["expected_for_requested_suites"]
    if len(frame)!=expected: raise ValueError(f"Expected {expected} results, found {len(frame)}")
    from cgx.graphs.manifests import read_jsonl,load_cases
    from .diagnostics import enrich_evaluator_results,write_diagnostic_tables
    cases=[case for split in config['splits'] if split.startswith('test_') for case in load_cases(root,split)]
    validate_result_matrix(frame,cases,config)
    graphs={row['configuration_id']:row['graph'] for row in read_jsonl(root/'manifests/public_graphs.jsonl')}
    frame=enrich_evaluator_results(frame,cases,graphs)
    output=root/"results"; output.mkdir(parents=True,exist_ok=True)
    with (output/'evaluation_input_hashes.jsonl').open('w') as evidence:
        for path in files:evidence.write(json.dumps({'path':str(path.relative_to(root)),'sha256':file_sha256(path)})+'\n')
    thresholds=read_json(root/'models/thresholds.json')
    if 'cost_to_quality' not in thresholds:raise ValueError('Validation quality target was not sealed before evaluation')
    write_diagnostic_tables(root,frame,thresholds['cost_to_quality'])
    frame.to_parquet(output/"per_case_results.parquet",index=False)
    scalar=frame.drop(columns=[c for c in ("prediction","errors") if c in frame])
    scalar.to_csv(output/"per_case_results.csv",index=False)
    grouping=["suite","method","policy_seed","byte_fraction","execution_cap","configuration_id"]
    metrics=["initial_nll_bits","final_nll_bits","score_improvement_bits","prior_nll_bits","correct_joint","correct_mechanism",
             "bytes_used","executions_used","calibrations_used","decisions_used","selection_seconds","inference_seconds","simulation_seconds",
             "timeout","abstained","target_cardinality"]
    metrics += [k for k in ('initial_bytes','recorder_seconds','warmup_seconds','end_to_end_seconds') if k in frame]
    systems=frame.groupby(grouping,dropna=False)[metrics].mean().reset_index()
    systems.to_csv(output/"per_configuration.csv",index=False)
    seeds=systems.groupby(grouping[:-1],dropna=False)[metrics].mean().reset_index()
    seeds.to_csv(output/"per_seed.csv",index=False)
    seeds.groupby(["suite","method","byte_fraction","execution_cap"],dropna=False)[metrics].agg(["mean","std"]).to_csv(output/"summary.csv")
    primary=frame[(frame.suite==config["evaluation"]["primary_suite"]) & (frame.byte_fraction==config["evaluation"]["primary_byte_fraction"]) & (frame.execution_cap==config['evaluation']['main_execution_cap'])]
    contrasts=[]
    comparators=[config["evaluation"]["primary_comparator"],config["evaluation"]["primary_secondary_comparator"],*config["evaluation"]["required_other_comparisons"]]
    for comparator in comparators:
        row=paired_bootstrap(primary,"gnn_adaptive",comparator,repetitions,config["seeds"]["bootstrap"])
        row["primary"]=comparator==config["evaluation"]["primary_comparator"]; contrasts.append(row)
    adjusted=holm([r["p_value_centered_bootstrap"] for r in contrasts[1:]])
    for row,p in zip(contrasts[1:],adjusted): row["secondary_holm_p"]=p
    atomic_json(output/"paired_statistics.json",contrasts)
    repair_files=sorted((root/"evaluation/repairs/cases").glob("*.json"))
    if repair_files:
        repair=pd.DataFrame([read_json(p)["row"] for p in repair_files]); repair.to_parquet(output/"repair_results.parquet",index=False)
        summaries=[]
        for keys,group in repair.groupby(["suite","method","policy_seed"],dropna=False):
            accepted=group[group.declared_completion]; n=len(accepted)
            summaries.append(dict(zip(["suite","method","policy_seed"],keys),coverage=n/len(group),
                risk=float((~accepted.audit_all_pass.astype(bool)).mean()) if n else None,
                yield_fraction=float(accepted.audit_all_pass.astype(bool).sum()/len(group)),
                audit_per_execution_failure_rate=float(1-accepted.audit_pass_count.sum()/accepted.audit_count.sum()) if n else None,
                mean_total_deployed_executions=float(group.total_deployed_executions.mean()),
                mean_total_deployed_bytes=float(group.total_deployed_bytes.mean()),mean_total_deployed_seconds=float(group.total_deployed_seconds.mean()),cases=len(group)))
        pd.DataFrame(summaries).to_csv(output/"repair_summary.csv",index=False)
    independent_path=root/'evaluation/independent_reader.jsonl'
    if not independent_path.exists():raise ValueError('Independent reader results are required before final aggregation')
    independent=pd.DataFrame(list(read_jsonl(independent_path)))
    join=['suite','configuration_id','case_id','method','policy_seed','byte_fraction','execution_cap']
    if independent.duplicated(join).any() or len(independent)!=len(frame):raise ValueError('Independent reader result matrix is incomplete/duplicated')
    reread=frame[join+['evidence_trace_hash','final_nll_bits']].merge(independent,on=join,suffixes=('_primary','_independent'),validate='one_to_one')
    if len(reread)!=len(frame) or not reread.evidence_trace_hash_primary.eq(reread.evidence_trace_hash_independent).all():raise ValueError('Independent reader did not reuse exactly the acquired transcripts')
    reread.to_parquet(output/'independent_reader_per_case.parquet',index=False)
    reader_systems=reread.groupby(grouping,dropna=False)[['final_nll_bits_primary','final_nll_bits_independent']].mean().reset_index()
    reader_systems.to_csv(output/'independent_reader_per_configuration.csv',index=False)
    reader_systems.groupby(grouping[:-1],dropna=False)[['final_nll_bits_primary','final_nll_bits_independent']].mean().reset_index().to_csv(output/'independent_reader_per_seed.csv',index=False)
    atomic_json(output/'independent_reader_evidence.json',{'input_sha256':file_sha256(independent_path),'rows':len(reread),'same_transcripts_verified':True,'reacquired_executions':0})
    resource_tables(root)
    write_result_hashes(root)
    return {"state":"passed","case_rows":len(frame),"configuration_rows":len(systems),"primary_contrast":contrasts[0]}

def figures(root,config):
    if config.get('protocol_id')=='CGX-2.0':
        from .v2 import figures_v2
        return figures_v2(root,config)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .methods import display_name, SHARED_DIAGNOSER_CAPTION
    root=Path(root); results=root/"results"
    frame=pd.read_csv(results/"per_seed.csv"); systems=pd.read_csv(results/"per_configuration.csv")
    out=results/"figures"; out.mkdir(parents=True,exist_ok=True)
    def save(fig,name,caption=SHARED_DIAGNOSER_CAPTION):
        fig.text(.5,.01,caption,ha='center',va='bottom',fontsize=8,wrap=True)
        fig.tight_layout(rect=(0,.08,1,1)); fig.savefig(out/f"{name}.png",dpi=180); fig.savefig(out/f"{name}.pdf"); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,suite in zip(axes,["test_seen","test_connections"]):
        for method,g in frame[frame.suite==suite].groupby("method"):
            means=g.groupby("byte_fraction").mean(numeric_only=True)
            reference=method=='full_recording_unrestricted_reference'
            ax.plot(means.index,means.final_nll_bits,linestyle='--' if reference else '-',marker='o',label=display_name(method,compact=True)+(' (infeasible reference below full cap)' if reference else ''))
        ax.set(title=suite,xlabel="Byte cap / four full-rate executions",ylabel="Final NLL (bits)")
    axes[1].legend(fontsize=6); save(fig,"F1_budget_nll")
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex='col',sharey='row')
    for column,suite in enumerate(['test_seen','test_connections']):
        for method,g in frame[frame.suite==suite].groupby('method'):
            means=g.groupby('byte_fraction').mean(numeric_only=True)
            reference=method=='full_recording_unrestricted_reference'
            label=display_name(method,compact=True)+(' (unrestricted; infeasible below full cap)' if reference else '')
            style='--' if reference else '-'
            axes[0,column].plot(means.bytes_used/1e6,means.final_nll_bits,style,marker='o',label=label)
            axes[1,column].plot(means.bytes_used/1e6,means.correct_joint,style,marker='o',label=label)
        axes[0,column].set(title=suite,ylabel='Final NLL (bits)')
        axes[1,column].set(xlabel='Actual incremental wire bytes (MB)',ylabel='Joint mechanism + location accuracy',ylim=(0,1))
    axes[0,1].legend(fontsize=5);save(fig,'F1_actual_bytes_nll_accuracy')
    selected=systems[(systems.suite=="test_connections") & (systems.byte_fraction==.25)]
    wide=selected.groupby(["configuration_id","method"]).final_nll_bits.mean().unstack()
    costs=selected.groupby(['configuration_id','method']).bytes_used.mean().unstack()
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True);differences=wide.gnn_adaptive-wide.transformer_adaptive
    axes[0].bar(range(len(differences)),differences);axes[0].axhline(0,color='black',linewidth=.8)
    axes[1].bar(range(len(differences)),(costs.gnn_adaptive-costs.transformer_adaptive)/1e6);axes[1].axhline(0,color='black',linewidth=.8)
    comparison=display_name('gnn_adaptive',compact=True)+' −\n'+display_name('transformer_adaptive',compact=True)
    axes[0].set(ylabel=comparison+'\nNLL difference (bits)')
    axes[1].set(ylabel=comparison+'\nRecorded bytes difference (MB)',xlabel='Every held-out configuration')
    axes[1].set_xticks(range(len(differences)),[str(i)[:10] for i in differences.index],rotation=60,fontsize=6)
    save(fig,'F2_all_systems')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for method in ["gnn_adaptive","gnn_fixed_plan"]:
        group=frame[(frame.suite=="test_connections") & (frame.method==method)].groupby("byte_fraction").mean(numeric_only=True)
        axes[0].plot(group.index,group.final_nll_bits,label=display_name(method,compact=True)); axes[1].plot(group.index,group.executions_used,label=display_name(method,compact=True))
    axes[0].set(ylabel="Final NLL (bits)",xlabel="Byte cap fraction"); axes[1].set(ylabel="Actual extra executions",xlabel="Byte cap fraction"); axes[0].legend(); save(fig,"F3_feedback")
    fig,ax=plt.subplots(figsize=(10,4))
    primary=frame[frame.byte_fraction==.25].groupby(["suite","method"]).final_nll_bits.mean().unstack()
    primary.rename(columns=lambda m:display_name(m,compact=True)).plot.bar(ax=ax); ax.set_ylabel("Final NLL (bits)"); ax.legend(fontsize=6); save(fig,"F4_suites")
    if (results/"repair_summary.csv").exists():
        repair=pd.read_csv(results/"repair_summary.csv"); fig,axes=plt.subplots(1,2,figsize=(10,4))
        for method,g in repair.groupby("method"):
            axes[0].scatter(g.coverage,g.risk,label=display_name(method,compact=True)); axes[1].scatter(g.mean_total_deployed_executions,g.yield_fraction,label=display_name(method,compact=True))
        axes[0].set(xlabel="Declared coverage",ylabel="Failure of all-20-pass audit"); axes[1].set(xlabel="Total deployed executions",ylabel="Yield"); axes[0].legend(fontsize=7); save(fig,"F5_repair_audit")
    thresholds_path=root/'models/thresholds.json'
    if thresholds_path.exists():
        thresholds=read_json(thresholds_path);curves=[]
        for key,rows in thresholds.get('curves',{}).items():
            if key.startswith('beta:'):continue
            method,seed=key.rsplit(':',1)
            for row in rows:curves.append(dict(row,method=method,policy_seed=seed,split='validation'))
        if curves:
            validation=pd.DataFrame(curves);validation.to_csv(results/'validation_risk_coverage.csv',index=False)
            fig,axes=plt.subplots(1,2,figsize=(12,4),sharex=True,sharey=True)
            for (method,seed),g in validation.groupby(['method','policy_seed']):
                g=g.sort_values('threshold');axes[0].plot(g.coverage,g.risk,marker='o',alpha=.7,label=f'{display_name(method,compact=True)}:{seed}')
            axes[0].set(title='Validation: full preregistered threshold grid',xlabel='Declared coverage',ylabel='Failure of all-20-pass audit')
            if (results/'repair_summary.csv').exists():
                heldout=pd.read_csv(results/'repair_summary.csv')
                for method,g in heldout.groupby('method'):axes[1].scatter(g.coverage,g.risk,label=display_name(method,compact=True))
            axes[1].set(title='Held-out: frozen selected thresholds only',xlabel='Declared coverage')
            axes[0].legend(fontsize=5);axes[1].legend(fontsize=6);save(fig,'F5_validation_curve_heldout_points')
    models,collection=resource_tables(root)
    if models:
        resources=pd.DataFrame(models);fig,axes=plt.subplots(1,3,figsize=(14,4))
        def training_label(row):
            if row.kind=='policies':
                method,separator,seed=row.run.rpartition('_seed')
                if not separator:return display_name(row.run,compact=True)
                method='cost_aware_adaptation' if method=='gnn_myopic' else method
                return display_name(method,compact=True)+' selector\nseed '+seed
            if row.run=='final':return 'Shared GNN diagnoser'
            if row.run=='independent':return 'Independent Transformer reader'
            if row.run.startswith('crossfit_'):return 'GNN training reader\nfold '+row.run.removeprefix('crossfit_')
            return row.run
        names=[training_label(row) for row in resources.itertuples(index=False)]
        axes[0].bar(range(len(resources)),resources.parameters/1e6);axes[0].set_ylabel('Parameters (millions)')
        axes[1].bar(range(len(resources)),resources.peak_gpu_memory_bytes/1024**3);axes[1].set_ylabel('Measured peak GPU memory (GiB)')
        duration=resources.job_allocation_seconds.fillna(resources.trainer_elapsed_seconds)
        axes[2].bar(range(len(resources)),duration/3600);axes[2].set_ylabel('Allocated job hours (incl. validation)')
        for ax in axes:ax.set_xticks(range(len(resources)),names,rotation=90,fontsize=5)
        save(fig,'F6_training_resources')
    fig,ax=plt.subplots(figsize=(10,4)); frame.groupby("method")[["selection_seconds","inference_seconds"]].mean().rename(index=lambda m:display_name(m,compact=True)).plot.bar(ax=ax)
    ax.set_ylabel("Measured seconds per case (excludes warmup)"); save(fig,"F6_latency")
    independent_table=results/'independent_reader_per_seed.csv'
    if independent_table.exists():
        readers=pd.read_csv(independent_table);fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
        for axis,suite in zip(axes,['test_seen','test_connections']):
            for method,g in readers[readers.suite==suite].groupby('method'):
                mean=g.groupby('byte_fraction').mean(numeric_only=True)
                line=axis.plot(mean.index,mean.final_nll_bits_primary,label=display_name(method,compact=True)+' / main GNN')[0]
                axis.plot(mean.index,mean.final_nll_bits_independent,'--',color=line.get_color(),label=display_name(method,compact=True)+' / independent Transformer')
            axis.set(title=suite,xlabel='Byte cap fraction',ylabel='Final NLL on identical transcripts (bits)')
        axes[1].legend(fontsize=5);save(fig,'F7_independent_reader',caption='Solid: shared final GNN diagnoser. Dashed: separate independent Transformer reader, using the same frozen records.')
    write_result_hashes(root)
    return {"state":"passed","figures":[str(p) for p in sorted(out.iterdir())]}

def export(root,out):
    root=Path(root); out=Path(out)
    if not (root/"results/hashes.json").exists(): raise ValueError("No aggregated results to export")
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in ["results","reports","commands","jobs","manifests","software"]:
            for path in sorted((root/directory).rglob("*")):
                if path.is_file() and path!=out: archive.write(path,path.relative_to(root))
        for name in ["protocol_lock.json","environment.json","models/frozen.json","models/selection.json","models/thresholds.json"]:
            archive.write(root/name,name)
        for name in ['private/private_cases.jsonl','assets/mujoco_menagerie/franka_emika_panda/LICENSE']:
            if (root/name).is_file():archive.write(root/name,name)
        archive.writestr('RELEASE_CONTENTS.txt','This release contains actual result tables, figures, provenance, commands, job states, public and private case manifests, pinned protocol/dependencies and source.tar.gz.\nSelected checkpoint hashes and relative paths are in models/selection.json. Checkpoint tensors, full acquired raw transcripts and mesh assets remain separately in CGX_DATA_ROOT; they are not silently represented as present in this ZIP. Asset commit/license are recorded in environment/protocol provenance.\n')
    with zipfile.ZipFile(out) as archive:
        bad=archive.testzip()
        if bad is not None:raise ValueError(f'Release archive CRC failure: {bad}')
    result={'state':'passed','path':str(out),'sha256':file_sha256(out),'archive_crc_checked':True,'result_hashes_sha256':file_sha256(root/'results/hashes.json')}
    atomic_json(root/'release/complete.json',result)
    return result
