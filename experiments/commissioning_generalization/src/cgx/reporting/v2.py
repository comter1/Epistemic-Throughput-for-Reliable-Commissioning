"""Four main-paper figures and an evidence-backed V2 result report."""
from pathlib import Path
import pandas as pd
from cgx.provenance import read_json
from .methods import COMPACT_NAMES, DISPLAY_NAMES, METHOD_DESCRIPTIONS, SHARED_DIAGNOSER_CAPTION, display_name

LABELS = COMPACT_NAMES
REPORT_ORDER = ('gnn_fixed_plan', 'gnn_depth2', 'transformer_adaptive',
                'cost_aware_adaptation', 'full_recording_feasible', 'gnn_adaptive')


def figures_v2(root,config,output_dir=None):
    """Render saved results; explicit output_dir leaves original artifacts intact."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .results import write_result_hashes,resource_tables
    root=Path(root);results=root/'results'
    presentation_only=output_dir is not None
    destination=Path(output_dir) if presentation_only else results
    if presentation_only and destination.resolve()==results.resolve():
        raise ValueError('Presentation output must differ from the original results directory')
    out=destination/'figures';out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(results/'per_seed.csv');colors=dict(zip(LABELS,plt.rcParams['axes.prop_cycle'].by_key()['color']))
    def save(fig,name,legend=False):
        if legend:
            handles,labels=fig.axes[-1].get_legend_handles_labels()
            fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,fontsize=8,frameon=False)
        fig.text(.5,.015,SHARED_DIAGNOSER_CAPTION,ha='center',va='bottom',fontsize=9)
        fig.tight_layout(rect=(0,.21 if legend else .08,1,1))
        fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'));plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),sharey=True)
    for ax,suite in zip(axes,['test_seen','test_connections']):
        for method,group in frame[frame.suite==suite].groupby('method'):
            point=group.groupby('execution_cap').final_nll_bits.agg(['mean','std']).fillna(0)
            ax.errorbar(point.index,point['mean'],yerr=point['std'],marker='o',label=LABELS[method],color=colors[method],capsize=2)
        ax.set(title='Seen topology' if suite=='test_seen' else 'Unseen topology (primary)',xlabel='Maximum additional diagnostic executions B',xticks=[1,2,4],ylabel='Final log loss (bits)')
    save(fig,'F1_budget_performance_seen_unseen',legend=True)
    target=read_json(root/'models/thresholds.json')['cost_to_quality']['target_nll_bits']
    matched=[]
    for method,group in frame[frame.suite=='test_connections'].groupby('method'):
        points=group.groupby('execution_cap')[['final_nll_bits','executions_used','bytes_used']].mean().reset_index()
        feasible=points[points.final_nll_bits<=target].sort_values(['executions_used','bytes_used','execution_cap'])
        if feasible.empty:matched.append(dict(method=method,status='not_reached',target_nll_bits=target,execution_cap=None,executions_used=None,bytes_used=None))
        else:matched.append(dict(method=method,status='reached',target_nll_bits=target,**feasible.iloc[0].to_dict()))
    table=pd.DataFrame(matched);table['display_name']=table.method.map(display_name)
    table.to_csv(destination/'V2_matched_quality_resources.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,metric,divisor,label in zip(axes,['executions_used','bytes_used'],[1,1e6],['Actual extra executions','Actual recorded bytes (MB)']):
        for i,row in enumerate(matched):
            if row['status']=='reached':ax.bar(i,row[metric]/divisor,color=colors[row['method']])
            else:ax.text(i,0,'Not reached',rotation=90,va='bottom',ha='center',fontsize=8)
        ax.set_xticks(range(len(matched)),[LABELS[r['method']] for r in matched],rotation=35,ha='right',fontsize=8);ax.set_ylabel(label)
    fig.suptitle(f'Unseen topology: validation-fixed target {target:.3f} bits; no extrapolation');save(fig,'F2_resources_to_matched_quality')
    primary=frame[frame.execution_cap==config['evaluation']['primary_execution_cap']]
    gap=primary[primary.suite.isin(['test_seen','test_connections'])].groupby(['method','policy_seed','suite'],dropna=False).final_nll_bits.mean().unstack('suite')
    gap['unseen_minus_seen_bits']=gap.test_connections-gap.test_seen
    gap_export=gap.copy();gap_export['display_name']=gap_export.index.get_level_values('method').map(display_name)
    gap_export.to_csv(destination/'V2_seen_unseen_gap.csv')
    means=gap.groupby(level='method').unseen_minus_seen_bits.agg(['mean','std']).fillna(0)
    fig,ax=plt.subplots(figsize=(12,6));ax.bar(range(len(means)),means['mean'],yerr=means['std'],color=[colors[m] for m in means.index],capsize=3)
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(range(len(means)),[LABELS[m] for m in means.index],rotation=30,ha='right')
    ax.set(ylabel='Unseen minus seen log loss (bits)',title='Transfer to unseen connections at B=4 (error bars: training-seed SD)');save(fig,'F3_seen_unseen_gap')
    repairs=pd.read_csv(results/'repair_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    for method,group in repairs.groupby('method'):
        axes[0].scatter(group.mean_total_deployed_executions,group.yield_fraction,label=LABELS[method],color=colors[method])
        axes[1].scatter(group.mean_total_deployed_bytes/1e6,group.yield_fraction,label=LABELS[method],color=colors[method])
    axes[0].set(xlabel='Total deployed executions, including public verification',ylabel='Declared completion and all 20 audit runs pass')
    axes[1].set(xlabel='Total deployed recorded bytes (MB)',ylabel='Declared completion and all 20 audit runs pass')
    save(fig,'F4_repair_audit_vs_total_cost',legend=True)
    if not presentation_only:resource_tables(root)
    stats=read_json(results/'paired_statistics.json');primary_stat=next(r for r in stats if r['primary'])
    summary=primary.groupby(['suite','method'])[['final_nll_bits','correct_joint','bytes_used','executions_used','abstained']].mean()
    summary['display_name']=summary.index.get_level_values('method').map(display_name)
    summary.to_csv(destination/'V2_primary_summary.csv')
    report,expanded_summary=_result_report(frame,config,primary_stat)
    expanded_summary['display_name']=expanded_summary.index.get_level_values('method').map(display_name)
    expanded_summary.to_csv(destination/'MAIN_METHOD_RESULTS.csv')
    report_path=destination/'MAIN_EXPERIMENT_V2_RESULTS.md';report_path.write_text(report)
    captions_path=destination/'FIGURE_CAPTIONS.md';captions_path.write_text(_figure_captions())
    if not presentation_only:write_result_hashes(root)
    return dict(state='passed',figures=[str(p) for p in sorted(out.iterdir())],main_report=str(report_path),
                captions=str(captions_path),output_dir=str(destination))


def _formatted(value, spec='.4f', divisor=1):
    """Missing display-only fields stay missing, including older test fixtures."""
    return 'N/A' if pd.isna(value) else format(value/divisor,spec)


def _result_report(frame,config,primary_stat):
    """Describe saved measurements; requested conclusions must match those values."""
    cap=config['evaluation']['primary_execution_cap']
    primary=frame[frame.execution_cap==cap]
    metrics=['final_nll_bits','correct_joint','bytes_used','executions_used',
             'calibrations_used','end_to_end_seconds','timeout','abstained']
    available=[metric for metric in metrics if metric in primary]
    summary=primary.groupby(['suite','method'])[available].mean().reindex(columns=metrics)
    seen=summary.loc['test_seen'];unseen=summary.loc['test_connections']
    fixed=unseen.loc['gnn_fixed_plan'];full=unseen.loc['full_recording_feasible']
    once_name=display_name('gnn_fixed_plan');gnn_name=display_name('gnn_adaptive')
    transformer_name=display_name('transformer_adaptive');full_name=display_name('full_recording_feasible')
    lines=['# MAIN EXPERIMENT V2 — executed results with clarified method names','',
        'This presentation uses the original saved results. Internal method identifiers, training seeds, '
        'checkpoints and protocol settings are unchanged. Original reports and figures retained in the '
        'source results directory use legacy display names.','',
        'All model selection, normalization and repair thresholds were frozen before test access. '
        'No test-time fitting was performed.',
        'Three training seeds; 32 source topologies; 8 seen, 16 structurally held-out, and 8 larger '
        'held-out topologies. B counts additional diagnostic executions.','',
        SHARED_DIAGNOSER_CAPTION+' After evidence collection, every main method gives its records '
        'to that same model. The main comparison therefore changes how evidence is selected and '
        'collected, not the final diagnostic model. Execution and computation limits still affect '
        'the measured outcome.','',
        '## What each method does','',
        '| Internal identifier | Full method name |','|---|---|']
    for method in REPORT_ORDER:lines.append(f'| `{method}` | {display_name(method)} |')
    lines.append('')
    for method in REPORT_ORDER:
        lines.extend([f'**{display_name(method)}.** {METHOD_DESCRIPTIONS[method]}',''])
    model=config['models']
    lines.extend([
        f"The shared diagnoser uses a GRU with hidden width {model['temporal_gru_hidden']} to read "
        f"each recorded interface time series. Those summaries are attached to the corresponding "
        f"components, and a {model['message_passing_layers']}-layer GNN with hidden width "
        f"{model['hidden_width']} passes information along the system connections. Its output "
        'assigns probabilities to healthy and every legal (component, fault type) target. '
        'The independent Relation-Transformer reader only re-reads frozen transcripts in a '
        'separate analysis; it is not the shared main diagnoser.','',
        f'## Main results on structurally held-out connections, B={cap}','',
        '| Evidence collection method | Final diagnoser | NLL ↓ (bits) | '
        'Joint fault+location accuracy ↑ | Recorded data (MB) | End-to-end time (s) |',
        '|---|---|---:|---:|---:|---:|'])
    for method in REPORT_ORDER:
        row=unseen.loc[method]
        lines.append(f'| {display_name(method,compact=True)} | Shared GNN | '
                     f'{row.final_nll_bits:.4f} | {row.correct_joint:.3%} | '
                     f'{row.bytes_used/1e6:.4f} | {_formatted(row.end_to_end_seconds)} |')
    lines.extend(['',
        'Recorded data means additional diagnostic wire bytes (decimal MB). End-to-end time '
        'is the saved diagnostic-acquisition wall time, including initialization, the common '
        'initial execution, warmup, selection and diagnosis; it ends before repair and public '
        'verification. These means retain timed-out cases. N/A means a field was not present '
        'in the supplied saved summary.','',
        f'**{once_name} is learned.** It learns from source systems and creates a new '
        'system-specific inspection plan from each test system structure, initial evidence '
        'and budget. It does not use one hand-written checklist for all systems.','',
        f'At B={cap}, {once_name} has seen/held-out NLL '
        f"{seen.loc['gnn_fixed_plan','final_nll_bits']:.4f}/{fixed.final_nll_bits:.4f} bits "
        f"and joint accuracy {seen.loc['gnn_fixed_plan','correct_joint']:.3%}/{fixed.correct_joint:.3%}."])
    if (abs(fixed.final_nll_bits-seen.loc['gnn_fixed_plan','final_nll_bits'])<.01
            and abs(fixed.correct_joint-seen.loc['gnn_fixed_plan','correct_joint'])<.01):
        lines.append('The observed performance is nearly unchanged between seen and held-out '
                     'connections in this V2 result, supporting transfer within this tested set.')
    lines.extend(['The scope is familiar component types in unseen type arrangements. '
                  'This is not an equivalence test or a claim about arbitrary machines.',''])
    if (unseen.final_nll_bits.idxmin()=='gnn_fixed_plan'
            and unseen.correct_joint.idxmax()=='gnn_fixed_plan'):
        lines.extend([f'At B={cap}, {once_name} has the lowest mean NLL and the highest '
                      'joint fault-type-and-location accuracy among the six current main methods '
                      'on the primary held-out connection suite.',''])
    relation='less than' if fixed.bytes_used<full.bytes_used else 'compared with'
    lines.extend([
        f'{once_name} records {fixed.bytes_used/1e6:.4f} MB per held-out case, {relation} '
        f'{full.bytes_used/1e6:.4f} MB for {full_name}. Both use the same learned final diagnoser.','',
        f'The corresponding mean calibration counts are {_formatted(fixed.calibrations_used)} '
        f'for {once_name} and {_formatted(full.calibrations_used)} for {full_name}. '
        'This compares complete evidence-acquisition plans, including calibration; it does '
        'not isolate the effect of recording fewer bytes.','',
        '## What changes as the execution budget increases','',
        'The following NLL values are saved means on the primary held-out connection suite. '
        'Lower is better. Each budget is an execution allowance; a method can stop earlier.',''])
    budgets=config['resources']['main_execution_budgets']
    lines.append('| Full method name | '+' | '.join(f'B={b}' for b in budgets)+' |')
    lines.append('|---|'+'---:|'*len(budgets))
    curves=frame[frame.suite=='test_connections'].groupby(['method','execution_cap']).final_nll_bits.mean()
    for method in ('gnn_fixed_plan','gnn_adaptive','transformer_adaptive','gnn_depth2','cost_aware_adaptation'):
        lines.append('| '+display_name(method)+' | '+
                     ' | '.join(f'{curves.loc[(method,b)]:.4f}' for b in budgets)+' |')
    worsening=[m for m in ('gnn_adaptive','transformer_adaptive') if any(
        curves.loc[(m,right)]>curves.loc[(m,left)] for left,right in zip(budgets,budgets[1:]))]
    if worsening:
        lines.extend(['',' and '.join(display_name(m) for m in worsening)+
            ' do not improve monotonically as B increases in these measurements. More allowed '
            'inspections therefore do not guarantee a better final prediction for these '
            'implemented methods. This is an observed result, not a general theorem.'])
    better_budgets=[b for b in budgets if b>min(budgets) and
                   curves.loc[('transformer_adaptive',b)]<curves.loc[('gnn_adaptive',b)]]
    if better_budgets:
        lines.extend(['',f"At B={', '.join(str(b) for b in better_budgets)}, {transformer_name} "
            f'has lower NLL than {gnn_name}. This comparison changes the model selecting '
            'inspections; both still use the same final GNN diagnoser.'])
    lines.extend(['',
        f'The preregistered paired contrast ({gnn_name} minus {transformer_name} log loss) '
        f"is {primary_stat['difference_nll_bits']:.4f} bits; the 95% configuration/case bootstrap "
        f"interval is [{primary_stat['ci_low']:.4f}, {primary_stat['ci_high']:.4f}]. "
        'Positive values mean the GNN inspection selector has higher loss.','',
        '## Runtime and interpretation','',
        f'| Full method name | Timeout rate at B={cap} |','|---|---:|'])
    for method in REPORT_ORDER:
        lines.append(f'| {display_name(method)} | {_formatted(unseen.loc[method,"timeout"],".3%")} |')
    lines.extend(['',
        'The saved evaluation ran diagnosis and inspection selection on CPU, with '
        f"a {config['resources']['selection_timeout_seconds_per_decision']:g}-second selection "
        f"limit and {config['resources']['diagnosis_timeout_seconds_per_call']:g}-second diagnosis "
        'limit per call. Timeouts and abstentions remain in every denominator. Results '
        'describe these complete implementations under the declared limits; they do not '
        'separate selection quality from computation failures.','',
        'These are learned-predictor log scores, not exact mutual information. Training-seed '
        'variation is separate from configuration/case uncertainty. The independent reader '
        'checks the same acquired transcripts without collecting new evidence. Twenty '
        'successful hidden audit runs define a protocol outcome, not a risk certificate.','',
        'The four main figures are in `figures/`; `FIGURE_CAPTIONS.md` provides captions '
        'that identify the shared final GNN diagnoser. Original per-case data, checkpoint '
        'hashes and the sealed protocol remain in the source experiment directory.'])
    return '\n'.join(lines)+'\n',summary


def _figure_captions():
    captions={
        'F1_budget_performance_seen_unseen':
            'Final diagnostic log loss versus the maximum number of additional executions. '
            'Error bars are training-seed standard deviations. Full Recording has only a '
            'registered B=4 reference point. Timed-out cases remain included.',
        'F2_resources_to_matched_quality':
            'Actual resources at a registered point meeting the validation-fixed quality '
            'target. The original selection rule and saved measurements are unchanged. '
            'Unreached targets remain marked; no unmeasured budget is inferred for Full Recording.',
        'F3_seen_unseen_gap':
            'Difference in diagnostic log loss between held-out and seen connections at B=4. '
            'Positive values mean higher loss on held-out connections. Error bars show '
            'training-seed standard deviations; model parameters are frozen.',
        'F4_repair_audit_vs_total_cost':
            'Declared completion followed by all-20-run audit success, plotted against deployed '
            'execution and recording costs including public verification. Points retain separate '
            'test suites and training seeds. Hidden audit costs are separate. Audit success '
            'is a protocol outcome, not a certified deployment-risk bound.',
    }
    lines=['# Main figure captions','','Compact labels describe evidence collection. Full method '
           'names are listed in `MAIN_EXPERIMENT_V2_RESULTS.md`.','']
    for name,caption in captions.items():
        lines.extend([f'## {name}','',SHARED_DIAGNOSER_CAPTION+' '+caption,'',
                      f'![{name}](figures/{name}.png)',''])
    return '\n'.join(lines)
