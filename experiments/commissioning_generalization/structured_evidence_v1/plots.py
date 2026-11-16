"""Eight required scientific figure families with standalone PDF/PNG exports."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cgx.provenance import atomic_json,file_sha256
from .reporting import METHODS

ORDER=['gnn_fixed_plan','gnn_adaptive','gnn_depth2','transformer_adaptive','cost_aware_adaptation']
COLORS=dict(zip(ORDER,plt.get_cmap('tab10').colors[:5]))
READERS={'original_as_deployed':('#777777',':'),'original_GNN_reread':('#d95f02','--'),
         'R-C':('#7570b3','-.'),'structured':('#1b9e77','-')}
SUITES=['test_seen','test_connections','test_size']
SUITE_LABELS={'test_seen':'Seen topology','test_connections':'Unseen connections (primary)','test_size':'Larger unseen topology'}


def _save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)


def _budget_figures(run,summary):
    out=run/'figures'
    def count(rows,column):
        values=rows[column].dropna().astype(int).unique()
        if len(values)!=1:raise ValueError(f'Budget figure has inconsistent {column} denominators')
        return int(values[0])
    for i,metric,title,scale in [(1,'accuracy','Exact fault type + location / healthy accuracy (%)',100),
        (2,'healthy_false_alarm_rate','Healthy false-alarm rate (%)',100),
        (3,'fault_miss_rate','Fault → healthy miss rate (%)',100),
        (4,'nll_bits','True-label NLL (bits)',1)]:
        fig,axes=plt.subplots(1,3,figsize=(17,4.6),sharey=False)
        for ax,suite in zip(axes,SUITES):
            part=summary[summary.suite==suite]
            learned=part[part.method.isin(ORDER)];full=part[part.method=='full_recording_feasible']
            seed_count=count(learned,'policy_seeds')
            if count(full,'rows')!=count(full,'physical_cases'):
                raise ValueError('Full Recording figure expects one run per physical case')
            denominator,label={'healthy_false_alarm_rate':('healthy_rows','Healthy'),
                'fault_miss_rate':('fault_rows','Faulty')}.get(metric,('rows','All-case'))
            panel_title=(f'{SUITE_LABELS[suite]}\nLearned: {seed_count} seeds; Full Recording: once per case\n'
                f'{label} N (learned / Full): {count(learned,denominator):,} / {count(full,denominator):,}')
            for method in ORDER:
                for version,linestyle in [('original','--'),('structured','-')]:
                    rows=part[(part.method==method)&(part.version==version)].sort_values('execution_cap')
                    ax.plot(rows.execution_cap,scale*rows[metric],marker='o' if version=='structured' else 'x',
                        color=COLORS[method],linestyle=linestyle,label=METHODS[method]+(' · original' if version=='original' else ' · structured'))
            for version,marker in [('original','x'),('structured','o')]:
                rows=part[(part.method=='full_recording_feasible')&(part.version==version)]
                ax.scatter(rows.execution_cap,scale*rows[metric],color='black',marker=marker,s=50,
                    label='Full Recording · '+version,zorder=5)
            ax.set(xlabel='Additional-execution budget B',xticks=[1,2,4]);ax.set_title(panel_title,fontsize=9);ax.grid(alpha=.2)
        axes[0].set_ylabel(title)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,-.17),fontsize=8)
        fig.suptitle('Original policies/readers vs retrained policies with Structured Evidence Diagnoser')
        fig.tight_layout();_save(fig,out/f'{i:02d}_{metric}_vs_budget')


def _plan_figure(run,summary):
    replay=pd.read_csv(run/'reports/reader_replay_summary.csv')
    replay=replay[(replay.cohort=='main')&(replay.suite=='test_connections')&(replay.reader=='structured')]
    stages=['original_deployed','structured_on_old_histories','structured_retrained']
    labels=['Original reader\n+ original policy','Structured reader\n+ old actions (untimed)',
        'Structured reader\n+ retrained policy (live)']
    plot_rows=[];fig,axes=plt.subplots(1,3,figsize=(18,5),sharey=False)
    for ax,cap in zip(axes,(1,2,4)):
        part=summary[(summary.suite=='test_connections')&(summary.execution_cap==cap)]
        for method in ('gnn_fixed_plan','gnn_adaptive'):
            selections=[part[(part.method==method)&(part.version=='original')],
                replay[(replay.method==method)&(replay.execution_cap==cap)],
                part[(part.method==method)&(part.version=='structured')]]
            if any(len(rows)!=1 for rows in selections):raise ValueError('Plan/Replan three-arm figure cell missing or duplicated')
            values=[rows.iloc[0] for rows in selections]
            if any(int(r.rows)!=4800 or int(r.physical_cases)!=1600 for r in values):
                raise ValueError('Plan/Replan three-arm figure changed the primary-suite denominator')
            for stage,r in zip(stages,values):
                plot_rows.append(dict(stage=stage,suite='test_connections',execution_cap=cap,method=method,
                    accuracy=r.accuracy,correct_count=int(r.correct_count),rows=int(r.rows),physical_cases=int(r.physical_cases)))
            yy=[r.accuracy*100 for r in values]
            ax.plot([0,1,2],yy,marker='o',color=COLORS[method],label=METHODS[method])
            for x,y,r in zip([0,1,2],yy,values):
                below=method=='gnn_fixed_plan'
                ax.annotate(f'{y:.2f}%\n{int(r.correct_count)}/{int(r.rows)}',(x,y),
                    xytext=(3,-6 if below else 6),textcoords='offset points',va='top' if below else 'bottom',fontsize=8)
        ax.margins(y=.12)
        ax.set(xticks=[0,1,2],xticklabels=labels,title=f'Primary suite · B={cap}',ylabel='Accuracy (%)',xlim=(-.15,2.4));ax.grid(alpha=.2)
        ax.tick_params(axis='x',labelsize=8)
    pd.DataFrame(plot_rows).to_csv(run/'figures/plan_once_vs_replan_plot_data.csv',index=False)
    axes[0].legend(fontsize=8)
    fig.suptitle('Plan-Once versus Replan: fixed-history reader replacement, then policy retraining\nThe final step includes changed acquisition and live deadline effects',fontsize=11)
    fig.tight_layout()
    _save(fig,run/'figures/05_plan_once_vs_replan')


def _healthy_figures(run):
    pairs=pd.read_csv(run/'reports/continuation_pairs.csv')
    candidates=pairs[(pairs.is_healthy)&(pairs.reader=='original_as_deployed')].copy()
    candidates['healthy_probability_drop']=candidates.first_p_healthy-candidates.terminal_p_healthy
    # Explicit illustrative selection, never used to estimate an aggregate.
    selected=candidates.sort_values(['healthy_probability_drop','source_trajectory'],ascending=[False,True]).groupby(
        ['cohort','method'],sort=True).head(2)
    selected.to_csv(run/'figures/healthy_example_selection.csv',index=False)
    frame=pd.read_parquet(run/'replay/per_case_predictions.parquet')
    frame=frame[frame.source_trajectory.isin(selected.source_trajectory)]
    frame.drop(columns=['full_log_probabilities','target_keys']).to_csv(run/'figures/healthy_trajectory_plot_data.csv',index=False)
    n=len(selected);fig,axes=plt.subplots((n+1)//2,2,figsize=(14,3.4*((n+1)//2)),squeeze=False)
    for ax,example in zip(axes.flat,selected.itertuples()):
        part=frame[frame.source_trajectory==example.source_trajectory]
        for reader,(color,style) in READERS.items():
            rows=part[part.reader==reader].sort_values('prefix_ordinal')
            ax.plot(rows.prefix_ordinal,rows.p_healthy,color=color,linestyle=style,marker='.',label=reader)
        ax.set(title=f'{example.cohort} · {METHODS[example.method]}\ncase {example.case_id[:14]} · seed {example.policy_seed}',
            xlabel='Acquisition prefix (execution or calibration)',ylabel='P(healthy)',ylim=(-.02,1.02));ax.grid(alpha=.2)
    for ax in list(axes.flat)[n:]:ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Illustrative healthy continuations: two largest original P(healthy) drops per cohort/method',fontsize=11)
    fig.tight_layout();_save(fig,run/'figures/06_healthy_trajectories')


def _invariance_figures(run):
    rows=pd.read_csv(run/'invariance/transformations.csv');dups=rows[rows.test=='A_duplicate_storage']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for _,part in dups.groupby('case_id'):
        part=part.sort_values('copies');axes[0].plot(part.copies,part.p_healthy,color='#1b9e77',alpha=.18,linewidth=.7)
    axes[0].set(xlabel='Copies of the same stored internal record (ID preserved)',ylabel='P(healthy)',ylim=(-.02,1.02),title=f'{dups.case_id.nunique()} real histories · all families')
    maxima=dups.groupby('copies').max_probability_error.max();axes[1].plot(maxima.index,maxima.values,marker='o',color='#1b9e77')
    axes[1].axhline(1e-6,color='gray',linestyle='--',label='Declared numerical tolerance')
    axes[1].set(xlabel='Duplicate copies',ylabel='Maximum absolute posterior probability change',
        title='Entire target-vector invariance',ylim=(-.05e-6,1.2e-6))
    axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();_save(fig,run/'figures/07_duplicate_storage_invariance')
    shared=pd.read_csv(run/'invariance/shared_clock.csv')
    candidate=shared.groupby('case_id').available_unique_message_contrasts.max().sort_values(ascending=False)
    case=sorted(candidate[candidate==candidate.max()].index)[0]
    part=shared[shared.case_id==case].sort_values('n');part.to_csv(run/'figures/shared_clock_example_plot_data.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].plot(part.n,np.sqrt(part.correct_marginal_average_variance)*1000,marker='o',label='Correct common clock latent')
    axes[0].plot(part.n,np.sqrt(part.incorrect_independent_clock_average_variance)*1000,marker='x',linestyle='--',label='Incorrect independent-clock debug model')
    axes[0].axhline(np.sqrt(part.shared_clock_variance.iloc[0])*1000,color='gray',linestyle=':',label='Learned common-clock prior floor')
    axes[0].set(xlabel='Distinct actually acquired packet contrasts',ylabel='Marginal transit-mean standard deviation (ms)',
        title='Uncalibrated healthy timing control');axes[0].set_xscale('log',base=2);axes[0].legend(fontsize=8)
    axes[1].plot(part.n,np.sqrt(part.posterior_relative_clock_variance)*1000,marker='o',color='#1b9e77')
    axes[1].set(xlabel='Distinct actually acquired packet contrasts',ylabel='Posterior SD(Ca − Cz) (ms)',
        title='Actual structured posterior; fault marginalized');axes[1].set_xscale('log',base=2)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle(f"Acquired ports {part.port_a.iloc[0]} / {part.port_z.iloc[0]} · case {case[:14]}\nCalibration removed for this control; identifiable conditional uncertainty may legitimately contract",fontsize=10)
    fig.tight_layout();_save(fig,run/'figures/08_shared_clock_uncertainty')


def render(run_dir):
    run=Path(run_dir);out=run/'figures';out.mkdir(parents=True,exist_ok=True)
    summary=pd.read_csv(run/'reports/main_summary.csv')
    _budget_figures(run,summary);_plan_figure(run,summary);_healthy_figures(run);_invariance_figures(run)
    files=sorted(out.glob('[0-9][0-9]_*.pdf'))
    if len(files)!=8:raise ValueError('Eight required figure families missing')
    result=dict(state='passed',required_figure_families=8,files={p.name:file_sha256(p) for p in files},
        png_files={p.name:file_sha256(p) for p in sorted(out.glob('[0-9][0-9]_*.png'))})
    atomic_json(out/'complete.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True)
    print(render(p.parse_args().run_dir),flush=True)
