"""Standalone research figures from completed artifacts; performs no inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ledger import TimedStage

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})

METHODS=('gnn_depth2','gnn_adaptive','transformer_adaptive','gnn_fixed_plan')
NAMES={'gnn_depth2':'GNN 2-Step Planner','gnn_adaptive':'GNN Replan-Each-Round',
       'transformer_adaptive':'Transformer Replan-Each-Round','gnn_fixed_plan':'GNN Plan-Once'}
SHORT={'gnn_depth2':'GNN\n2-Step Planner','gnn_adaptive':'GNN\nReplan-Each-\nRound',
       'transformer_adaptive':'Transformer\nReplan-Each-\nRound','gnn_fixed_plan':'GNN\nPlan-Once'}
COLORS=dict(zip(METHODS,('#d95f02','#1b9e77','#7570b3','#e7298a')))
CASE=['configuration_id','case_id']
TRAJECTORY=CASE+['method_key','policy_seed']


def save(fig, run, stem):
    directory=run/'figures';directory.mkdir(exist_ok=True)
    for extension in ('png','pdf'):
        fig.savefig(directory/(stem+'.'+extension),dpi=180,bbox_inches='tight')
    plt.close(fig)


def equal_config(frame, metric):
    """Average all seeds within physical case, then cases within configuration."""
    return frame.groupby(CASE)[metric].mean().groupby('configuration_id').mean().mean()


def curve_table(frame):
    healthy=frame[frame.is_healthy].copy()
    maximum=float(healthy.bytes_used.max())
    axes={'decision_index':np.arange(7),'executions_used':np.arange(5),'bytes_used':np.linspace(0,maximum,21)}
    rows=[]
    for identity,part in healthy.groupby(TRAJECTORY+['diagnostic_reader'],sort=True):
        part=part.sort_values('decision_index')
        identity=dict(zip(TRAJECTORY+['diagnostic_reader'],identity))
        for axis,points in axes.items():
            values=part[axis].to_numpy()
            for point in points:
                # Actual observations only. A terminated trajectory contributes its
                # last actual prediction, rather than disappearing from the cohort.
                index=np.searchsorted(values,point,side='right')-1
                current=part.iloc[max(0,index)]
                rows.append(dict(identity,axis=axis,point=float(point),false_alarm=float(current.false_alarm),
                    p_healthy=float(current.p_healthy),continuing=float(values[-1]>point),
                    actual_prefix_index=int(current.decision_index)))
    dense=pd.DataFrame(rows)
    groups=['method_key','diagnostic_reader','axis','point']
    cases=dense.groupby(groups+CASE)[['false_alarm','p_healthy']].mean().reset_index()
    configs=cases.groupby(groups+['configuration_id'])[['false_alarm','p_healthy']].mean().reset_index()
    summary=configs.groupby(groups)[['false_alarm','p_healthy']].mean().reset_index()
    counts=dense.groupby(groups).agg(continuing_seed_trajectories=('continuing','sum'),
        fixed_seed_trajectory_denominator=('continuing','size'),physical_cases=('case_id','nunique')).reset_index()
    return summary.merge(counts,on=groups,validate='one_to_one')


def prefixes(run):
    path=run/'prefix_predictions.parquet'
    if not path.exists():return
    frame=pd.read_parquet(path)
    curves=curve_table(frame);curves.to_csv(run/'prefix_fixed_denominator_curves.csv',index=False)
    fig,axes=plt.subplots(3,3,figsize=(14,10),gridspec_kw={'height_ratios':[2,2,1.25]},sharex='col',sharey='row')
    labels={'decision_index':'Acquired-observation decision index','executions_used':'Additional robot executions',
            'bytes_used':'Additional acquired bytes (MiB)'}
    main=curves[curves.diagnostic_reader=='main_gnn']
    for column,axis in enumerate(labels):
        for method in METHODS:
            rows=main[(main.axis==axis)&(main.method_key==method)].sort_values('point')
            x=rows.point.to_numpy()/(2**20 if axis=='bytes_used' else 1)
            for row,metric,scale in ((0,'false_alarm',100),(1,'p_healthy',1),(2,'continuing_seed_trajectories',1)):
                axes[row,column].plot(x,rows[metric]*scale,color=COLORS[method],label=NAMES[method],linewidth=1.8)
                axes[row,column].grid(alpha=.18)
        axes[2,column].set_xlabel(labels[axis])
    axes[0,0].set_ylabel('Healthy false alarm (%)')
    axes[1,0].set_ylabel('Mean q(healthy)')
    axes[2,0].set_ylabel('Still acquiring\n(seed trajectories)')
    axes[0,0].set_ylim(bottom=0);axes[1,0].set_ylim(0,1)
    handles,labels_=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels_,loc='upper center',ncol=2,bbox_to_anchor=(.5,.965),frameon=False)
    fig.suptitle('Same original records, progressively reread by the frozen main GNN Diagnoser',y=1.005)
    fig.text(.5,.005,'400 healthy cases per method, 3 seeds; seed → case → equal configuration averages. '
        'Stopped trajectories retain their last actual prediction.\nAt equal execution counts, all already acquired calibration observations are included. '
        'No observations are generated or extrapolated.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.90));save(fig,run,'01_prefix_trajectories')


def fixed_transcripts(run):
    path=run/'fixed_transcript_reread.csv'
    if not path.exists():return
    data=pd.read_csv(path)
    variants=[]
    for reader,label in [('main_gnn','Main GNN, full reread'),('independent_transformer','Independent Transformer, full reread')]:
        part=data[data.diagnostic_reader==reader].copy()
        part['healthy_false_alarm']=part.reread_false_alarm.astype(float)
        part['fault_miss']=part.reread_missed_fault.astype(float)
        part['joint_correct']=(part.reread_predicted_target_key==part.true_target_key).astype(float)
        variants.append((label,part))
    original=data[data.diagnostic_reader=='main_gnn'].copy()
    original['healthy_false_alarm']=original.original_false_alarm.astype(float)
    original['fault_miss']=original.original_missed_fault.astype(float)
    original['joint_correct']=(original.original_predicted_target_key==original.true_target_key).astype(float)
    original['nll_bits']=original.original_nll_bits
    variants.insert(0,('Original GNN return (2 s limits)',original))
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    metrics=[('healthy_false_alarm','healthy','Healthy false alarm (%)',100),
             ('fault_miss','faulty','True-fault missed detection (%)',100),
             ('joint_correct','faulty','Correct fault type and location (%)',100),
             ('nll_bits','faulty','True-fault log-loss (bits)',1)]
    rows=[];positions=np.arange(len(METHODS));width=.24
    for ax,(metric,group,label,scale) in zip(axes.flat,metrics):
        for index,(name,part) in enumerate(variants):
            selected=part[(part.true_target_key=='healthy') if group=='healthy' else (part.true_target_key!='healthy')]
            values=[]
            for method in METHODS:
                source=selected[selected.method_key==method]
                value=float(equal_config(source,metric));values.append(value*scale)
                rows.append(dict(variant=name,metric=metric,subgroup=group,method_key=method,value=value,
                                 physical_cases=source.case_id.nunique(),seed_trajectory_rows=len(source)))
            ax.bar(positions+(index-1)*width,values,width,label=name,color=('#aaaaaa','#377eb8','#984ea3')[index])
        ax.set_xticks(positions,[SHORT[m] for m in METHODS]);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.98))
    fig.suptitle('Same acquired transcript: original returned prediction versus complete frozen reread',y=1.015)
    fig.text(.5,.005,'400 healthy and 96 fault cases per method; all 3 seeds and all original timeouts retained. '
             'Bars use equal configuration weights after averaging seeds within case.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.94));save(fig,run,'02_fixed_transcript_readers')
    pd.DataFrame(rows).to_csv(run/'replay_fixed_summary.csv',index=False)


def stories(run):
    path=run/'six_case_stories.json'
    if not path.exists():return
    data=json.loads(path.read_text());fig,axes=plt.subplots(3,2,figsize=(13,11),sharey=True)
    categories={'healthy_to_fault':'Healthy → fault at terminal','always_healthy':'Always healthy','fault_then_recovered':'Fault → healthy recovery'}
    for ax,story in zip(axes.flat,data):
        identity=story['identity'];rows=pd.DataFrame(story['prefixes'])
        ax.plot(rows.decision_index,rows.p_healthy,'o-',color='#1b9e77',label='q(healthy)')
        ax.plot(rows.decision_index,rows.max_fault_probability,'s-',color='#d95f02',label='Largest fault probability')
        for _,r in rows.iterrows():
            if 'calibration' in r.action_kind:
                ax.axvspan(r.decision_index-.12,r.decision_index+.12,color='#e6ab02',alpha=.22)
        ax.set_xticks(rows.decision_index,[f'{int(r.decision_index)}\nE{int(r.executions_used)} C{int(r.calibrations_used)}'
                                          for _,r in rows.iterrows()])
        ax.set_ylim(0,1);ax.grid(alpha=.2)
        ax.set_title(categories[story['category']]+'\n'+identity['case_id'][:12]+' · '+NAMES[identity['method_key']]+f" · seed {identity['policy_seed']}",fontsize=10)
        last=rows.iloc[-1]
        ax.text(.02,.03,'Final argmax: '+last.predicted_target_key+'\nLargest fault: '+last.max_fault_target,
                transform=ax.transAxes,fontsize=8,color='#444444',bbox={'facecolor':'white','alpha':.75,'edgecolor':'none'})
        ax.set_xlabel('Decision index; cumulative executions (E) / calibrations (C)')
        ax.set_ylabel('Probability')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,.985))
    fig.suptitle('Six original trajectory stories, main GNN Diagnoser reread; all truly healthy',y=1.01)
    physical_cases=len({story['identity']['case_id'] for story in data})
    fig.text(.5,.005,f'Six trajectories from {physical_cases} physical cases: hash order with global case deduplication; category order fixed as shown; ties use method then seed. '
             'Yellow bands mark acquired calibration.\nFalse alarm means largest target probability is a fault, not q(healthy) < 0.5. '
             'Case stories are descriptive, not independent statistical samples.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95));save(fig,run,'03_six_case_stories')


def interventions(run):
    path=run/'intervention_outcomes.parquet'
    if not path.exists():return
    data=pd.read_parquet(path)
    if set(data.arm)!=set('ABCD'):return
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    metrics=[('healthy_false_alarm','healthy','Healthy false alarm (%)',100),('fault_miss','faulty','True-fault missed detection (%)',100),
             ('final_nll_bits','faulty','True-fault log-loss (bits)',1),('executions_used','all','Additional robot executions',1),
             ('calibrations_used','all','Calibrations acquired',1),('bytes_used','all','Additional acquired bytes (MiB)',1/2**20)]
    for ax,(metric,group,label,scale) in zip(axes.flat,metrics):
        for method in METHODS[:2]:
            values=[]
            for arm in 'ABCD':
                part=data[(data.method_key==method)&(data.arm==arm)]
                if group!='all':part=part[part.is_healthy if group=='healthy' else ~part.is_healthy]
                values.append(equal_config(part,metric)*scale)
            ax.plot(list('ABCD'),values,'o-',label=NAMES[method],color=COLORS[method])
        ax.set_ylabel(label);ax.grid(alpha=.2);ax.set_ylim(bottom=0)
        if metric=='executions_used':ax.set_ylim(0,4.1)
        if metric=='calibrations_used':ax.set_ylim(0,2.1)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,.99))
    fig.suptitle('Same frozen selector: evidence input × calibration allocation (discovery)',y=1.025)
    fig.text(.5,.005,'A: current evidence, free calibration; B: initial evidence, free calibration; '
        'C: current evidence, prepaid calibration; D: initial evidence, prepaid calibration.\n'
        'Every final GNN diagnosis uses all acquired records. Identical resource caps; 32 healthy + 32 fault cases, 3 seeds; equal configuration weights.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.94));save(fig,run,'04_discovery_four_groups')


def effects(run):
    path=run/'paired_effects.csv'
    if not path.exists():return
    data=pd.read_csv(path)
    selected=data[(data.metric.isin(['healthy_false_alarm','fault_miss'])) & (~data.contrast.str.startswith('('))].copy()
    # The other two runtime contrasts repeat an already displayed same-record
    # comparison or its algebraic difference. All contrasts remain in the CSV.
    selected=selected[(selected.phase!='runtime_discovery')|(selected.contrast=='full_unlimited_A-original')]
    if selected.empty:return
    phases={'same_transcript':'Same transcript','same_record_calibration':'Calibration, same records',
            'discovery':'Discovery','confirmation':'Fresh confirmation',
            'stopping_existing_trajectories':'Existing transcript, fixed STOP','runtime_discovery':'Original limits versus full run'}
    contrasts={'independent_transformer-original':'Transformer reader − original',
               'main_gnn-original':'GNN reread − original','remove-retain':'Remove calibration − retain',
               'supplement_to_two_pairs-retain':'Supplement calibration − retain','E-A':'Freeze q − A',
               'S-A':'Stop after first robot execution − A',
               'STOP_after_first_execution-final':'First-execution STOP − actual terminal',
               'full_unlimited_A-original':'Unlimited A − original return'}
    selected['label']=selected.phase.map(lambda x:phases.get(x,x))+' | '+selected.method_key.map(NAMES)+' | '+selected.contrast.map(lambda x:contrasts.get(x,x))
    order=selected[['phase','label']].drop_duplicates().sort_values(['phase','label']).label.tolist()
    fig,axes=plt.subplots(1,2,figsize=(15,max(6,.32*len(order))),sharey=True)
    for ax,metric,title in zip(axes,['healthy_false_alarm','fault_miss'],['Healthy false alarm change (pp)','True-fault missed detection change (pp)']):
        for y,label in enumerate(order):
            rows=selected[(selected.label==label)&(selected.metric==metric)]
            if rows.empty:continue
            row=rows.iloc[0];point=row.difference*100
            ax.errorbar(point,y,xerr=[[max(0,point-row.paired_ci_low*100)],[max(0,row.paired_ci_high*100-point)]],
                        fmt='o',color=COLORS.get(row.method_key,'#333333'),capsize=3)
        ax.axvline(0,color='black',linewidth=.8);ax.set_xlabel(title);ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(np.arange(len(order)),order,fontsize=9);axes[0].invert_yaxis()
    fig.suptitle('Paired interventions and same-transcript frozen rereads')
    fig.text(.5,.005,'Intervention minus paired baseline. 95% hierarchical intervals resample configurations then physical cases; '
             'all seeds stay together. Negative means fewer errors.\nDiscovery and fresh-case confirmation are separate; '
             'reader changes and route changes are not additive explanations.\nA: current evidence/free calibration; B: initial evidence/free calibration; '
             'C: current evidence/prepaid calibration; D: initial evidence/prepaid calibration.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.97));save(fig,run,'05_paired_effects')


def fresh_confirmation(run):
    path=run/'fresh_intervention_outcomes.parquet'
    if not path.exists():return
    data=pd.read_parquet(path)
    if not {'A','S'}<=set(data.arm):return
    stats_path=run/'paired_effects.csv'
    stats=pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
    fig,axes=plt.subplots(2,3,figsize=(14,8.5))
    metrics=[('healthy_false_alarm','healthy','Healthy false alarm (%)',100),
             ('fault_miss','faulty','True-fault missed detection (%)',100),
             ('correct_joint','faulty','Correct fault type and location (%)',100),
             ('final_nll_bits','faulty','True-fault log-loss (bits)',1),
             ('executions_used','all','Additional robot executions',1),
             ('bytes_used','all','Additional acquired bytes (MiB)',1/2**20)]
    positions=np.arange(2);width=.33;intervals_present=False
    for ax,(metric,group,label,scale) in zip(axes.flat,metrics):
        for offset,arm in enumerate(('A','S')):
            values=[];lower=[];upper=[]
            for method in METHODS[:2]:
                part=data[(data.method_key==method)&(data.arm==arm)]
                if group!='all':part=part[part.is_healthy if group=='healthy' else ~part.is_healthy]
                value=equal_config(part,metric)*scale;values.append(value)
                matching=stats[(stats.phase=='confirmation')&(stats.method_key==method)&(stats.contrast=='S-A')&
                    (stats.metric==metric)&(stats.subgroup==group)] if not stats.empty else pd.DataFrame()
                if len(matching)==1:
                    row=matching.iloc[0];stem='baseline' if arm=='A' else 'intervention'
                    lower.append(max(0,value-row[stem+'_ci_low']*scale));upper.append(max(0,row[stem+'_ci_high']*scale-value))
                    intervals_present=True
                else:lower.append(0);upper.append(0)
            ax.bar(positions+(offset-.5)*width,values,width,yerr=[lower,upper],capsize=3,
                label='A: full frozen policy' if arm=='A' else 'S: STOP after first additional execution',
                color=('#377eb8','#e41a1c')[offset])
        ax.set_xticks(positions,[SHORT[m] for m in METHODS[:2]]);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,.985))
    fig.suptitle('New-case confirmation: a prespecified stopping rule on the identical realized prefix',y=1.02)
    ci='95% hierarchical intervals; ' if intervals_present else ''
    fig.text(.5,.005,ci+'64 new healthy + 64 new fault cases, 3 seeds per method; seeds averaged within physical case, then equal configuration weights.\n'
        'A/S share the identical realized first segment, frozen GNN Diagnoser and B=4 initial resource input; earlier A STOPs are retained.\n'
        'S adds no observations or model calls; its policy cost is the retained prefix. No separate S wall-clock timing is implied.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.075,1,.94));save(fig,run,'06_fresh_stopping_confirmation')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--stage',choices=['replay','interventions','all'],default='all');args=parser.parse_args()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    with TimedStage(args.run,'figures_'+args.stage):
        if args.stage in ('replay','all'):prefixes(args.run);fixed_transcripts(args.run);stories(args.run)
        if args.stage in ('interventions','all'):interventions(args.run);effects(args.run);fresh_confirmation(args.run)
    print('Figures completed: '+args.stage,flush=True)


if __name__=='__main__':main()
