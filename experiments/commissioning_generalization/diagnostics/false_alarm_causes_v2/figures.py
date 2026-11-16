"""Scientific figures generated only from saved V2 result tables."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'gnn_adaptive':'GNN Replan-Each-Round','gnn_depth2':'GNN 2-Step Planner'}
CELLS=['000','100','010','001','110','101','011','111']
LABELS=['base','P','C','E','P+C','P+E','C+E','P+C+E']


def save(fig,path,caption):
    fig.text(.5,.012,caption,ha='center',va='bottom',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.97))
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)


def reader(run):
    data=pd.read_csv(run/'reader_factorial_per_case.csv',dtype={'cell':str})
    endpoint=run/'reader_real_endpoint_effects.csv'
    actual=pd.read_csv(endpoint,dtype={'cell':str}) if endpoint.exists() else data[(data.comparison=='real_prefix')&data.is_final]
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for i,(method,name) in enumerate(NAMES.items()):
        for j,(title,part) in enumerate([('Copy first additional execution',data[(data.comparison=='copy_first')&data.is_healthy]),
                                       ('Real first execution to endpoint',actual[actual.is_healthy])]):
            part=part[part.method_key==method];ax=axes[i,j]
            grouped=part.groupby('cell').agg(p=('p_healthy','mean'),fa=('false_alarm','sum'),n=('false_alarm','size'))
            grouped=grouped.reindex(CELLS)
            ax.bar(range(8),100*grouped.p,color=['#63758b','#6481bc','#c06a54','#638b67','#9566a2','#4792a1','#ba9650','#374d68'])
            for x,(_,r) in enumerate(grouped.iterrows()):
                ax.text(x,100*r.p+1,f'{int(r.fa)}/{int(r.n)}',ha='center',fontsize=8)
            ax.set_xticks(range(8),LABELS,rotation=25);ax.set_ylim(0,100)
            ax.set_ylabel('Mean P(healthy), %');ax.set_title(f'{name}\n{title}',fontsize=11)
    save(fig,run/'figures/01_reader_eight_calculations',
         'Unchanged GNN Diagnoser weights. Labels above bars: healthy false alarms / eligible seed runs.\n'
         '64 distinct healthy physical cases, 3 seeds per method. P = pooled GRU vectors; C = count fields; E = measured edges.\n'
         'Cases with no additional execution are reported separately.')


def values(run):
    data=pd.read_csv(run/'continuing_value_calibration.csv')
    data=data[data.grouping=='method_key/q_bin']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,(method,name) in zip(axes,NAMES.items()):
        p=data[data.method_key==method]
        for teacher,color in [('oof','#4d78b0'),('final','#b65f49')]:
            d=p[p.teacher==teacher].sort_values('original_q')
            ax.errorbar(d.original_q,d.realized_return,
                yerr=np.maximum(0,np.array([d.realized_return-d.realized_return_ci_low,d.realized_return_ci_high-d.realized_return])),
                fmt='o-',color=color,capsize=3,label=f'{teacher.upper()} reader return')
        upper=max(1.,float(p.original_q.max()));ax.plot([0,upper],[0,upper],':',color='gray',label='Predicted = observed')
        ax.axhline(0,color='black',linewidth=.7);ax.set_xlabel('Original predicted continuation benefit, bits')
        ax.set_ylabel('Observed matched-horizon score change, bits');ax.set_title(name);ax.legend(fontsize=8)
    save(fig,run/'figures/02_predicted_and_observed_benefit',
         'Source validation only; identical acquired continuations scored by two frozen readers.\n'
         'Replan: actual remaining decisions; 2-Step: at most two decisions, including calibration. Configuration/case bootstrap 95% intervals.')


def four_cells(run):
    data=pd.read_csv(run/'correction_2x2_source_validation.csv')
    data=data[data.step==2000]
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,(method,name) in zip(axes,NAMES.items()):
        p=data[data.method_key==method].set_index('cell')
        keys=[['T00','T01'],['T10','T11']]
        matrix=np.array([[p.loc[k].final_nll_bits for k in row] for row in keys])
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=max(1.,matrix.max()))
        for i,row in enumerate(keys):
            for j,key in enumerate(row):
                r=p.loc[key]
                ax.text(j,i,f'{key}: {r.final_nll_bits:.3f} bits\nFA {int(r.false_alarms)}/{int(r.healthy_denominator)}\nMiss {int(r.missed_faults)}/{int(r.faulty_denominator)}',
                        ha='center',va='center',fontsize=11,color='white' if matrix[i,j]>.6*max(1.,matrix.max()) else 'black')
        ax.set_xticks([0,1],['OOF q','FINAL q']);ax.set_yticks([0,1],['OOF labels','FINAL labels'])
        ax.set_xlabel('q input used during correction training');ax.set_ylabel('Reader defining continuation labels')
        ax.set_title(name)
    save(fig,run/'figures/03_four_matched_training_cells',
         'SOURCE validation: 16 healthy + 48 faulty physical cases, each with 3 policy seeds. All cells shown at exactly 2,000 updates.\n'
         'All deployment/replay inputs use FINAL q; original backbone, support, initialization, optimizer, and minibatch order are matched.')


def confirmation(run):
    data=pd.read_csv(run/'confirmation_counts_and_rates.csv')
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    order=['A0','S1','R1','G1','RG1','R-live','RG-live']
    for i,(method,name) in enumerate(NAMES.items()):
        for j,(metric,title) in enumerate([('healthy_false_alarm','Healthy called faulty'),('fault_miss','Faulty called healthy'),('correct_joint','Correct fault type + location')]):
            p=data[(data.method_key==method)&(data.metric==metric)].set_index('arm')
            p=p.reindex([a for a in order if a in p.index]);ax=axes[i,j]
            ax.bar(range(len(p)),100*p.rate,color=['#6b7c90','#b7bfc8','#74a3b5','#bc7759','#6d9b77','#af90b5','#b7a66d'][:len(p)])
            for x,(_,r) in enumerate(p.iterrows()):ax.text(x,100*r.rate+1,f'{int(r.numerator)}/{int(r.denominator)}',ha='center',fontsize=8)
            ax.set_xticks(range(len(p)),p.index,rotation=25);ax.set_ylim(0,min(105,max(20,100*p.rate.max()+12)))
            ax.set_ylabel('Conditional rate, %');ax.set_title(f'{name}\n{title}',fontsize=10)
    save(fig,run/'figures/04_fresh_errors_and_correct_localization',
         'Fresh confirmation: 64 distinct healthy + 64 distinct faulty physical cases on 16 previously examined held-out configurations.\n'
         'Each method/arm has 3 policy seeds per case: conditional denominator 192.\n'
         '2-Step G1/RG1/RG-live use the prespecified experimental fallback: its gate did not pass SOURCE miss guardrails.')


def run(run_dir):
    run=Path(run_dir);(run/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    reader(run);values(run);four_cells(run);confirmation(run)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);run(p.parse_args().run_dir)
