"""Publication-exportable reader figures from completed result tables only."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'gnn_adaptive':'GNN Replan-Each-Round','gnn_depth2':'GNN 2-Step Planner'}
CELLS=['000','100','010','001','110','101','011','111']
LABELS=['Original','P only','C only','E only','P+C','P+E','C+E','P+C+E']
COLORS=['#5b6470','#277da8','#dd802f','#4b9b72','#ad6644','#438d9b','#88964e','#9d4351']


def draw(run):
    run=Path(run);out=run/'figures';out.mkdir(exist_ok=True)
    frame=pd.read_csv(run/'reader_factorial_per_case.csv',dtype={'cell':str})
    endpoint=pd.read_csv(run/'reader_real_endpoint_effects.csv',dtype={'cell':str})
    for title,filename,groups in [
        ('Same acquired records; no new measurement','reader_duplicate_eight_cells',
         [('First-execution block copied',frame[frame.comparison=='copy_first']),
          ('Last-execution block copied',frame[frame.comparison=='copy_last'])]),
        ('Real later measurements: separated reader paths','reader_real_endpoint_eight_cells',
         [('All healthy cases',endpoint),('Initially correct healthy cases',endpoint[endpoint.initially_correct_healthy])])]:
        fig,axes=plt.subplots(2,2,figsize=(15,9),sharex=True)
        for row,method in enumerate(NAMES):
            for col,(groupname,part) in enumerate(groups):
                ax=axes[row,col];part=part[(part.method_key==method)&part.is_healthy]
                table=part.groupby('cell').agg(n=('false_alarm','sum'),N=('case_id','size'),q=('p_healthy','mean')).reindex(CELLS)
                rates=100*table.n/table.N
                ax.bar(np.arange(8),rates,color=COLORS)
                for i,(_,r) in enumerate(table.iterrows()):
                    ax.text(i,100*r.n/r.N+.8,f'{int(r.n)}/{int(r.N)}\nqH={r.q:.3f}',ha='center',va='bottom',fontsize=8)
                ax.set_ylim(0,max(43,float(rates.max())+11));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
                ax.set_title(f'{NAMES[method]}\n{groupname}',fontsize=11)
                ax.set_ylabel('Healthy false alarms (%)');ax.set_xticks(np.arange(8),LABELS,rotation=25,ha='right')
        fig.suptitle(title,fontsize=15)
        fig.text(.5,.015,'Bars: argmax is a fault. qH: mean p(healthy). 64 distinct healthy physical cases, 3 seeds per method; initially correct subset: 57 cases.\nP = pooled GRU vectors; C = counts and observed status; E = measured edges. Synthetic swaps are explanatory probes, not deployable policies.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(out/(filename+'.png'),dpi=180);fig.savefig(out/(filename+'.pdf'));plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);draw(p.parse_args().run)
