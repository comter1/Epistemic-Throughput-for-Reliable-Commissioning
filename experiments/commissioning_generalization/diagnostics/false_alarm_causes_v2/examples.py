"""Deterministic illustrations, separate from population-level comparisons."""
import argparse
import json
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cgx.data.store import Store
from cgx.provenance import atomic_json,hash_json
from diagnostics.false_alarm_causes.interventions import load_blob,dump_blob

KEY=['case_id','method_key','policy_seed']


def run(run_dir):
    run=Path(run_dir);out=run/'case_examples';out.mkdir(exist_ok=True)
    data=pd.read_csv(run/'confirmation_per_case.csv')
    a=data[data.arm=='A0'].set_index(KEY);s=data[data.arm=='S1'].set_index(KEY)
    candidates=[]
    for priority,arm in enumerate(['G1','RG1','R1']):
        if arm not in set(data.arm):continue
        for _,row in data[data.arm==arm].iterrows():
            key=tuple(row[k] for k in KEY);base=a.loc[key]
            if row.is_healthy and base.healthy_false_alarm and not row.healthy_false_alarm:
                candidates.append(dict(row.to_dict(),example_kind='prevented_healthy_false_alarm',arm_priority=priority,
                    example_hash=hash_json(['CGX-FA-CAUSE-V2:illustration',key,arm])))
    candidates=sorted(candidates,key=lambda r:(r['arm_priority'],r['example_hash']))
    selected=[];seen=set()
    for row in candidates:
        if row['case_id'] in seen:continue
        selected.append(row);seen.add(row['case_id'])
        if len(selected)==2:break
    useful=[]
    for _,row in data[data.arm=='G1'].iterrows():
        key=tuple(row[k] for k in KEY);first=s.loc[key]
        if not row.is_healthy and row.correct_joint and not first.correct_joint and row.executions_used>first.executions_used:
            useful.append(dict(row.to_dict(),example_kind='useful_continuation_preserved',
                example_hash=hash_json(['CGX-FA-CAUSE-V2:illustration',key,'G1'])))
    if useful:selected.append(sorted(useful,key=lambda r:r['example_hash'])[0])
    decisions=pd.read_csv(run/'confirmation_decisions.csv')
    store=Store(run);examples=[]
    fig,axes=plt.subplots(3,2,figsize=(15,12),gridspec_kw={'width_ratios':[1,1.7]})
    for index in range(3):
        ax,table=axes[index]
        if index>=len(selected):
            ax.axis('off');table.axis('off');ax.text(.1,.5,'No qualifying example; none invented.',fontsize=12)
            continue
        row=selected[index];case,method,seed=(row[k] for k in KEY)
        validity=row.get('gate_selected_as_deployable_improvement')
        validity=None if pd.isna(validity) else (validity=='True' if isinstance(validity,str) else bool(validity))
        path=run/'trajectories/confirmation/cases'/f'{hash_json([case,method,int(seed)])}.json.gz'
        item=load_blob(path)
        prefix=item['prefixes'];keep=int(row['retained_prefix_ordinal'])
        pp=[p['prediction'] for p in prefix];truth=row['true_target_key']
        healthy=[float(np.exp2(p['log_probabilities_base2'][p['target_keys'].index('healthy')])) for p in pp]
        true=[float(np.exp2(p['log_probabilities_base2'][p['target_keys'].index(truth)])) for p in pp]
        steps=list(range(len(prefix)))
        ax.plot(steps,healthy,'o-',color='#bc7759',label='Original P(healthy)')
        if truth!='healthy':ax.plot(steps,true,'s-',color='#547bad',label='Original P(true fault)')
        ax.axvline(keep,color='#527c58',linestyle='--',label=f'{row["arm"]} retained prefix')
        ax.scatter([keep],[row['p_healthy']],color='#527c58',marker='*',s=120,label=f'{row["arm"]} final P(healthy)',zorder=5)
        if keep<len(prefix)-1:ax.axvspan(keep+.05,len(prefix)-1,color='#dddddd',alpha=.45)
        ax.set_ylim(0,1.05);ax.set_xticks(steps);ax.set_xlabel('Acquired-action prefix (calibration also counts)')
        ax.set_ylabel('Probability');ax.legend(fontsize=8,loc='best')
        ax.set_title(f'{row["example_kind"]}\n{method}, seed {seed}; evaluator truth: {truth}',fontsize=10)
        d=decisions[(decisions.case_id==case)&(decisions.method_key==method)&(decisions.policy_seed==seed)].sort_values('prefix_ordinal')
        lines=['prefix  exec  records  originalQ  gateQ   next ports / action']
        for _,decision in d.iterrows():
            number=lambda v:'terminal' if pd.isna(v) else f'{v:+.3f}'
            ports=', '.join(json.loads(decision.proposed_ports))
            action=ports or str(decision.proposed_action_kind)
            text=f'{int(decision.prefix_ordinal):>3}     {int(decision.executions_used):>2}  {int(decision.total_record_count):>7}  {number(decision.original_q):>9}  {number(decision.selected_gate_prediction):>7}  {action}'
            lines.extend(textwrap.wrap(text,width=94,subsequent_indent=' '*48,replace_whitespace=False))
        lines+=['',f'Selected arm: {row["arm"]}; retained prefix: {keep}',
                f'Final prediction: {row["predicted_target_key"]}',
                'Gray area: baseline continuation after the gate; unavailable to stopped method.']
        if validity is False:
            lines.append('Gate status: experimental fallback; source miss guardrail failed.')
        table.axis('off');table.text(0,1,'\n'.join(lines),va='top',family='DejaVu Sans Mono',fontsize=7.5)
        stem=f'{index+1:02d}_{case[:12]}_{method}_seed{seed}'
        d.to_csv(out/(stem+'_decisions.csv'),index=False)
        dump_blob(out/(stem+'_actual_A0_trajectory.json.gz'),item)
        dump_blob(out/(stem+'_actual_retained_public_history.json.gz'),store.load_history(prefix[keep]['history_hash']))
        summary={k:row[k] for k in KEY+['arm','example_kind','example_hash','true_target_key','predicted_target_key','retained_prefix_ordinal']}
        summary.update(original_prediction=a.loc[(case,method,seed)].predicted_target_key,
            first_execution_prediction=s.loc[(case,method,seed)].predicted_target_key,
            gate_selected_as_deployable_improvement=validity,
            true_label_used_only_by_evaluator=True,decision_table=stem+'_decisions.csv')
        atomic_json(out/(stem+'_summary.json'),summary);examples.append(summary)
    fig.suptitle('Actual prefixes: prevented mistakes and retained useful continuation',fontsize=14)
    fig.text(.5,.01,'Illustrations selected by fixed arm priority and case hash; never used for training or selection. Population tables contain every case.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.97))
    fig.savefig(run/'figures/05_actual_case_examples.png',dpi=180,bbox_inches='tight')
    fig.savefig(run/'figures/05_actual_case_examples.pdf',bbox_inches='tight');plt.close(fig)
    manifest=dict(selection_rule='G1 then RG1 then R1; deterministic domain-separated hash; distinct physical cases; healthy A0 error corrected. Useful continuation: G1 correct faulty endpoint, S1 incorrect, strictly more actual executions; hash order.',
        requested_prevented_examples=2,available_prevented_physical_cases=len({r['case_id'] for r in candidates}),
        requested_useful_examples=1,available_useful_physical_cases=len({r['case_id'] for r in useful}),
        selected=examples,missing_examples_reported_without_fabrication=True,
        labels_used_only_for_posthoc_illustration=True)
    atomic_json(out/'selection.json',manifest)
    (out/'README.md').write_text('这些是完整统计之后按预定哈希规则选取的插图案例，不参与模型选择。每例保存实际 A0 轨迹、实际保留的公开历史和逐步 ports/记录数/q/原分数/门控分数；真标签只在评估摘要。\n\n'+
        f"可用的不同健康纠错案例 {manifest['available_prevented_physical_cases']} 个；可用的保留有益继续案例 {manifest['available_useful_physical_cases']} 个。若不足要求，图明确留空而不伪造。\n")
    print({k:manifest[k] for k in ('available_prevented_physical_cases','available_useful_physical_cases')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);run(p.parse_args().run_dir)
