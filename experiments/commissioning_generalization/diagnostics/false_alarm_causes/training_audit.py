"""Read-only source coverage, finite-horizon reward, and OOF/final-reader audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import socket
import sqlite3

import numpy as np
import pandas as pd

from cgx.data.store import Store
from cgx.provenance import atomic_json, hash_json
from cgx.training.engine import transition_key
from .core import inference_settings, load_reader, predict
from .ledger import Ledger, TimedStage


FIELDS=('executions','calibrations','bytes','decisions','unique_ports',
        'repeated_port_visits','repeated_actions','max_single_record_bytes','raw_records')


def readonly_db(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro&immutable=1',uri=True)


def family(target):
    return 'healthy' if target=='healthy' else target.split(':',1)[1]


def audit_rows(rows, scores):
    """Audit complete source trajectories; accepts iterables for reproducible unit tests."""
    trajectories=defaultdict(list)
    for row in rows:trajectories[row['trajectory_id']].append(row)
    prefix_rows=[];trajectory_rows=[];action_rows=[]
    checks=Counter();failures=[];same_history=defaultdict(dict)
    for tid,transitions in trajectories.items():
        transitions.sort(key=lambda r:r['step'])
        first=transitions[0]
        ports=[];actions=[];costs=Counter();maxbytes=0;rawrecords=0;rewards=[];score_rows=[]
        for position,row in enumerate(transitions):
            action=row['action'];kind=action['kind'];charge=row['charged_cost']
            checks['transitions']+=1
            def verify(name,passed):
                checks[name+'_checked']+=1
                if not passed:
                    checks[name+'_failed']+=1
                    if len(failures)<100:failures.append(dict(check=name,trajectory_id=tid,step=row['step']))
            verify('resource_subtraction',all(row['resources_after'][k]==row['resources_before'][k]-charge[k] for k in charge))
            verify('nonnegative_resources',min(row['resources_after'].values())>=0)
            verify('charged_within_declared_bound',all(charge[k]<=action['cost_bound'][k] for k in charge))
            verify('consecutive_steps',row['step']==position)
            if position:
                verify('history_chain',transitions[position-1]['next_history_hash']==row['history_hash'])
                verify('resource_chain',transitions[position-1]['resources_after']==row['resources_before'])
            verify('terminal_only_at_end',not row['terminal'] or position==len(transitions)-1)
            if kind=='stop':
                verify('stop_zero_cost',all(v==0 for v in charge.values()))
                verify('stop_unchanged_evidence',row['history_hash']==row['next_history_hash'])
                verify('stop_terminal',row['terminal'])
            if row['resources_after']['decisions']==0:verify('zero_horizon_terminal',row['terminal'])
            costs.update(charge)
            ports.extend(action['ports'])
            if action['ports']:actions.append(row['action_key'])
            maxbytes=max(maxbytes,charge['bytes'])
            # Wire format is exactly 128 bytes/record and 160 bytes/calibration burst.
            record_bytes=charge['bytes']-160*charge['calibrations']
            verify('fixed_wire_record_size',record_bytes>=0 and record_bytes%128==0)
            rawrecords+=record_bytes//128
            metrics={k:int(costs[k]) for k in ('executions','calibrations','bytes','decisions')}
            metrics.update(unique_ports=len(set(ports)),repeated_port_visits=len(ports)-len(set(ports)),
                           repeated_actions=len(actions)-len(set(actions)),max_single_record_bytes=maxbytes,raw_records=rawrecords)
            meta=dict(split=first['split'],configuration_id=first['configuration_id'],case_id=first['parent_case_id'],
                      trajectory_id=tid,source_fold=first['source_fold'],family=family(first['true_target_key']),
                      healthy=first['true_target_key']=='healthy',execution_cap=first['execution_cap'],
                      byte_fraction=first['byte_fraction'],behavior=first['behavior'])
            prefix_rows.append(dict(**meta,step=row['step'],history_hash=row['next_history_hash'],**metrics))
            action_rows.append(dict(**meta,kind=kind,calibration=bool(charge['calibrations']),step=row['step']))
            key=transition_key(row)
            score=scores.get(key)
            if score is not None:
                rewards.append(score['reward']);score_rows.append(score)
                checks['scored_transitions']+=1
                if kind=='stop':verify('stop_zero_reward',abs(score['reward'])<1e-10)
                if len(score_rows)>1:
                    verify('adjacent_logscore_consistency',abs(-score_rows[-2]['next_nll']-(-score['next_nll']-score['reward']))<1e-6)
            # Sampling frame is public history identity; no outcome-based selection.
            if row['step']==0 and kind!='stop':same_history[(first['split'],row['history_hash'])][row['action_key']]=row
        checks['trajectories']+=1
        if not transitions[-1]['terminal']:
            checks['last_transition_not_marked_terminal']+=1
            if len(failures)<100:failures.append(dict(check='last_transition_not_marked_terminal',trajectory_id=tid))
        error=None
        if len(score_rows)==len(transitions):
            first_lp=-score_rows[0]['next_nll']-score_rows[0]['reward']
            last_lp=-score_rows[-1]['next_nll']
            error=abs(sum(rewards)-(last_lp-first_lp))
            checks['telescoping_trajectories']+=1
            if error>1e-6:checks['telescoping_failed']+=1
        trajectory_rows.append(dict(**meta,**metrics,transitions=len(transitions),terminal=bool(transitions[-1]['terminal']),
                                    total_reward_bits=sum(rewards) if score_rows else None,
                                    telescoping_abs_error=error))
    return prefix_rows,trajectory_rows,action_rows,dict(checks),failures,same_history


def source_summary(frame):
    rows=[]
    for (split,fam),group in frame.groupby(['split','family']):
        for field in FIELDS:
            values=group[field].to_numpy()
            rows.append(dict(split=split,family=fam,field=field,n=len(values),mean=float(values.mean()),
                             minimum=float(values.min()),q01=float(np.quantile(values,.01)),
                             q05=float(np.quantile(values,.05)),median=float(np.median(values)),
                             q95=float(np.quantile(values,.95)),q99=float(np.quantile(values,.99)),maximum=float(values.max())))
    return pd.DataFrame(rows)


def support_comparison(prefixes, current, trajectories=None):
    """Marginal tail positions are descriptive, not evidence of causal distribution shift."""
    out=[]
    for _,row in current.iterrows():
        fam=family(row['true_target_key'])
        scopes=[('all_source_prefixes',prefixes[(prefixes['split']=='train')&(prefixes['family']==fam)]),
                             ('B4_full_byte_source_prefixes',prefixes[(prefixes['split']=='train')&(prefixes['family']==fam)&
                                  (prefixes['execution_cap']==4)&(prefixes['byte_fraction']==1.)]),
                             ('B4_full_byte_same_execution_count_prefixes',prefixes[(prefixes['split']=='train')&(prefixes['family']==fam)&
                                  (prefixes['execution_cap']==4)&(prefixes['byte_fraction']==1.)&
                                  (prefixes['executions']==row['executions'])])]
        if trajectories is not None:
            scopes.append(('B4_full_byte_terminal_trajectories',trajectories[(trajectories['split']=='train')&(trajectories['family']==fam)&
                           (trajectories['execution_cap']==4)&(trajectories['byte_fraction']==1.)]))
        for scope,source in scopes:
            for field in FIELDS:
                values=source[field].to_numpy();value=float(row[field])
                lower=float((values<=value).mean()) if len(values) else float('nan')
                upper=float((values>=value).mean()) if len(values) else float('nan')
                out.append(dict(case_id=row['case_id'],configuration_id=row['configuration_id'],method=row['method'],
                                policy_seed=row['policy_seed'],family=fam,reference=scope,field=field,value=value,
                                n_source=len(values),percentile_le=lower,percentile_ge=upper,
                                outside_observed_range=bool(value<values.min() or value>values.max()) if len(values) else None,
                                marginal_tail_le_1pct=bool(min(lower,upper)<=.01),
                                original_false_alarm=row['original_false_alarm'],
                                complete_reread_false_alarm=row['complete_reread_false_alarm']))
    return pd.DataFrame(out)


def finalize_tables(run_dir):
    """Join completed input probes without repeating training or reader calls."""
    run_dir=Path(run_dir)
    prefixes=pd.read_parquet(run_dir/'training_prefix_coverage.parquet')
    trajectories=pd.read_csv(run_dir/'training_trajectory_summaries.csv')
    current=pd.read_csv(run_dir/'input_history_coverage.csv')
    support_comparison(prefixes,current,trajectories).to_csv(run_dir/'training_support_comparison.csv',index=False)
    joint=prefixes.groupby(['split','family','execution_cap','byte_fraction','executions','calibrations']).size().rename('n_prefixes').reset_index()
    joint.to_csv(run_dir/'training_joint_execution_calibration_coverage.csv',index=False)
    gains=pd.read_csv(run_dir/'crossfit_final_reward_comparison.csv')
    resolutions=[]
    for threshold in [0.,1e-6,1e-5,1e-4,1e-3]:
        for split,subset in gains.groupby('split'):
            total=resolved=discordant=0
            for _,group in subset.groupby('history_hash'):
                for a,b in itertools.combinations(group.to_dict('records'),2):
                    total+=1
                    od=a['gain_oof_bits']-b['gain_oof_bits'];fd=a['gain_final_bits']-b['gain_final_bits']
                    if min(abs(od),abs(fd))>threshold:
                        resolved+=1;discordant+=od*fd<0
            resolutions.append(dict(split=split,min_action_gap_bits=threshold,total_pairs=total,
                                    resolved_pairs=resolved,discordant_resolved_pairs=discordant))
    pd.DataFrame(resolutions).to_csv(run_dir/'crossfit_final_ranking_resolution.csv',index=False)
    train=trajectories[trajectories['split']=='train']
    means=train.groupby('family')[['executions','calibrations','unique_ports','repeated_actions','bytes']].mean()
    no_cal=train.groupby('family')['calibrations'].agg(n='size',zero=lambda x:int((x==0).sum()),two=lambda x:int((x==2).sum()))
    text='\n\n补充：完成输入探针后的覆盖/排序对照\n\n'
    text+='训练有28个STOP transitions / 160829 (0.0174%)。STOP值和mask实现通过检查，但停止行为在原source replay中非常稀少；不能单独据此归因当前误报。完整训练轨迹校准数均值接近2，详细为：\n\n'+means.to_string()+'\n\n'+no_cal.to_string()+'\n\n'
    text+='OOF/final原始排序翻转包括极近tie。以下固定精度网格只表明读者排序差的数值大小，不做最优阈值选择；OOF重算与原cache收益的最大偏差另在逐动作表中保留。\n\n'+pd.DataFrame(resolutions).to_string(index=False)+'\n\n'
    text+='coverage新增同执行次数前缀与B4完整字节终止轨迹对照，防止把早期0校准与晚期0校准混为一谈。端口覆盖从训练实际动作声明的端口集合累计；当前历史从真实记录端口累计，二者粒度差异需保留。\n'
    path=run_dir/'training_audit.md'
    original=path.read_text().split('\n\n补充：完成输入探针后的覆盖/排序对照')[0]
    path.write_text(original+text)


def audit_crossfit_assignment(root,run_dir):
    """Check all source configurations/checkpoint hashes; no reader forwards."""
    root,run_dir=Path(root),Path(run_dir)
    with TimedStage(run_dir,'crossfit_assignment_audit'):
        trajectories=pd.read_csv(run_dir/'training_trajectory_summaries.csv')
        source=trajectories[trajectories['split']=='train']
        validation=set(trajectories.loc[trajectories['split']=='validation','configuration_id'])
        final=load_reader(root,'final','cpu');records=[];expected={}
        for fold in sorted(source.source_fold.unique()):
            model=load_reader(root,f'crossfit_{int(fold)}','cpu')
            heldout=set(source.loc[source.source_fold==fold,'configuration_id'])
            trained=set(model.training_configurations)
            records.append(dict(fold=int(fold),held_out_configurations=sorted(heldout),
                                trained_configuration_count=len(trained),overlap_with_heldout=sorted(heldout&trained),
                                overlap_with_validation=sorted(validation&trained),checkpoint_hash=model.checkpoint_hash,
                                different_from_final=model.checkpoint_hash!=final.checkpoint_hash))
            expected[int(fold)]=model.checkpoint_hash
        folds=dict(zip(source.trajectory_id,source.source_fold.astype(int)))
        db=readonly_db(root/'training/scores.sqlite');mismatch=0;total=0
        for trajectory,reader_hash in db.execute('SELECT trajectory,reader_hash FROM scores'):
            total+=1;mismatch+=trajectory not in folds or expected.get(folds.get(trajectory))!=reader_hash
        db.close()
        result=dict(state='passed' if not mismatch and all(not r['overlap_with_heldout'] and not r['overlap_with_validation'] for r in records) else 'failed',
                    folds=records,scored_transitions=total,reader_hash_mismatches=mismatch,
                    final_checkpoint_hash=final.checkpoint_hash,final_training_configuration_count=len(final.training_configurations),
                    final_validation_overlap=sorted(validation&set(final.training_configurations)),model_forward_calls=0)
        atomic_json(run_dir/'crossfit_assignment_checks.json',result)
        print(json.dumps(result),flush=True)


def compare_readers(root,run_dir,groups,device,ledger):
    # Eight train and four validation distinct configurations, <=4 actions/history.
    selected=[]
    for split,limit in [('train',8),('validation',4)]:
        candidates=[(key,rs) for key,rs in groups.items() if key[0]==split and len(rs)>=2]
        candidates.sort(key=lambda x:hash_json(['oof-final-order-v1',x[0]]))
        seen=set()
        for (sp,ref),options in candidates:
            cfg=next(iter(options.values()))['configuration_id']
            if cfg in seen:continue
            seen.add(cfg)
            rows=sorted(options.values(),key=lambda r:hash_json(['oof-final-actions-v1',r['action_key']]))[:4]
            selected.append(dict(split=sp,configuration_id=cfg,history_hash=ref,
                                 source_fold=rows[0]['source_fold'],transitions=rows))
            if len(seen)>=limit:break
    atomic_json(Path(run_dir)/'manifests/oof_final_comparison.json',dict(
        rule='Fixed hash; >=2 actually recorded alternatives at identical initial history; distinct configurations; no reward/outcome selection.',
        validation_reader_rule='crossfit_0, fixed in advance; all validation configurations excluded from all training readers',groups=selected))
    store=Store(root);models={'final':load_reader(root,'final',device)};rows=[];groupsummary=[]
    cache={}
    def infer(name,ref):
        key=(name,ref)
        if key not in cache:
            if name not in models:models[name]=load_reader(root,name,device)
            h=store.load_history(ref)
            if 'prediction' in h:raise AssertionError('Unexpected prediction in stored history')
            cache[key]=predict(models[name],h,ledger,detail='training_oof_final')[0]
        return cache[key]
    scoredb=readonly_db(Path(root)/'training/scores.sqlite')
    for group in selected:
        name='crossfit_'+str(group['source_fold'] if group['split']=='train' else 0)
        before={n:infer(n,group['history_hash']) for n in ('final',name)}
        if group['configuration_id'] in models[name].training_configurations:raise AssertionError('OOF exclusion failed')
        gains=[]
        for row in group['transitions']:
            target=row['true_target_key'];after={n:infer(n,row['next_history_hash']) for n in ('final',name)}
            def true_lp(p):return p['log_probabilities_base2'][p['target_keys'].index(target)]
            gr={n:true_lp(after[n])-true_lp(before[n]) for n in before}
            score=scoredb.execute('SELECT reward,reader_hash FROM scores WHERE id=?',(transition_key(row),)).fetchone()
            if score and score[1]!=models[name].checkpoint_hash:raise AssertionError('Scored OOF checkpoint differs')
            result=dict(split=group['split'],configuration_id=group['configuration_id'],history_hash=group['history_hash'],
                        next_history_hash=row['next_history_hash'],action_key=row['action_key'],action_kind=row['action']['kind'],
                        true_target_key=target,reader_oof=name,oof_checkpoint_hash=models[name].checkpoint_hash,
                        final_checkpoint_hash=models['final'].checkpoint_hash,oof_configuration_excluded=True,
                        gain_oof_bits=gr[name],gain_final_bits=gr['final'],gain_difference_bits=gr['final']-gr[name],
                        stored_training_reward_bits=score[0] if score else None,
                        current_oof_minus_stored_reward_bits=gr[name]-score[0] if score else None)
            rows.append(result);gains.append(result)
        pairs=list(itertools.combinations(gains,2))
        discordant=sum((a['gain_oof_bits']-b['gain_oof_bits'])*(a['gain_final_bits']-b['gain_final_bits'])<0 for a,b in pairs)
        best_oof=max(gains,key=lambda r:r['gain_oof_bits'])['action_key']
        best_final=max(gains,key=lambda r:r['gain_final_bits'])['action_key']
        groupsummary.append(dict(split=group['split'],configuration_id=group['configuration_id'],history_hash=group['history_hash'],
                                 n_actions=len(gains),n_pairs=len(pairs),discordant_pairs=discordant,
                                 top_action_changed=best_oof!=best_final))
        print('OOF/final',group['split'],group['configuration_id'],len(gains),flush=True)
    scoredb.close()
    pd.DataFrame(rows).to_csv(Path(run_dir)/'crossfit_final_reward_comparison.csv',index=False)
    pd.DataFrame(groupsummary).to_csv(Path(run_dir)/'crossfit_final_ranking_summary.csv',index=False)
    return groupsummary


def run(root,run_dir,device='cuda:0'):
    root,run_dir=Path(root),Path(run_dir)
    inference_settings();ledger=Ledger(run_dir)
    with TimedStage(run_dir,'training_audit'):
        scoredb=readonly_db(root/'training/scores.sqlite')
        cols=['id','reward','next_nll','reader_hash','trajectory','step','history','next_history']
        scores={r[0]:dict(zip(cols,r)) for r in scoredb.execute('SELECT '+','.join(cols)+' FROM scores')}
        scoredb.close()
        print('Loaded original scores',len(scores),flush=True)
        results=[];allgroups={};allchecks=Counter();failures=[]
        for split in ('train','validation'):
            db=readonly_db(root/'training'/f'replay_{split}.sqlite')
            result=audit_rows((json.loads(r[0]) for r in db.execute('SELECT data FROM transitions')),scores)
            db.close();results.append(result);allgroups.update(result[-1]);allchecks.update(result[3]);failures.extend(result[4])
            print('Audited',split,result[3]['transitions'],flush=True)
        prefixes=pd.DataFrame([r for result in results for r in result[0]])
        trajectories=pd.DataFrame([r for result in results for r in result[1]])
        actions=pd.DataFrame([r for result in results for r in result[2]])
        prefixes.to_parquet(run_dir/'training_prefix_coverage.parquet',index=False)
        trajectories.to_csv(run_dir/'training_trajectory_summaries.csv',index=False)
        source_summary(prefixes).to_csv(run_dir/'training_prefix_coverage_summary.csv',index=False)
        source_summary(trajectories).to_csv(run_dir/'training_coverage_summary.csv',index=False)
        association=actions.groupby(['split','family','kind']).size().rename('n').reset_index()
        association['within_family_fraction']=association['n']/association.groupby(['split','family'])['n'].transform('sum')
        association.to_csv(run_dir/'training_action_label_association.csv',index=False)
        calibration=trajectories.groupby(['split','family','calibrations']).size().rename('n').reset_index()
        calibration['within_family_fraction']=calibration['n']/calibration.groupby(['split','family'])['n'].transform('sum')
        calibration.to_csv(run_dir/'training_calibration_label_association.csv',index=False)
        atomic_json(run_dir/'training_reward_checks.json',dict(counts=dict(allchecks),failure_examples=failures[:100],
            reward_source='Original scores.sqlite, no rescore or training',
            q_input_path='engine.fit_policy.prediction reads same OOF predictions table used for rewards',
            validation_reward_note='Validation replay has no cached training rewards; excluded from full telescoping denominator'))
        comparison_path=run_dir/'input_history_coverage.csv'
        if comparison_path.exists():support_comparison(prefixes,pd.read_csv(comparison_path)).to_csv(run_dir/'training_support_comparison.csv',index=False)
        ranks=compare_readers(root,run_dir,allgroups,device,ledger)
        summary=pd.DataFrame(ranks)
        text='# 原训练数据与奖励审计\n\n'
        text+=f"只读扫描原 train/validation 共 {allchecks['transitions']} transitions、{allchecks['trajectories']} trajectories。原奖励覆盖 {allchecks['scored_transitions']} transitions；{allchecks['telescoping_trajectories']} 条完整轨迹核查 log-score 奖励相加。\n\n"
        text+='代码中训练 reward 与策略显式 q-input 均来自三折 topology OOF reader；部署统一使用 final GNN，权重并不相同。此区别是协议设计，不自动等于实现错误。depth2 训练时以固定规则采样1/2剩余决策，myopic为1；Double-Q在terminal或剩余决策为0时不给continuation，STOP固定0。逐条预算、链路、终止和reward检查见 training_reward_checks.json；不重写原数据。\n\n'
        text+='coverage 同时保存所有source prefixes和完整轨迹。端口数、重复动作/端口访问、记录字节/条数来自实际charged_cost及动作记录；初始共用执行不计入追加资源。训练行为 mixture 与字节/执行上限分布本就不同于当前学习策略，所以分布差异只能作为覆盖证据。当前轨迹分别相对于同标签所有train前缀和B4/完整字节train前缀报告边缘percentile、1%尾部和观测范围之外；这些不是因果检验，也不等于联合状态密度。\n\n'
        text+='OOF/final比较只用原train/validation中相同history真实已有的至少两种动作结果，固定hash选8 train+4 validation配置，每组最多4动作；并非对这些历史展开新物理分支。validation固定选crossfit_0，其配置没有进入reader训练。完整manifest先于额外forward落盘。比较的是下一步真实目标log-score增量，不能当成多步Q误差。\n\n'
        if not summary.empty:text+=summary.groupby('split')[['n_actions','n_pairs','discordant_pairs','top_action_changed']].sum().to_string()+'\n'
        (run_dir/'training_audit.md').write_text(text)
        atomic_json(run_dir/'training_audit_status.json',dict(state='completed',host=socket.gethostname(),device=device,
                     source_transitions=allchecks['transitions'],source_trajectories=allchecks['trajectories'],
                     reader_ranking_groups=len(ranks),ledger_snapshot=ledger.snapshot()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',required=True);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--finalize-only',action='store_true')
    parser.add_argument('--crossfit-checks-only',action='store_true')
    args=parser.parse_args()
    if args.crossfit_checks_only:audit_crossfit_assignment(args.data_root,args.run_dir)
    elif args.finalize_only:finalize_tables(args.run_dir)
    else:run(args.data_root,args.run_dir,args.device)
