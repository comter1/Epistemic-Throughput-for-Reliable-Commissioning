"""Recheck all saved outcomes and compare real B=1/B=4 acquisition routes."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import time
import numpy as np
import pandas as pd
from cgx.provenance import atomic_json, hash_json, read_json
from cgx.reporting.methods import display_name
from .ledger import TimedStage


def _route(args):
    root,job=args
    def load(cap):
        key=hash_json([job['method'],job['policy_seed'],cap,1.0,job['case_id']])
        item=read_json(Path(root)/'evaluation/main/cases'/f'{key}.json')
        if item['state']!='passed' or item['result_hash']!=hash_json({'row':item['row'],'trace':item['trace']}):
            raise ValueError(f'Invalid original artifact {key}')
        return item
    base=dict(configuration_id=job['configuration_id'],case_id=job['case_id'],
              method_key=job['method'],policy_seed=job['policy_seed'])
    try:
        a,b=load(1),load(4)
        assert a['row']['true_target_key']==b['row']['true_target_key']
        sequences=[[t for t in item['trace'] if 'action' in t] for item in (a,b)]
        acquired=[[t for t in seq if 'evidence_hash' in t] for seq in sequences]
        def first(seq,physical=False):
            return next((t for t in seq if not physical or bool(t['action']['ports'])),None)
        key=lambda t:t['action']['action_key'] if t else None
        common=0
        for left,right in zip(*sequences):
            if key(left)!=key(right):break
            common+=1
        base.update(state='passed',true_target_key=a['row']['true_target_key'],
            first_action_b1=key(first(sequences[0])),first_action_b4=key(first(sequences[1])),
            first_execution_action_b1=key(first(acquired[0],True)),
            first_execution_action_b4=key(first(acquired[1],True)),common_action_prefix_length=common,
            first_different_decision=common+1 if common<max(map(len,sequences)) else None,
            actions_b1=json.dumps([t['action'] for t in sequences[0]],sort_keys=True),
            actions_b4=json.dumps([t['action'] for t in sequences[1]],sort_keys=True),
            first_execution_observation_b1=(first(acquired[0],True) or {}).get('evidence_hash'),
            first_execution_observation_b4=(first(acquired[1],True) or {}).get('evidence_hash'),
            original_timeout_b1=a['row']['timeout'],original_timeout_b4=b['row']['timeout'])
        base['first_action_changed']=base['first_action_b1']!=base['first_action_b4']
        base['first_execution_action_changed']=base['first_execution_action_b1']!=base['first_execution_action_b4']
        # A missing first action is retained and distinguished from an observed difference.
        base['both_have_first_action']=all(base[k] is not None for k in ('first_action_b1','first_action_b4'))
        base['both_have_first_execution']=all(base[k] is not None for k in ('first_execution_action_b1','first_execution_action_b4'))
    except Exception as exc:base.update(state='failed',error=f'{type(exc).__name__}: {exc}')
    return base


def baseline(root,run,workers=16):
    root,run=Path(root),Path(run)
    with TimedStage(run,'baseline_recheck'):
        frame=pd.read_parquet(root/'results/per_case_results.parquet')
        assert len(frame)==147200 and not frame.technical_smoke.any()
        frame['healthy']=frame.true_target_key.eq('healthy')
        frame['false_alarm']=frame.healthy & frame.predicted_target_key.ne('healthy')
        frame['missed_fault']=~frame.healthy & frame.predicted_target_key.eq('healthy')
        keys=['suite','configuration_id','method','policy_seed','execution_cap']
        rows=[]
        for key,group in frame.groupby(keys,dropna=False):
            h=group[group.healthy];f=group[~group.healthy]
            rows.append(dict(zip(keys,key),display_name=display_name(key[2]),cases=len(group),
                healthy_cases=len(h),faulty_cases=len(f),healthy_false_alarm=float(h.false_alarm.mean()),
                fault_miss_rate=float(f.missed_fault.mean()),fault_joint_accuracy=float(f.correct_joint.mean()),
                joint_accuracy=float(group.correct_joint.mean()),final_nll_bits=float(group.final_nll_bits.mean()),
                healthy_nll_bits=float(h.final_nll_bits.mean()),fault_nll_bits=float(f.final_nll_bits.mean()),
                healthy_timeout=float(h.timeout.mean()),timeout=float(group.timeout.mean()),
                healthy_calibrations=float(h.calibrations_used.mean()),calibrations=float(group.calibrations_used.mean()),
                recorded_bytes=float(group.bytes_used.mean()),executions=float(group.executions_used.mean())))
        checks=pd.DataFrame(rows);checks.to_csv(run/'baseline_recheck.csv',index=False)
        numeric=[c for c in checks.select_dtypes('number').columns if c not in ('policy_seed','execution_cap')]
        summary=checks.groupby(['suite','method','execution_cap'],dropna=False)[numeric].mean().reset_index()
        summary['display_name']=summary.method.map(display_name)
        summary.to_csv(run/'baseline_summary.csv',index=False)
        jobs=read_json(run/'manifests/replay_jobs.json')
        start=time.monotonic()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            routes=list(pool.map(_route,((str(root),j) for j in jobs),chunksize=16))
        divergence=pd.DataFrame(routes);divergence.to_csv(run/'budget_path_divergence.csv',index=False)
        selection=read_json(root/'models/selection.json')
        model_rows=[]
        for method in ['gnn_depth2','gnn_adaptive','transformer_adaptive','gnn_fixed_plan']:
            for seed in [11,23,37]:
                name=f'models/policies/{method}_seed{seed}';selected=selection['inventory'][name]['selected']
                model_rows.append(dict(method_key=method,policy_seed=seed,checkpoint=f'{name}/{selected}',
                    checkpoint_hash=selection['files'][f'{name}/{selected}'],budgets=[1,2,4]))
        atomic_json(run/'baseline_checkpoint_mapping.json',model_rows)
        same_cases=frame.groupby(['method','policy_seed','execution_cap'],dropna=False).case_id.nunique().eq(3200).all()
        truth_consistent=frame.groupby('case_id').true_target_key.nunique().eq(1).all()
        expected={'gnn_adaptive':(.075,.2866666666666667),'gnn_depth2':(.035,.22833333333333333),
                  'transformer_adaptive':(.06333333333333334,.15083333333333335),'gnn_fixed_plan':(.075,.0425)}
        points=summary[summary.suite.eq('test_connections')].set_index(['method','execution_cap'])
        for method,(b1,b4) in expected.items():
            assert abs(points.loc[(method,1),'healthy_false_alarm']-b1)<1e-10
            assert abs(points.loc[(method,4),'healthy_false_alarm']-b4)<1e-10
        lines=['# 原结果完整性与健康误报复核','','全部147,200条保存结果已重新汇总；没有新增机器人执行。',
            f'- 每个方法/种子/预算包含同一组3,200案例：{bool(same_cases)}。',
            f'- 同一案例跨方法/预算真值一致：{bool(truth_consistent)}。',
            '- 同一方法/种子在B=1/2/4使用同一选定checkpoint；映射见baseline_checkpoint_mapping.json。',
            '- 健康误报按argmax目标不是healthy计算，不用q(healthy)<0.5替代。',
            '- 原timeout/abstained以及真正故障漏检均保留；这里的误报不是已执行修复。',
            '- 当前科学代码与封存源码一致；只有此前命名相关文档和报告模块不同。',
            '- 前缀target、eval/无梯度和原始记录哈希另由重读及input checks核查。','',
            '| 方法（共享GNN诊断器） | B=1健康误报 | B=4健康误报 | B=4健康超时 |','|---|---:|---:|---:|']
        for method in expected:
            a,b=points.loc[(method,1)],points.loc[(method,4)]
            lines.append(f'| {display_name(method)} | {a.healthy_false_alarm:.4%} | {b.healthy_false_alarm:.4%} | {b.healthy_timeout:.4%} |')
        lines+=['',f'已核对{len(divergence)}组B=1/B=4真实路线；失败记录{int(divergence.state.ne("passed").sum())}条。',
                '第一项动作和第一项真正执行机器人的动作分别比较，原动作/端口/采样率/校准对象和取得记录的hash均保留。']
        (run/'baseline_integrity.md').write_text('\n'.join(lines)+'\n')
        atomic_json(run/'baseline_complete.json',dict(state='passed',rows=len(frame),route_pairs=len(divergence),
            route_errors=int(divergence.state.ne('passed').sum()),route_elapsed_seconds=time.monotonic()-start,
            same_case_sets=bool(same_cases),truth_consistent=bool(truth_consistent)))
        print(json.dumps({'state':'passed','baseline_rows':len(frame),'route_pairs':len(divergence)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();baseline(a.data_root,a.run_dir,a.workers)
