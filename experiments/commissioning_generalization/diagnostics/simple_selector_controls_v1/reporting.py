"""Completed-matrix reporting for simple selection controls.

All prediction vectors are retained. Seeds are averaged within physical cases
before paired statistics; displayed one-pass budget references are never counted
as independent executed trials. This module does not acquire or diagnose data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from cgx.provenance import atomic_json, file_sha256, hash_json, now, read_json
from structured_evidence_v1.replay import prediction_metrics
from structured_evidence_v1.reporting import aggregate as existing_aggregate
from structured_evidence_v1.reporting import timeout_annotations

METHODS = {
    'full_once_100': 'One Full Execution',
    'full_once_100_cal2': 'One Full Execution + Calibration',
    'random_legal': 'Random Legal Checks',
    'coverage_forward': 'Fixed Graph Coverage',
    'fixed_template_validation': 'Validation-Selected Fixed Plan',
    'interface_rule_se': 'Hand-Written Graph Rule',
    'gnn_fixed_plan': 'GNN Plan-Once',
    'gnn_adaptive': 'GNN Replan-Each-Round',
    'gnn_depth2': 'GNN 2-Step Planner',
    'cost_aware_adaptation': 'GNN Cost-Aware One-Step',
    'transformer_adaptive': 'Transformer Replan-Each-Round',
    'full_recording_feasible': 'Four-Execution Full Recording',
}
SIMPLE = tuple(list(METHODS)[:6])
ONE_PASS = ('full_once_100', 'full_once_100_cal2')
LEARNED = ('gnn_fixed_plan', 'gnn_adaptive', 'gnn_depth2',
           'cost_aware_adaptation', 'transformer_adaptive')
SUITES = ('test_connections', 'test_seen', 'test_size')
CASES = {'test_connections': 1600, 'test_seen': 800, 'test_size': 800}
KEY = ['suite', 'configuration_id', 'case_id', 'method', 'execution_cap', 'policy_seed']
PRIMARY = tuple((learned, simple) for learned in ('gnn_adaptive', 'transformer_adaptive')
                for simple in ('fixed_template_validation', 'interface_rule_se'))
RESOURCES = ('bytes_used', 'executions_used', 'calibrations_used', 'decisions_used',
             'selection_seconds', 'inference_seconds', 'end_to_end_seconds',
             'completed_read_seconds', 'distinct_port_coverage')
PAIR_METRICS = ('accuracy', 'nll_bits', 'healthy_false_alarm_rate', 'fault_miss_rate',
                'fault_joint_accuracy', *RESOURCES, 'full_trace_used')
SEED = 671917
REPLICATES = 2000


def _missing(value):
    return value is None or isinstance(value, float) and np.isnan(value)


def _container(value, default):
    if _missing(value):
        return default
    return json.loads(value) if isinstance(value, str) else value


def _plain(value):
    """Parquet list/struct columns may be NumPy arrays/scalars after decoding."""
    if isinstance(value, np.ndarray):
        return [_plain(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(x) for x in value]
    return value


def key(row):
    return tuple(None if _missing(row.get(k)) else
                 int(row[k]) if k in ('policy_seed', 'execution_cap') else row[k]
                 for k in KEY)


def normalize(frame, mode, origin):
    """Use the frozen vector as the authority, including prior fallbacks."""
    records = []
    for source in frame.to_dict('records'):
        row = dict(source)
        prediction = row.pop('prediction')
        row.update(prediction_metrics(prediction, row['true_target_key']))
        row.update(timeout_annotations(row.get('errors'), row.get('timeout', False)))
        row.update(result_mode=mode, origin=origin,
                   policy_seed=None if _missing(row.get('policy_seed')) else int(row['policy_seed']),
                   is_cap_reference=False, evaluated_execution_cap=int(row['execution_cap']),
                   fault_family='healthy' if row['true_target_key'] == 'healthy'
                   else row['true_target_key'].split(':', 1)[1])
        row['row_key'] = hash_json(list(key(row)))
        row['underlying_trial_key'] = row.get('execution_source_key') or row['row_key']
        for field in RESOURCES:
            if field not in row:
                row[field] = np.nan
        if 'full_trace_used' not in row:
            row['full_trace_used'] = np.nan
        if 'completed_read_seconds' not in source:
            row['completed_read_seconds'] = np.nan
        if 'sample_rate_counts' not in row or _missing(row['sample_rate_counts']):
            row['sample_rate_counts'] = dict(Counter(str(int(x)) for x in
                _container(row.get('sample_rates'), [])))
        for field, default in [('sample_rates', []), ('sample_rate_counts', {}), ('chosen_action_keys', []),
                               ('action_family_counts', {})]:
            row[field] = json.dumps(_plain(_container(row.get(field), default)), sort_keys=True)
        for field in ('reused_rollout', 'completion_reused_probability',
                      ):
            row[field] = bool(row.get(field, False))
        # Old imported rows lack a reliable stage-specific flag; unknown is not
        # silently converted into evidence that no early termination occurred.
        row['earlier_termination_affected_acquisition'] = row.get(
            'earlier_termination_affected_acquisition', np.nan)
        for field, source_field in (
                ('extension_actual_new_robot_attempts', 'new_robot_execution_attempts'),
                ('extension_actual_cache_hits', 'physical_cache_hits'),
                ('extension_actual_new_physical_seconds', 'actual_new_physical_seconds')):
            row[field] = 0. if origin == 'imported' or row['reused_rollout'] else row.get(source_field, np.nan)
        records.append(row)
    result = pd.DataFrame(records)
    if result.duplicated(KEY).any():
        raise ValueError('Repeated method/case/cap/seed row')
    return result


def _validate_matrix(frame, suites, origin):
    """Check exact physical-case support and seed sets, not just row totals."""
    methods = SIMPLE if origin == 'new' else (*LEARNED, 'full_recording_feasible')
    expected = set()
    for suite in suites:
        case_rows = frame[frame.suite == suite][['configuration_id', 'case_id', 'fault_family']].drop_duplicates()
        if len(case_rows) != CASES[suite]:
            raise ValueError(f'{suite}: case denominator mismatch')
        if set(case_rows.fault_family) != {'healthy', 'cache_stale', 'queue_hold', 'adapter_scale'}:
            raise ValueError('Fault-family support changed')
        for method in methods:
            caps = (1,) if method in ONE_PASS else (4,) if method == 'full_recording_feasible' else (1, 2, 4)
            seeds = (11, 23, 37) if method in LEARNED or method == 'random_legal' else (None,)
            for row in case_rows.itertuples(index=False):
                for cap in caps:
                    for seed in seeds:
                        expected.add((suite, row.configuration_id, row.case_id, method, cap, seed))
    actual = {key(row) for row in frame.to_dict('records')}
    if actual != expected or len(frame) != len(expected):
        raise ValueError(f'Incomplete {origin} matrix: actual={len(actual)}, expected={len(expected)}')
    return expected


def cap_views(frame, verified=False):
    pieces = [frame]
    for method in ONE_PASS:
        source = frame[(frame.method == method) & (frame.execution_cap == 1)]
        if source.empty:
            continue
        row_verified = ('one_pass_cap_views_verified' in source and
                        source.one_pass_cap_views_verified.fillna(False).all())
        if not (verified or row_verified):
            raise ValueError('One-pass cap views require execution/identity/semantics verification')
        if (source.executions_used.dropna() > 1).any():
            raise ValueError('One-pass row consumed multiple executions')
        for cap in (2, 4):
            view = source.copy()
            view['execution_cap'] = cap
            view['is_cap_reference'] = True
            view['cap_reference_note'] = 'Same B1 trial feasible at a larger cap; no new execution or independent trial'
            pieces.append(view)
    return pd.concat(pieces, ignore_index=True)


def summary(frame, groups):
    result = []
    for values, part in frame.groupby(groups, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(groups, values), **existing_aggregate(part))
        for field in tuple(row):
            if field.startswith('sum_known_'):
                row['historical_or_canonical_reference_' + field] = row.pop(field)
        row['compute_accounting_scope'] = ('Inherited compute timings/counts describe the original canonical '
            'rollout; imported and reused aliases are not fresh work. Global actual work uses the durable ledger.')
        row.update(public_method=METHODS[part.method.iloc[0]],
                   reference_view_rows=int(part.is_cap_reference.sum()),
                   unique_underlying_trials=part.underlying_trial_key.nunique(),
                   full_trace_known_rows=int(part.full_trace_used.notna().sum()),
                   full_trace_count=int(part.full_trace_used.fillna(False).astype(bool).sum()),
                   full_trace_fraction=part.full_trace_used.mean(),
                   mean_distinct_port_coverage=part.distinct_port_coverage.mean(),
                   mean_completed_read_seconds=part.completed_read_seconds.mean(),
                   completed_read_seconds_known_rows=int(part.completed_read_seconds.notna().sum()),
                   reused_rollout_rows=int(part.reused_rollout.sum()),
                   completion_reused_probability_rows=int(part.completion_reused_probability.sum()),
                   earlier_termination_rows=int(part.earlier_termination_affected_acquisition.fillna(False).sum()),
                   earlier_termination_known_rows=int(part.earlier_termination_affected_acquisition.notna().sum()))
        for field in ('extension_actual_new_robot_attempts', 'extension_actual_cache_hits',
                      'extension_actual_new_physical_seconds'):
            row['sum_'+field] = part[field].sum(min_count=1)
            row[field+'_known_rows'] = int(part[field].notna().sum())
        if 'completion_status' in part:
            row['completed_read_unresolved_rows'] = int(part.completion_status.isin(
                ('failed', 'unresolved', 'unreplayable', 'watchdog_timeout', 'unresolved_kept_online')).sum())
        result.append(row)
    return pd.DataFrame(result)


def metric_frame(frame):
    result = frame.copy()
    result['accuracy'] = result.correct.astype(float)
    result['healthy_false_alarm_rate'] = result.healthy_false_alarm.where(result.is_healthy)
    result['fault_miss_rate'] = result.fault_miss.where(~result.is_healthy)
    result['fault_joint_accuracy'] = result.joint_fault_correct.where(~result.is_healthy)
    for column in PAIR_METRICS:
        result[column] = pd.to_numeric(result[column], errors='coerce')
    return result


def paired(left, right, repetitions=REPLICATES, seed=SEED):
    """Right minus left; average each method's seeds independently within case."""
    groups = ['configuration_id', 'case_id', 'fault_family']
    a = metric_frame(left).groupby(groups)[list(PAIR_METRICS)].mean()
    b = metric_frame(right).groupby(groups)[list(PAIR_METRICS)].mean()
    if not a.index.equals(b.index):
        raise ValueError('Paired methods do not contain identical physical cases')
    delta = b - a
    configurations = sorted(delta.index.get_level_values(0).unique())
    rng = np.random.default_rng(seed)
    choice = rng.integers(0, len(configurations), (repetitions, len(configurations)))
    draws = np.full((repetitions, len(configurations), len(PAIR_METRICS)), np.nan)
    points = []
    for ci, config in enumerate(configurations):
        part = delta.xs(config)
        points.append(part.mean().to_numpy())
        rr, slots = np.where(choice == ci)
        total = np.zeros((len(rr), len(PAIR_METRICS)))
        count = np.zeros_like(total)
        for _, family in part.groupby(level='fault_family'):
            values = family.to_numpy(dtype=float)
            indices = rng.integers(0, len(values), (len(rr), len(values)))
            samples = values[indices]
            total += np.nansum(samples, axis=1)
            count += np.isfinite(samples).sum(axis=1)
        draws[rr, slots] = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        sample = np.nanmean(draws, axis=1)
        point = np.nanmean(points, axis=0)
    rows = []
    for i, metric in enumerate(PAIR_METRICS):
        values = sample[:, i]
        valid = values[np.isfinite(values)]
        quantiles = np.quantile(valid, (.025, .975, .05, .95)) if len(valid) else [np.nan] * 4
        rows.append(dict(metric=metric, difference=float(point[i]), ci_low=quantiles[0],
                         ci_high=quantiles[1], one_sided_95_lower=quantiles[2],
                         one_sided_95_upper=quantiles[3], physical_cases=len(delta),
                         configurations=len(configurations), known_paired_cases=int(delta[metric].notna().sum()),
                         repetitions=repetitions, bootstrap_seed=seed,
                         within_case_seed_average=True, random_and_training_seeds_not_paired_by_index=True,
                         interval_scope='Individual descriptive paired intervals; no simultaneous significance claim'))
    return rows


def within_tolerances(accuracy_lower, miss_upper):
    return bool(accuracy_lower >= -.01 and miss_upper <= .01)


def _comparison_job(job):
    metadata, left, right = job
    intervals = paired(left, right)
    configuration_rows = []
    if metadata['execution_cap'] == 4 and (metadata['learned_method'], metadata['simple_method']) in PRIMARY:
        keys = ['configuration_id', 'case_id', 'fault_family']
        aa = metric_frame(left).groupby(keys)[list(PAIR_METRICS)].mean()
        bb = metric_frame(right).groupby(keys)[list(PAIR_METRICS)].mean()
        for config, delta in (bb - aa).groupby(level=0):
            configuration_rows.append(dict(metadata, configuration_id=config,
                physical_cases=len(delta), **{m: delta[m].mean() for m in PAIR_METRICS}))
    return metadata, intervals, configuration_rows


def comparisons(views, workers=1):
    differences, practical, configuration_rows, jobs = [], [], [], []
    # Workers need scalars only, never raw histories or full probability vectors.
    columns = ['configuration_id', 'case_id', 'fault_family', 'is_healthy', 'correct',
               'healthy_false_alarm', 'fault_miss', 'joint_fault_correct',
               'nll_bits', 'full_trace_used', *RESOURCES]
    for mode in ('online', 'completed_read'):
        for suite in sorted(views.suite.unique()):
            for cap in (1, 2, 4):
                part = views[(views.result_mode == mode) & (views.suite == suite) & (views.execution_cap == cap)]
                for learned in ('gnn_adaptive', 'transformer_adaptive'):
                    left = part.loc[part.method == learned, columns]
                    for simple in SIMPLE:
                        right = part.loc[part.method == simple, columns]
                        principal = suite == 'test_connections' and cap == 4 and (learned, simple) in PRIMARY
                        metadata = dict(result_mode=mode, suite=suite, execution_cap=cap,
                                        learned_method=learned, simple_method=simple,
                                        principal_comparison=principal, direction='simple_minus_learned')
                        jobs.append((metadata, left, right))
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) if workers > 1 else None
    try:
        results = pool.map(_comparison_job, jobs, chunksize=1) if pool else map(_comparison_job, jobs)
        for metadata, intervals, configs in results:
            differences.extend(dict(metadata, **row) for row in intervals)
            configuration_rows.extend(configs)
            by_metric = {row['metric']: row for row in intervals}
            acc, miss = by_metric['accuracy'], by_metric['fault_miss_rate']
            comparable = within_tolerances(acc['one_sided_95_lower'], miss['one_sided_95_upper'])
            practical.append(dict(metadata, accuracy_tolerance=.01, miss_tolerance=.01,
                accuracy_difference=acc['difference'], accuracy_one_sided_lower=acc['one_sided_95_lower'],
                miss_difference=miss['difference'], miss_one_sided_upper=miss['one_sided_95_upper'],
                accuracy_within_tolerance=acc['one_sided_95_lower'] >= -.01,
                miss_within_tolerance=miss['one_sided_95_upper'] <= .01,
                comparable_within_declared_tolerances=bool(comparable),
                scope='Individual 95% bounds, not joint or simultaneous safety guarantees'))
    finally:
        if pool:
            pool.shutdown()
    return pd.DataFrame(differences), pd.DataFrame(practical), pd.DataFrame(configuration_rows)


def action_usage(frame):
    rows = []
    for values, part in frame.groupby(['origin', 'suite', 'method', 'execution_cap'], dropna=False):
        metadata = dict(zip(['origin', 'suite', 'method', 'execution_cap'], values))
        families, rates = {}, {}
        for row in part.to_dict('records'):
            for name, count in _container(row.get('action_family_counts'), {}).items():
                if not _missing(count):
                    families[name] = families.get(name, 0) + int(count)
            for rate, count in _container(row.get('sample_rate_counts'), {}).items():
                label = str(rate)
                if not _missing(count):
                    rates[label] = rates.get(label, 0) + int(count)
        for kind, items in [('action_family', families), ('sample_rate', rates)]:
            total = sum(items.values())
            for name, count in sorted(items.items()):
                rows.append(dict(metadata, category=kind, value=name, occurrences=count,
                    occurrence_denominator=total, fraction=count / total if total else np.nan,
                    planned_rows=len(part), physical_cases=part.case_id.nunique(),
                    uniformity_note='Random samples uniformly over distinct affordable non-STOP action keys, not families'))
        rows.append(dict(metadata, category='full_trace_per_case', value='any',
            occurrences=int(part.full_trace_used.fillna(False).astype(bool).sum()),
            occurrence_denominator=int(part.full_trace_used.notna().sum()),
            fraction=part.full_trace_used.mean(), planned_rows=len(part), physical_cases=part.case_id.nunique()))
    return pd.DataFrame(rows)


def _rate(row, numerator, denominator):
    n, d = int(row[numerator]), int(row[denominator])
    return f'{n}/{d}（{100*n/d:.2f}%）' if d else f'{n}/{d}（无对应案例）'


def rank_groups(table, metric, higher_is_better):
    ordered = table.sort_values(metric, ascending=not higher_is_better)
    groups, anchor = [], None
    for row in ordered.itertuples():
        value = float(getattr(row, metric))
        if anchor is None or abs(value-anchor) > 1e-12:
            groups.append([])
            anchor = value
        groups[-1].append(row.method)
    return tuple(tuple(sorted(group)) for group in groups)


def findings(run, out, table, paired_rows, practical, phase):
    settings = read_json(run / 'validation/selected_settings.json')
    def cell(mode, method, cap, suite='test_connections'):
        selected = table[(table.result_mode == mode) & (table.method == method) &
                         (table.execution_cap == cap) & (table.suite == suite)]
        if len(selected) != 1:
            raise ValueError('Findings table cell missing or duplicated')
        return selected.iloc[0]
    def costs(row):
        return (f"平均追加执行 {row.mean_executions_used:.3f} 次、记录 {row.mean_bytes_used/1e6:.3f} MB、"
                f"校准 {row.mean_calibrations_used:.3f} 次")
    lines = ['# 简单检查规则与学习选择器：已完成的对照结果',
        '所有方法使用同一个已经拟合并冻结的 Structured Evidence 诊断器。简单方法没有训练选择网络，但整个诊断流程并非没有学习。原 Panda/MuJoCo 任务、案例、动作菜单、物理随机流和预算保持不变；本轮没有重训网络或重新拟合诊断器。',
        'online 表遵守原两秒选择/诊断时限。completed_read 表只完成同一批已保存证据的最终诊断，不补采数据，也不恢复因早期超时而放弃的动作。每个错误、超时、弃权和未解决复读都保留在分母中。',
        '主测试的确定性方法每格为 1,600 个案例（400 健康、1,200 故障）；随机与学习方法每格为 4,800 次种子运行，仍只有同一批 1,600 个物理仿真案例。统计先在各方法内部对同案例的三个种子取平均，再对配置、配置内故障类别分层案例做 2,000 次配对 bootstrap。Random seed 11 与训练 seed 11 不被当作特殊配对。',
        '这些是已经研究过的测试配置上的直接扩展，不是从未查看过的新最终测试集。以下区间为单项比较的描述性区间；不作四项比较同时显著的声明。1 个百分点准确率和漏检容差在新测试结果前固定，非劣结论仅对应单项单侧 95% 界，不是部署安全保证。',
        '## 1. 一次完整记录能解决多少主测试案例？']
    for method in ONE_PASS:
        for mode in ('online', 'completed_read'):
            row = cell(mode, method, 1)
            lines.append(f"- {METHODS[method]} / {mode}：正确 {_rate(row,'correct_count','rows')}；健康误报 {_rate(row,'false_alarm_count','healthy_rows')}；故障漏检 {_rate(row,'miss_count','fault_rows')}；NLL {row.nll_bits:.6f} bit；{costs(row)}。")
    lines.append('这两个方法只运行 B=1 条件。图表中的 B=2/4 是同一次执行在更大预算下仍可行的引用，不是新的独立试验；校准最多选择两个合法时钟对，并非校准全部时钟。')
    lines.append('## 2. Random 表现如何，多久会选择完整记录？')
    for cap in (1, 2, 4):
        for mode in ('online', 'completed_read'):
            row = cell(mode, 'random_legal', cap)
            lines.append(f"- B={cap} / {mode}：正确 {_rate(row,'correct_count','rows')}；NLL {row.nll_bits:.6f} bit；选择过 full_trace {int(row.full_trace_count)}/{int(row.full_trace_known_rows)}；{costs(row)}。")
    lines.append('Random 对当前可负担的非 STOP 动作键均匀抽样，包括 full_trace、不同采样率、校准与重复测量；这不等于对动作类别均匀抽样。完整类别/采样率频率见 action_family_and_full_trace_usage.csv。')
    lines += ['## 3. 验证集选出了哪些固定模板？',
              '全部 512 个原验证案例在 B=1/2/4 上比较预先指定的八个模板和三个停止选项；按配置平衡的部署 NLL、字节、执行、校准和固定选项顺序打破平局。测试案例不参与选择。',
              '```json', json.dumps(settings, ensure_ascii=False, indent=2), '```',
              '若所选模板与 mandatory baseline 的计划相同，复用同一已验证 rollout，并在记录中保留 identity/alias；这不是额外独立证据。',
              '## 4. 人工图规则能否匹配学习选择器？']
    for mode in ('online', 'completed_read'):
        for method in ('interface_rule_se', 'gnn_adaptive', 'transformer_adaptive'):
            row = cell(mode, method, 4)
            lines.append(f"- {METHODS[method]} B=4 / {mode}：正确 {_rate(row,'correct_count','rows')}；健康误报 {_rate(row,'false_alarm_count','healthy_rows')}；故障漏检 {_rate(row,'miss_count','fault_rows')}；故障联合定位 {_rate(row,'fault_joint_correct_count','fault_rows')}。")
    lines.append('图规则复用既有 interface_rule 的公式，使用当前 SE 诊断、已测接口不一致、覆盖和合法校准信息；只增加验证集选择的停止阈值，并且至少执行一次后才能应用置信停止。代码没有另行增加一个声称存在的传播搜索。')
    lines.append('图规则也保留原实现的同分处理：多个动作分数相同时选目录中排在最前的动作。因此改变节点数组或目录顺序可能改变同分选择；本轮不声称它对任意图重标号或目录重排严格等变，也没有为改善结果另换规则。')
    for row in practical[(practical.suite == 'test_connections') & (practical.execution_cap == 4) &
                         practical.principal_comparison].itertuples():
        result = '达到预先规定的两项容差' if row.comparable_within_declared_tolerances else '未由这两个单侧界共同建立容差内可比'
        lines.append(f"- {row.result_mode}，{METHODS[row.simple_method]} 相对 {METHODS[row.learned_method]}：准确率差 {row.accuracy_difference*100:+.3f} pp，单侧下界 {row.accuracy_one_sided_lower*100:+.3f} pp；漏检差 {row.miss_difference*100:+.3f} pp，单侧上界 {row.miss_one_sided_upper*100:+.3f} pp；{result}。")
    lines.append('## 5. 相似质量下，学习选择器究竟节省什么？')
    for mode in ('online', 'completed_read'):
        for learned, simple in PRIMARY:
            a, b = cell(mode, learned, 4), cell(mode, simple, 4)
            p = practical[(practical.result_mode == mode) & (practical.suite == 'test_connections') &
                          (practical.execution_cap == 4) & (practical.learned_method == learned) & (practical.simple_method == simple)].iloc[0]
            status = '简单规则达到已声明质量容差' if p.comparable_within_declared_tolerances else '质量容差尚未共同建立'
            deltas = {metric: float(b['mean_'+metric] - a['mean_'+metric]) for metric in
                      ('bytes_used', 'executions_used', 'calibrations_used')}
            lines.append(f"- {mode}，{METHODS[learned]} 对 {METHODS[simple]}（{status}）：学习方法 {costs(a)}；简单方法 {costs(b)}。简单减学习：字节 {deltas['bytes_used']/1e6:+.3f} MB，执行 {deltas['executions_used']:+.3f}，校准 {deltas['calibrations_used']:+.3f}；各项配对区间见 paired_differences.csv。")
            lines.append(f"  历史运行的平均选择 / 在线诊断 / 端到端秒数：学习方法 {a.mean_selection_seconds:.4f} / {a.mean_inference_seconds:.4f} / {a.mean_end_to_end_seconds:.4f}；简单方法 {b.mean_selection_seconds:.4f} / {b.mean_inference_seconds:.4f} / {b.mean_end_to_end_seconds:.4f}。完成复读的额外时间另列在表中，未把不同机器负载下的秒数差当作受控速度优势。")
            if p.comparable_within_declared_tolerances and all(v <= 0 for v in deltas.values()):
                lines.append('  在这些容差和这三项实际资源上，此比较未建立训练选择器的收益；不能据此否定共同诊断器的研究。')
    lines.append('字节、执行和校准分别报告，不合成为任意加权总成本。跨运行 wall time 只作描述，因为机器负载、缓存状态和并发不完全匹配；复用预测不被宣称为算法加速。早期停止导致记录不同的情况单独标记。')
    lines.append('## 6. 完成同一记录的诊断后，排序是否保持？')
    for method in (*SIMPLE, *LEARNED, 'full_recording_feasible'):
        a, b = cell('online', method, 4), cell('completed_read', method, 4)
        lines.append(f"- {METHODS[method]}：准确率 {100*a.accuracy:.3f}% → {100*b.accuracy:.3f}%；NLL {a.nll_bits:.6f} → {b.nll_bits:.6f} bit；正式超时 {int(a.timeout_count)}/{int(a.rows)}。")
    for metric, label, higher in [('accuracy', '准确率', True), ('nll_bits', 'NLL', False)]:
        orders = []
        for mode in ('online', 'completed_read'):
            ordered = rank_groups(table[(table.result_mode == mode) & (table.suite == 'test_connections') &
                (table.execution_cap == 4)], metric, higher)
            orders.append(ordered)
            lines.append(f"- {label} / {mode} 从优到劣：" + ' > '.join(
                ' = '.join(METHODS[m] for m in group) for group in ordered) + '。')
        lines.append(f"{label}排序{'保持一致' if orders[0] == orders[1] else '发生变化'}；只将绝对差不超过 1e−12 的值记为并列。这里是观测排序，统计不确定性仍以配对区间为准。")
    lines.append('如果两者只被期限分开，这是当前运行时的结果，不能被解释为完整记录缺少故障信息。completed_read 仍然没有恢复此前放弃的采集。earlier_termination_rows 只标记可继续时发生的初始化、选择、采集或中途诊断终止，不把最终诊断超时算作采集终止；这个标记不证明无故障时必定还会多选一个动作。旧导入行没有可靠细分标记时保留未知，并给出 known_rows。')
    lines.append('## 7. 已见连接和更大系统上的结论是否保持？')
    if phase == 'primary':
        lines.append('本文件是主测试优先交付。已见连接和更大系统尚未纳入该文件，不标记为已运行或已通过；完成后见 FINDINGS_CN.md。')
    else:
        for suite in ('test_seen', 'test_size'):
            lines.append(f'### {suite}')
            for mode in ('online', 'completed_read'):
                for method in (*SIMPLE, 'gnn_adaptive', 'transformer_adaptive'):
                    row = cell(mode, method, 4, suite)
                    lines.append(f"- {METHODS[method]} / {mode}：正确 {_rate(row,'correct_count','rows')}；健康误报 {_rate(row,'false_alarm_count','healthy_rows')}；故障漏检 {_rate(row,'miss_count','fault_rows')}；NLL {row.nll_bits:.6f} bit；{costs(row)}。")
                for learned, simple in PRIMARY:
                    selected = practical[(practical.result_mode == mode) & (practical.execution_cap == 4) &
                        (practical.learned_method == learned) & (practical.simple_method == simple)]
                    source = selected[selected.suite == 'test_connections'].iloc[0]
                    transfer = selected[selected.suite == suite].iloc[0]
                    status = '满足两项容差界' if transfer.comparable_within_declared_tolerances else '未共同建立容差内可比'
                    consistent = '一致' if source.comparable_within_declared_tolerances == transfer.comparable_within_declared_tolerances else '不一致'
                    lines.append(f"- 描述性迁移对照 {mode}，{METHODS[simple]} 相对 {METHODS[learned]}：准确率单侧下界 {transfer.accuracy_one_sided_lower*100:+.3f} pp，漏检单侧上界 {transfer.miss_one_sided_upper*100:+.3f} pp；{status}，与主套件的容差判定{consistent}。这不是额外预声明主要显著性检验。")
    lines.append('## 8. 当前结果支持论文的哪一部分？')
    full = cell('completed_read', 'full_once_100', 1)
    lines.append('此前相同旧轨迹上的诊断器替换与本轮检查选择比较是不同问题。共同诊断器的高准确率不能单独证明学习选择器必要，本轮也不能证明一切可能的非学习规则都失败。')
    if full.accuracy >= .99:
        lines.append(f'一次无需校准的完整记录已正确解决 {int(full.correct_count)}/{int(full.rows)} 个主测试案例。当前动作与字节上限下，是否必须学习选择位置受到较弱检验；可以建议另做预先规定的更紧记录预算研究，但本轮不运行或调节该研究。')
    else:
        lines.append(f'一次完整记录的完成诊断正确率为 {100*full.accuracy:.3f}%，仍需依据具体规则、配对差异与实际资源来判断选择是否有益。')
    principal = practical[(practical.suite == 'test_connections') & (practical.execution_cap == 4) &
                          (practical.result_mode == 'completed_read') & practical.principal_comparison]
    lines.append(f"四项预声明主要比较中，{int(principal.comparable_within_declared_tolerances.sum())}/4 的简单规则分别满足所声明的准确率/漏检容差界。该计数是各单项结论的描述，不是联合显著性声明。第5问逐项列出学习方法是否节省记录、执行或校准；不把诊断改进自动写成选择优势。")
    lines += ['## 文件与局限',
        '所有方法的共同诊断器、模型/代码哈希、原始证据路径及声明见 source_inventory.json、selection_manifest.json、protocol.json。逐案例概率见 reports/normalized_per_case.parquet 和 evaluation/ 下的原始行。所有新旧方法、所有预算和已完成的套件均在表格中保留，不按结果优劣筛选。',
        '报告中的 one-pass cap views 仅是同一记录在更大预算下的可行引用。硬件和缓存不同的 wall time 不是受控速度比较；概率阈值不是正确性保证。大型原始观测和检查点保留在原存储，ZIP中的定位索引不能替代实际测量值。']
    target = run / ('FINDINGS_PRIMARY_CN.md' if phase == 'primary' else 'FINDINGS_CN.md')
    target.write_text('\n\n'.join(lines) + '\n')
    return target


def run_reports(run_dir, phase='all', workers=1):
    run = Path(run_dir)
    marker = read_json(run / 'evaluation' / f'phase_{phase}_complete.json')
    if marker['state'] != 'passed':
        raise ValueError('Requested evaluation phase is incomplete')
    imports = read_json(run / 'evaluation/import_complete.json')
    if imports['state'] != 'passed':
        raise ValueError('Imported learned evidence is incomplete')
    for manifest, field in ((marker, 'outputs_sha256'), (imports, 'output_hashes')):
        for name, digest in manifest[field].items():
            if file_sha256(run / 'evaluation' / name) != digest:
                raise ValueError(f'Evaluation output changed after completion: {name}')
    selection = run / 'selection_manifest.json'
    if read_json(selection)['state'] != 'sealed':
        raise ValueError('Validation selection must be sealed')
    suites = ('test_connections',) if phase == 'primary' else SUITES
    out = run / 'reports' / 'primary' if phase == 'primary' else run / 'reports'
    out.mkdir(parents=True, exist_ok=True)
    frames, inputs = [], {}
    for mode, imported_name in [('online', 'imported_learned_rows'),
                                ('completed_read', 'imported_completed_read')]:
        for origin, name in [('new', f'{phase}_per_case_{mode}'), ('imported', imported_name)]:
            path = run / 'evaluation' / f'{name}.parquet'
            if not path.exists() and phase == 'all' and origin == 'new':
                path = run / 'evaluation' / f'per_case_{mode}.parquet'
            inputs[str(path.relative_to(run))] = file_sha256(path)
            raw = pd.read_parquet(path)
            raw = raw[raw.suite.isin(suites)]
            profile_path = run / 'evaluation/imported_action_profile.parquet'
            if origin == 'imported' and profile_path.exists():
                profile = pd.read_parquet(profile_path)
                join = ['cell_key'] if 'cell_key' in raw and 'cell_key' in profile else KEY
                extra = [c for c in profile if c not in raw or c in join]
                raw = raw.merge(profile[extra], on=join, how='left', validate='one_to_one')
                inputs[str(profile_path.relative_to(run))] = file_sha256(profile_path)
            normalized = normalize(raw, mode, origin)
            _validate_matrix(normalized, suites, origin)
            frames.append(normalized)
    frame = pd.concat(frames, ignore_index=True)
    per_case = out / 'normalized_per_case.parquet'
    frame.to_parquet(per_case, index=False)
    verified = bool(marker.get('one_pass_cap_views_verified', False))
    views = cap_views(frame, verified=verified)
    groups = ['result_mode', 'origin', 'suite', 'method', 'execution_cap']
    table = summary(views, groups)
    table.to_csv(out / 'main_comparison.csv', index=False)
    error_columns = [c for c in table if c in groups or any(s in c for s in
        ('rows', 'count', 'accuracy', 'alarm', 'miss', 'nll', 'brier', 'ece', 'physical_cases'))]
    table[error_columns].to_csv(out / 'errors_with_denominators.csv', index=False)
    resource_columns = [c for c in table if c in groups or c.startswith('mean_') or
                        c.endswith(('_known_rows', '_unknown_rows')) or c in ('rows', 'physical_cases', 'reference_view_rows')]
    table[resource_columns].to_csv(out / 'resources_at_each_budget.csv', index=False)
    summary(views, groups + ['configuration_id']).to_csv(out / 'per_configuration.csv', index=False)
    summary(views, groups + ['policy_seed']).to_csv(out / 'per_seed.csv', index=False)
    action_usage(frame[frame.result_mode == 'online']).to_csv(out / 'action_family_and_full_trace_usage.csv', index=False)
    differences, practical, configs = comparisons(views, workers)
    differences.to_csv(out / 'paired_differences.csv', index=False)
    practical.to_csv(out / 'practical_comparability.csv', index=False)
    configs.to_csv(out / 'per_configuration_paired_differences.csv', index=False)
    work = summary(frame[frame.result_mode == 'online'], ['origin', 'suite', 'method', 'execution_cap'])
    work['timing_scope'] = 'Descriptive cross-run time; imported/cache-reused work is not a speedup claim'
    work.to_csv(out / 'reused_vs_new_work.csv', index=False)
    report = findings(run, out, table, differences, practical, phase)
    outputs = {str(p.relative_to(run)): file_sha256(p) for p in sorted(out.glob('*'))
               if p.is_file() and p.name != 'complete.json'}
    outputs[str(report.relative_to(run))] = file_sha256(report)
    result = dict(state='passed', phase=phase, completed_at=now(),
                  new_executed_rows=len(frame[(frame.origin == 'new') & (frame.result_mode == 'online')]),
                  imported_rows=len(frame[(frame.origin == 'imported') & (frame.result_mode == 'online')]),
                  all_planned_denominators_retained=True, cap_views_are_not_independent_trials=True,
                  primary_comparisons=[list(x) for x in PRIMARY], bootstrap_repetitions=REPLICATES,
                  accuracy_tolerance=.01, fault_miss_tolerance=.01,
                  within_case_seed_averaging=True, simultaneous_significance_claims=False,
                  reporting_workers=workers,
                  input_hashes={**inputs, 'selection_manifest.json': file_sha256(selection)}, output_hashes=outputs)
    atomic_json(out / 'complete.json', result)
    return result


def engineering_checks(output):
    """Small deterministic accounting fixtures, never scientific results."""
    rows = []
    for config in range(2):
        for family in ('healthy', 'cache_stale', 'queue_hold', 'adapter_scale'):
            for case in range(2):
                healthy = family == 'healthy'
                base = dict(configuration_id=str(config), case_id=f'{config}-{family}-{case}',
                    fault_family=family, is_healthy=healthy, correct=1,
                    healthy_false_alarm=0, fault_miss=0, joint_fault_correct=int(not healthy),
                    nll_bits=.1, full_trace_used=1)
                base.update({field: 1. for field in RESOURCES})
                rows.append(base)
    deterministic = pd.DataFrame(rows)
    learned = pd.concat([deterministic.assign(policy_seed=seed) for seed in (11, 23, 37)], ignore_index=True)
    exact = paired(learned, deterministic, repetitions=200)
    assert all(abs(x['difference']) < 1e-12 for x in exact)
    assert all(x['physical_cases'] == 16 for x in exact)
    assert all(abs(x['ci_low']) < 1e-12 and abs(x['ci_high']) < 1e-12 for x in exact)
    changed = deterministic.copy();changed['bytes_used'] = .5;changed['executions_used'] = 2.
    effects = {x['metric']: x for x in paired(learned, changed, repetitions=200)}
    assert effects['bytes_used']['difference'] == -.5 and effects['executions_used']['difference'] == 1.
    assert effects['healthy_false_alarm_rate']['known_paired_cases'] == 4
    assert effects['fault_miss_rate']['known_paired_cases'] == 12
    renamed = learned.assign(policy_seed=np.resize([103, 7, 88], len(learned)))
    assert [x['difference'] for x in paired(renamed, deterministic, repetitions=200)] == [x['difference'] for x in exact]
    assert within_tolerances(-.01, .01)
    assert within_tolerances(.02, -.02)
    assert not within_tolerances(-.010001, -.02)
    assert not within_tolerances(.02, .010001)
    assert not within_tolerances(np.nan, .0)
    assert not within_tolerances(.0, np.nan)
    assert json.dumps(_plain(np.array([np.int64(10), np.int64(100)]))) == '[10, 100]'
    result = dict(state='passed', synthetic_engineering_only=True, physical_executions=0,
        checks=['independent_within_case_seed_average', 'no_4800_vs_1600_independent_bootstrap',
                'conditional_healthy_fault_denominators', 'signed_separate_resource_differences',
                'seed_numbers_not_cross_method_pairing', 'identical_methods_zero_interval',
                'noninferiority_bound_directions_and_inclusive_boundaries', 'unknown_is_not_comparable',
                'parquet_numpy_arrays_are_serializable'],
        implementation_sha256=file_sha256(__file__))
    atomic_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--phase', choices=('primary', 'all'), default='all')
    parser.add_argument('--engineering-checks', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    result = engineering_checks(Path(args.run_dir) / 'tests/reporting_engineering.json') if args.engineering_checks else run_reports(args.run_dir, args.phase, args.workers)
    print(json.dumps(result, indent=2), flush=True)
