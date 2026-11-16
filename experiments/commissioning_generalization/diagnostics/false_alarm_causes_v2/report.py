"""Generate Chinese result explanations from audited, completed V2 tables."""
from pathlib import Path
import argparse
import json
import pandas as pd
from cgx.provenance import read_json
from .ledger import Ledger

NAMES={'gnn_adaptive':'GNN Replan-Each-Round','gnn_depth2':'GNN 2-Step Planner'}
ARMS={'A0':'原选择器与原诊断器','S1':'第一次追加执行后停止','R1':'完整 A0 记录，用 R* 重读',
      'G1':'按源验证预案固定的继续收益门控','RG1':'G1 的同一前缀，用 R* 重读',
      'R-live':'R* 同时供给选择器和最终诊断','RG-live':'R-live 路径上的收益门控'}


def count(value,total):return f'{int(value)}/{int(total)}（{100*value/total:.2f}%）' if total else '不适用'


def prose(parts):
    """Keep adjacent Markdown table rows contiguous, paragraphs separated."""
    text='';previous_table=False
    for part in parts:
        if not part.strip():continue
        table=part.lstrip().startswith('|')
        if text:text+='\n' if table and previous_table else '\n\n'
        text+=part;previous_table=table
    return text+'\n'


def counts_table(data):
    lines=['| 选择模型 | 比较 | 健康误报 | 故障判健康 | 类型和位置均正确 |',
           '|---|---|---:|---:|---:|']
    for method in NAMES:
        for arm in ARMS:
            p=data[(data.method_key==method)&(data.arm==arm)].set_index('metric')
            if p.empty:continue
            values=[count(p.loc[k].numerator,p.loc[k].denominator) for k in ('healthy_false_alarm','fault_miss','correct_joint')]
            lines.append('| '+' | '.join([NAMES[method],arm]+values)+' |')
    return '\n'.join(lines)


def arm_counts(data,method,arm):
    p=data[(data.method_key==method)&(data.arm==arm)].set_index('metric')
    return '；'.join(label+count(p.loc[key].numerator,p.loc[key].denominator)
        for key,label in [('healthy_false_alarm','误报 '),('fault_miss','漏检 '),('correct_joint','类型+位置正确 ')])


def cause_table(counts,calibration,four,live_enabled):
    """The protocol's causal-evidence checklist, filled with observed outcomes."""
    q=[]
    for method in NAMES:
        r=calibration[(calibration.method_key==method)&(calibration.grouping=='method_key')&
            (calibration.teacher=='final')&calibration.positive_q_only.eq(True)].iloc[0]
        q.append(f'{NAMES[method]} {int(r.positive_q_negative_return_count)}/{int(r.positive_q_prefixes)} 个原正 Q 前缀实际 FINAL 收益为负')
    label=four[(four.phase=='fresh_replay')&(four.contrast=='mean_FINAL_label-minus-OOF_label')&
        (four.metric=='final_nll_bits')&(four.subgroup=='all')]
    label_text='；'.join(f'{NAMES[r.method_key]} 的 FINAL 标签 NLL 主效应 {r.difference:+.4f} bit'
        for _,r in label.iterrows())
    route='未运行 live，不能从 final-only 推断路线效果'
    if live_enabled:
        route='；'.join(f'{NAMES[m]} 的 R-live：{arm_counts(counts,m,"R-live")}' for m in NAMES)
    rows=[
        ('重复端口的平均权重', 'V1 首次追加执行正确的健康子组，两方法 P-only 均 0/171 新误报；R-P 源验证误报 27/96，高于原始 26/96，未通过选择', '输出确实依赖 P；当前证据不支持 P 是该健康子组主要致错路径。复制结果混合 C/E 不能单独证明 P'),
        ('记录计数字段', '同一子组 C-only 新误报 49/171、44/171；完整真实变化 51/171、31/171。新确认 R-C：'+ '；'.join(f'{NAMES[m]} {arm_counts(counts,m,"R1")}' for m in NAMES), '支持 C 对冻结网络的因果输入影响及源选择规则的新案例改善；不等于记录越多必然越差'),
        ('时间戳/校准边汇总', '同一子组 E-only 新误报 3/171、4/171；C+E 为 51/171、31/171；八格重构误差为 0', 'E 与 C 存在相互作用，不能把全部变化都归入平均或计数'),
        ('继续动作收益高估', '；'.join(q), '支持匹配实际路径上的分数偏差；G1 的新案例结果检验可预测性，但 G1 未优于 S1'),
        ('训练/部署 reader 不一致', label_text+'；两个主效应的 95% 区间均跨 0，完整条件效应和区间见下表及 CSV', '本轮匹配小头的标签效应点估计改善，但总体效应仍不确定；不能仅由 T11 对旧模型的比较推出原 RL 训练是唯一原因'),
        ('STOP 样本稀少直接致错', '两种实际选择器终端 STOP score/loss/gradient 均为 0；固定继续样本归一化时追加 STOP 的梯度差为 0', '当前不支持直接归因；28 条 STOP 记录不足以证明原因，也未执行无信息的 STOP 过采样训练'),
        ('中途反馈改变检查路线', route, 'live 在相同封存案例上改变中间 q 并实际重新采集；须与 R1 一起读，不能从一个错误轨迹宣称所有反馈有害')]
    return '\n'.join(['| 提出的解释 | 本轮实际证据 | 结论及不足以支持的推断 |','|---|---|---|']+
        ['| '+' | '.join(r)+' |' for r in rows])


def effect_table(data,contrasts,metrics):
    lines=['| 选择模型 | 配对比较 | 指标 | 差值与 95% 区间 |','|---|---|---|---:|']
    labels={'healthy_false_alarm':'健康误报','fault_miss':'故障判健康','correct_joint':'故障类型+位置正确',
            'final_nll_bits':'真实目标 log loss','executions_used':'追加执行次数'}
    for _,r in data.iterrows():
        if r.contrast not in contrasts or (r.metric,r.subgroup) not in metrics:continue
        rate=r.metric in ('healthy_false_alarm','fault_miss','correct_joint');scale=100 if rate else 1
        unit=' 个百分点' if rate else (' bit' if r.metric=='final_nll_bits' else ' 次')
        value=f'{scale*r.difference:+.3f} [{scale*r.paired_ci_low:+.3f}, {scale*r.paired_ci_high:+.3f}]{unit}'
        lines.append(f'| {NAMES[r.method_key]} | {r.contrast} | {labels.get(r.metric,r.metric)} | {value} |')
    return '\n'.join(lines)


def write_resources(run):
    ledger=Ledger(run).snapshot();c=ledger['counts'];success=ledger['successful']
    combined=sum(c[k] for k in ('model_forward','feature_build','reader_partial_forward'))
    text=['# V2 资源使用与实际计数','',
        ('初期按附件使用 `ssh.cloud.sdu.dk:2404` 的两张 B200；随后用户明确授权使用全部可用节点，分布式 dispatch 的具体端口/GPU/窗口见 hardware_authorization_override.json 与 dispatch manifest。2379 的空闲 B200 纳入加速；2982 两卡有其他用户作业，保持其运行。本机只协调与编辑。'
         if (run/'manifests/hardware_authorization_override.json').exists() else '计算节点为 `ssh.cloud.sdu.dk:2404`，仅 GPU 0/1 两张 B200；每卡最多一个独立任务。本机只协调与编辑。'),
        'CPU 使用远程节点：A 特征构造最多 24 workers，每个物理 rollout job 最多 32 workers；各节点总 workers 不超过其 CPU quota，BLAS 每进程单线程。','',
        '| 项目 | 实际尝试 | 成功/备注 | 硬上限 |','|---|---:|---|---:|',
        f"| 新物理机器人执行，含 initial | {c['robot_execution']} | {success['robot_execution']} 成功 | 16000 |",
        f"| 校准 | {c['calibration']} | {success['calibration']} 成功 | 10000 |",
        f"| 完整逐历史 forward | {c['model_forward']} | 按 logical item，不按 batch API 次数 | — |",
        f"| 原始特征构造 | {c['feature_build']} | 缓存复用不重复构造 | — |",
        f"| 另列 partial forward | {c['reader_partial_forward']} | 本轮 hybrid 也保守计入 forward | — |",
        f'| forward + 原始特征合计 | {combined} | 工程检查和失败尝试也计入 | 200000 |',
        f"| 小型评分头 fits | {c['head_fit']} | 每头 2000 updates，总 {c['head_update']} | 24 |",
        f"| 评分头训练 GPU 占用 | {c['training_gpu_seconds']/3600:.4f} GPU 小时 | 所有 fit 的实测占卡时间相加 | 48 GPU 小时 |",
        '| 原诊断器/选择器骨干重训 | 0 | 原权重逐哈希保护 | 0 |',
        f"| 显式记录的缓存命中 | {c['cache_hit']} | 未包含批内/内存 reader 去重；不减方法逻辑采集成本 | — |",'',
        '停止组复用实际已有前缀，因此其执行数、bytes、calibration 是合法逻辑资源量；没有给它们伪造独立物理 wall-clock。',
        'source/fresh/live 的 timing JSON 给出 coordinator wall time、worker CPU time 和模型调用时间。stage_timings.jsonl 的 process CPU 只覆盖主进程；未测量的 A 子进程 CPU 总量不作推算。','',
        f"已测量 GPU 阶段（包括冻结推理与数据准备）的墙钟时间相加为 {read_json(run/'tests/final_audit.json')['measured_gpu_stage_allocation_upper_seconds']/3600:.3f} GPU 小时。另把从 run 建立到最终审计的全部时长都按 {read_json(run/'tests/final_audit.json')['conservative_reservation_card_count']} 卡持续占用计算，严格保守上界为 {read_json(run/'tests/final_audit.json')['conservative_gpu_reservation_bound_seconds']/3600:.3f} GPU 小时，也低于 48 小时。小头优化占用与这些数不能重复相加。批量逐历史调用仍逐项计入 200000 上限。",'',
        '完整原子账本见 resource_ledger.json；工程 profile 与预估见 reader_workload_forecast.json、source_workload_forecast.json 和 head_data 的训练速率文件。预估不是实际耗时。']
    (run/'RESOURCE_USAGE.md').write_text('\n'.join(text)+'\n')


def run(run_dir):
    run=Path(run_dir)
    assert read_json(run/'tests/final_audit.json')['state']=='passed'
    assert read_json(run/'tests/confirmation_statistics.json')['state']=='passed'
    counts=pd.read_csv(run/'confirmation_counts_and_rates.csv')
    paired=pd.read_csv(run/'paired_effects.csv');four=pd.read_csv(run/'correction_2x2_paired_effects.csv')
    source4=pd.read_csv(run/'correction_2x2_source_validation.csv')
    calibration=pd.read_csv(run/'continuing_value_calibration.csv')
    reader=read_json(run/'manifests/source_reader_selection.json')
    choices=read_json(run/'manifests/source_variant_selection.json')
    table=counts_table(counts)
    reader_notes=(run/'reader_mechanism_notes_CN.md').read_text().split('\n',1)[1].replace('\n## ','\n### ')
    reader_notes=reader_notes.replace('新 source-only 候选选择及 fresh 确认仍需独立完成。','source-only 选择与独立 fresh 确认结果见下文。')
    opening='我们改变了读取记录的计数字段及派生观察状态，或“是否继续”的评分头，同时在主比较中保持原诊断器权重、选择器骨干、非 STOP 动作排序和物理噪声机制不变。在另行封存的 64 个不同健康系统案例与 64 个不同故障系统案例中，得到以下计数。每个案例跑三个策略 seed，所以每种方法每组的健康/故障分母各为 192；它们不是 192 台独立系统。'
    if not choices['reader_available']:
        opening=opening.replace('读取记录的计数字段及派生观察状态，或“是否继续”的评分头','“是否继续”的评分头')
        opening+=' 读数规则候选未通过源验证，主确认没有强行加入读数修正。'
    parts=[opening,'',table,'',
        '主比较中，仅改计数读取的 R1 同时降低了两种选择器的健康误报和故障漏检点估计。G1 相比 A0 也改善，但两种选择器都没有优于简单的 S1：Replan 的 G1/S1 误报同为 12/192，漏检为 24/192 对 18/192；2-Step 的 G1/S1 误报为 17/192 对 12/192，漏检同为 18/192。2-Step 的 G1 仍是未通过源验证约束的预定实验候选，不能因 fresh 改善而改称可部署方案。',
        '两条原始完整管线为 **GNN Replan-Each-Round + GNN Diagnoser** 和 **GNN 2-Step Planner + GNN Diagnoser**。R* 是另外标记的冻结权重读数规则，不把它称为未修改的共同诊断器。',
        '\n'.join(f'- **{arm}**：{description}。' for arm,description in ARMS.items() if arm in set(counts.arm)),
        '健康误报按所有合法目标中 argmax 是否为 healthy 判断，不用 p(healthy)<0.5 代替；故障正确要求类型和位置同时正确。',
        '', '确认集故意各含 50% 健康/故障。这里报告条件错误率；不将整体准确率直接与健康占比不同的原主实验比较。新案例来自已经在 V1 检查过的 16 个 held-out configurations，并非全新未检查的拓扑集合。','',
        '## 哪条计算把正确的健康判断推成故障','',
        reader_notes,'',
        '## STOP 梯度与继续收益','',
        '实际代码把 STOP 输出严格遮罩成 0。两个已训练选择器的合法终端 STOP 行均得到零损失、骨干/评分头/辅助头零梯度；追加 STOP 并保持继续样本归一化不变，梯度不变。混合 batch 均值只按继续行占比稀释梯度，测试没有 optimizer step。详情见 stop_gradient_audit.json。',
        '因此，原训练集 STOP 仅 28 行是数据事实，不能据此认定它是根因。本轮训练校正的是继续动作相对于零分 STOP 的收益。','',
        '| 源验证选择模型 | 原正 Q、实际 FINAL return 为负的前缀 | 原 Q 均值 | 实际 FINAL return 均值 |',
        '|---|---:|---:|---:|']
    for method in NAMES:
        p=calibration[(calibration.method_key==method)&(calibration.grouping=='method_key')&(calibration.teacher=='final')&calibration.positive_q_only.eq(True)].iloc[0]
        parts.append(f"| {NAMES[method]} | {int(p.positive_q_negative_return_count)}/{int(p.positive_q_prefixes)} | {p.original_q:+.4f} bit | {p.realized_return:+.4f} bit |")
    parts+=['','上述均值先在 case×seed 内平均前缀，再平均 seed、case、configuration。原 Q 均值高于同口径的实际 FINAL 收益均值，支持这批实际继续路径上存在收益高估；正 Q 后出现一次负回报本身不能证明校准偏差，也不能证明选择器当时可从公开历史知道应该停止。门控在新案例上的结果才检验这种偏差能否被事先预测。',
        '收益标签严格使用同一已获得历史、同一 source 真标签，分别由原 cross-fit OOF reader 和最终部署 GNN 计算；Replan 用实际剩余决策范围，2-Step 最多两步，校准也占决策。详细分组、每家族/执行次数/Q bins、STOP/资源终止均保留。','',
        '## 只用源案例选择，随后封存','',
        '源材料为 256 个新训练案例（16 个原 TRAIN 配置）和 64 个新验证案例（8 个原 VALIDATION 配置），四家族均衡。最终 reader 可能见过 source 配置，但没有见过这些新案例；每个 OOF reader 的原配置排除映射逐项验证。所有 V1 与两次 fresh 的标签均不参与拟合或选择。',
        '24 个独立评分头从各自原头初始化，仅优化这些头，每头 2000 次更新；四格同一配置/案例/前缀支持、初始化、batch 顺序、Huber 损失及 AdamW 设置。0 步仅为原头对照，不参与实际修正选择，这一解释在 source outcome 生成前封存。','',
        '| 格子 | 训练 backbone 的 q 输入 | 收益标签 reader | 部署 q 输入 |','|---|---|---|---|',
        '| T00 | OOF | OOF | FINAL |','| T01 | FINAL | OOF | FINAL |',
        '| T10 | OOF | FINAL | FINAL |','| T11 | FINAL | FINAL | FINAL |','',
        f"读数候选的源验证选择结果：**{reader['selected_variant'] or '无候选通过；R1/RG1 不可作为改善组'}**。原始/R-P/R-C/R-PC 的完整成功或失败结果均在 reader_candidate_validation.csv 及 summary 中。"]
    parts.append('选中的 R-C 把公共计数改为每组件不同已观察端口数、每端口平均 block 长度和缺失比例，保留原 log1p 与派生 observed-status 计算。所有真实测量仍进入原 GRU 平均和原时间戳/校准边；没有把重复检查获得的新测量当成旧记录删掉。')
    for method,choice in choices['selected_heads'].items():
        parts.append(f"{NAMES[method]} 的 G1 固定为 **{choice['cell']}，{choice['step']} 步**；状态 `{choice['status']}`。"+
            ('该候选满足预先规定的源验证点估计约束；这不是统计安全保证。' if choice['guardrails_passed'] else '所有候选均未通过源验证约束，按预案确认 T11@2000；不称其为可部署改善。'))
    rc=pd.read_csv(run/'reader_candidate_validation.csv')
    rsummary=pd.read_csv(run/'reader_candidate_validation_summary.csv').set_index('variant')
    rt=['读数规则选择合并两种方法和三个 seed：源验证有 16 个健康、48 个故障不同物理案例；每个规则的运行分母分别为 96/288。约束在这一预定合并口径上判断，不代表每种方法单独获得安全保证。','',
        '| 读数规则 | 健康误报 | 故障判健康 | 类型+位置正确 | joint NLL | 通过三项约束 |',
        '|---|---:|---:|---:|---:|---|']
    for variant,p in rc.groupby('variant',sort=False):
        h=p[p.is_healthy];f=p[~p.is_healthy];r=rsummary.loc[variant]
        rt.append(f'| {variant} | {count(h.false_alarm.sum(),len(h))} | {count(f.missed_fault.sum(),len(f))} | {count(f.joint_fault_correct.sum(),len(f))} | {r.nll:.4f} bit | {bool(r.passes_guardrails)} |')
    parts.append('\n'.join(rt))
    parts.append('源验证中另有一个跨批量形状的 FP32 数值差：384 条历史里仅 1 条 native-vs-stored 概率误差超过 1e−6，实测最大 1.050716e−6，argmax 全部相同；按原混合 batch 重构该项误差为 0。对跨 batch 单独记录明确的 2e−6 工程上界，所有超 1e−6 项仍按原 batch 验证；A 内部 P/C/E 的 1e−6 重构门槛没有改变。没有删除案例。完整记录见 reader_candidate_validation_notes_CN.md。')
    batch=read_json(run/'tests/reader_fresh_batch_alignment.json')
    parts.append(f"fresh 最终重读另对 {batch['unique_histories']} 条不同实际历史核对了 native 原诊断器与 A0 保存结果：argmax 改变 {batch['argmax_changes']} 条，最大概率差 {batch['max_abs_probability_error']:.7g}；{batch['over_1e_6_count']} 条超过 1e−6，{batch['over_2e_6_count']} 条超过预先记录的跨 batch 2e−6 上界。该检查复用缓存，没有改变规则、选择或预测；完整误差列表见 tests/reader_fresh_batch_alignment.json。")
    metrics=[('healthy_false_alarm','healthy'),('fault_miss','faulty'),('correct_joint','faulty'),('final_nll_bits','all')]
    fresh4=pd.read_csv(run/'correction_2x2_fresh_replay.csv');fresh4=fresh4[fresh4.step==2000]
    matrix=['| 选择模型 | 格子，固定2000步 | 健康误报 | 故障判健康 | 故障类型+位置正确 |',
            '|---|---|---:|---:|---:|']
    for _,r in fresh4.iterrows():
        matrix.append(f'| {NAMES[r.method_key]} | {r.cell} | {count(r.false_alarms,r.healthy_denominator)} | {count(r.missed_faults,r.faulty_denominator)} | {count(r.joint_correct_faults,r.faulty_denominator)} |')
    parts+=['','## 标签 reader 与 q 输入是否各自有帮助','',
        '以下是四格都训练到 2000 步后的 fresh 配对结果。标签主效应在两种 q 训练输入上取均值；q 主效应在两种标签 reader 上取均值。所有格子部署时仍收到 FINAL q。没有根据此表重新选获胜格子。','',
        '\n'.join(matrix),'',
        effect_table(four[four.phase=='fresh_replay'],['T11-T00','mean_FINAL_label-minus-OOF_label','mean_FINAL_q_training-minus-OOF_q_training','(T11-T01)-(T10-T00)'],metrics),'',
        '在固定的两种 q 训练输入下，FINAL 标签相对 OOF 标签的 NLL 点估计都更低；但 fresh 标签主效应的两个 95% 区间均跨 0，不能宣称总体改善已经确定。改变 q 训练输入的 NLL 方向也不一致：Replan 略降，2-Step 上升。完整条件差值保留在 correction_2x2_paired_effects.csv，不能把标签或 q 的 NLL 点估计等同于所有分类错误同时下降。',
        '这些对照分离指定的标签 reader/q 输入选择；四格都改用了原策略实际 continuation 的监督标签。即使 T11 更好，也不能把全部效果归给 STOP 行数量或证明原训练程序的唯一致错原因。若区间包括恶化方向，不能把点估计写成“保证不增加漏检”。','',
        '## 新案例中的停止、重读及组合','',
        effect_table(paired,['G1-A0','R1-A0','RG1-A0','(RG1-G1)-(R1-A0)','G1-S1'],metrics+[('executions_used','all')]),'',
        '区间使用 2000 次分层 bootstrap：先平均同一物理案例的三个 seed，再抽配置、在配置内按健康/故障分层抽案例；各组始终配对。只报告效应大小和区间，不作多重比较显著性宣称。S1 是预定的第一次追加执行后停止，不是事后用真值挑最好前缀。',
        'R1/RG1 只改变最终读数；G1 只决定是否允许原选择器已经提出的动作。人工 P/C/E 张量混合不属于任何可部署组。',
        'fixed-fault 的 anchor 明确分开：A 的真实前缀对照取首次追加执行时最可疑目标，复制记录对照取未复制的完整终点最可疑目标，fresh 表取 A0 完整终点最可疑目标。各自对照内保持该目标身份，不把不同 anchor 的数值混合比较。']
    if choices['optional_live']['enabled']:
        parts+=['','另行封存并运行了 R-live/RG-live：R* 同时改变选择器每一步收到的 q。评分头仍来自原 reader 的 source 收益监督；这里检验联合部署，不能声称其数值对 R* 收益也已校准。',
            'live 并非所有指标都改善：2-Step 的 R-live 误报为 10/192（R1 为 12/192），但故障漏检为 25/192（A0 为 24/192，R1 为 23/192）。因此，最终读数改善不能直接推成中间反馈路线也全面改善。',
            effect_table(paired,['R-live-A0','RG-live-A0'],metrics)]
    else:parts+=['','未启用可选 live 路由分支；这一决定在 fresh outcome 打开前封存。因此不从 final-only 结果推断改变中间诊断反馈后的路线表现。']
    parts+=['','## 各种解释得到什么程度的支持','',cause_table(counts,calibration,four,choices['optional_live']['enabled'])]
    parts+=['','## 剩余错误与未支持的解释','',
        'remaining_errors.csv 逐运行保留 S1/G1/R1/RG1（以及已启用 live）剩余的健康误报、故障漏检、故障类型/位置错误，另标记相对 A0 新增的错误。case_examples 按固定哈希选取可展示的实际轨迹；没有满足条件的正例会明确注明，绝不伪造。',
        '已经定位的是冻结网络的输入路径，以及源监督停止规则在新案例的具体效果。计数字段影响不是单调的“越查越故障”规律；复制早期广覆盖记录甚至可降低误报。固定原故障位置的 log odds 与最可能故障位置的 log odds 也不是同一量。',
        '仍未证明：STOP 稀缺是根因、所有反馈都应关闭、更多独立测量本身必定有害、此次小头监督等价于原 RL 过程的严格因果重训对照、改计数必能修复所有健康错误。若新案例改善有限，应保留失败结果，再区分首次执行前已错误的诊断、计数分布依赖、以及剩余 continuation 收益的条件预测误差；不在确认集上继续调规则。','',
        '## 可复核材料','',
        '输入路径/完整概率/八格/Shapley、source 每个前缀标签、24 头训练记录、四格同迭代结果、fresh 每案例/配置/seed 计数、分层配对区间、原子资源账本、所有原文件 before/after 哈希、测试、命令和代码均在本包。大型原 checkpoint 与全部原始数据留在服务器，以经验证的 hash 和路径引用；本包不伪称包含这些大资产。']
    (run/'FINDINGS_CN.md').write_text(prose(parts))
    executive=[opening,table,
        '实际早期正确健康子组（V1：57 个不同案例，各方法 171 次运行）中，C-only 产生49/171、44/171新增误报；全部真实终点变化为51/171、31/171。该计算归因与可部署纠错的效果分开评价。',
        f"源验证读数选择：{reader['selected_variant'] or '无合格读数修正'}。G1 改善原模型，但两种方法都没有优于简单 S1；2-Step 的 G1 未通过源约束，保留实验候选标签。原 STOP 梯度为零，不能把 STOP 样本少直接说成根因。",effect_table(paired,['G1-A0','G1-S1'],metrics),
        '完整解释见 FINDINGS_CN.md；误报和漏检必须一起看。确认数据未用于 refit 或重新选择。']
    (run/'EXECUTIVE_COUNTS_CN.md').write_text('\n\n'.join(executive)+'\n')
    (run/'METHOD_NAMES.md').write_text('\n'.join(['# 方法命名','',
        '原始 raw keys 不变。主管线为 GNN Replan-Each-Round + GNN Diagnoser、GNN 2-Step Planner + GNN Diagnoser。','',
        '| raw key | 选择证据的方法 |','|---|---|',
        '| gnn_adaptive | GNN Replan-Each-Round |','| gnn_depth2 | GNN 2-Step Planner |',
        '| gnn_fixed_plan | GNN Plan-Once |','| transformer_adaptive | Transformer Replan-Each-Round |',
        '| cost_aware_adaptation | GNN Cost-Aware One-Step |','| full_recording_feasible | Full Recording |','',
        'GNN 2-Step Planner 表示已训练模型的两决策 horizon，不是穷举树搜索。独立 Transformer diagnoser 是另一读数器，不能与选择证据的 Transformer 混用。',
        'R-P/R-C/R-PC 使用原 GNN Diagnoser 权重，但其读数计算经修改，必须单列。']))
    write_resources(run)
    (run/'README.md').write_text('''# CGX_FA_CAUSE_V2

先读 EXECUTIVE_COUNTS_CN.md 的计数，再读 FINDINGS_CN.md 的证据与边界。

本包完成冻结读数路径分解、实际 STOP 梯度检查、320 个新 source 案例、24 个匹配小头训练，以及另行封存的128个新确认案例。原主实验/V1/原权重未改动。原完整管线是 GNN Replan-Each-Round + GNN Diagnoser 和 GNN 2-Step Planner + GNN Diagnoser。

每方法每比较有64个健康和64个故障物理案例，各3seed，条件分母各192。三seed不是三个独立物理系统。新案例使用已在V1看过的16个 held-out 配置；并非全新拓扑测试。

关键表：confirmation_counts_and_rates.csv、paired_effects.csv、correction_2x2_paired_effects.csv、reader_real_endpoint_full_population_counts.csv、continuing_value_calibration.csv。失败候选也完整保留。

manifests/ 包含先于outcome封存的样本、训练规则及source选择；provenance/ 保存原始保护哈希/版本/环境；commands/ 和 code/ 是复现入口，tests/ 是实际测试结果。PACKAGE_MANIFEST.json 和 SHA256SUMS 覆盖归档内文件。LARGE_ASSETS.json 明确列出留在服务器上的大数据与原权重，并不假装这些文件包含在压缩包内。

生成的数据与checkpoint目录：以 source_checks.json 中 original_data_root 和本次 run 路径为准。执行代码以 experiments/commissioning_generalization 为工作目录，PYTHONPATH=.:src；远程环境及精确命令见 provenance/environment_2404.json、commands/。
''')
    print('Chinese result reports generated from completed audited artifacts',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);run(p.parse_args().run_dir)
