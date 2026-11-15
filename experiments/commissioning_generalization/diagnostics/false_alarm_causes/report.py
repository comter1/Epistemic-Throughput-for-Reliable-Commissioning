"""Write the Chinese evidence report from completed, paired result tables."""
from pathlib import Path
import argparse
import json
import pandas as pd

NAMES={'gnn_adaptive':'GNN Replan-Each-Round','gnn_depth2':'GNN 2-Step Planner',
       'gnn_fixed_plan':'GNN Plan-Once','transformer_adaptive':'Transformer Replan-Each-Round'}
FOCUS=['gnn_adaptive','gnn_depth2']


def pct(value):return f'{100*value:.2f}%'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def run(run):
    run=Path(run);effects=pd.read_csv(run/'paired_effects.csv')
    fresh=pd.read_parquet(run/'fresh_intervention_outcomes.parquet')
    assert len(fresh)==1536 and set(fresh.arm)=={'A','S'}
    def effect(phase,method,contrast,metric,subgroup):
        rows=effects[(effects.phase==phase)&(effects.method_key==method)&(effects.contrast==contrast)&
                     (effects.metric==metric)&(effects.subgroup==subgroup)]
        assert len(rows)==1,(phase,method,contrast,metric,subgroup)
        return rows.iloc[0]
    fresh_rows=[];cost_rows=[];opening=[];remaining=[]
    for method in FOCUS:
        fa=effect('confirmation',method,'S-A','healthy_false_alarm','healthy')
        miss=effect('confirmation',method,'S-A','fault_miss','faulty')
        joint=effect('confirmation',method,'S-A','correct_joint','faulty')
        nll=effect('confirmation',method,'S-A','final_nll_bits','faulty')
        opening.append(f"{NAMES[method]} 的健康误报从 {pct(fa.baseline)} 变为 {pct(fa.intervention)}，"
                       f"差 {100*fa.difference:+.2f} 个百分点；真正故障漏检从 {pct(miss.baseline)} 变为 {pct(miss.intervention)}。")
        remaining.append(f"{NAMES[method]} 停止后仍有 {int(fa.cases_with_any_remaining_positive)}/64 个健康案例至少一个seed误报（{int(fa.remaining_positive_seed_rows)}/192条seed运行）")
        for metric,label,subgroup in [('healthy_false_alarm','健康误报','healthy'),('fault_miss','故障漏检','faulty'),('correct_joint','故障联合定位正确','faulty')]:
            e=effect('confirmation',method,'S-A',metric,subgroup)
            fresh_rows.append([NAMES[method],label,pct(e.baseline),pct(e.intervention),
                f'{100*e.difference:+.2f}',f'[{100*e.paired_ci_low:+.2f}, {100*e.paired_ci_high:+.2f}]'])
        for metric,label,divisor in [('executions_used','追加机器人执行',1),('calibrations_used','校准次数',1),('bytes_used','记录MB（十进制）',1e6)]:
            e=effect('confirmation',method,'S-A',metric,'all')
            cost_rows.append([NAMES[method],label,f'{e.baseline/divisor:.3f}',f'{e.intervention/divisor:.3f}'])
    discovery=pd.read_parquet(run/'intervention_outcomes.parquet');discovery_rows=[]
    for method in FOCUS:
        for arm in 'ABCD':
            part=discovery[(discovery.method_key==method)&(discovery.arm==arm)]
            health=part[part.is_healthy];fault=part[~part.is_healthy]
            discovery_rows.append([NAMES[method],arm,pct(health.healthy_false_alarm.mean()),pct(fault.fault_miss.mean()),
                pct(fault.correct_joint.mean()),f'{fault.final_nll_bits.mean():.3f}',f'{part.calibrations_used.mean():.3f}'])
    reread=pd.read_csv(run/'fixed_transcript_reread.csv');reread_rows=[]
    for method in NAMES:
        for reader in ['main_gnn','independent_transformer']:
            part=reread[(reread.method_key==method)&(reread.diagnostic_reader==reader)]
            if part.empty:continue
            h=part[part.true_target_key=='healthy'];f=part[part.true_target_key!='healthy']
            reread_rows.append([NAMES[method],reader,pct(h.original_false_alarm.mean()),pct(h.reread_false_alarm.mean()),
                pct(f.original_missed_fault.mean()),pct(f.reread_missed_fault.mean())])
    branch=pd.read_csv(run/'action_value_checks.csv');branch=branch[branch.policy_selected]
    branch_rows=[]
    for (method,healthy),part in branch.groupby(['method_key','is_healthy']):
        means=part.groupby('evidence_prefix_hash').matched_horizon_sample_gain_bits.mean()
        branch_rows.append([NAMES[method],'健康' if healthy else '故障',len(means),
            f'{part.predicted_q.mean():.3f}',f'{means.mean():.3f}',f'{int((means<0).sum())}/{len(means)}'])
    q=pd.read_csv(run/'q_input_action_checks.csv')
    q_summary=q.groupby('method_key').action_changed.mean()
    ledger=json.loads((run/'resource_ledger.json').read_text())['counts']
    failed=int((discovery.state!='passed').sum()+(fresh.state!='passed').sum())
    text=f'''# CGX-FA-CAUSE-V1：健康误报原因调查

当前已经通过单项改变检查的环节是**何时停止继续取证**：保持冻结模型、B=4 的公开资源信息和原来的首段路线，只在第一次追加机器人执行完成诊断后停止。{' '.join(opening)}

这是新任务与噪声下的配对结果，具体区间见下表。没有发现一个可以归罪于消息配对、时钟单位或旧概率泄漏的实现错误；也没有把所有剩余误报解释完。改变停止环节验证的是当前冻结模型组合继续调查的实际后果，并未单独证明平均池化、计数特征或训练分布中的哪一项是根本原因。

## 1. 新案例确认：单一、非预知的停止规则

128 个新物理案例，16 个原未见连接配置，每配置4健康＋4故障；三个原策略种子先在同案例内平均，再对配置等权。每方法64个健康、64个故障案例；种子不是额外独立物理案例。没有改阈值或重训。A 是取消原2秒限制、保留240秒调用防挂死上限的完整原策略；S 是首个追加执行已得到诊断即STOP。

S 使用新生成 A 轨迹的真实共同前缀：首个执行前，A与S的模型、输入、动作规则和随机流完全相同。停止位置只由“执行次数达到1”决定，不看标签、置信度、是否误诊或后来记录。若A在首个追加执行前已STOP，S保留这个更早终点；768条S中23条执行0次，其余745条执行1次。这是可精确配对的公开停止规则，**没有独立重采样一次S，也没有虚构S的独立墙钟时间**。新增取证成本仍按前缀实际费用报告。选择规则在任何fresh仿真之前封存于 `manifests/confirmation_selection.json`；该停止对照是在发现阶段提出的，不是原V2主协议中的另一种已训练方法。

差值均为 S−A；区间为2000次“配置→配置内案例”的配对bootstrap，单位为百分点。

{table(['选择模型；主读者均为GNN','指标','A','S','差值pp','95%配对区间pp'],fresh_rows)}

{table(['选择模型','平均实际成本（所有新案例）','A','S'],cost_rows)}

完整逐配置、逐种子和逐案例结果见 `confirmation_per_configuration.csv`、`confirmation_per_seed.csv`、`fresh_case_confirmation.csv`。故障漏检的区间用于判断剩余不确定性；观察到不升不等于证明任何部署条件下都不会上升。

{'；'.join(remaining)}。本次改动没有消除全部误报。

## 2. 原数据明确排除了什么

全部147,200条原主评价核对通过。B=1/B=4四个原健康误报数字完全复现。5,952条B=4轨迹的原记录均按哈希成功重建，共34,334个实际前缀、18,266个唯一公开历史；两个冻结读者输出68,668行，缺失0。原STOP/超时轨迹在实际终点结束，没有补造未来记录。

同一个主GNN完整重读后，四个方法的最终健康误报和故障漏检均不变。只有1条原轨迹存在实际已得记录比有效输出多1个observation，补读也未改变其argmax。因此，原2秒限制确实存在，但“已取得记录没来得及读”不能解释这里的健康误报增加。完整不限时A仍可能继续取得更多记录，须与同记录重读分开；配对分解见 `runtime_attribution.csv`。

{table(['选择模型','同记录读者','原健康误报','重读健康误报','原故障漏检','重读故障漏检'],reread_rows)}

独立Transformer降低了部分路线的健康误报，同时提高故障漏检，并降低这批故障案例的联合定位表现。不能把换读者叫作无代价修复。主GNN健康平均 q(healthy) 也不是全程单调下降：多数案例可以更自信，同时少数案例严重误诊；不能用平均概率或信息论口号替代逐案例指标。

在同一初始记录下，64个发现案例×2模型×3种子的B=1/B=4候选评分检查中，合法动作集合没有改变；首次动作改变为adaptive1/192、depth2 2/192。全5,952对原路线另有完整比较，不能把B=1终点直接当作B=4的第一执行前缀。

## 3. 同模型四组成对干预

64个发现案例（32健康、32故障）×2模型×3种子×4组=1,536次调查，全部保留。
A读取最新记录/q；B选择器保留初始记录/q，但仍使用真实剩余资源、已执行action keys及合法动作；C是A加事先按公开拓扑规则预付最多2次校准；D是B加同一校准计划。所有组最终GNN均读取实际完整记录。B/D不是另一个训练好的GNN Plan-Once。

{table(['选择模型','组','健康误报','故障漏检','故障联合定位正确','故障NLL bits','平均校准次数'],discovery_rows)}

B改变了路线，并把两个方法的健康误报都降至8.33%，代价是故障漏检都升至25%。C对adaptive没有降低误报，对depth2只有较小下降。给同一批普通记录额外补实际带噪校准，adaptive误报31.25%→31.25%，depth2 21.875%→20.833%；相应漏检16.667%→16.667%、14.583%→15.625%。超出原预算的增补已逐条标记，不能算同预算胜利。全部差值、交互项(C−A)−(D−B)及区间均在 `paired_effects.csv`。

仅改变显式q、保持同一原始前缀和合法动作集合，下一动作改变比例：adaptive {pct(q_summary['gnn_adaptive'])}、depth2 {pct(q_summary['gnn_depth2'])}。对全部384条轨迹各取一个预先按hash选出的可行动前缀，未遍历全部前缀；全A轨迹另保存疑似区域触及频率与合法动作基准，full-trace不算针对性追查。这证明q会影响选择，却不足以断言健康误报全由“追着错误怀疑走”造成。B的漏检代价及缺少fresh B/q-only确认必须一并保留。

## 4. 继续动作的评分与记录读法

32个固定hash前缀，两个方法各16个，覆盖全部16配置，健康/故障各半。每前缀最多4动作、3次新测量噪声，共384个分支结果。下表先在同一状态的3次噪声重复上求平均；STOP累计额外收益是0。depth2匹配至多2次剩余决策，adaptive匹配实际剩余决策，并采用同一冻结继续策略。

{table(['选择模型','真值组','前缀数','选中动作平均预测Q','匹配决策范围的平均实际收益bits','继续平均收益为负的前缀'],branch_rows)}

这里的实际收益是给定真实案例的样本log-score变化，不是对所有潜在故障状态积分的精确期望Q；训练Q奖励来自OOF读者，而表中是final GNN部署收益。不能用单个负收益证明策略事前已经知道应该STOP。它提供的是继续价值偏乐观的具体样本证据；新的统一STOP对照才检验实际效果。

58项输入路径检查全部通过：token只在各observation内匹配，分箱是每端口每observation最多256箱，旧块不因新块重分箱，时钟差方向/单位、有限值、padding、目标映射和图重排均已核对。主诊断器不消费预算向量，也不读先前q；策略网络才显式接收q。块顺序反转不改变argmax。

复制已取得首执行块则会改变输出：seed11的32健康/32故障中，adaptive误报12/32→4/32而漏检3/32→6/32；depth2误报8/32→4/32而漏检5/32→6/32。复制同时改变序列平均权重、计数，以及块内若有校准时的相关表示；没有新增测量。这是读法敏感性证据，不是正式部署改进，也不能把效果全归给一个池化算子。未实施或宣称单项代码修复，故无 `confirmed_fixes.diff`。

## 5. 训练侧线索及结论边界

只读审计180,843条train/validation transitions、36,864条轨迹；预算、终止、历史链和32,768条训练轨迹奖励相加均通过。160,829条训练reward的读者hash全部匹配对应OOF fold，配置留出关系正确。STOP训练transition仅28条；32,768条完整训练轨迹中32,744条用满2次校准。不过同执行次数的训练前缀确实包含0/1次校准，不能称“模型从没见过”。发现样本里误报案例的平均校准数甚至并非更低。

OOF与final在相同训练/验证历史上的动作收益排序有差别：train8组里5组、validation4组里1组翻转；其中若干接近数值tie，详见分辨率表。未重训，也未用测试真值选择更有利的读者。这些仍是机制线索，尚未被单项训练改动证明为误报原因。

已经确认/检查的是冻结组合的停止环节，以及读者对多次执行记录的敏感性；没有得到支持的是“单纯补读超时遗漏就能解决”“多补两对校准就能解决全部误报”“已有明确消息或时钟代码bug”。剩余的最小下一步是单独分离计数特征与序列池化权重，或在严格source-only条件下检查继续价值目标与部署读者的差异，再做独立确认；本次没有用重训掩盖现象。

## 6. 完整性与资源

发现/新案例结果行失败合计{failed}，预选分母完整。CPU/GPU FP32核对包括22,493个可与原CPU分布直接对齐的主GNN前缀，argmax改变0；最大概率差约1.764e-6，另对接近并列的前缀进行了单线程CPU复核。原权重、协议、阈值、结果以及任务开始前的工作区字节级保护检查见 `provenance/final_integrity_verification.json`。

本次账本：新增机器人执行尝试{ledger['robot_execution']:,}，校准尝试{ledger['calibration']:,}，模型forward {ledger['model_forward']:,}，特征构建{ledger['feature_build']:,}，后两项合计{ledger['model_forward']+ledger['feature_build']:,}/120,000，机器人上限18,000。缓存只复用已经实际取得且动作/序号/随机流一致的记录；每臂逻辑费用照扣。GPU/CPU细分及未测边界见 `RESOURCE_USAGE.md`，未把同步等待时间冒称GPU kernel时间。

运算使用2379/2982/2404三个远端节点，共7张B200；本机用于协调和代码整理。全部命令、代码、清单、6个不同物理案例及原记录、图和逐案例表随包提供。所有结论限于这些冻结模型、系统范围和条件指标，不是精确ET、Bayes最优或一般性的“自适应无效”结论。
'''
    (run/'FINDINGS_CN.md').write_text(text)
    (run/'README_CN.md').write_text('''# GPT Pro 分析包：CGX-FA-CAUSE-V1

先读 `FINDINGS_CN.md`，再核对 `paired_effects.csv` 与 `fresh_case_confirmation.csv`。原始任务全文为 `REQUEST.md`。

本包包含完整双读者逐前缀预测、四组发现干预、128新案例A/S配对确认、同记录校准、动作分支与q输入检查、训练/输入审计、图、六个原始案例、可重现代码及清单。

关键口径：健康误报是argmax不是healthy；种子先在物理案例内平均，配置等权。A/S确认中S是A真实首段的非预知STOP规则，不是独立重新执行；S没有独立墙钟时间。原主结果未改，新增不限2秒参考与原CPU时限结果分开。

建议审阅：新案例STOP效果与漏检区间；同记录读者和改变取证路线的区别；B减少误报但增加漏检；匹配决策范围分支的样本收益与精确期望Q的区别；平均池化/计数的机制尚未单独分离。请不要把这些证据升级成已定位单一实现bug或已证明所有自适应调查有害。

文件哈希见 `SHA256SUMS` / `PACKAGE_MANIFEST.json`。checkpoint张量、网格资产、全体原始记录和大缓存不随包重复发送；原始根目录和哈希见INPUT_MANIFEST，六个案例的真实公共history在case_examples。新干预的完整原始Store仍在原运行目录public/；每臂trace和全部统计预测已打包。
''')
    print('FINDINGS_CN.md and README_CN.md generated from completed tables',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();run(a.run_dir)
