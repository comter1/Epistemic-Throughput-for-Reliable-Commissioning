# 从这里开始：MAIN EXPERIMENT V2

当前正式协议是 CGX-2.0；旧大规模方案和临时单种子 P48 方案已归档。
先读 `MAIN_EXPERIMENT_V2.md`、`../configs/protocol.json` 和 V2 运行手册。
用户已确认 B 表示追加诊断执行次数；字节、校准和时间分别报告。

方法名称和完整解释见 [METHODS.md](METHODS.md)。六种主方法均使用同一个冻结的
GNN 最终诊断器，名称前半部分说明如何采集证据：

- **GNN Plan-Once + GNN Diagnoser**：训练好的 GNN 根据新系统结构、初始证据和预算，
  为该系统一次制定完整检查计划，执行途中不根据新观测调整后续步骤。
- **GNN Replan-Each-Round + GNN Diagnoser**：GNN 每次读入新证据后重新选择下一项检查。
- **Transformer Replan-Each-Round + GNN Diagnoser**：Transformer 每轮选择检查，最终诊断仍由同一个 GNN 完成。
- **GNN 2-Step Planner + GNN Diagnoser**：选择下一项检查时，也考虑再下一项检查的价值；2-Step 指规划时域，不是网络层数。
- **GNN Cost-Aware One-Step + GNN Diagnoser**：GNN 估计下一项检查的诊断价值，选择时同时考虑检查成本。
- **Full Recording + GNN Diagnoser**：按全记录规则采集允许的信号，最终仍使用同一个训练好的 GNN 诊断。

GNN Plan-Once 是学习方法，计划随系统和初始证据变化，并非所有系统共用的手写清单。
独立 Relation-Transformer reader 只用于复读冻结证据，不是主实验最终诊断器。
内部方法 key 保持原样；已封存报告保留旧展示名称，新展示产物使用上述名称。

训练32×128、验证8×64、已见8×100、未见16×100、更大未见8×100。
保留六种主方法、三种子和 B=1/2/4。每个模型最多10万步，每2000步验证，
连续10次无足够改善时停止，采用最佳验证检查点。测试前冻结全部选择。

当前可用三块 B200：本机两块，以及 leiyo@ssh.cloud.sdu.dk:2127 的一块；
一块卡运行一个独立训练任务。数据位于共享的 /home/leiyo/work/cgx_data_v2。
原有数据经过精确匹配和哈希验证后复用，未选入数据不进入 V2 训练索引。
不要改动三个旧实验目录，不要缩成 toy，也不要以工程检查替代正式结果。

从 `../STATUS.md` 和外部 main_run/status.json 确认实际进度。
按 V2 手册继续到全部正式冻结测试、修复审计、四张主图和结果包。
不预设任何方法获胜；没有实际运行的工作不得写为通过或完成。
