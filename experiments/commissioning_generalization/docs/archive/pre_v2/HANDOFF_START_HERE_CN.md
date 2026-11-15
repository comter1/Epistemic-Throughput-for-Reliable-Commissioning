# 小飒主实验：交给 Codex 的实施包

**版本：CGX-1.0 / 2026-11-13。**

这是正式主实验的实施协议，不是再开展一轮判断题目是否值得做的探索性实验。
目标是在已有系统配置上训练取证策略，冻结权重，检验它在未见连接上的表现。
硬件按同一台服务器上的 **2 块 NVIDIA B200 GPU** 配置；不使用 LLM API，不训练机器人控制器。

## 给 Codex 的任务

先读根目录的 `CODEX_TASK.md`，再读
`overlay/experiments/commissioning_generalization/AGENTS.md` 和 `docs/00_PROTOCOL.md`。
让 Codex 在**你现有的小飒代码仓库**中创建：

- 分支：`feat/commissioning-generalization-main`
- 目录：`experiments/commissioning_generalization/`

本包的 `overlay/` 是复制到仓库中的文件。不要覆盖仓库根目录的 AGENTS.md。
不要改动原来的 `finite_delay_benchmark_complete/`、`finite-delay-plotting/` 和 `result_review/`。
已有有限模型用于回归检查，不替代新的 MuJoCo 主实验。

## 可以直接粘贴给 Codex

```text
请解压并阅读这个实施包，从 CODEX_TASK.md 开始。在当前小飒仓库中创建
feat/commissioning-generalization-main 分支，只在
experiments/commissioning_generalization/ 下扩展主实验，保留原有实验不变。
严格按照本包的协议、配置、数据接口和验收清单实现完整管线。
先完成工程正确性、数据隔离和环境检查，然后启动正式训练/测试任务；
不要停在重新建议一个 toy/pilot 实验，也不要等结果有利才继续。
服务器有两块 B200，优先一块卡一个任务并行运行，不要默认 DDP。
正式实验是否成功，以未见系统上的实际结果为准，不预设 GNN 一定赢。
每个阶段保存真实命令、日志、配置/模型/数据哈希、失败原因和运行状态。
不得把未实现的 CLI、未运行的测试或未算出的最优值写成已完成。
```

## 主实验已经固定的内容

1. 128 个训练配置、24 个验证配置；5 个测试组，每组 20 个配置。
2. 训练配置 8–16 个实际执行组件；规模测试 24–32 个组件。
3. 训练每配置 256 个案例，验证/测试每配置 128 个案例；健康及三类故障各占 25%。
4. 目标是“健康”或“哪个组件发生哪种故障”，不是只有四类的固定输出。
5. GNN 与 Transformer 等共享诊断器、动作清单、数据和预算。测试不重训。
6. 主条件最多追加 4 次物理仿真运行、2 次校准操作；最多 6 次决策（校准可单独执行）。
7. 观测字节预算为统一四次全速记录容量的 0%、10%、25%、50%、100%。
8. 学习对照 5 组、每组 5 个训练种子；核心比较以未见连接为主。
9. 主仿真报告 log-loss、定位、成本及修复复测；只有原有限模型可称精确信息增益。

“4 次执行”和“6 次决策”是专门澄清后的两个量，不再混用 round：
只测时钟不需要重跑机械臂，但会消耗校准与决策额度；一次组合动作可以同时消耗一轮执行和一次校准。

## 文件导航

| 文件 | 内容 |
|---|---|
| `CODEX_TASK.md` | 分支、范围、执行顺序、完成条件 |
| `docs/00_PROTOCOL.md` | 主假设、证据边界、哪些结论允许报告 |
| `docs/01_SYSTEM_AND_FAULTS.md` | 实际执行的软件链、控制器、故障、校准公式 |
| `docs/02_GRAPHS_AND_SPLITS.md` | 图生成、未见连接、时钟/实现测试、标签与数据隔离 |
| `docs/03_OBSERVATIONS_AND_BUDGETS.md` | 观测按钮、采样、字节成本、轮数、停止 |
| `docs/04_MODELS_AND_LEARNING.md` | q、Q、GRU、GNN、Transformer、Bellman 更新 |
| `docs/05_DATA_AND_EVALUATION.md` | 训练数据、基线、统计、修复与隐藏复测 |
| `docs/06_IMPLEMENTATION_AND_RUNBOOK.md` | 目录、接口、必须实现的 CLI、运行顺序 |
| `docs/07_COMPUTE_2XB200.md` | 双卡排程、CPU/存储、工作量和吞吐估算 |
| `docs/08_ACCEPTANCE_TESTS.md` | 防泄漏、物理故障、预算、RL、泛化的验收测试 |
| `configs/protocol.json` | 机器可读主配置，数值的唯一权威来源 |
| `schemas/`、`examples/` | JSON 接口定义与示例（示例不是正式数据） |
| `contracts/interfaces.py` | 不依赖 MuJoCo 的 Python 类型接口草案 |
| `tools/` | 可直接运行的完整性、配置和环境检查工具 |
| `bootstrap_branch.sh` | 安全创建分支/复制实施包，默认只预览 |

## 交付状态

本包是**方案、配置、接口和实施辅助工具**；没有声称已实现新的 MuJoCo 环境、训练好 GNN，或在你的 B200 上运行。
已经检查的内容见 `BUNDLE_VALIDATION.md`。所有代码文件都有明确用途，不把脚手架假装成完成的实验。

原仓库快照已经实际查看。已知的旧文件及哈希见 `repository_snapshot.json`。
新目录中的编程验收和小批量冒烟检查是防止错误长跑的工程步骤，不是“结果不好就不做主实验”的筛选。
