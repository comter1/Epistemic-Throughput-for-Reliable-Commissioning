# Section 3.2 实验代码补齐版

基于你提供的 benchmark.py 修改。原始文件没有覆盖。本包只有代码和说明，没有生成的实验结果。
本次交付只做 Python 语法解析检查；没有运行实验，也没有运行附带测试。请自行执行下面的测试与实验。
建议 Python 3.10+。

## 修改内容

- 基础独立噪声模型、225 个隐藏状态、精确后验和式 (24) 保留。
- 新增 `--noise-mode shared_round`，同时实现共享噪声生成和精确联合似然。
- 完整 bundle 固定执行一次，单独报告实际成本；校准成本 1/2/4 分别对应 4/6/10，即使超过预算网格 0..6 也绘制结果点。
- 另有两个局部观测+校准 bundle 的单次成本点。
- 新增校准与 bundle 的 2×2 Greedy 消融，以及同一案例差值的配对近似 95% CI。
- DP 与 Greedy 在完全相同动作库和轮数下精确枚举比较；明确标注参考范围。
- 保存逐案例信息增益、成本和轮数，便于你自己分析。
- sweep 支持噪声、偏移、先验、校准成本、共享噪声、多个 seed，以及可选较长时域。

## 共享噪声的具体假设（需要写入论文）

截图只要求另设共享噪声条件，没有规定共享机制。这里选取一个明确的压力测试条件：
每个 acquisition round 抽一次 e∈{-1,0,1}，概率仍为 (p,1-2p,p)，
同轮 bundle 的所有测量（包括校准）都加同一个 e；不同轮之间、与初始 O0 的噪声之间独立。
隐藏延迟和偏移仍在案例内固定。基础条件则每个测量独立抽噪声。

这不是“论文已经指定的唯一共享噪声模型”。此设置下，同轮 Os-Ds=s、Oq-Dq=q，
所以完整 bundle 可以完全识别 Y。这是该共享机制的数学结果，不能推广为普遍的校准优势。
单项测量策略在两个噪声条件下的边际模型相同；共享机制主要影响同轮 bundle。
跨轮持续共享的噪声不属于此实现，需要扩展隐藏状态并修改后验模型。

## 安装与检查（由你执行）

在解压后的代码目录打开终端：

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_benchmark.py
```

测试覆盖：先验、独立/共享联合似然的独立构造校验、共享噪声消除性质、重复采样、
完整 bundle 成本、预算、同库 DP 与 Greedy、初始单独校准的信息量。

## 运行

先做小样本运行检查（不用于论文结论）：

```bash
python benchmark.py --n 20 --out results/check
```

基础实验和共享噪声分别运行：

```bash
python benchmark.py --n 1000 --seed 42 --out results/base
python benchmark.py --n 1000 --seed 42 --noise-mode shared_round --out results/shared
```

全部逐因素扫描，默认两轮、预算 0..6：

```bash
python sweep.py --n 1000 --seeds 42 43 44 --out results/sweeps
```

小规模 DP 与相同单项动作库下 Greedy 比较：

```bash
python benchmark.py --n 1000 --bundle-size 1 --dp --dp-library single --dp-max-budget 2 --out results/dp_single
```

若需要在默认允许 bundle 的动作库下做小预算 DP：

```bash
python benchmark.py --n 1000 --dp --dp-library matched --dp-max-budget 2 --out results/dp_matched
```

`matched` 只表示对主程序所配置动作库、预算和时域最优，不是无限时域的全局前沿。
DP 是枚举计算，使用浮点运算；外层初始观测也做精确求和，因此 DP 表不附 Monte Carlo CI。
普通策略和 bundle 点仍使用式 (24) 的 Monte Carlo 估计和区间。
不要直接把单项受限 DP 与默认 bundle Greedy 当作同一优化问题。

可选较长时域（先自行确认计算量可承受）：

```bash
python sweep.py --n 1000 --seeds 42 43 44 --rounds 2 3 --out results/horizons
```

大 bundle 的联合结果枚举和长时域 DP 可能很慢。没有默认启用长时域 DP。
`--bundle-size 1` 会改变实验动作库，不能仅作为加速选项而忽略其科学含义。
独立的固定 bundle 成本点仍会生成，且明确不属于单项动作库曲线。

## 输出解释

每个条件产生：

| 文件 | 内容 |
|---|---|
| main_results.csv | 四个原策略及三个 Greedy 消融版本，预算 0..6 |
| bundle_points.csv | 完整和局部 bundle 各执行一次，在实际成本处评估 |
| information_gain_vs_budget.png | 主曲线与独立 bundle 点；完整 bundle 的成本 10 会显示 |
| per_case_results.csv | case_id、策略、预算、真实类别、初始观察、单例增益、成本、轮数 |
| paired_comparisons.csv | 四组消融对比的配对均值差与近似 95% CI |
| config.json | 参数、seed、噪声定义、依赖版本 |
| dp_reference.csv | 可选：DP、同动作库 Greedy 精确期望和差值 |

Greedy 消融四组：greedy（校准+bundle）、greedy_no_calibration（无校准+bundle）、
greedy_single（校准+单项）、greedy_single_no_calibration（无校准+单项）。
均使用同样的先验、案例、初始证据、目标和轮数；比较衡量的是这些冻结 Greedy 策略的表现，
不能解释成各动作库的最优价值差。

同一条件内复用案例和噪声 seed 进行配对。不同条件改变生成模型，不能仅凭相同 case_id 就做跨条件配对。
多个 seed 分目录输出，不自动合并。普通 CI 是均值 ±1.96×标准误，配对 CI 基于案例内差值；
没有进行多重比较校正。n 的选择由你根据区间精度决定，默认值不代表正式统计精度保证。

Repeat 每轮只测一次总延迟，两轮最多两次，所以高预算处可能平台化；这与轮数约束一致。
Greedy 使用当前一步的总期望信息增益，不按成本相除，也不等于 DP。
噪声扫描同时改变测量和校准噪声，沿用原代码对“each noise draw”的解释；
噪声 0.1/0.3、偏移半径 1/3、非均匀先验 0.7/0.1/0.1/0.1 是研究者选择，非截图指定数值。
成本是抽象单位，不代表实际工程时间节省。
