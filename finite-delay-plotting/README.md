# 从原始实验ZIP复现汇总图

这份脚本就是本次报告的数据处理和绘图代码。它读取你运行实验产生的逐案例CSV，重新计算统计量并绘制两张报告图，不需要重新运行实验，不需要GPU，也不执行ZIP中的实验代码。

已用 `complete_20261111_234532_for_review.zip` 验证：合并统计、配对差值、种子均值、DP表和恢复的NLL，与之前报告的数据一致（数值容差1e-12）；两张图已重新生成并检查。不同字体、Matplotlib版本和浮点环境可能产生细微像素差异，不代表数据变化。

## 1. 最简单的使用方法

将原始结果ZIP放在本文件夹的上一层：

```text
工作目录/
├── complete_20261111_234532_for_review.zip
└── finite-delay-plotting/
    ├── analyze_results.py
    ├── requirements.txt
    ├── requirements-tested.txt
    ├── verification_environment.json
    └── README.md
```

进入 `finite-delay-plotting` 文件夹，使用Python 3.10或更新版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python analyze_results.py \
  --input ../complete_20261111_234532_for_review.zip \
  --out figures
```

运行完成后打开 `figures` 文件夹。输出目录必须为空或尚不存在，防止覆盖旧分析。如果想重画，换成 `--out figures_v2`。

本次验证环境是Python3.12.7、NumPy1.26.4、pandas2.3.3、Matplotlib3.10.6。需要使用相同依赖版本时，可在单独的Python3.12环境中安装 `requirements-tested.txt`。没有必要更改原实验环境。

## 2. 在原UCloud服务器上运行

上传并解压本代码包到 `/work/finite` 后，原结果打包文件如果仍位于此前的位置，可以执行：

```bash
cd /work/finite/finite-delay-plotting
source /work/finite/.venv/bin/activate
python -m pip install -r requirements.txt

python analyze_results.py \
  --input /work/finite/results/complete_20261111_234532_for_review.zip \
  --out /work/finite/results/report_figures
```

如果ZIP位置不同，只需要修改 `--input`。本程序只是后处理，不提交新的模拟实验。

## 3. 输出哪些文件

| 文件 | 用途 |
|---|---|
| results_overview.png | Markdown中的四面板结果总览 |
| results_overview.svg | 相同总览的矢量版，放大清晰 |
| all_conditions.png | Markdown中的九条件比较图 |
| all_conditions.svg | 九条件比较图矢量版 |
| pooled_summary.csv | 合并后的均值、区间、成本、轮数和种子范围 |
| paired_effects.csv | 策略间的案例内配对差值和区间 |
| seed_means.csv | 每个种子各自的均值 |
| dp_reference.csv | 原始DP参考表副本；输入有DP时生成 |
| recovered_nll.csv | 从初始模型概率和已保存收益恢复的样本NLL |
| initial_entropy.csv | 初始模型的精确条件熵与样本初始NLL |
| integrity_checks.csv | 每组原始统计核对结果 |
| run_configs.json | 原始实验配置汇总 |
| archive_audit.txt | 输入ZIP哈希与核对摘要 |
| analysis_details.txt | 详细数值日志 |
| analysis_environment.json | 本次分析环境、输入及脚本哈希 |

PNG适合直接插入幻灯片和网站；SVG适合矢量展示。脚本无交互窗口，适用于服务器。

## 4. 统计和绘图分别在哪里

`analyze_results.py` 中：

- `validate_design()`：检查是否为九条件、每条件具有相同种子集合，且同条件跨种子模型参数一致。
- `analyze()`：读取ZIP；核对原有统计；合并案例并重新计算区间；计算配对差值；恢复初始模型信息。
- `make_plots()`：读取新生成的统计CSV，绘制四面板与九面板图，输出PNG和SVG。
- `main()`：处理输入输出路径、保存日志和环境。

统计方法：

```text
每条件每策略预算点：合并不同seed的案例
均值 = 所有案例收益之和 / 案例数
标准误 = 案例收益样本标准差 / sqrt(案例数)
95%区间 = 均值 ± 1.96 × 标准误

配对比较：
先在同一seed、case_id、budget内计算 A收益 − B收益
再对这些差值计算均值与标准误
```

没有把三个置信区间取平均，也没有把不同预算下的同一案例当作独立案例。区间是逐点近似区间，未作多重比较校正。

`recovered_nll.csv` 的初始概率由已知有限模型独立计算；最终样本NLL=初始样本NLL−信息增益均值。它不是重新执行仿真，也不是新的模型训练。

## 5. 使用范围

这份布局针对当前两轮、预算0–6、最大bundle长度4的九条件扫描。可使用其他种子编号或相同配置下不同样本量，但各组样本量应相同、条件与种子集合应完整。

不要把多次运行、多个时域或重复seed混进同一输入ZIP。只有一个DP参考文件时原样导出；若有多个DP文件会要求你明确区分范围，不静默合并。

不要用 `python -O` 执行，否则完整性断言会被禁用；脚本会拒绝这种方式。

图中的所有收益、误差棒和成本点都来自输入数据。代码里的固定数字主要是实验布局（预算0–6、两轮）和统计方法（1.96），没有把报告中的0.631等结果硬写成绘图数据。

## 6. 更新此前Markdown中的图片

此前文章包的图片名用了连字符，而绘图脚本输出使用下划线。对应关系：

```text
figures/results_overview.png  → 文章的 assets/results-overview.png
figures/all_conditions.png   → 文章的 assets/all-conditions.png
```

复制并改名即可更新正文图片。也可以不改文件名，直接修改Markdown的图片链接。

## 7. 向老师说明的复现流程

> 原始实验由benchmark.py和sweep.py运行，保存逐案例结果。后处理由analyze_results.py读取原始结果ZIP，在同一条件内合并三个种子的案例，计算信息增益和配对置信区间，并生成汇总图。绘图过程中不增加模拟数据、不修改实验结果。
