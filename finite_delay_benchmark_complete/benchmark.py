"""Section 3.2：有限延迟与时钟偏移基准实验。每段实现都附中文说明。"""
from __future__ import annotations  # 允许类型标注延迟求值。
import argparse  # 用于命令行参数解析。
import csv  # 用于保存实验结果表格。
import itertools  # 用于枚举有限离散状态与观察结果。
import json  # 用于保存实验参数。
from dataclasses import dataclass  # 用于定义清晰的配置和实验案例。
from functools import lru_cache  # 用于缓存重复使用的似然矩阵。
from pathlib import Path  # 用于管理跨平台的文件路径。
from typing import Sequence  # 用于表示动作序列的类型。
import matplotlib  # 导入绘图库。
matplotlib.use("Agg")  # 使用无界面绘图后端，适合服务器或命令行。
import matplotlib.pyplot as plt  # 用于绘制实验曲线。
import numpy as np  # 用于数组、向量化概率计算。

# -----------------------------------------------------------------------------
# 一、实验配置：论文确定的参数与我们选择的实现参数。
# -----------------------------------------------------------------------------
@dataclass(frozen=True)  # 冻结配置对象，避免实验运行过程中被意外修改。
class Config:  # 定义单个实验条件。
    noise_mode: str = "independent"  # shared_round：同轮共享噪声，不同轮独立。
    noise_side_prob: float = 0.2  # 基础噪声：P(-1)=P(+1)=0.2，P(0)=0.6。
    offset_radius: int = 2  # 基础时钟偏移取整数 -2,-1,0,1,2。
    class_prior: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)  # 故障类别先验均匀。
    calibration_cost: int = 2  # 基础校准成本为 2；敏感性扫描还考虑 1、4。
    max_rounds: int = 2  # 初始精确规划只考虑至多两轮观察。

    def validate(self) -> None:  # 检查用户提供的参数是否合理。
        assert self.noise_mode in ("independent", "shared_round")
        assert 0.0 <= self.noise_side_prob <= 0.5, "噪声两侧概率必须在 [0,0.5]。"  # 保证概率有效。
        assert self.offset_radius >= 0, "时钟偏移半径不能为负。"  # 保证离散支持合法。
        assert len(self.class_prior) == 4, "四种故障需要四个先验概率。"  # 保证类别数量一致。
        assert all(x > 0 for x in self.class_prior), "先验概率必须严格大于零。"  # 避免零概率配置。
        assert np.isclose(sum(self.class_prior), 1.0), "先验概率之和必须为 1。"  # 检查先验归一化。
        assert self.calibration_cost >= 1, "校准成本必须为正整数。"  # 检查动作成本。
        assert self.max_rounds >= 1, "最少允许一轮观察。"  # 检查决策时域。


NAMES = ("Healthy", "Stale", "Queued", "Both")  # 固定四种故障类别及其顺序。
MEASURES = ("T", "Os", "Oq", "Ds", "Dq")  # T=重复总延迟，另外四种为分项测量和校准。

@dataclass(frozen=True)  # 单个模拟案例是一个不可修改的数据对象。
class Case:  # 表示仿真器内部真实状态以及给 Agent 的初始观察。
    theta_index: int  # 当前真正抽到的隐藏配置在状态空间里的索引，禁止给策略。
    y: int  # 当前真实故障类别，禁止给策略，评价器可以读取。
    x0: int  # 初始总延迟观察，允许给策略。
    seed: int  # 单个案例的随机种子，用于可复现实验。


class Model:  # 构建有限概率模型、似然函数、采样器和精确贝叶斯推断器。
    def __init__(self, cfg: Config):  # 初始化一个明确的实验条件。
        cfg.validate()  # 确保所有配置合法。
        self.cfg = cfg  # 保存实验参数。
        states = []  # 建立每个合法隐藏配置对应的一行。
        weights = []  # 建立每个配置的先验概率权重。
        labels = []  # 保存每个配置对应的故障类别。
        delays_by_class = (((0, 0),), ((1, 0), (2, 0)), ((0, 1), (0, 2)), ((1, 1), (1, 2), (2, 1), (2, 2)))  # 论文给定的合法延迟组合。
        offsets = range(-cfg.offset_radius, cfg.offset_radius + 1)  # 时钟偏移取值集合。
        offset_count = len(offsets) ** 2  # 两个独立时钟偏移的组合数量。
        for y, delays in enumerate(delays_by_class):  # 先均匀或按自定义先验抽故障类别。
            for s, q in delays:  # 在类别内部均匀选择激活延迟。
                for ds in offsets:  # 枚举 State 观测时钟偏移。
                    for dq in offsets:  # 枚举 Queue 观测时钟偏移。
                        states.append((s, q, ds, dq))  # 添加完整隐藏配置 theta。
                        labels.append(y)  # 记录 theta 对应的故障类别。
                        weights.append(cfg.class_prior[y] / len(delays) / offset_count)  # 遵守先抽类别、再抽延迟的论文先验。
        self.states = np.asarray(states, dtype=np.int16)  # 转换为 shape=(隐藏状态数,4) 的数组。
        self.labels = np.asarray(labels, dtype=np.int8)  # 转换成每个 theta 的类别标签。
        self.prior = np.asarray(weights, dtype=np.float64)  # 转换成数值稳定的浮点先验。
        assert np.isclose(self.prior.sum(), 1.0), "隐藏配置先验不归一。"  # 检查构造正确。
        self.label_matrix = (self.labels[:, None] == np.arange(4)[None, :]).astype(float)  # 用矩阵一次性汇总四类后验。
        self.noise_values = np.array([-1, 0, 1], dtype=np.int16)  # 所有测量的基础噪声支持。
        p = cfg.noise_side_prob  # 获取两侧噪声概率。
        self.noise_probs = np.array([p, 1 - 2 * p, p], dtype=float)  # 对应 -1、0、1 的概率。
        self.base = {  # 将每种测量写为隐藏变量的确定性部分。
            "T": self.states[:, 0] + self.states[:, 1],  # 重复总延迟为 s+q。
            "Os": self.states[:, 0] + self.states[:, 2],  # State 测量为 s+delta_s。
            "Oq": self.states[:, 1] + self.states[:, 3],  # Queue 测量为 q+delta_q。
            "Ds": self.states[:, 2],  # State 校准为 delta_s。
            "Dq": self.states[:, 3],  # Queue 校准为 delta_q。
        }  # 每项最终都要再加一次新抽取的测量噪声。
        self.single_likelihood = {}  # 保存单项测量的所有结果和条件似然。
        for name in MEASURES:  # 遍历五种允许的普通/校准观察。
            low = int(self.base[name].min() - 1)  # 最小可能结果取决于隐藏值和 -1 噪声。
            high = int(self.base[name].max() + 1)  # 最大可能结果取决于隐藏值和 +1 噪声。
            values = np.arange(low, high + 1, dtype=np.int16)  # 枚举全部可能观察值。
            differences = values[:, None] - self.base[name][None, :]  # 构建 结果-隐藏状态 的噪声差矩阵。
            likelihood = np.zeros(differences.shape, dtype=float)  # 初始化 P(观察值|隐藏状态)。
            for noise, probability in zip(self.noise_values, self.noise_probs):  # 遍历三种噪声。
                likelihood += (differences == noise) * probability  # 为匹配该噪声的结果填入概率。
            self.single_likelihood[name] = (values, likelihood)  # 结果维度为行，隐藏 theta 为列。

    def action_cost(self, action: tuple[str, ...]) -> int:  # 计算整组观察动作的标量成本。
        return sum(self.cfg.calibration_cost if x.startswith("D") else 1 for x in action)  # 普通测量 1，校准依配置计价。

    def action_library(self, include_calibration: bool = True, max_bundle_size: int = 4) -> list[tuple[str, ...]]:  # 生成合法的非空观察组合。
        candidates = MEASURES if include_calibration else ("T", "Os", "Oq")  # 可选是否允许时钟校准。
        actions = []  # 保存候选动作。
        for length in range(1, min(max_bundle_size, len(candidates)) + 1):  # 限制组合最多包含多少种测量。
            actions.extend(itertools.combinations(candidates, length))  # 生成所有不重复的同轮测量组合。
        return actions  # 动作可以在后续轮次重复选择，以研究重复测量的价值。

    @lru_cache(maxsize=256)  # 相同 Bundle 的似然矩阵只计算一次。
    def bundle_likelihood(self, action: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:  # 返回所有可能结果与 P(结果|theta)。
        if not action:  # 空动作代表 STOP，不能当作普通观测。
            raise ValueError("STOP 没有测量似然矩阵。")  # 防止错误调用。
        if self.cfg.noise_mode == "shared_round":
            # 同一轮每个分量加同一个 e；枚举精确联合分布，不能乘边际似然。
            bases = np.column_stack([self.base[name] for name in action])
            samples = (bases[:, None, :] + self.noise_values[None, :, None]).reshape(-1, len(action))
            outcomes, inverse = np.unique(samples, axis=0, return_inverse=True)
            joint = np.zeros((len(outcomes), len(self.prior)))
            np.add.at(joint, (inverse, np.repeat(np.arange(len(self.prior)), 3)),
                      np.tile(self.noise_probs, len(self.prior)))
            positive = np.any(joint > 0, axis=1)
            return outcomes[positive], joint[positive]
        values, probs = self.single_likelihood[action[0]]  # 首个测量的结果列表和似然矩阵。
        outcomes = values[:, None]  # 把单测量结果转为二维结果元组。
        joint = probs  # 起始联合似然就是第一个测量的似然。
        for name in action[1:]:  # 依次拼接 Bundle 的其他测量。
            next_values, next_probs = self.single_likelihood[name]  # 获取下一个测量的似然。
            outcomes = np.concatenate((np.repeat(outcomes, len(next_values), axis=0), np.tile(next_values, len(joint))[:, None]), axis=1)  # 枚举笛卡尔积结果。
            joint = (joint[:, None, :] * next_probs[None, :, :]).reshape(-1, len(self.prior))  # 仅在给定 theta 后按独立测量噪声相乘。
        positive = np.any(joint > 0, axis=1)  # 删除对所有 theta 都不可能的结果。
        return outcomes[positive], joint[positive]  # 返回 (可能结果数, Bundle 长度) 与 (可能结果数,隐藏状态数)。

    def posterior(self, prior_or_posterior: np.ndarray, action: tuple[str, ...], observation: Sequence[int]) -> np.ndarray:  # 观测后更新完整  theta 后验。
        outcomes, likelihood = self.bundle_likelihood(action)  # 取出此动作的联合观察似然。
        wanted = np.asarray(observation, dtype=np.int16)  # 将实际观察转成整数数组。
        indices = np.flatnonzero(np.all(outcomes == wanted, axis=1))  # 精确查找对应的结果元组。
        if len(indices) != 1:  # 检查这个观察是否属于公开概率模型支持集。
            raise ValueError(f"非法观察结果：{action} -> {observation}")  # 阻止使用模型之外的数据。
        unnormalized = prior_or_posterior * likelihood[int(indices[0])]  # 贝叶斯分子：先验乘似然。
        normalizer = float(unnormalized.sum())  # 计算总证据概率。
        if normalizer <= 0:  # 禁止对零概率事件做条件化。
            raise ValueError("观察历史在此概率模型下不可能发生。")  # 明确报告配置错误。
        return unnormalized / normalizer  # 返回所有 theta 的精确后验分布。

    def y_posterior(self, theta_posterior: np.ndarray) -> np.ndarray:  # 将完整 theta 分布转为四类故障分布。
        return theta_posterior @ self.label_matrix  # 汇总相同机制类别的概率权重。

    def entropy_y(self, theta_posterior: np.ndarray) -> float:  # 计算当前故障类别的不确定性（bits）。
        probabilities = self.y_posterior(theta_posterior)  # 获取四种机制的边际后验。
        nonzero = probabilities[probabilities > 0]  # 避免计算 log(0)。
        return float(-(nonzero * np.log2(nonzero)).sum())  # 使用 Shannon Entropy 定义。

    def expected_gain(self, theta_posterior: np.ndarray, action: tuple[str, ...]) -> float:  # 计算某 Bundle 的精确条件期望信息增益。
        _, likelihood = self.bundle_likelihood(action)  # 获取所有可能观察及其条件似然。
        numerators = likelihood * theta_posterior[None, :]  # 构造每个观察的未归一化 theta 后验。
        outcome_probs = numerators.sum(axis=1)  # 得到 P(观察结果|当前历史)。
        valid = outcome_probs > 0  # 忽略当前历史下概率为零的观察结果。
        if not np.any(valid):  # 安全检查：至少应有一个可能结果。
            return 0.0  # 理论上不会发生，仅作为边界保护。
        label_masses = numerators[valid] @ self.label_matrix  # 按故障标签求和，得到每个结果的联合权重。
        y_probs = label_masses / outcome_probs[valid, None]  # 计算每个可能结果对应的四类后验。
        log_y = np.zeros_like(y_probs)  # 为 log2(概率) 申请数组。
        np.log2(y_probs, out=log_y, where=y_probs > 0)  # 只在严格正概率的位置求对数。
        conditional_entropies = -(y_probs * log_y).sum(axis=1)  # 每个可能结果出现时的条件熵。
        expected_entropy = float(outcome_probs[valid] @ conditional_entropies)  # 对所有观察结果按概率求平均。
        gain = self.entropy_y(theta_posterior) - expected_entropy  # 信息增益=观察前熵-期望观察后熵。
        return max(0.0, float(gain))  # 消除浮点舍入产生的极小负数。

    def draw_case(self, rng: np.random.Generator) -> Case:  # 按给定先验生成一个独立测试案例。
        theta_index = int(rng.choice(len(self.prior), p=self.prior))  # 根据不等权的  theta 先验采样。
        y = int(self.labels[theta_index])  # 从真实 theta 得到真实故障标签。
        x0 = int(self.base["T"][theta_index] + rng.choice(self.noise_values, p=self.noise_probs))  # 生成初始总延迟测量。
        seed = int(rng.integers(0, 2**31 - 1))  # 为后续观察生成独立随机种子。
        return Case(theta_index, y, x0, seed)  # 将真相保存在 Environment / Evaluator 侧。

    def initial_posterior(self, x0: int) -> np.ndarray:  # Agent 根据公开初始证据计算后验。
        return self.posterior(self.prior, ("T",), (x0,))  # 不使用真实 theta_index，也不使用真实 y。

    def observe(self, case: Case, action: tuple[str, ...], rng: np.random.Generator) -> tuple[int, ...]:  # Environment 生成一次合法观察。
        if self.cfg.noise_mode == "shared_round":
            noise = int(rng.choice(self.noise_values, p=self.noise_probs))
            return tuple(int(self.base[name][case.theta_index] + noise) for name in action)
        return tuple(int(self.base[name][case.theta_index] + rng.choice(self.noise_values, p=self.noise_probs)) for name in action)  # 每个测量使用同一案例固定偏移和新的独立噪声。


# -----------------------------------------------------------------------------
# 二、策略：都只能看公开后验、已用轮数和剩余预算，绝不能读取真实 theta。
# -----------------------------------------------------------------------------
def choose_action(model: Model, posterior: np.ndarray, budget: int, rounds_left: int, policy: str, action_library: list[tuple[str, ...]], step: int) -> tuple[str, ...]:  # 给出下一观察组合。
    feasible = [a for a in action_library if model.action_cost(a) <= budget]  # 严格掩码所有超预算动作。
    if not feasible or rounds_left <= 0:  # 没有可执行动作或已到轮数上限。
        return ()  # 用空元组代表 STOP。
    if policy == "stop":  # 不获取额外证据的零成本基线。
        return ()  # 直接停止。
    if policy == "repeat":  # 重复总延迟的基线。
        return ("T",) if ("T",) in feasible else ()  # 仅使用普通测量。
    if policy == "fixed":  # 固定测量计划，不根据观测值适应性改变动作。
        proposed = ("Os",) if step % 2 == 0 else ("Oq",)  # 在 State 和 Queue 两类测量之间轮换。
        return proposed if proposed in feasible else ()  # 如果不合法则停止。
    if policy in ("full_bundle", "local_s", "local_q"):
        proposed = {"full_bundle": ("Os", "Oq", "Ds", "Dq"),
                    "local_s": ("Os", "Ds"), "local_q": ("Oq", "Dq")}[policy]
        return proposed if step == 0 and proposed in feasible else ()
    if policy == "greedy":  # 根据当前后验选择单步期望信息增益最大的动作。
        values = [model.expected_gain(posterior, a) for a in feasible]  # 为每个合法 Bundle 计算精确条件 EIG。
        best = int(np.argmax(values))  # 找到最高增益动作的下标。
        return feasible[best] if values[best] > 1e-10 else ()  # 若没有可观测增益则停止。
    raise ValueError(f"未知策略：{policy}")  # 明确拒绝未实现的策略。


def evaluate_policy(model: Model, cases: list[Case], policy: str, budget: int, action_library: list[tuple[str, ...]], records: list[dict] | None = None) -> dict:  # 运行固定策略的 Monte Carlo 测试。
    gains = []  # 保存每个测试案例公式 (24) 的 log-ratio。
    costs = []  # 保存每个案例实际花费的观察成本。
    accuracy = []  # 保存最终最大后验类别是否预测正确。
    actions_used = []  # 保存每个案例执行了多少轮观察。
    for case_id, case in enumerate(cases):  # 在完全相同的隐藏案例集合中比较不同策略。
        rng = np.random.default_rng(case.seed + 1009)  # 为本案例生成可重复的观察噪声。
        posterior = model.initial_posterior(case.x0)  # Agent 只根据公开初始证据构建后验。
        initial_true_prob = float(model.y_posterior(posterior)[case.y])  # Evaluator 才使用真实 y 记录初始概率。
        remaining = budget  # 每个测试案例从相同预算出发。
        used = 0  # 已经执行的观察次数初始化为 0。
        for step in range(model.cfg.max_rounds):  # 最多执行配置允许的观察轮数。
            action = choose_action(model, posterior, remaining, model.cfg.max_rounds - step, policy, action_library, step)  # Policy 决策不接触真相。
            if not action:  # 如果选择 STOP，就停止采集。
                break  # 退出当前案例的观察循环。
            observation = model.observe(case, action, rng)  # Environment 使用真实 theta 产生观察结果。
            posterior = model.posterior(posterior, action, observation)  # 根据真实收集到的观察更新精确后验。
            remaining -= model.action_cost(action)  # 扣除本次 Bundle 的实际成本。
            used += 1  # 记录已进行的一轮观察。
        assert remaining >= 0, "发现超预算策略！"  # 严格验证 almost-sure 预算约束。
        final_probabilities = model.y_posterior(posterior)  # 将最终完整 theta 后验汇总成四类概率。
        final_true_prob = float(final_probabilities[case.y])  # Evaluator 读取真实类别对应的最终概率。
        gains.append(np.log2(final_true_prob / initial_true_prob))  # 公式 (24)：单例可正可负。
        costs.append(budget - remaining)  # 记录实际花费而不是自动假设花满预算。
        accuracy.append(int(np.argmax(final_probabilities) == case.y))  # 仅作辅助指标，不代替信息增益。
        if records is not None:
            records.append({"case_id": case_id, "policy": policy, "budget": budget,
                            "y": case.y, "x0": case.x0, "gain_bits": float(gains[-1]),
                            "cost": costs[-1], "rounds": used})
        actions_used.append(used)  # 统计本例执行轮数。
    values = np.asarray(gains, dtype=float)  # 转为 NumPy 数组以计算样本统计量。
    standard_error = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else float("nan")  # 计算样本均值的标准误。
    return {  # 返回适合写入 CSV 的数值指标。
        "noise_mode": model.cfg.noise_mode,
        "policy": policy,  # 策略名称。
        "budget": budget,  # 本组预算上限。
        "calibration_cost": model.cfg.calibration_cost,  # 当前校准成本。
        "noise_side_prob": model.cfg.noise_side_prob,  # 当前单侧噪声概率。
        "offset_radius": model.cfg.offset_radius,  # 当前时钟偏移范围半径。
        "class_prior": "/".join(str(x) for x in model.cfg.class_prior),  # 记录当前故障先验。
        "n": len(cases),  # 独立测试案例数量。
        "information_gain_bits": float(values.mean()),  # Monte Carlo 信息增益估计。
        "ci95_low": float(values.mean() - 1.96 * standard_error),  # 正态近似的 95% 置信区间下端。
        "ci95_high": float(values.mean() + 1.96 * standard_error),  # 正态近似的 95% 置信区间上端。
        "mean_cost": float(np.mean(costs)),  # 平均实际观察成本。
        "mean_rounds": float(np.mean(actions_used)),  # 平均观察轮数。
        "accuracy": float(np.mean(accuracy)),  # 作为辅助诊断指标的最高后验类别准确率。
    }  # 返回一条结果记录。


# -----------------------------------------------------------------------------
# 三、有限深度 DP：仅在小型动作集合上进行精确规划，不冒充全局大动作库最优。
# -----------------------------------------------------------------------------
def dp_optimal_gain(model: Model, posterior: np.ndarray, budget: int, rounds_left: int, action_library: list[tuple[str, ...]]) -> tuple[float, tuple[str, ...]]:  # 当前后验下求有限时域最优增益与首动作。
    memo: dict[tuple[bytes, int, int], tuple[float, tuple[str, ...]]] = {}  # 缓存重复的后验与预算状态。
    def solve(p: np.ndarray, b: int, rounds: int) -> tuple[float, tuple[str, ...]]:  # 动态规划递归状态。
        if rounds <= 0:  # 如果没有剩余观察轮数。
            return 0.0, ()  # 未来信息增益为零，选择 STOP。
        key = (p.tobytes(), b, rounds)  # 用完整浮点后验字节、预算和轮数作为缓存键。
        if key in memo:  # 如果已经计算过这个信念状态。
            return memo[key]  # 直接复用规划结果。
        best_value = 0.0  # STOP 的未来收益为零，是永远合法的候选动作。
        best_action: tuple[str, ...] = ()  # 默认最优动作是 STOP。
        for a in action_library:  # 遍历事先声明的有限参考动作库。
            if model.action_cost(a) > b:  # 动作不能超过剩余预算。
                continue  # 忽略不合法动作。
            outcomes, likelihood = model.bundle_likelihood(a)  # 获取动作所有可能的结果和似然。
            masses = likelihood * p[None, :]  # 构造观察与当前隐藏状态的联合概率权重。
            probabilities = masses.sum(axis=1)  # 计算各个可能观察结果的条件概率。
            value = 0.0  # 初始化当前动作的期望累计价值。
            h_before = model.entropy_y(p)  # 当前信念状态中的故障熵。
            for j in np.flatnonzero(probabilities > 0):  # 只遍历可能发生的观察结果。
                next_p = masses[j] / probabilities[j]  # 该观察结果对应的精确后验。
                immediate = h_before - model.entropy_y(next_p)  # 此具体结果的即时熵变化，可为负。
                future, _ = solve(next_p, b - model.action_cost(a), rounds - 1)  # 根据新历史考虑后续最优观察。
                value += float(probabilities[j]) * (immediate + future)  # 根据结果概率求期望总信息增益。
            if value > best_value + 1e-12:  # 如果这个动作比之前更好。
                best_value, best_action = float(value), a  # 更新最优值与最优首动作。
        memo[key] = (best_value, best_action)  # 保存该状态的最优决策。
        return memo[key]  # 返回当前状态最优信息增益及其首动作。
    return solve(posterior, budget, rounds_left)  # 从给定初始后验开始求解。


def exact_policy_gain(model, posterior, budget, rounds_left, library, policy="greedy", step=0):
    """枚举同一动作库、同一时域的冻结策略价值，供 DP 公平比较。"""
    action = choose_action(model, posterior, budget, rounds_left, policy, library, step)
    if not action:
        return 0.0
    _, likelihood = model.bundle_likelihood(action)
    masses = likelihood * posterior[None, :]
    probabilities = masses.sum(axis=1)
    value = 0.0
    for j in np.flatnonzero(probabilities > 0):
        next_p = masses[j] / probabilities[j]
        immediate = model.entropy_y(posterior) - model.entropy_y(next_p)
        future = exact_policy_gain(model, next_p, budget-model.action_cost(action),
                                   rounds_left-1, library, policy, step+1)
        value += float(probabilities[j]) * (immediate + future)
    return float(value)


def run_dp_reference(model, max_budget, output_dir, library=None):
    # library=None 明确表示仅五种单项测量。传入主动作库时仍只对该库/时域最优。
    actions = library if library is not None else [(m,) for m in MEASURES]
    scope = "configured_library" if library is not None else "single_measurements"
    x0_values, likelihood = model.single_likelihood["T"]
    x0_probabilities = likelihood @ model.prior
    rows = []
    for budget in range(max_budget + 1):
        optimal = greedy = 0.0
        for i in np.flatnonzero(x0_probabilities > 0):
            posterior = model.initial_posterior(int(x0_values[i]))
            value, _ = dp_optimal_gain(model, posterior, budget, model.cfg.max_rounds, actions)
            optimal += float(x0_probabilities[i]) * value
            greedy += float(x0_probabilities[i]) * exact_policy_gain(
                model, posterior, budget, model.cfg.max_rounds, actions)
        rows.append({"budget": budget, "scope": scope, "rounds": model.cfg.max_rounds,
                     "noise_mode": model.cfg.noise_mode,
                     "actions": json.dumps(actions), "dp_gain_bits": optimal,
                     "greedy_same_library_gain_bits": greedy, "gap_bits": optimal-greedy,
                     "estimator": "exact_enumeration_no_MC_interval"})
        print(f"[DP {scope}] budget={budget}, optimal={optimal:.6f}, greedy={greedy:.6f}")
    write_csv(output_dir / "dp_reference.csv", rows)


# -----------------------------------------------------------------------------
# 四、输出：CSV、实验曲线及单元自检。
# -----------------------------------------------------------------------------
def write_csv(filename: Path, rows: list[dict]) -> None:  # 将结果保存成可用于分析的 CSV 文件。
    if not rows:  # 检查是否有数据。
        return  # 没有数据就不写入空文件。
    with filename.open("w", newline="", encoding="utf-8-sig") as file:  # UTF-8 BOM 便于部分表格软件正确识别中文。
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))  # 使用第一条结果的键作为表头。
        writer.writeheader()  # 写入列名。
        writer.writerows(rows)  # 写入所有结果行。


def plot_curves(rows: list[dict], output_dir: Path) -> None:  # 绘制各策略的信息增益随预算变化的曲线。
    fig, ax = plt.subplots(figsize=(8, 5))  # 创建一个独立绘图画布。
    policies = list(dict.fromkeys(row["policy"] for row in rows))  # 获得去重且保持顺序的策略名。
    for name in policies:  # 每种策略绘制一条线。
        subset = sorted((r for r in rows if r["policy"] == name), key=lambda x: x["budget"])  # 按预算从小到大排序。
        x = np.array([r["budget"] for r in subset], dtype=float)  # 提取预算横坐标。
        y = np.array([r["information_gain_bits"] for r in subset], dtype=float)  # 提取信息增益纵坐标。
        low = np.array([r["ci95_low"] for r in subset], dtype=float)  # 提取置信区间下端。
        high = np.array([r["ci95_high"] for r in subset], dtype=float)  # 提取置信区间上端。
        if len(subset) == 1:
            ax.errorbar(x, y, yerr=np.vstack((y-low, high-y)), fmt="D", capsize=4, label=name)
        else:
            ax.plot(x, y, marker="o", label=name)  # 绘制信息增益曲线。
        ax.fill_between(x, low, high, alpha=0.12)  # 标记 95% 近似置信区间。
    ax.set_xlabel("Acquisition budget (cost units)")  # 横轴是预算。
    ax.set_ylabel("Estimated information gain (bits)")  # 纵轴是公式 (24) 的估计。
    ax.set_title("Finite delay-and-clock benchmark")  # 设置图题。
    ax.axhline(0, linewidth=0.8, linestyle="--")  # 添加零信息增益参考线。
    ax.grid(alpha=0.25)  # 添加淡网格便于读数。
    ax.legend()  # 标注各条策略曲线。
    fig.tight_layout()  # 自动调整边距。
    fig.savefig(output_dir / "information_gain_vs_budget.png", dpi=180)  # 保存高清 PNG。
    plt.close(fig)  # 关闭画布以释放内存。


def self_test() -> None:  # 执行若干无需实验数据的正确性检查。
    model = Model(Config())  # 创建论文默认概率设置。
    assert len(model.states) == 225, "默认隐藏状态数量应为 225。"  # 检查枚举数量。
    assert np.allclose(model.prior @ model.label_matrix, [0.25] * 4), "故障类先验不均匀。"  # 检查每类概率。
    for name in MEASURES:  # 遍历所有单项测量。
        _, probabilities = model.bundle_likelihood((name,))  # 读取此测量的概率矩阵。
        assert np.allclose(probabilities.sum(axis=0), 1), f"{name} 的条件似然未归一。"  # 给定每个 theta，各结果概率之和必须为 1。
    for action in [("Os", "Ds"), ("Os", "Oq"), ("Os", "Oq", "Ds", "Dq")]:  # 测试典型 Bundles。
        _, probabilities = model.bundle_likelihood(action)  # 生成联合条件似然矩阵。
        assert np.allclose(probabilities.sum(axis=0), 1), f"{action} 的联合似然未归一。"  # 复核独立噪声联合分布。
    for x0 in [-1, 0, 1, 2, 3, 4, 5]:  # 对一批可能的初始测量进行检查。
        try:  # 有些 x0 可能在指定支持下不出现。
            p = model.initial_posterior(x0)  # 根据初始测量计算后验。
        except ValueError:  # 该 x0 属于零概率事件。
            continue  # 跳过不可能的观察值。
        assert np.isclose(p.sum(), 1), "后验没有归一化。"  # 检查贝叶斯更新。
        assert 0 <= model.entropy_y(p) <= 2 + 1e-12, "四分类熵应位于 0 到 2 bits。"  # 检查熵的理论范围。
        assert model.expected_gain(p, ("Os",)) >= -1e-12, "期望信息增益不能为负。"  # 检查互信息性质。
    print("自检通过：225 个配置、先验、测量似然、Bundle 似然、后验及信息增益检查。")  # 展示通过信息。


def paired_comparisons(records, output_dir):
    """同一案例的策略差值 CI；保留配对关系而不是独立相减两个 CI。"""
    indexed = {(r["policy"], r["budget"], r["case_id"]): r["gain_bits"] for r in records}
    pairs = [("greedy", "greedy_no_calibration"), ("greedy", "greedy_single"),
             ("greedy_single", "greedy_single_no_calibration"),
             ("greedy_no_calibration", "greedy_single_no_calibration")]
    rows = []
    for a, b in pairs:
        for budget in sorted({r["budget"] for r in records}):
            differences = [v-indexed[(b, budget, i)] for (name, bud, i), v in indexed.items()
                           if name == a and bud == budget and (b, budget, i) in indexed]
            if len(differences) < 2:
                continue
            values = np.asarray(differences)
            mean = float(values.mean())
            half = float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
            rows.append({"policy_a": a, "policy_b": b, "budget": budget,
                         "n": len(values), "mean_difference_bits": mean,
                         "ci95_low": mean-half, "ci95_high": mean+half})
    write_csv(output_dir / "paired_comparisons.csv", rows)


def run_experiment(args: argparse.Namespace) -> None:
    if args.n < 2 or args.max_budget < 0 or args.dp_max_budget < 0 or not 1 <= args.bundle_size <= 5:
        raise ValueError("需要 n>=2、非负预算及 bundle-size 在 1..5 内。")
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    self_test()
    cfg = Config(noise_mode=args.noise_mode, noise_side_prob=args.noise,
                 offset_radius=args.offset, class_prior=tuple(args.prior),
                 calibration_cost=args.calibration_cost, max_rounds=args.rounds)
    model = Model(cfg)
    rng = np.random.default_rng(args.seed)
    cases = [model.draw_case(rng) for _ in range(args.n)]
    library = model.action_library(True, args.bundle_size)
    variants = [(name, name, library) for name in ("stop", "repeat", "fixed", "greedy")]
    variants += [
        ("greedy_no_calibration", "greedy", model.action_library(False, args.bundle_size)),
        ("greedy_single", "greedy", model.action_library(True, 1)),
        ("greedy_single_no_calibration", "greedy", model.action_library(False, 1))]
    rows, records = [], []
    for budget in range(args.max_budget + 1):
        for label, policy, actions in variants:
            per_case = []
            result = evaluate_policy(model, cases, policy, budget, actions, per_case)
            result["policy"] = label
            for record in per_case:
                record["policy"] = label
            records.extend(per_case)
            rows.append(result)
            print(f"budget={budget} | {label} | gain={result['information_gain_bits']:+.4f}", flush=True)
    # 单独固定执行一次，不依赖 Greedy 是否选中，也不受公共网格上限或 bundle-size 限制。
    bundle_rows = []
    for policy, action in [("full_bundle", ("Os", "Oq", "Ds", "Dq")),
                           ("local_s", ("Os", "Ds")), ("local_q", ("Oq", "Dq"))]:
        result = evaluate_policy(model, cases, policy, model.action_cost(action), [action], records)
        bundle_rows.append(result)
    write_csv(output_dir / "main_results.csv", rows)
    write_csv(output_dir / "bundle_points.csv", bundle_rows)
    write_csv(output_dir / "per_case_results.csv", records)
    paired_comparisons(records, output_dir)
    plot_curves(rows + bundle_rows, output_dir)
    metadata = {**vars(args), "shared_noise_definition":
                "one common additive noise per acquisition round; independent across rounds and from O0",
                "ci_method": "mean +/- 1.96 sample standard error; paired differences for ablations",
                "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__}
    (output_dir / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dp:
        run_dp_reference(model, args.dp_max_budget, output_dir,
                         library if args.dp_library == "matched" else None)
    print(f"结果已保存到：{output_dir.resolve()}")


def make_parser() -> argparse.ArgumentParser:  # 定义命令行参数。
    parser = argparse.ArgumentParser(description="Section 3.2 有限延迟与时钟偏移实验")  # 帮助信息。
    parser.add_argument("--noise-mode", choices=["independent", "shared_round"], default="independent")
    parser.add_argument("--dp-library", choices=["single", "matched"], default="single",
                        help="single 为单项受限参考；matched 使用主实验配置的动作库，可能很慢")
    parser.add_argument("--n", type=int, default=120, help="独立 Monte Carlo 案例数；正式实验建议增加")  # 控制统计精度。
    parser.add_argument("--seed", type=int, default=42, help="随机种子")  # 控制可复现性。
    parser.add_argument("--noise", type=float, default=0.2, help="P(噪声=-1)=P(噪声=+1)")  # 支持噪声敏感性扫描。
    parser.add_argument("--offset", type=int, default=2, help="时钟偏移范围为 [-offset,offset]")  # 支持偏移范围变化。
    parser.add_argument("--prior", type=float, nargs=4, default=[0.25] * 4, help="四种故障的先验概率")  # 支持非均匀先验。
    parser.add_argument("--calibration-cost", type=int, default=2, choices=[1, 2, 4], help="校准动作成本")  # 论文指定三种校准成本。
    parser.add_argument("--max-budget", type=int, default=6, help="最大预算，默认从 0 扫到 6")  # 论文预算网格。
    parser.add_argument("--rounds", type=int, default=2, help="最大观察轮数，最初建议 2")  # 论文初始两轮设置。
    parser.add_argument("--bundle-size", type=int, default=4, help="每轮 Bundle 最多包含多少种测量；可设为 1 加速")  # 控制联合动作大小。
    parser.add_argument("--out", type=str, default="results/base", help="结果输出目录")  # 保存实验产物。
    parser.add_argument("--dp", action="store_true", help="额外计算指定动作库和轮数下的 DP 参考")  # 显式请求 DP。
    parser.add_argument("--dp-max-budget", type=int, default=2, help="受限 DP 计算到的最大预算；大预算计算较慢")  # 控制 DP 成本。
    return parser  # 返回配置好的命令行解析器。


if __name__ == "__main__":  # 只有直接执行文件时才运行实验。
    arguments = make_parser().parse_args()  # 从命令行获取实验设置。
    run_experiment(arguments)  # 启动完整实验流程。
