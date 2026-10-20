# Poisson Critic / MH Policy Gradient 预实验结果

训练随机种子：[17, 43, 89]；每个冻结 kernel 独立评估 8 条链，每条 5000 步；3 个任务、2 种方法。

数值均为跨训练种子的均值 ± 标准误（SE）；样本少，不能当作确定性排名。主表的 ASV 除以统计函数真实边际方差，越低越好。x0、x1 用于训练，x0² 和右半空间指示函数仅用于评估。

## gaussian

| 方法 | x0 归一化 ASV | x1 归一化 ASV | x0² 归一化 ASV | 最差 R-hat | 最小 ESS/s（含训练） |
|---|---:|---:|---:|---:|---:|
| Neural ESJD | 1.94 ± 0.37 | 1.76 ± 0.47 | 6.59 ± 1.74 | 1.002 | 3959.6 |
| Poisson actor-critic | 1.18 ± 0.07 | 1.02 ± 0.02 | 3.35 ± 0.63 | 1.001 | 5146.2 |

## banana

| 方法 | x0 归一化 ASV | x1 归一化 ASV | x0² 归一化 ASV | 最差 R-hat | 最小 ESS/s（含训练） |
|---|---:|---:|---:|---:|---:|
| Neural ESJD | 24.12 ± 15.19 | 40.70 ± 27.82 | 43.01 ± 30.16 | 1.017 | 257.5 |
| Poisson actor-critic | 16.40 ± 2.73 | 27.18 ± 8.29 | 27.63 ± 7.98 | 1.007 | 211.6 |

**诊断警示：本任务部分方法/种子的 R-hat >1.01。其短链 ASV 估计及下述效率比不应被视为可靠的真实渐近效率比；应同时检查 ESS、独立链均值误差和冷启动图。**

## mixture

| 方法 | x0 归一化 ASV | x1 归一化 ASV | x0² 归一化 ASV | 最差 R-hat | 最小 ESS/s（含训练） |
|---|---:|---:|---:|---:|---:|
| Neural ESJD | 1.39 ± 0.02 | 2.96 ± 0.28 | 2.65 ± 0.10 | 1.001 | 2108.2 |
| Poisson actor-critic | 2.01 ± 0.20 | 3.35 ± 0.28 | 3.39 ± 0.33 | 1.001 | 1600.8 |

## 解释范围与局限

- 主实验从独立的精确 π 样本开始，仅用于隔离固定 kernel 的平稳态效率；训练和调参完全不调用精确采样器或真实矩。双峰另有全部从左峰出发的压力测试。
- 所有方法使用相同链数、pilot 及 warmup 转移步数；学习方法另有 Actor proposal 和反向传播成本，因此同时报告 ESS/步、ESS/密度调用和包含训练的 ESS/秒。MALA 还使用目标梯度，调用数单列。
- 主 ASV 来自带链间方差修正的 Geyer 序列估计；同时提供两种 batch size 的 batch means，以及 N·Var(独立链均值)。后者每个训练种子仅有少量独立链，误差很大；batch 太短会低估慢相关，不能将 Critic 的训练目标当成验证 ASV。
- Critic 为线性/二次/平滑阈值特征加可训练 MLP；末层用带 ridge 的 Galerkin 方程求解。这是原变分目标的有限维正则化近似，不是原无限维 Poisson 方程的精确解。Galerkin residual 只衡量同批次的投影残差，不是全状态空间的 Bellman residual。
- Actor 使用原 MH-specific score、独立 proposal 批次和冻结 Critic；当前训练样本的中心化、非稳态、有限 Critic 表达能力、ridge 和正标量归一化都可能影响优化。
- 任务只有二维、训练种子很少，结果只能判断是否值得继续研究，无法支持普遍优于经典 MCMC 的结论。

## 输出文件

- `metrics.csv`：每个任务/方法/种子/统计函数的完整指标。
- `summary.csv`：按训练统计函数汇总的比较。
- `runs/`：逐次配置、历史、冻结参数、样本及诊断，支持追溯。
- `figures/`：ASV、成本、覆盖、轨迹、自相关、训练和冷启动图。

## 图

![01_asymptotic_variance](figures/01_asymptotic_variance.png)

![02_cost_efficiency](figures/02_cost_efficiency.png)

![03_target_coverage](figures/03_target_coverage.png)

![04_trace_and_acf](figures/04_trace_and_acf.png)

![05_asv_crosscheck](figures/05_asv_crosscheck.png)

![06_training_diagnostics](figures/06_training_diagnostics.png)

![07_cold_start](figures/07_cold_start.png)
