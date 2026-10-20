# 算法、理论对应与实现边界

## 从单函数到多个统计函数

对每个训练统计函数 gj=fj−π(fj)，Poisson 方程为 (I−Kθ)hj=gj。
共同目标是各标准化统计函数渐近方差的平均：

```text
Cj(h) = 2 Eπ[gj(x) hj(x)] − 1/2 Eπq[α(x,y) (hj(y)−hj(x))²]
F(θ,h) = mean_j [2 Cj(hj) − Varπ(fj)]
min_θ max_h F(θ,h)
```

输入数据的均值、标准差来自共同 pilot，并非目标分布的解析矩。
每个训练批次重新减去 gj 的样本均值；hj 也减去批次均值。
常数对 Dirichlet energy 无影响，这样可避免有限样本中的常数方向漂移，但批次中心化仍是有限样本近似。

## Critic：小型神经特征加 Galerkin 求解

令 Ψφ(x) 为 9 个固定特征与 8 个 MLP 输出的拼接：标准化坐标、平方、交叉项、两组平滑阈值，加一个 2→16→8 的 Tanh MLP。
两头 Critic 为 h(x)=Ψφ(x)C。末层没有偏置，因为 Poisson 解只在相差常数意义下唯一。

定义：

```text
A = 1/2 mean[α ΔΨ ΔΨᵀ]
B = mean[(Ψ(x)−mean Ψ(x)) (g(x)−mean g(x))ᵀ]
C* = (A + λI)^(-1) B, λ = 0.001
```

这是最大化 `2 tr(Bᵀ C)−tr(Cᵀ A C)−λ||C||²` 的精确末层解。
固定 C*，用同一个变分目标训练非线性特征，交替 3 次，然后再次求解 C*。
固定 C 时 ridge 对特征梯度为常数，因此不需要在 `loss()` 中再加这一常数。
Actor 阶段不对 C* 或 Critic 特征求导。

这保留原研究的 Poisson Critic / MH policy-gradient 流程，同时避免让普通 SGD 从零学习一个可能幅值很大的 Critic。
它只是在**正则化有限特征空间**中求解，无法声称精确恢复整个状态空间上的 Poisson solution。
训练日志的 Galerkin residual 来自同一训练批次，并不能证明泛化或梯度无偏。

## Actor：完整接受流的 score

proposal 是 `N(x + Lpilot dθ(Lpilot⁻¹(x−center)), Lθ Lθᵀ)`。
drift 使用 2→16→2 Tanh 网络，输出幅度在 pilot 白化坐标下有界。
Lθ 是全下三角 Cholesky，可学习对角尺度和相关结构。初始 drift 为零，尺度为 1.5。
该 proposal 有非零 Lebesgue 密度，MH 正反向密度使用相同实现。

对采样得到的固定 x、y，以及固定 Critic，计算：

```text
log α = min(0, log π(y) − log π(x) + log qθ(x|y) − log qθ(y|x))
R = mean_j [(hj(y)−hj(x))²]
loss = −mean[ stop(α R) · (log qθ(y|x) + log α) ]
```

对 θ 求导给出 `−E[α R ∇θ log(qθα)]` 的 Monte Carlo 形式。
当 MH ratio <1 时，正向 score 与 log α 中的负正向 score 抵消，留下反向 score；ratio >1 时留下正向 score。
`y` 必须 detach，不能对 proposal 的采样路径求导；α 作为权重也必须 detach，避免重复求导。
在 ratio=1 的不可微集合使用 PyTorch 分支导数；连续非退化情形中该集合通常为零测度。

每轮先用当前冻结 proposal 收集链转移。按相邻链成对分组：0/1、4/5等训练 Critic，2/3、6/7等提供 Actor 状态，并另采新 y。
这种分组让两个子集都含双峰的左右初值，避免奇偶拆分与初始模式分配重合；train_chains 要求是4的倍数。
Critic 和 Actor 不共用 proposal 随机数。它们共享当前 kernel 和 pilot，因此这种拆分不能消除所有有限样本误差。

实际更新中按本批 reward 的均值除以一个正标量，再做梯度裁剪、Adam 与参数投影。
在单批次内，这一标量保持 score 方向；由于它依赖随机批次，**归一化后的梯度不能称为原始无偏估计器**。
`kernels.mh_score_loss` 本身保留未归一化的数学形式，有限状态精确测试直接核对这一形式。

## 工程约束

- λI 正则、float64 和线性求解用于缓解小 spectral gap 的条件数问题。
- Critic 更新多于 Actor，避免 Actor 在远未拟合的 Critic 上连续更新。
- 不用接受率奖励辅助 Poisson Actor，避免把研究目标偷换为其他代理目标。
- ESJD 使用相同 MH score 及相同 proposal，只将 reward 换为标准化坐标跳跃距离。
- 训练后完全冻结 proposal；最终 ESS、ASV 等只看新链，绝不混入训练链。
- 这些约束不构成在线自适应过程的遍历性证明。本项目分析的是适配结束后的固定 kernel。

## 正确性证据

`tests/test_mathematics.py` 使用可精确求解的三状态可逆 MH 链：

1. 解线性方程得到 h，再自动微分真实 ASV；
2. 用冻结 h 的 MH score 求导，逐参数比较；
3. 验证 ASV 的变分公式和 Critic gap 等于误差 Dirichlet energy；
4. 连续非对称神经 proposal 验证详细平衡、反向 score 和 detach 机制；
5. 检查正则化 Critic 末层的一阶最优条件。

参考：[Sherlock 原文](https://arxiv.org/abs/1809.01903)、[Baxter–Bartlett 原文](https://arxiv.org/abs/1106.0665)。
