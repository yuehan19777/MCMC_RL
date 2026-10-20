# Poisson Critic 与 MH Policy Gradient：可复现预实验

这是对 `docs/research_idea_original.txt` 中研究方案的实现和小规模检验。训练目标直接来自可逆链的渐近方差变分表示；Actor 使用包含 MH 接受机制的 score，而不是接受率奖励。最终结果、分析和图见 [预实验导读](docs/FINDINGS.md) 和 [最终比较报告](results/final_comparison/REPORT.md)。

## 快速运行

需要 Python 3.10+ 和 CPU PyTorch。推荐在独立环境中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\run_experiment.py --config configs\smoke.json --output results\my_smoke
.\.venv\Scripts\python.exe scripts\run_experiment.py --config configs\pilot_v2.json --output results\my_pilot
```

当前电脑已在下面这个现有环境中完整执行，默认 Anaconda 的 `python` **没有 torch**：

```powershell
& 'C:\Users\16209\anaconda3\envs\pytor\python.exe' -m pytest -q
& 'C:\Users\16209\anaconda3\envs\pytor\python.exe' scripts\run_experiment.py --config configs\pilot_v2.json --output results\my_pilot
```

也可以安装后使用 `poisson-mcmc --config configs/pilot_v2.json --output results/my_pilot`。
输出目录非空时会停止，防止覆盖实验；相同代码和配置可以加 `--resume` 恢复。
只重新绘图/汇总已有结果用 `--plots-only`；不会重新运行采样。
所有相对路径以当前工作目录为基准，因此应在项目根目录执行命令。

## 实验范围

| 任务 | 定义 | 主要难点 |
|---|---|---|
| Gaussian | 相关系数 0.95，两个坐标标准差 1 和 3 | 线性相关、条件数约 112 |
| Banana | x0=2z0，x1=0.5z1+0.35(x0²−4) | 弯曲、局部曲率不同 |
| Mixture | 0.5 N((-4,0),diag(0.49,1.44)) + 0.5 N((4,0),diag(0.49,1.44)) | 双峰之间低密度屏障 |

三种经典对照：调尺度的随机游走 MH（RWM）、调协方差与尺度的 Adaptive Metropolis（AM）、预条件 MALA。
另有 neural ESJD：与 Poisson 共用 proposal 架构、初始化、Adam 和训练步数，主要改变优化信号。
AM 适配后冻结；本项目不将无限期自适应链当成固定平稳 kernel 分析。

最终配置为 3 个训练种子，每个方法每个种子训练 16 条并行链，pilot 256 步，随后 128 轮×32 步。
每个冻结 kernel 用 8 条独立链各采样 5000 步；双峰额外运行 2500 步单峰冷启动测试。
没有维度扫描、任务搜索或大规模超参数搜索，预实验重点是算法正确性和趋势。

训练统计函数是标准化后的 x0、x1，中心和尺度均从共同 pilot 估计。留出统计函数为 x0² 和 1{x0>0}。
多头 Critic 将两个标准化统计函数的渐近方差相加，这是单 f 推导的直接扩展，**不意味着所有 f 的方差都会下降**。

## 核心实现

- `src/poisson_mcmc/targets.py`：目标密度、解析梯度、独立参考采样和解析矩。
- `proposals.py`：可求正反向密度的高斯 proposal；学习方法包含全协方差与有界神经 drift。
- `kernels.py`：所有方法共用的 MH 转移及完整 MH-specific score。
- `critic.py`：变分 Poisson Critic，训练非线性特征，解析求解正则化末层。
- `training.py`：共同 pilot、经典方法调参、Critic/Actor 交替更新及按链拆分训练数据。
- `diagnostics.py`、`evaluation.py`：独立评估，谱方差、batch means、链均值方差、ESS、rank/folded split R-hat、RMSE 和冷启动。
- `experiment.py`：配置、种子、逐次产物、断点恢复和环境/源码指纹。
- `plotting.py`、`reporting.py`：自动生成图、CSV 和报告。
- `tests/test_mathematics.py`：详细平衡、理论梯度、变分恒等式、Critic 一阶条件、AR(1) 理论 ASV 和解析矩验证。

算法细节和近似偏差见 [ALGORITHM.md](docs/ALGORITHM.md)；预算与指标解释见 [EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 输出组织

```text
configs/                 # 受控的 smoke / pilot 配置
docs/                    # 理论、实验协议、结果讨论、用户原始思路
src/poisson_mcmc/         # 可安装、职责明确的实现
tests/                   # 数学与统计性质测试
scripts/                 # 源码目录直接运行入口
results/refined_learned/
  manifest.json          # 环境、源码 SHA256、配置指纹
  config.json
  metrics.csv            # 每任务/方法/种子/统计函数的指标
  summary.csv
  REPORT.md
  figures/               # 七张比较图
  runs/<task>/<method>/seed_<seed>/
    checkpoint.pt        # 冻结参数，使用 torch.load(..., weights_only=True)
    history.json         # 每轮训练和 Critic 诊断
    result.json          # 该次独立实验的所有数值
    samples.npz          # 完整评估轨迹、统计函数、自相关与压力测试
```

`results/final_comparison/` 汇总最终的六方法比较：三种初始经典方法、额外的预条件 RWM 步长网格检查、ESJD 和 Poisson。
样本通过相对路径引用原实验目录，没有复制整份数据。`results/README.md` 说明各轮记录的用途与修正过程。
`requirements-tested.txt` 记录本次实际使用的包版本；未修改用户原来的 Python 环境。

补充经典步长检查、重绘最终对比和恢复采样的入口：

```powershell
python scripts\check_classical_scale.py --config configs\pilot_v2.json --output results\my_scale_check
python scripts\compare_results.py --main results\pilot_v2 --learned results\refined_learned --scale-check results\classical_scale_check --output results\my_comparison
python scripts\sample_checkpoint.py results\refined_learned\runs\gaussian\poisson\seed_17\checkpoint.pt --output results\new_draws.npz
```

若从头重做最终整套实验，可将第一条主实验输出中的两个学习方法也用于最终对比，
即将 `compare_results.py` 的 `--main` 与 `--learned` 同时设为新的主实验目录。
恢复采样默认从分散状态开始并丢弃500步；短 burn-in 不保证任意复杂目标都已收敛。

## 理论来源

- [Sherlock：Reversible Markov chains: variational representations and ordering](https://arxiv.org/abs/1809.01903)。
- [Baxter & Bartlett：Infinite-Horizon Policy-Gradient Estimation](https://arxiv.org/abs/1106.0665)。原工作为平均奖励梯度；本项目借鉴 score 思想，不等同于直接套用 GPOMDP。
- [Stan Reference Manual：Posterior Analysis](https://mc-stan.org/docs/reference-manual/analysis.html)。本项目参考其 ESS 和 R-hat 思路；原统计函数 ESS 不等于 Stan 的 rank-normalized bulk/tail ESS。
