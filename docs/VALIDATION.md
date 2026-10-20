# 验证记录

环境：Windows、Python3.12.7、CPU、float64；依赖精确版本见`requirements-tested.txt`。

最终代码执行 `python -m pytest -q`，结果为 **14 passed**。
测试覆盖目标解析梯度与自动微分的一致性、三类目标上的非对称MH详细平衡、
有限状态精确Poisson梯度和接受流score的一致性、变分恒等式与Dirichlet误差、
proposal样本及reward的detach、正则化末层的一阶最优条件、AR(1)理论渐近方差、
非混合链R-hat告警、参考采样矩，以及双峰中Critic/Actor两组各自包含两类初值。

另外实际运行了`sample_checkpoint.py`，分别恢复神经proposal和MALA检查点，各采样4链×100步。
输出保存在`results/validation/checkpoint_neural.npz`和`checkpoint_mala.npz`。
这两份小文件用于确认恢复路径可执行，不参与模型效果判断。

最终54组产物由`scripts/audit_results.py`独立检查样本形状、有限值、统计函数重算、
ASV单位和成本计数、冻结参数、结果唯一性及源码快照指纹；结果保存在`results/validation/audit.json`。
