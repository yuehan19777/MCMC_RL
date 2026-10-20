"""Aggregate raw run records into CSV tables and a short, honest results report."""
import csv
from pathlib import Path

import numpy as np

from .plotting import LABELS, mean_se
from .targets import STAT_NAMES


def write_tables(root: Path, records: list[dict]) -> None:
    rows = []
    for record in records:
        for metric in record["metrics"]:
            row = {"task": record["task"], "method": record["method"], "seed": record["seed"],
                   "statistic": STAT_NAMES[metric["stat_index"]], **metric, **record["metadata"]}
            rows.append(row)
    with (root / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(root: Path, cfg: dict, records: list[dict]) -> None:
    text = ["# Poisson Critic / MH Policy Gradient 预实验结果", "",
            f"训练随机种子：{cfg['seeds']}；每个冻结 kernel 独立评估 {cfg['eval_chains']} 条链，"
            f"每条 {cfg['eval_steps']} 步；{len(cfg['tasks'])} 个任务、{len(cfg['methods'])} 种方法。",
            "", "数值均为跨训练种子的均值 ± 标准误（SE）；样本少，不能当作确定性排名。"
            "主表的 ASV 除以统计函数真实边际方差，越低越好。x0、x1 用于训练，"
            "x0² 和右半空间指示函数仅用于评估。", ""]
    summary_rows = []
    for task in cfg["tasks"]:
        text += [f"## {task}", "",
                 "| 方法 | x0 归一化 ASV | x1 归一化 ASV | x0² 归一化 ASV | 最差 R-hat | 最小 ESS/s（含训练） |",
                 "|---|---:|---:|---:|---:|---:|"]
        method_scores = {}
        for method in cfg["methods"]:
            subset = [r for r in records if r["task"] == task and r["method"] == method]
            values = [mean_se([r["metrics"][k]["normalized_asv"] for r in subset]) for k in range(3)]
            rhat = max(r["metrics"][k]["rhat"] for r in subset for k in range(4))
            rate, _ = mean_se([min(r["metrics"][k]["ess_per_second_total"] for k in (0, 1)) for r in subset])
            formatted = [f"{v:.2f} ± {se:.2f}" for v, se in values]
            rhat_display = f"{rhat:.3f}" if rhat < 1e6 else ">10⁶（未混合）"
            text.append(f"| {LABELS[method]} | {' | '.join(formatted)} | {rhat_display} | {rate:.1f} |")
            # Aggregate trained-statistic score: arithmetic mean of normalized ASV.
            score = np.mean([np.mean([r["metrics"][k]["normalized_asv"] for k in (0, 1)]) for r in subset])
            method_scores[method] = score
            summary_rows.append({"task": task, "method": method, "trained_mean_normalized_asv": score,
                                 "min_ess_per_second_total": rate, "max_rhat": rhat})
        if any(max(r["metrics"][k]["rhat"] for r in records if r["task"] == task for k in range(4)) > 1.01 for _ in [0]):
            text += ["", "**诊断警示：本任务部分方法/种子的 R-hat >1.01。其短链 ASV 估计及下述效率比不应被视为可靠的真实渐近效率比；应同时检查 ESS、独立链均值误差和冷启动图。**"]
        best_classic = min((m for m in ("rwm", "am", "mala") if m in method_scores),
                           key=lambda m: method_scores[m], default=None)
        if "poisson" in method_scores and best_classic:
            ratio = method_scores[best_classic] / method_scores["poisson"]
            text += ["", f"按 x0、x1 归一化 ASV 的算术平均，Poisson 相对于本次估计值最低的经典对照 "
                     f"{LABELS[best_classic]} 的**观测估计比**为 **{ratio:.2f}×**（>1 表示该指标较好，可靠性须结合诊断）。"]
            if "esjd" in method_scores:
                ratio = method_scores["esjd"] / method_scores["poisson"]
                text += [f"相对于相同 neural proposal 的 ESJD 消融，效率比为 **{ratio:.2f}×**。"]
        text.append("")

    text += ["## 解释范围与局限", "",
             "- 主实验从独立的精确 π 样本开始，仅用于隔离固定 kernel 的平稳态效率；训练和调参"
             "完全不调用精确采样器或真实矩。双峰另有全部从左峰出发的压力测试。",
             "- 所有方法使用相同链数、pilot 及 warmup 转移步数；学习方法另有 Actor proposal 和反向传播成本，"
             "因此同时报告 ESS/步、ESS/密度调用和包含训练的 ESS/秒。MALA 还使用目标梯度，调用数单列。",
             "- 主 ASV 来自带链间方差修正的 Geyer 序列估计；同时提供两种 batch size 的 batch means，"
             "以及 N·Var(独立链均值)。后者每个训练种子仅有少量独立链，误差很大；"
             "batch 太短会低估慢相关，不能将 Critic 的训练目标当成验证 ASV。",
             "- Critic 为线性/二次/平滑阈值特征加可训练 MLP；末层用带 ridge 的 Galerkin 方程求解。"
             "这是原变分目标的有限维正则化近似，不是原无限维 Poisson 方程的精确解。"
             "Galerkin residual 只衡量同批次的投影残差，不是全状态空间的 Bellman residual。",
             "- Actor 使用原 MH-specific score、独立 proposal 批次和冻结 Critic；当前训练样本的"
             "中心化、非稳态、有限 Critic 表达能力、ridge 和正标量归一化都可能影响优化。",
             "- 任务只有二维、训练种子很少，结果只能判断是否值得继续研究，无法支持普遍优于经典 MCMC 的结论。",
             "", "## 输出文件", "",
             "- `metrics.csv`：每个任务/方法/种子/统计函数的完整指标。",
             "- `summary.csv`：按训练统计函数汇总的比较。",
             "- `runs/`：逐次配置、历史、冻结参数、样本及诊断，支持追溯。",
             "- `figures/`：ASV、成本、覆盖、轨迹、自相关、训练和冷启动图。",
             "", "## 图", ""]
    for name in sorted((root / "figures").glob("*.png")):
        text += [f"![{name.stem}](figures/{name.name})", ""]
    (root / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
