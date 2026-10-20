"""Combine the refined study and classical scale check without duplicating samples."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from poisson_mcmc.experiment import json_write
from poisson_mcmc.plotting import COLORS, LABELS, make_plots, mean_se


def read_records(root: Path, destination: Path) -> list[dict]:
    records = []
    for path in sorted((root / "runs").rglob("result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["artifact"] = Path(os.path.relpath(path.parent.resolve(), destination.resolve())).as_posix()
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, default=Path("results/pilot_v2"))
    parser.add_argument("--scale-check", type=Path, default=Path("results/classical_scale_check"))
    parser.add_argument("--initial", type=Path, default=Path("results/pilot_v1"))
    parser.add_argument("--learned", type=Path, default=Path("results/refined_learned"))
    parser.add_argument("--output", type=Path, default=Path("results/final_comparison"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = [r for r in read_records(args.main, args.output) if r["method"] not in ("esjd", "poisson")]
    records += [r for r in read_records(args.learned, args.output) if r["method"] in ("esjd", "poisson")]
    records += read_records(args.scale_check, args.output)
    cfg = json.loads((args.main / "config.json").read_text(encoding="utf-8"))
    cfg["methods"] = ["rwm", "am", "mala", "cov_grid", "esjd", "poisson"]
    expected = len(cfg["tasks"]) * len(cfg["methods"]) * len(cfg["seeds"])
    if len(records) != expected:
        raise ValueError(f"Expected {expected} completed runs, got {len(records)}")
    LABELS["cov_grid"] = "Cov-RWM (ESJD grid)"
    COLORS["cov_grid"] = "#d67055"
    make_plots(args.output, cfg, records)
    rows = []
    for task in cfg["tasks"]:
        for method in cfg["methods"]:
            subset = [r for r in records if r["task"] == task and r["method"] == method]
            scores = [np.mean([r["metrics"][k]["normalized_asv"] for k in (0, 1)]) for r in subset]
            avg, se = mean_se(scores)
            rows.append({"task": task, "method": method, "mean_normalized_asv": avg, "asv_seed_se": se,
                         "x0_normalized_asv": np.mean([r["metrics"][0]["normalized_asv"] for r in subset]),
                         "x1_normalized_asv": np.mean([r["metrics"][1]["normalized_asv"] for r in subset]),
                         "min_ess_per_second_total": np.mean([min(r["metrics"][k]["ess_per_second_total"] for k in (0, 1)) for r in subset]),
                         "max_rhat": max(r["metrics"][k]["rhat"] for r in subset for k in range(4))})
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib.pyplot as plt
    # Separate traces prevent six overlaid trajectories from obscuring sticking.
    fig, axes = plt.subplots(7, len(cfg["tasks"]), figsize=(13, 14), squeeze=False,
                              gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1, 1.5]})
    for j, task in enumerate(cfg["tasks"]):
        for i, method in enumerate(cfg["methods"]):
            rec = next(r for r in records if r["task"] == task and r["method"] == method)
            data = np.load(args.output / rec["artifact"] / "samples.npz")
            ax = axes[i, j]
            ax.plot(data["states"][:1500, 0, 0], color=COLORS[method], lw=0.65)
            if j == 0:
                ax.set_ylabel(LABELS[method], fontsize=8)
            if i == 0:
                ax.set_title(f"{task}: one x0 chain")
            if i < 5:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Transition")
            axes[6, j].plot(data["acf"][:250, 0], color=COLORS[method], label=LABELS[method])
        axes[6, j].set(xlabel="Lag", ylabel="x0 autocorrelation", ylim=(-0.2, 1.05))
    axes[6, -1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output / "figures/04_trace_and_acf.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, len(cfg["tasks"]), figsize=(12, 3.6), squeeze=False)
    short_labels = ["RWM", "AM", "MALA", "Cov-RWM", "ESJD", "Poisson"]
    for ax, task in zip(axes[0], cfg["tasks"]):
        task_rows = [next(r for r in rows if r["task"] == task and r["method"] == m)
                     for m in cfg["methods"]]
        ax.bar(range(6), [r["mean_normalized_asv"] for r in task_rows],
               yerr=[r["asv_seed_se"] for r in task_rows], capsize=3,
               color=[COLORS[m] for m in cfg["methods"]])
        ax.set_xticks(range(6), short_labels, rotation=25, ha="right")
        ax.set(title=task, yscale="log", ylabel="Mean normalized ASV (x0, x1)")
    fig.suptitle("Final pilot: lower is better; mean +/- SE, 3 training seeds", y=1.02)
    fig.tight_layout()
    fig.savefig(args.output / "figures/08_summary.png", bbox_inches="tight")
    plt.close(fig)
    json_write(args.output / "records.json", records)
    json_write(args.output / "manifest.json", {"main_study": str(args.main.resolve()),
                                               "classical_scale_check": str(args.scale_check.resolve()),
                                               "initial_study": str(args.initial.resolve()),
                                               "corrected_learned_study": str(args.learned.resolve()),
                                               "config": cfg, "sample_files_are_linked_not_duplicated": True})
    text = ["# 最终预实验比较（第二轮 + 经典步长敏感性检查）", "",
            "训练后的固定 kernel，以两个训练统计函数归一化渐近方差的平均作为汇总指标；越低越好。"
            "这里只做探索性预实验，第二轮参数根据第一轮观察调整，未进行独立确认性检验。"
            "学习方法使用最终修正的成对链分组；经典方法复用第二轮结果（该修正不涉及经典方法）。", "",
            "| 任务 | 方法 | 平均归一化 ASV ± SE | 最小 ESS/s（含训练） | 最差 R-hat |",
            "|---|---|---:|---:|---:|"]
    for row in rows:
        rhat = f"{row['max_rhat']:.3f}" if row["max_rhat"] < 1e6 else ">10⁶（未混合）"
        text.append(f"| {row['task']} | {LABELS[row['method']]} | {row['mean_normalized_asv']:.2f} ± "
                    f"{row['asv_seed_se']:.2f} | {row['min_ess_per_second_total']:.1f} | {rhat} |")
    text += ["", "Cov-RWM 使用共同 pilot 的协方差，在8个尺度候选上用训练期 ESJD 选择尺度；"
             "总训练转移预算与第二轮其他方法相同，没有调用解析矩或独立目标采样器选尺度。", "",
             "R-hat >1.01 的结果须谨慎解释；明显未混合的经典链，其短链 ASV 不能看作真实渐近方差，"
             "不应把由此形成的巨大比值当成可靠速度提升。", "", "## Poisson 的相对观测结果", ""]
    for task in cfg["tasks"]:
        task_rows = {r["method"]: r for r in rows if r["task"] == task}
        best = min(("rwm", "am", "mala", "cov_grid"), key=lambda m: task_rows[m]["mean_normalized_asv"])
        p = task_rows["poisson"]["mean_normalized_asv"]
        text.append(f"- {task}：相对本次估计最低的经典对照 {LABELS[best]}，ASV 观测比 "
                    f"{task_rows[best]['mean_normalized_asv']/p:.2f}×；相对同架构 ESJD，"
                    f"{task_rows['esjd']['mean_normalized_asv']/p:.2f}×。")
    if args.initial.exists():
        initial = read_records(args.initial, args.output)
        text += ["", "## 一次受控迭代", "",
                 "v1→v2：每批步数16→32、轮数96→128、Actor LR 0.012→0.008、Critic ridge 0.001→0.003。"
                 "所有对照同步增加训练转移预算，使用相同种子；最终又修正了与双峰初始分配重合的奇偶链拆分。"
                 "多个因素同时改变，因此不能拆分各因素贡献。早期结果只作为开发过程记录。", "",
                 "| 任务 | v1 Poisson 平均归一化 ASV | v2 Poisson 平均归一化 ASV |", "|---|---:|---:|"]
        for task in cfg["tasks"]:
            old = [np.mean([r["metrics"][k]["normalized_asv"] for k in (0, 1)])
                   for r in initial if r["task"] == task and r["method"] == "poisson"]
            new = next(r["mean_normalized_asv"] for r in rows if r["task"] == task and r["method"] == "poisson")
            text.append(f"| {task} | {np.mean(old):.2f} | {new:.2f} |")
    text += ["", "详细科学解读见项目 `docs/FINDINGS.md`；完整逐次指标在各输入实验目录的 metrics.csv。", "", "## 图", ""]
    for path in sorted((args.output / "figures").glob("*.png")):
        text += [f"![{path.stem}](figures/{path.name})", ""]
    (args.output / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    print(args.output.resolve() / "REPORT.md")


if __name__ == "__main__":
    main()
