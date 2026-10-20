"""Small, fixed set of comparison plots; all values come from held-out chains."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .targets import STAT_NAMES, Target


LABELS = {"rwm": "RWM", "am": "Adaptive Metropolis", "mala": "MALA",
          "esjd": "Neural ESJD", "poisson": "Poisson actor-critic"}
COLORS = {"rwm": "#8a8a8a", "am": "#e6a23c", "mala": "#5084bf",
          "esjd": "#7d66af", "poisson": "#159c88"}


def mean_se(values: list[float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, 0.0
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def make_plots(root: Path, cfg: dict, records: list[dict]) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 120})
    out = root / "figures"
    out.mkdir(exist_ok=True)
    tasks, methods = cfg["tasks"], cfg["methods"]

    def subset(task, method):
        return [r for r in records if r["task"] == task and r["method"] == method]

    fig, axes = plt.subplots(len(tasks), 4, figsize=(16, 3.2 * len(tasks)), squeeze=False)
    for i, task in enumerate(tasks):
        for k, stat in enumerate(STAT_NAMES):
            ax = axes[i, k]
            values = [mean_se([r["metrics"][k]["normalized_asv"] for r in subset(task, m)]) for m in methods]
            ax.bar(range(len(methods)), [v[0] for v in values],
                   yerr=[v[1] for v in values], color=[COLORS[m] for m in methods], capsize=3)
            ax.set_yscale("log")
            ax.set_xticks(range(len(methods)), [LABELS[m] for m in methods], rotation=28, ha="right", fontsize=8)
            ax.set_title(f"{task}: {stat}" + (" (trained)" if k < 2 else " (held out)"))
            ax.set_ylabel("ASV / true Var(f), lower is better")
    fig.suptitle("Frozen-kernel asymptotic variance; bars = mean +/- SE across training seeds", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "01_asymptotic_variance.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, len(tasks), figsize=(5 * len(tasks), 7), squeeze=False)
    for j, task in enumerate(tasks):
        for i, metric in enumerate(("ess_per_second", "ess_per_second_total")):
            ax = axes[i, j]
            values = [mean_se([min(r["metrics"][k][metric] for k in (0, 1))
                               for r in subset(task, m)]) for m in methods]
            ax.bar(range(len(methods)), [v[0] for v in values], yerr=[v[1] for v in values],
                   color=[COLORS[m] for m in methods], capsize=3)
            ax.set_xticks(range(len(methods)), [LABELS[m] for m in methods], rotation=30, ha="right", fontsize=8)
            ax.set_title(f"{task}: " + ("sampling only" if i == 0 else "pilot + training + sampling"))
            ax.set_ylabel("Min ESS/second over trained statistics")
    fig.tight_layout()
    fig.savefig(out / "02_cost_efficiency.png")
    plt.close(fig)

    fig, axes = plt.subplots(len(tasks), len(methods), figsize=(3.1 * len(methods), 3 * len(tasks)), squeeze=False)
    import torch
    for i, task in enumerate(tasks):
        reference = subset(task, methods[0])[0]
        points = np.load(root / reference["artifact"] / "samples.npz")["states"]
        lower = np.quantile(points.reshape(-1, 2), 0.003, axis=0) - 0.5
        upper = np.quantile(points.reshape(-1, 2), 0.997, axis=0) + 0.5
        gx, gy = np.meshgrid(np.linspace(lower[0], upper[0], 100),
                             np.linspace(lower[1], upper[1], 100))
        grid = torch.from_numpy(np.stack((gx.ravel(), gy.ravel()), -1))
        lp = Target(task).log_prob(grid).numpy().reshape(gx.shape)
        for j, method in enumerate(methods):
            ax = axes[i, j]
            rec = subset(task, method)[0]
            points = np.load(root / rec["artifact"] / "samples.npz")["states"]
            ax.contour(gx, gy, np.exp(lp - lp.max()), levels=[0.03, 0.2, 0.6], colors="#b8c3cc", linewidths=0.6)
            ax.scatter(points[::5, :, 0].ravel(), points[::5, :, 1].ravel(), s=2,
                       color=COLORS[method], alpha=0.2, rasterized=True)
            ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), title=f"{task}: {LABELS[method]}")
    fig.tight_layout()
    fig.savefig(out / "03_target_coverage.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, len(tasks), figsize=(5 * len(tasks), 6.5), squeeze=False)
    for j, task in enumerate(tasks):
        for method in methods:
            rec = subset(task, method)[0]
            data = np.load(root / rec["artifact"] / "samples.npz")
            axes[0, j].plot(data["states"][:1500, 0, 0], color=COLORS[method], lw=0.7, alpha=0.8, label=LABELS[method])
            axes[1, j].plot(data["acf"][:250, 0], color=COLORS[method], label=LABELS[method])
        axes[0, j].set(title=f"{task}: one x0 chain", xlabel="Transition", ylabel="x0")
        axes[1, j].set(title=f"{task}: x0 autocorrelation", xlabel="Lag", ylabel="rho", ylim=(-0.2, 1.05))
    axes[1, -1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "04_trace_and_acf.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)
    for j, task in enumerate(tasks):
        ax = axes[0, j]
        for mi, method in enumerate(methods):
            for offset, metric, marker in ((-0.12, "asv_spectral", "o"),
                                            (0.0, "asv_batch", "s"),
                                            (0.12, "asv_replicate", "^")):
                values = [np.mean([r["metrics"][k][metric] / r["metrics"][k]["true_variance"]
                                    for k in (0, 1)]) for r in subset(task, method)]
                avg, se = mean_se(values)
                ax.errorbar(mi + offset, avg, yerr=se, fmt=marker, color=COLORS[method], capsize=3,
                            label=metric.replace("asv_", "") if mi == 0 else None)
        ax.set_xticks(range(len(methods)), [LABELS[m] for m in methods], rotation=30, ha="right", fontsize=8)
        ax.set(title=task, ylabel="Mean normalized ASV over x0,x1", yscale="log")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "05_asv_crosscheck.png")
    plt.close(fig)

    if "poisson" in methods:
        fig, axes = plt.subplots(len(tasks), 3, figsize=(14, 3 * len(tasks)), squeeze=False)
        import json
        for i, task in enumerate(tasks):
            for seed_index, rec in enumerate(subset(task, "poisson")):
                history = json.loads((root / rec["artifact"] / "history.json").read_text())
                for j, metric in enumerate(("critic_objective", "galerkin_residual", "acceptance")):
                    axes[i, j].plot([h[metric] for h in history], label=f"seed {rec['seed']}", alpha=0.8)
                    axes[i, j].set(title=f"{task}: {metric}", xlabel="Actor round")
            axes[i, -1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "06_training_diagnostics.png")
        plt.close(fig)

    if "mixture" in tasks and cfg["stress_steps"] > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for method in methods:
            occupancy = np.stack([np.load(root / r["artifact"] / "samples.npz")["cold_occupancy"]
                                   for r in subset("mixture", method)]).mean(0)
            running = np.cumsum(occupancy) / np.arange(1, len(occupancy) + 1)
            axes[0].plot(running, label=LABELS[method], color=COLORS[method])
            times = np.concatenate([np.load(root / r["artifact"] / "samples.npz")["cold_first_passage"]
                                    for r in subset("mixture", method)])
            survival = [(times > t).mean() for t in range(cfg["stress_steps"])]
            axes[1].plot(survival, label=LABELS[method], color=COLORS[method])
        axes[0].axhline(0.5, ls="--", color="black", lw=0.8)
        axes[0].set(title="Mixture: all chains start in left mode", xlabel="Transition", ylabel="Running right-mode occupancy")
        axes[1].set(title="First passage to right mode (right censored)", xlabel="Transition", ylabel="Fraction not yet switched")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "07_cold_start.png")
        plt.close(fig)
