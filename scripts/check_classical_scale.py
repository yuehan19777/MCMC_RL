"""Check whether classical results depend on acceptance-rate scale tuning.

Fit a covariance-preconditioned RWM by a finite ESJD scale grid, with exactly the
same pilot and adaptation transition budget as the selected experiment config.
No reference moments or reference samples are used for selecting the scale.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from poisson_mcmc.evaluation import evaluate
from poisson_mcmc.experiment import json_write
from poisson_mcmc.kernels import collect
from poisson_mcmc.proposals import FixedGaussian
from poisson_mcmc.reporting import write_tables
from poisson_mcmc.targets import Target
from poisson_mcmc.training import Fit, common_pilot, generator


SCALES = (0.25, 0.4, 0.65, 1.0, 1.6, 2.5, 4.0, 6.5)


def grid_fit(target, cfg, seed, pilot):
    started = time.perf_counter()
    budget = cfg["rounds"] * cfg["steps_per_round"]
    if budget % len(SCALES):
        raise ValueError("Adaptation steps must be divisible by the grid size")
    steps = budget // len(SCALES)
    history, best_score, best_proposal = [], -np.inf, None
    for index, scale in enumerate(SCALES):
        proposal = FixedGaussian(pilot.covariance, scale)
        data, _ = collect(target, proposal, pilot.final_x.clone(), steps,
                          generator(seed + 30000 + index))
        reward = ((data.proposal - data.x) /
                  pilot.covariance.diagonal().sqrt()).square().mean(-1)
        score = (data.alpha * reward).mean().item()
        history.append({"scale": scale, "training_esjd": score,
                        "acceptance": data.accepted.double().mean().item()})
        if score > best_score:
            best_score, best_proposal = score, proposal
    best_proposal.eval()
    return Fit(best_proposal, None, history, time.perf_counter() - started,
               budget * cfg["train_chains"], 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot_v2.json"))
    parser.add_argument("--output", type=Path, default=Path("results/classical_scale_check"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new, empty output directory")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(cfg["threads"])
    args.output.mkdir(parents=True, exist_ok=True)
    json_write(args.output / "config.json", cfg)
    source_files = [Path(__file__), *sorted((Path(__file__).resolve().parents[1] / "src/poisson_mcmc").glob("*.py"))]
    json_write(args.output / "manifest.json", {
        "description": "Covariance RWM, scale selected on training ESJD only",
        "scales": SCALES, "config": cfg,
        "source_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in source_files)).hexdigest()})
    records = []
    for task_name in cfg["tasks"]:
        target = Target(task_name)
        for seed in cfg["seeds"]:
            pilot = common_pilot(target, cfg, seed)
            fitted = grid_fit(target, cfg, seed, pilot)
            metrics, metadata, arrays = evaluate(target, fitted, cfg, seed, pilot)
            relative = Path("runs") / task_name / "cov_grid" / f"seed_{seed}"
            run_dir = args.output / relative
            run_dir.mkdir(parents=True)
            json_write(run_dir / "history.json", fitted.history)
            np.savez_compressed(run_dir / "samples.npz", **arrays)
            torch.save({"proposal_state_dict": fitted.proposal.state_dict(),
                        "critic_state_dict": None, "fixed_log_scale": fitted.proposal.log_scale,
                        "method": "cov_grid", "target": task_name, "config": cfg,
                        "seed": seed}, run_dir / "checkpoint.pt")
            record = {"task": task_name, "method": "cov_grid", "seed": seed,
                      "artifact": relative.as_posix(), "metrics": metrics, "metadata": metadata}
            json_write(run_dir / "result.json", record)
            records.append(record)
            tau = np.mean([m["normalized_asv"] for m in metrics[:2]])
            print(f"{task_name} seed={seed} selected_scale={np.exp(fitted.proposal.log_scale):.2f} "
                  f"mean_tau={tau:.2f}", flush=True)
    write_tables(args.output, records)
    json_write(args.output / "records.json", records)


if __name__ == "__main__":
    main()
