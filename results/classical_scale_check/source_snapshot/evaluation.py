"""Stationary, frozen-kernel evaluation plus a one-mode startup stress test."""
import time

import numpy as np
import torch

from .diagnostics import summarize
from .kernels import step
from .targets import Target
from .training import Fit, Pilot, generator


@torch.no_grad()
def evaluate(target: Target, fitted: Fit, cfg: dict, seed: int,
             pilot: Pilot) -> tuple[list[dict], dict, dict[str, np.ndarray]]:
    rng = generator(seed + 10000)
    # Exact pi draws are only an evaluation device, shared starts across methods.
    x = target.reference_sample(cfg["eval_chains"], generator(seed + 9000))
    states = torch.empty(cfg["eval_steps"], cfg["eval_chains"], 2)
    accepted = 0.0
    crossings = torch.zeros(cfg["eval_chains"])
    start = time.perf_counter()
    for t in range(cfg["eval_steps"]):
        item = step(target, fitted.proposal, x, rng)
        crossings += ((x[:, 0] > 0) != (item.next_x[:, 0] > 0))
        x = item.next_x
        states[t] = x
        accepted += item.accepted.double().sum().item()
    seconds = time.perf_counter() - start
    stats = target.statistics(states).numpy()
    means, variances = target.moments()
    total_seconds = seconds + pilot.seconds + fitted.seconds
    rows, acfs = summarize(stats, means.numpy(), variances.numpy(), seconds, total_seconds)
    n = cfg["eval_steps"] * cfg["eval_chains"]
    # This MH implementation computes pi twice per proposal, no cached pi(x).
    train_proposals = pilot.transitions + fitted.transitions + fitted.extra_proposals
    evaluation_calls = 2 * n
    total_calls = 2 * (train_proposals + n)
    for row in rows:
        row["ess_per_1000_logp"] = row["ess"] / evaluation_calls * 1000
        row["ess_per_1000_logp_total"] = row["ess"] / total_calls * 1000
    metadata = {
        "evaluation_seconds": seconds,
        "pilot_seconds": pilot.seconds,
        "adaptation_seconds": fitted.seconds,
        "total_seconds": total_seconds,
        "evaluation_transitions": n,
        "training_transitions": pilot.transitions + fitted.transitions,
        "training_extra_proposals": fitted.extra_proposals,
        "logp_calls_total": total_calls,
        "gradient_calls_total": 3 * (fitted.transitions + n) if getattr(fitted.proposal, "target", None) else 0,
        "acceptance": accepted / n,
        "crossings_per_chain": crossings.mean().item(),
        "zero_crossing_chains": int((crossings == 0).sum()),
    }
    arrays = {"states": states.numpy(), "statistics": stats, "acf": acfs}
    if target.name == "mixture" and cfg["stress_steps"] > 0:
        stress_meta, stress_arrays = cold_start(target, fitted, cfg, seed)
        metadata.update(stress_meta)
        arrays.update(stress_arrays)
    return rows, metadata, arrays


@torch.no_grad()
def cold_start(target: Target, fitted: Fit, cfg: dict, seed: int) -> tuple[dict, dict]:
    rng = generator(seed + 11000)
    x = torch.randn(cfg["eval_chains"], 2, generator=rng) * 0.1
    x[:, 0] -= 4.0
    sentinel = cfg["stress_steps"] + 1
    first = torch.full((cfg["eval_chains"],), sentinel, dtype=torch.int64)
    occupancy = []
    for t in range(cfg["stress_steps"]):
        x = step(target, fitted.proposal, x, rng).next_x
        right = x[:, 0] > 0
        first = torch.where(right & (first == sentinel), t + 1, first)
        occupancy.append(right.double().mean().item())
    occupancy = np.array(occupancy)
    return {"cold_first_passage_median_censored": float(first.clamp(max=cfg["stress_steps"]).double().median()),
            "cold_not_switched_fraction": float((first == sentinel).double().mean()),
            "cold_right_occupancy": float(occupancy.mean())}, {
                "cold_occupancy": occupancy, "cold_first_passage": first.numpy()}
