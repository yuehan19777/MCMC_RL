"""Common pilot, classical warmup, and two-time-scale actor/critic updates."""
from dataclasses import dataclass
import math
import time

import torch

from .critic import PoissonCritic
from .kernels import collect, mh_score_loss, step
from .proposals import FixedGaussian, GaussianProposal, NeuralGaussian
from .targets import Target


@dataclass
class Pilot:
    center: torch.Tensor
    covariance: torch.Tensor
    final_x: torch.Tensor
    rwm_scale: float
    seconds: float
    transitions: int


@dataclass
class Fit:
    proposal: GaussianProposal
    critic: PoissonCritic | None
    history: list[dict]
    seconds: float
    transitions: int
    extra_proposals: int


def generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def critic_chain_mask(chains: int) -> torch.Tensor:
    """Keep adjacent pairs together, so both groups cover both mixture starts."""
    if chains < 4 or chains % 4:
        raise ValueError("train_chains must be a positive multiple of four")
    return (torch.arange(chains) // 2) % 2 == 0


def regularized_covariance(x: torch.Tensor) -> torch.Tensor:
    cov = torch.cov(x.T)
    return 0.95 * cov + 0.05 * torch.diag(cov.diagonal()) + 1e-4 * torch.eye(2)


def common_pilot(target: Target, cfg: dict, seed: int) -> Pilot:
    start = time.perf_counter()
    rng = generator(seed)
    x = target.initial(cfg["train_chains"], rng)
    proposal = FixedGaussian(torch.eye(2), 1.0)
    states = []
    for t in range(cfg["pilot_steps"]):
        item = step(target, proposal, x, rng)
        x = item.next_x
        proposal.log_scale += 0.05 * (item.accepted.double().mean().item() - 0.35)
        if t >= cfg["pilot_steps"] // 2:
            states.append(x)
    points = torch.cat(states)
    return Pilot(points.mean(0), regularized_covariance(points), x.clone(),
                 math.exp(proposal.log_scale), time.perf_counter() - start,
                 cfg["pilot_steps"] * cfg["train_chains"])


def fit(target: Target, method: str, cfg: dict, seed: int, pilot: Pilot) -> Fit:
    start = time.perf_counter()
    torch.manual_seed(seed + 1000)  # same network initialization for ESJD and Poisson
    rng = generator(seed + 2000)
    x = pilot.final_x.clone()
    learned = method in ("poisson", "esjd")
    critic = None
    if learned:
        proposal = NeuralGaussian(pilot.center, pilot.covariance, 1.5, cfg["actor_width"])
        optimizer = torch.optim.Adam(proposal.parameters(), lr=cfg["actor_lr"])
        if method == "poisson":
            critic = PoissonCritic(pilot.center, pilot.covariance.diagonal().sqrt(),
                                    cfg["critic_width"], cfg["critic_features"],
                                    cfg["critic_ridge"])
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg["critic_lr"])
    elif method == "rwm":
        proposal = FixedGaussian(torch.eye(2), pilot.rwm_scale)
    elif method == "am":
        proposal = FixedGaussian(pilot.covariance, 1.5)
    elif method == "mala":
        proposal = FixedGaussian(pilot.covariance, 0.4, target)
    else:
        raise ValueError(method)

    history = []
    extra_proposals = 0
    covariance_states = []
    for r in range(cfg["rounds"]):
        data, x = collect(target, proposal, x, cfg["steps_per_round"], rng)
        row = {"round": r, "acceptance": data.accepted.double().mean().item()}
        if learned:
            # Split by chain: no proposal used to fit h is reused for the score.
            chain_ids = torch.arange(len(data.x)) % cfg["train_chains"]
            critic_mask = critic_chain_mask(cfg["train_chains"])[chain_ids]
            actor_x = data.x[~critic_mask]
            with torch.no_grad():
                actor_y = proposal.sample(actor_x, rng)
            extra_proposals += len(actor_x)
            if critic is not None:
                cx, cy, ca = data.x[critic_mask], data.proposal[critic_mask], data.alpha[critic_mask]
                # pi(f) is unknown: center only using the current training states.
                g = (cx - pilot.center) / pilot.covariance.diagonal().sqrt()
                g = g - g.mean(0)
                for _ in range(cfg["critic_steps"]):
                    critic.solve(cx, cy, ca, g)
                    critic_optimizer.zero_grad()
                    loss = critic.loss(cx, cy, ca, g)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
                    critic_optimizer.step()
                row.update(critic.solve(cx, cy, ca, g))
                with torch.no_grad():
                    reward = (critic(actor_y) - critic(actor_x)).square().mean(-1)
            else:
                # Standardized ESJD avoids arbitrary coordinate units favoring one axis.
                reward = ((actor_y - actor_x) /
                          pilot.covariance.diagonal().sqrt()).square().mean(-1)
            # Positive, stopped scalar normalization preserves direction per batch;
            # it changes step size, not the underlying MH-specific score identity.
            normalizer = reward.mean().detach().clamp_min(1e-8)
            optimizer.zero_grad()
            actor_loss = mh_score_loss(target, proposal, actor_x, actor_y,
                                       reward / normalizer)
            actor_loss.backward()
            row["gradient_norm"] = float(torch.nn.utils.clip_grad_norm_(proposal.parameters(), 2.0))
            optimizer.step()
            proposal.project()
            row["reward_scale"] = normalizer.item()
        else:
            desired = 0.574 if method == "mala" else 0.35
            rate = 0.25 / (1 + r / 20)**0.6
            proposal.log_scale += rate * (row["acceptance"] - desired)
            proposal.log_scale = max(-5.0, min(3.0, proposal.log_scale))
            if method == "am":
                # Adapt covariance using all warmup states after the common pilot.
                covariance_states.append(data.next_x)
                if (r + 1) % 8 == 0:
                    covariance = regularized_covariance(torch.cat(covariance_states))
                    proposal.covariance.copy_(covariance)
                    proposal.base_chol.copy_(torch.linalg.cholesky(covariance))
        row["proposal_scale"] = proposal.cholesky().diagonal().tolist()
        history.append(row)

    proposal.eval()
    for parameter in proposal.parameters():
        parameter.requires_grad_(False)
    if critic is not None:
        critic.eval()
    return Fit(proposal, critic, history, time.perf_counter() - start,
               cfg["rounds"] * cfg["steps_per_round"] * cfg["train_chains"], extra_proposals)
