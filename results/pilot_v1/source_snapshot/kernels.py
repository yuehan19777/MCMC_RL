"""A single MH implementation shared by every method."""
from dataclasses import dataclass

import torch

from .proposals import GaussianProposal
from .targets import Target


@dataclass
class Transition:
    x: torch.Tensor
    proposal: torch.Tensor
    next_x: torch.Tensor
    alpha: torch.Tensor
    accepted: torch.Tensor


def log_acceptance(target: Target, proposal: GaussianProposal,
                   x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    forward = proposal.log_prob(y, x)
    reverse = proposal.log_prob(x, y)
    ratio = target.log_prob(y) - target.log_prob(x) + reverse - forward
    return ratio.clamp(max=0.0), forward


@torch.no_grad()
def step(target: Target, proposal: GaussianProposal, x: torch.Tensor,
         generator: torch.Generator) -> Transition:
    y = proposal.sample(x, generator)
    log_alpha, _ = log_acceptance(target, proposal, x, y)
    accepted = torch.rand(x.shape[0], generator=generator).log() < log_alpha
    next_x = torch.where(accepted[:, None], y, x)
    return Transition(x, y, next_x, log_alpha.exp(), accepted)


@torch.no_grad()
def collect(target: Target, proposal: GaussianProposal, x: torch.Tensor,
            steps: int, generator: torch.Generator) -> tuple[Transition, torch.Tensor]:
    records = []
    for _ in range(steps):
        item = step(target, proposal, x, generator)
        records.append(item)
        x = item.next_x
    data = Transition(*(torch.cat([getattr(r, key) for r in records], dim=0)
                        for key in Transition.__dataclass_fields__))
    return data, x


def mh_score_loss(target: Target, proposal: GaussianProposal,
                  x: torch.Tensor, y: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
    """Gradient = -E[alpha * reward * grad log(q_forward * alpha)].

    The sampled points, alpha weight and reward MUST be stopped. Differentiating
    alpha a second time, or differentiating through y, changes this estimator.
    log_alpha includes the reverse density, even for an asymmetric neural proposal.
    """
    x, y = x.detach(), y.detach()
    log_alpha, forward = log_acceptance(target, proposal, x, y)
    weight = (log_alpha.exp() * reward.detach()).detach()
    return -(weight * (forward + log_alpha)).mean()
