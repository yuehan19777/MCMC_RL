"""Gaussian proposals with exact forward and reverse densities."""
import math

import torch
from torch import nn

from .targets import Target


def normal_log_prob(y: torch.Tensor, mean: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
    whitened = torch.linalg.solve_triangular(chol, (y - mean).T, upper=False).T
    return (-0.5 * whitened.square().sum(-1) - chol.diagonal().log().sum()
            - y.shape[-1] * 0.5 * math.log(2 * math.pi))


class GaussianProposal(nn.Module):
    def mean(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def cholesky(self) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        noise = torch.randn(x.shape, generator=generator, dtype=x.dtype)
        return self.mean(x) + noise @ self.cholesky().T

    def log_prob(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return normal_log_prob(y, self.mean(x), self.cholesky())


class FixedGaussian(GaussianProposal):
    def __init__(self, covariance: torch.Tensor, scale: float = 1.0,
                 target: Target | None = None):
        super().__init__()
        self.register_buffer("covariance", covariance.clone())
        self.register_buffer("base_chol", torch.linalg.cholesky(covariance))
        self.log_scale = math.log(scale)
        self.target = target

    def cholesky(self) -> torch.Tensor:
        return self.base_chol * math.exp(self.log_scale)

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        if self.target is None:
            return x
        epsilon2 = math.exp(2 * self.log_scale)
        return x + 0.5 * epsilon2 * (self.target.grad_log_prob(x) @ self.covariance)


class NeuralGaussian(GaussianProposal):
    """Full covariance plus bounded state-dependent neural drift in whitened units.

    Both learned objectives use exactly this family, initialization and optimizer.
    It has no target-specific transport or oracle target gradient.
    """
    def __init__(self, center: torch.Tensor, covariance: torch.Tensor,
                 scale: float, width: int = 16):
        super().__init__()
        self.register_buffer("center", center.clone())
        self.register_buffer("base_chol", torch.linalg.cholesky(covariance))
        self.log_diag = nn.Parameter(torch.full((2,), math.log(scale)))
        self.off_diag = nn.Parameter(torch.zeros(()))
        self.drift = nn.Sequential(nn.Linear(2, width), nn.Tanh(), nn.Linear(width, 2))
        nn.init.zeros_(self.drift[-1].weight)
        nn.init.zeros_(self.drift[-1].bias)

    def cholesky(self) -> torch.Tensor:
        zero = self.off_diag * 0
        relative = torch.stack((self.log_diag[0].exp(), zero,
                                self.off_diag, self.log_diag[1].exp())).reshape(2, 2)
        return self.base_chol @ relative

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.linalg.solve_triangular(self.base_chol, (x - self.center).T,
                                         upper=False).T
        drift = 3.0 * torch.tanh(self.drift(z) / 3.0)
        return x + drift @ self.base_chol.T

    @torch.no_grad()
    def project(self) -> None:
        self.log_diag.clamp_(-2.5, 1.8)
        self.off_diag.clamp_(-3.0, 3.0)
