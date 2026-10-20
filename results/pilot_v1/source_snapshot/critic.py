"""Multi-output variational Poisson critic with a Galerkin last-layer solve."""
import torch
from torch import nn


class PoissonCritic(nn.Module):
    """h(x) = features_phi(x) @ coefficients, one head per trained statistic.

    Linear, quadratic, smooth threshold and neural features provide a small,
    interpretable basis. All feature-map parameters are trained by the supplied
    variational objective; the quadratic last-layer subproblem is solved exactly
    with a documented ridge penalty to handle near singular Dirichlet matrices.
    """
    def __init__(self, center: torch.Tensor, scale: torch.Tensor,
                 width: int = 16, neural_features: int = 8, ridge: float = 1e-3):
        super().__init__()
        self.register_buffer("center", center.clone())
        self.register_buffer("scale", scale.clone())
        self.features_net = nn.Sequential(nn.Linear(2, width), nn.Tanh(),
                                          nn.Linear(width, neural_features), nn.Tanh())
        self.register_buffer("coefficients", torch.zeros(9 + neural_features, 2))
        self.ridge = ridge

    def features(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.center) / self.scale
        z0, z1 = z.unbind(-1)
        fixed = torch.stack((z0, z1, z0.square(), z1.square(), z0 * z1,
                             torch.tanh(z0), torch.tanh(z1),
                             torch.tanh(3 * z0), torch.tanh(3 * z1)), -1)
        return torch.cat((fixed, self.features_net(z)), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x) @ self.coefficients

    @torch.no_grad()
    def solve(self, x: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor,
              g: torch.Tensor) -> dict[str, float]:
        fx, fy = self.features(x), self.features(y)
        centered = fx - fx.mean(0)
        delta = fy - fx
        # C(c)=2*b^T*c-c^T*A*c; A=1/2 E[alpha delta_phi delta_phi^T].
        a = 0.5 * delta.T @ (alpha[:, None] * delta) / len(x)
        b = centered.T @ (g - g.mean(0)) / len(x)
        penalty = self.ridge * torch.eye(a.shape[0], dtype=a.dtype)
        c = torch.linalg.solve(a + penalty, b)
        if not torch.isfinite(c).all():
            raise FloatingPointError("Nonfinite Poisson coefficients")
        self.coefficients.copy_(c)
        h = centered @ c
        dh = delta @ c
        objective = (2 * (g * h).mean(0) - 0.5 *
                     (alpha[:, None] * dh.square()).mean(0)).mean()
        relative_residual = ((a @ c - b).norm() / b.norm().clamp_min(1e-12)).item()
        return {"critic_objective": objective.item(),
                "galerkin_residual": relative_residual,
                "critic_coefficient_norm": c.norm().item(),
                "dirichlet_condition": torch.linalg.cond(a + penalty).item()}

    def loss(self, x: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor,
             g: torch.Tensor) -> torch.Tensor:
        h = self(x)
        h = h - h.mean(0)
        dh = self(y) - self(x)
        objective = 2 * (g * h).mean(0) - 0.5 * (alpha[:, None] * dh.square()).mean(0)
        return -objective.mean()
