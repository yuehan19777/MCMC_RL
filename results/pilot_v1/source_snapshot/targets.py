"""Analytic two-dimensional targets. Reference sampling is evaluation-only."""
from dataclasses import dataclass
import math

import torch


STAT_NAMES = ("x0", "x1", "x0_squared", "right_half")


@dataclass(frozen=True)
class Target:
    name: str

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "gaussian":
            z0 = x[..., 0]
            z1 = (x[..., 1] / 3.0 - 0.95 * z0) / math.sqrt(1 - 0.95**2)
            return -0.5 * (z0.square() + z1.square())
        if self.name == "banana":
            z0 = x[..., 0] / 2.0
            z1 = (x[..., 1] - 0.35 * (x[..., 0].square() - 4.0)) / 0.5
            return -0.5 * (z0.square() + z1.square())
        if self.name == "mixture":
            x0, x1 = x.unbind(-1)
            components = torch.stack((-0.5 * ((x0 - 4) / 0.7).square(),
                                      -0.5 * ((x0 + 4) / 0.7).square()), -1)
            return torch.logsumexp(components, -1) - 0.5 * (x1 / 1.2).square()
        raise ValueError(f"Unknown target: {self.name}")

    def grad_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        # Analytic gradients keep the MALA cost explicit and avoid an autograd tape.
        if self.name == "gaussian":
            a = 1.0 / (1 - 0.95**2)
            return torch.stack((-a * (x[..., 0] - 0.95 * x[..., 1] / 3),
                                -a / 3 * (x[..., 1] / 3 - 0.95 * x[..., 0])), -1)
        if self.name == "banana":
            residual = x[..., 1] - 0.35 * (x[..., 0].square() - 4)
            return torch.stack((-x[..., 0] / 4 + 2.8 * x[..., 0] * residual,
                                -4 * residual), -1)
        if self.name == "mixture":
            # posterior mean of the component center = 4*tanh(4*x0 / .7^2)
            center = 4 * torch.tanh(4 * x[..., 0] / 0.49)
            return torch.stack((-(x[..., 0] - center) / 0.49,
                                -x[..., 1] / 1.44), -1)
        raise ValueError(self.name)

    def initial(self, chains: int, generator: torch.Generator) -> torch.Tensor:
        """Dispersed, deliberately nonstationary training starts; no reference draws."""
        x = torch.randn(chains, 2, generator=generator) * 1.5
        if self.name == "mixture":
            x[:, 0] += torch.where(torch.arange(chains) % 2 == 0, -4.0, 4.0)
        return x

    def reference_sample(self, n: int, generator: torch.Generator) -> torch.Tensor:
        z = torch.randn(n, 2, generator=generator)
        if self.name == "gaussian":
            return torch.stack((z[:, 0], 3 * (0.95 * z[:, 0] +
                                math.sqrt(1 - 0.95**2) * z[:, 1])), -1)
        if self.name == "banana":
            x0 = 2 * z[:, 0]
            return torch.stack((x0, 0.5 * z[:, 1] + 0.35 * (x0.square() - 4)), -1)
        if self.name == "mixture":
            sign = 2 * torch.randint(0, 2, (n,), generator=generator) - 1
            return torch.stack((4 * sign + 0.7 * z[:, 0], 1.2 * z[:, 1]), -1)
        raise ValueError(self.name)

    @staticmethod
    def statistics(x: torch.Tensor) -> torch.Tensor:
        return torch.stack((x[..., 0], x[..., 1], x[..., 0].square(),
                            (x[..., 0] > 0).to(x.dtype)), -1)

    def moments(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact means and marginal variances; never used to train/tune a kernel."""
        if self.name == "gaussian":
            second, v1, v_square = 1.0, 9.0, 2.0
        elif self.name == "banana":
            second, v1, v_square = 4.0, 0.25 + 0.35**2 * 32, 32.0
        elif self.name == "mixture":
            second, v1, v_square = 16.49, 1.44, 4 * 16 * 0.49 + 2 * 0.49**2
        else:
            raise ValueError(self.name)
        return (torch.tensor([0.0, 0.0, second, 0.5]),
                torch.tensor([second, v1, v_square, 0.25]))
