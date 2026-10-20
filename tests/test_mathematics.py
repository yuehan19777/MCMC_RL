"""Tests focus on scientific identities and diagnostics, not line coverage."""
import numpy as np
import pytest
import torch

from poisson_mcmc.critic import PoissonCritic
from poisson_mcmc.diagnostics import batch_variance, rank_split_rhat, spectral_variance
from poisson_mcmc.kernels import log_acceptance, mh_score_loss
from poisson_mcmc.proposals import NeuralGaussian
from poisson_mcmc.targets import Target
from poisson_mcmc.training import critic_chain_mask


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def test_critic_and_actor_groups_both_cover_mixture_initial_modes():
    target = Target("mixture")
    x = target.initial(16, torch.Generator().manual_seed(17))
    mask = critic_chain_mask(16)
    assert mask.sum() == 8
    for group in (mask, ~mask):
        assert (x[group, 0] < 0).any()
        assert (x[group, 0] > 0).any()


@pytest.mark.parametrize("name", ["gaussian", "banana", "mixture"])
def test_analytic_target_gradient(name):
    x = torch.tensor([[-2.1, 0.7], [0.3, -1.1], [4.0, 2.0]], requires_grad=True)
    target = Target(name)
    expected, = torch.autograd.grad(target.log_prob(x).sum(), x)
    torch.testing.assert_close(target.grad_log_prob(x), expected)


@pytest.mark.parametrize("name", ["gaussian", "banana", "mixture"])
def test_asymmetric_mh_detailed_balance(name):
    torch.manual_seed(6)
    proposal = NeuralGaussian(torch.zeros(2), torch.tensor([[1., .3], [.3, 2.]]), 1.2)
    with torch.no_grad():
        proposal.drift[-1].weight.normal_(0, 0.3)
    x, y = torch.randn(100, 2), torch.randn(100, 2)
    target = Target(name)
    loga, logq = log_acceptance(target, proposal, x, y)
    reverse_loga, reverse_logq = log_acceptance(target, proposal, y, x)
    torch.testing.assert_close(target.log_prob(x) + logq + loga,
                               target.log_prob(y) + reverse_logq + reverse_loga)


def test_exact_finite_state_variational_and_policy_gradient():
    """Compare the full differentiated Poisson solve and the MH flow score."""
    pi = torch.tensor([0.2, 0.3, 0.5])
    logits = torch.tensor([[0.1, 1.2, -0.3], [-0.4, 0.5, 0.2], [0.6, -0.8, 0.3]], requires_grad=True)
    q = logits.softmax(-1)
    flow = torch.minimum(pi[:, None] * q, pi[None, :] * q.T)
    offdiag = 1 - torch.eye(3)
    k = flow / pi[:, None] * offdiag
    k = k + torch.diag(1 - k.sum(1))
    g = torch.tensor([-1., 0.4, 0.16])
    g = g - (pi * g).sum()
    h = torch.linalg.solve(torch.eye(3) - k + pi[None, :], g)
    exact_asv = 2 * (pi * g * h).sum() - (pi * g.square()).sum()
    exact_gradient, = torch.autograd.grad(exact_asv, logits, retain_graph=True)
    delta2 = (h.detach()[None, :] - h.detach()[:, None]).square()
    log_alpha = (pi.log()[None, :] - pi.log()[:, None] + q.T.log() - q.log()).clamp(max=0)
    score_loss = -((pi[:, None] * q * log_alpha.exp() * delta2).detach() *
                   (q.log() + log_alpha)).sum()
    score_gradient, = torch.autograd.grad(score_loss, logits, retain_graph=True)
    torch.testing.assert_close(score_gradient, exact_gradient, atol=1e-10, rtol=1e-10)
    energy = 0.5 * (flow * delta2).sum()
    objective = 2 * (pi * g * h).sum() - energy
    torch.testing.assert_close(2 * objective - (pi * g.square()).sum(), exact_asv)
    approximate = torch.tensor([0.3, -0.5, 0.1])
    dh = approximate[None, :] - approximate[:, None]
    approx_objective = 2 * (pi * g * approximate).sum() - 0.5 * (flow * dh.square()).sum()
    error = approximate - h
    error_energy = 0.5 * (flow * (error[None, :] - error[:, None]).square()).sum()
    torch.testing.assert_close(objective - approx_objective, error_energy)


def test_continuous_score_has_reverse_density_and_no_pathwise_gradient():
    torch.manual_seed(123)
    target = Target("gaussian")
    p = NeuralGaussian(torch.zeros(2), torch.eye(2), 0.8)
    x = torch.tensor([[0.1, 0.3], [1.1, 3.1], [-0.4, -0.9]], requires_grad=True)
    y = p.sample(x, torch.Generator().manual_seed(9))
    reward = y.square().sum(-1)  # deliberately attached; implementation must stop it
    loss = mh_score_loss(target, p, x, y, reward)
    assert torch.autograd.grad(loss, x, allow_unused=True, retain_graph=True)[0] is None
    parameter = p.log_diag
    actual, = torch.autograd.grad(loss, parameter)
    xd, yd, rd = x.detach(), y.detach(), reward.detach()
    la, forward = log_acceptance(target, p, xd, yd)
    ratio = target.log_prob(yd) - target.log_prob(xd) + p.log_prob(xd, yd) - forward
    chosen_score = torch.where(ratio > 0, forward, p.log_prob(xd, yd))
    expected_loss = -((la.exp() * rd).detach() * chosen_score).mean()
    expected, = torch.autograd.grad(expected_loss, parameter)
    torch.testing.assert_close(actual, expected)


def test_critic_quadratic_optimum_matches_regularized_first_order_condition():
    torch.manual_seed(42)
    c = PoissonCritic(torch.zeros(2), torch.ones(2), neural_features=4, ridge=1e-3)
    x, y = torch.randn(2000, 2), torch.randn(2000, 2)
    alpha = torch.ones(2000)
    g = x - x.mean(0)
    c.solve(x, y, alpha, g)
    coeff = c.coefficients.clone().requires_grad_()
    fx, fy = c.features(x).detach(), c.features(y).detach()
    hx = (fx - fx.mean(0)) @ coeff
    dh = (fy - fx) @ coeff
    objective = (2 * g * hx).mean(0) - 0.5 * dh.square().mean(0) - c.ridge * coeff.square().sum(0)
    grad, = torch.autograd.grad(objective.sum(), coeff)
    assert grad.abs().max() < 1e-9


def test_ar1_asymptotic_variance_and_rhat():
    rng = np.random.default_rng(77)
    n, m, rho = 16000, 16, 0.8
    x = np.empty((n, m))
    x[0] = rng.normal(size=m)
    for i in range(1, n):
        x[i] = rho * x[i-1] + np.sqrt(1-rho**2) * rng.normal(size=m)
    sigma, ess, _ = spectral_variance(x)
    truth = (1 + rho) / (1 - rho)
    assert abs(sigma / truth - 1) < 0.12
    assert abs(batch_variance(x) / truth - 1) < 0.18
    assert rank_split_rhat(x) < 1.01
    shifted = x.copy()
    shifted[:, :8] += 4
    assert rank_split_rhat(shifted) > 1.3
    assert ess > 0


@pytest.mark.parametrize("name", ["gaussian", "banana", "mixture"])
def test_reference_moments(name):
    target = Target(name)
    x = target.reference_sample(160000, torch.Generator().manual_seed(81))
    f = target.statistics(x)
    mean, variance = target.moments()
    assert torch.all(torch.abs((f.mean(0) - mean) / variance.sqrt()) < 0.02)
    torch.testing.assert_close(f.var(0), variance, rtol=0.06, atol=0.01)
