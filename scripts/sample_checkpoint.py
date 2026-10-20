"""Restore a saved frozen proposal and draw new samples without retraining."""
import argparse
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from poisson_mcmc.kernels import step
from poisson_mcmc.proposals import FixedGaussian, NeuralGaussian
from poisson_mcmc.targets import Target


def restore(path: Path):
    torch.set_default_dtype(torch.float64)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved["proposal_state_dict"]
    target = Target(saved["target"])
    if saved["method"] in ("poisson", "esjd"):
        cov = state["base_chol"] @ state["base_chol"].T
        proposal = NeuralGaussian(state["center"], cov, 1.0, saved["config"]["actor_width"])
    else:
        proposal = FixedGaussian(state["covariance"], math.exp(saved["fixed_log_scale"]),
                                 target if saved["method"] == "mala" else None)
    proposal.load_state_dict(state)
    proposal.eval()
    for parameter in proposal.parameters():
        parameter.requires_grad_(False)
    return target, proposal


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--burnin", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.draws, args.chains) < 1 or args.burnin < 0:
        parser.error("draws/chains must be positive; burnin must be nonnegative")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    torch.set_num_threads(1)
    target, proposal = restore(args.checkpoint)
    rng = torch.Generator().manual_seed(args.seed)
    x = target.initial(args.chains, rng)
    samples, accepted = [], 0
    for t in range(args.burnin + args.draws):
        data = step(target, proposal, x, rng)
        x = data.next_x
        if t >= args.burnin:
            samples.append(x.numpy())
            accepted += data.accepted.sum().item()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, states=np.stack(samples), seed=args.seed,
                        acceptance=accepted / (args.draws * args.chains))
    print(f"Saved {args.draws} draws x {args.chains} chains to {args.output}")


if __name__ == "__main__":
    main()
