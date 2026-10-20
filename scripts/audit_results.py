"""Audit scientific artifact integrity independently of the training loop."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from poisson_mcmc.targets import Target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=Path("results/final_comparison"))
    parser.add_argument("--output", type=Path, default=Path("results/validation/audit.json"))
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    root = args.comparison.resolve()
    records = json.loads((root / "records.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    expected = {(t, m, s) for t in cfg["tasks"] for m in cfg["methods"] for s in cfg["seeds"]}
    actual = [(r["task"], r["method"], r["seed"]) for r in records]
    assert len(actual) == len(set(actual)) and set(actual) == expected
    for record in records:
        folder = root / record["artifact"]
        with np.load(folder / "samples.npz") as data:
            states, stats = data["states"], data["statistics"]
            assert states.shape == (cfg["eval_steps"], cfg["eval_chains"], 2)
            assert stats.shape == (*states.shape[:-1], 4)
            assert np.isfinite(states).all() and np.isfinite(stats).all()
            target = Target(record["task"])
            np.testing.assert_array_equal(target.statistics(torch.from_numpy(states)).numpy(), stats)
            assert data["acf"].shape == (cfg["eval_steps"], 4)
            if record["task"] == "mixture":
                assert data["cold_occupancy"].shape == (cfg["stress_steps"],)
                assert np.all((data["cold_first_passage"] >= 1) &
                              (data["cold_first_passage"] <= cfg["stress_steps"] + 1))
        means, variances = target.moments()
        for index, metric in enumerate(record["metrics"]):
            assert all(np.isfinite(v) for v in metric.values())
            np.testing.assert_allclose(metric["true_mean"], means[index].item())
            np.testing.assert_allclose(metric["true_variance"], variances[index].item())
            np.testing.assert_allclose(metric["normalized_asv"],
                                       metric["asv_spectral"] / metric["true_variance"])
        meta = record["metadata"]
        assert 0 <= meta["acceptance"] <= 1
        assert meta["training_transitions"] == (cfg["pilot_steps"] + cfg["rounds"] * cfg["steps_per_round"]) * cfg["train_chains"]
        assert meta["evaluation_transitions"] == cfg["eval_steps"] * cfg["eval_chains"]
        assert meta["logp_calls_total"] == 2 * (meta["training_transitions"] +
                                                   meta["training_extra_proposals"] + meta["evaluation_transitions"])
        checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert all(torch.isfinite(tensor).all() for tensor in checkpoint["proposal_state_dict"].values())
    verified_snapshots = []
    for key in ("main_study", "initial_study", "corrected_learned_study"):
        experiment = Path(manifest[key])
        source = experiment / "source_snapshot"
        saved = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
        fingerprint = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes()
                                               for p in sorted(source.glob("*.py")))).hexdigest()
        assert fingerprint == saved["source_sha256"], experiment
        verified_snapshots.append(str(source))
    scale_root = Path(manifest["classical_scale_check"])
    scale_source = scale_root / "source_snapshot"
    scale_saved = json.loads((scale_root / "manifest.json").read_text(encoding="utf-8"))
    grid_script = scale_source / "check_classical_scale.py"
    grid_modules = sorted(p for p in scale_source.glob("*.py") if p.name != grid_script.name)
    grid_hash = hashlib.sha256(grid_script.read_bytes() + b"".join(p.read_bytes() for p in grid_modules)).hexdigest()
    assert grid_hash == scale_saved["source_sha256"]
    verified_snapshots.append(str(scale_source))
    result = {"status": "passed", "unique_runs": len(records),
              "metrics_rows": sum(len(r["metrics"]) for r in records),
              "stationary_evaluation_transitions": sum(r["metadata"]["evaluation_transitions"] for r in records),
              "verified_source_snapshots": verified_snapshots,
              "checks": ["sample shapes and finite values", "statistics recalculated from saved states",
                         "true moments and ASV units", "density call and transition budgets",
                         "checkpoint restoration and finite parameters", "unique task/method/seed tuples",
                         "source SHA256 snapshots"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
