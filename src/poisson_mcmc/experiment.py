"""CLI orchestration; per-run files are written before aggregate plotting."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy
import torch

from .evaluation import evaluate
from .plotting import make_plots
from .reporting import write_report, write_tables
from .targets import STAT_NAMES, Target
from .training import common_pilot, fit


def json_write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--output", type=Path, default=Path("results/pilot_v1"))
    parser.add_argument("--resume", action="store_true", help="Reuse completed runs with identical config and source")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(cfg["threads"])
    # Load optimizer infrastructure outside per-method timings (first import is costly).
    torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.001)
    source = Path(__file__).parent
    source_hash = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in sorted(source.glob("*.py")))).hexdigest()
    fingerprint = hashlib.sha256((json.dumps(cfg, sort_keys=True) + source_hash).encode()).hexdigest()
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()) and not (args.resume or args.plots_only):
        parser.error(f"Output is not empty: {root}. Use a new directory or --resume.")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint and not args.plots_only:
            parser.error("Source/config changed. Keep old results and choose a new output directory.")
    else:
        json_write(manifest_path, {"config": cfg, "fingerprint": fingerprint, "source_sha256": source_hash,
                                  "python": sys.version, "executable": sys.executable,
                                  "platform": platform.platform(), "torch": torch.__version__,
                                  "numpy": np.__version__, "scipy": scipy.__version__,
                                  "device": "cpu", "dtype": "float64", "statistics": STAT_NAMES,
                                  "protocol": "stationary held-out evaluation; frozen kernel; no oracle training"})
        json_write(root / "config.json", cfg)
        snapshot = root / "source_snapshot"
        snapshot.mkdir(exist_ok=True)
        for path in source.glob("*.py"):
            (snapshot / path.name).write_bytes(path.read_bytes())
    records = []
    started = time.perf_counter()
    for task_name in cfg["tasks"]:
        target = Target(task_name)
        for seed in cfg["seeds"]:
            pilot = None
            for method in cfg["methods"]:
                relative = Path("runs") / task_name / method / f"seed_{seed}"
                run_dir = root / relative
                result_path = run_dir / "result.json"
                if result_path.exists() and (args.resume or args.plots_only):
                    records.append(json.loads(result_path.read_text(encoding="utf-8")))
                    continue
                if args.plots_only:
                    raise FileNotFoundError(result_path)
                if pilot is None:
                    pilot = common_pilot(target, cfg, seed)
                fitted = fit(target, method, cfg, seed, pilot)
                metrics, metadata, arrays = evaluate(target, fitted, cfg, seed, pilot)
                run_dir.mkdir(parents=True, exist_ok=True)
                json_write(run_dir / "history.json", fitted.history)
                np.savez_compressed(run_dir / "samples.npz", **arrays)
                torch.save({"proposal_state_dict": fitted.proposal.state_dict(),
                            "critic_state_dict": fitted.critic.state_dict() if fitted.critic else None,
                            "fixed_log_scale": getattr(fitted.proposal, "log_scale", None),
                            "method": method, "target": task_name, "config": cfg,
                            "seed": seed}, run_dir / "checkpoint.pt")
                record = {"task": task_name, "method": method, "seed": seed,
                          "artifact": relative.as_posix(), "metrics": metrics, "metadata": metadata}
                json_write(result_path, record)
                records.append(record)
                tau = np.mean([row["normalized_asv"] for row in metrics[:2]])
                print(f"{task_name:8s} seed={seed:3d} {method:7s} mean_tau={tau:8.2f} "
                      f"accept={metadata['acceptance']:.3f} "
                      f"fit={fitted.seconds:.1f}s eval={metadata['evaluation_seconds']:.1f}s", flush=True)
    write_tables(root, records)
    make_plots(root, cfg, records)
    write_report(root, cfg, records)
    print(f"Completed {len(records)} runs in {time.perf_counter() - started:.1f}s. {root / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
