"""Independent evaluation diagnostics, never computed from the training critic."""
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata


def autocovariance(values: np.ndarray) -> np.ndarray:
    """Biased FFT autocovariance, values shape (draw, chain)."""
    n = values.shape[0]
    centered = values - values.mean(0)
    size = 1 << (2 * n - 1).bit_length()
    fft = np.fft.rfft(centered, n=size, axis=0)
    return np.fft.irfft(fft.conj() * fft, n=size, axis=0)[:n] / n


def spectral_variance(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Geyer initial positive/monotone sequence, with between-chain correction.

    Return sigma^2, ESS, and rho. This is a raw-statistic estimator; the R-hat
    reported separately is rank normalized and folded. ESS is allowed above N.
    """
    n, m = values.shape
    acov = autocovariance(values)
    w = np.var(values, axis=0, ddof=1).mean()
    b = n * np.var(values.mean(0), ddof=1) if m > 1 else 0.0
    var_plus = (n - 1) / n * w + b / n
    if var_plus <= 1e-15:
        return float("nan"), float("nan"), np.full(n, np.nan)
    rho = 1 - (w - acov.mean(1)) / var_plus
    rho[0] = 1
    pairs = []
    for k in range(0, n - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair <= 0:
            break
        pairs.append(min(pair, pairs[-1]) if pairs else pair)
    tau = max(-1 + 2 * np.sum(pairs), 1.0 / np.log10(max(n * m, 10)))
    return float(var_plus * tau), float(n * m / tau), rho


def batch_variance(values: np.ndarray, batch_size: int | None = None) -> float:
    """Non-overlapping batch-means asymptotic variance, average over chains."""
    n, m = values.shape
    size = batch_size or max(2, int(np.sqrt(n)))
    batches = n // size
    if batches < 2:
        return float("nan")
    means = values[:batches * size].reshape(batches, size, m).mean(1)
    return float(size * np.var(means, axis=0, ddof=1).mean())


def _basic_rhat(x: np.ndarray) -> float:
    n = x.shape[0]
    w = np.var(x, axis=0, ddof=1).mean()
    b = n * np.var(x.mean(0), ddof=1)
    if w == 0:
        return float("inf") if b > 0 else float("nan")
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def rank_split_rhat(x: np.ndarray) -> float:
    n, m = x.shape
    half = n // 2
    split = np.concatenate((x[:half], x[-half:]), axis=1)

    def normalize(v: np.ndarray) -> np.ndarray:
        ranks = rankdata(v.ravel(), method="average")
        z = ndtri((ranks - 3.0 / 8) / (len(ranks) + 1.0 / 4))
        return z.reshape(v.shape)

    return max(_basic_rhat(normalize(split)),
               _basic_rhat(normalize(np.abs(split - np.median(split)))))


def summarize(samples: np.ndarray, means: np.ndarray, variances: np.ndarray,
              eval_seconds: float, total_seconds: float) -> tuple[list[dict], np.ndarray]:
    n, chains, stats = samples.shape
    rows, acfs = [], []
    for k in range(stats):
        x = samples[..., k]
        sigma2, ess, acf = spectral_variance(x)
        empirical = n * np.var(x.mean(0), ddof=1)
        errors = x.mean(0) - means[k]
        rows.append({
            "stat_index": k,
            "asv_spectral": sigma2,
            "asv_batch": batch_variance(x),
            "asv_batch_long": batch_variance(x, max(2, n // 10)),
            "asv_replicate": float(empirical),
            "normalized_asv": sigma2 / variances[k],
            "ess": ess,
            "ess_per_draw": ess / (n * chains),
            "ess_per_second": ess / eval_seconds,
            "ess_per_second_total": ess / total_seconds,
            "bias": float(errors.mean()),
            "chain_rmse": float(np.sqrt(np.mean(errors**2))),
            "standardized_rmse": float(np.sqrt(np.mean(errors**2) / variances[k])),
            "rhat": rank_split_rhat(x),
            "true_mean": means[k],
            "true_variance": variances[k],
        })
        acfs.append(acf)
    return rows, np.stack(acfs, -1)
