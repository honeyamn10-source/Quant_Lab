"""Bootstrap framework: IID, block and stationary resampling with CIs."""

from __future__ import annotations

import numpy as np


def _stat(returns: np.ndarray, stat_fn) -> float:
    v = stat_fn(returns)
    if np.isscalar(v) or np.ndim(v) == 0:
        return float(v)
    return float(v)


def iid_bootstrap(returns, stat_fn, n_iter: int = 1000, seed: int = 0) -> np.ndarray:
    r = np.asarray(returns, dtype=float)[~np.isnan(np.asarray(returns, dtype=float))]
    rng = np.random.default_rng(seed)
    n = len(r)
    if n == 0:
        return np.zeros(0)
    out = np.empty(n_iter)
    for i in range(n_iter):
        sample = r[rng.integers(0, n, size=n)]
        out[i] = _stat(sample, stat_fn)
    return out


def block_bootstrap(
    returns, stat_fn, block_size: int = 10, n_iter: int = 1000, seed: int = 0
) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    rng = np.random.default_rng(seed)
    n = len(r)
    if n == 0:
        return np.zeros(0)
    bs = int(block_size)
    out = np.empty(n_iter)
    for i in range(n_iter):
        blocks = int(np.ceil(n / bs))
        idx = rng.integers(0, n - bs + 1, size=blocks)
        sample = np.concatenate([r[j : j + bs] for j in idx])[:n]
        out[i] = _stat(sample, stat_fn)
    return out


def stationary_bootstrap(
    returns, stat_fn, mean_block: float = 5.0, n_iter: int = 1000, seed: int = 0
) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    rng = np.random.default_rng(seed)
    n = len(r)
    if n == 0:
        return np.zeros(0)
    prob = 1.0 / max(float(mean_block), 1.0)
    out = np.empty(n_iter)
    for i in range(n_iter):
        idx = np.empty(n, dtype=int)
        pos = int(rng.integers(0, n))
        idx[0] = pos
        for t in range(1, n):
            pos = int(rng.integers(0, n)) if rng.random() < prob else (pos + 1) % n
            idx[t] = pos
        out[i] = _stat(r[idx], stat_fn)
    return out


def confidence_interval(samples: np.ndarray, alpha: float = 0.05) -> dict:
    """Two-sided bootstrap confidence interval."""
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        return {"lo": 0.0, "hi": 0.0, "mean": 0.0}
    lo, hi = np.quantile(samples, alpha / 2.0), np.quantile(samples, 1.0 - alpha / 2.0)
    return {"lo": float(lo), "hi": float(hi), "mean": float(np.mean(samples))}


def bootstrap_sharpe(
    returns, n_iter: int = 1000, seed: int = 0, block_size: int | None = None
) -> dict:
    """Bootstrap the Sharpe ratio (block bootstrap if block_size given)."""
    from quantlab.validation.metrics import sharpe_ratio

    fn = lambda r: sharpe_ratio(r)  # noqa: E731
    samples = (
        block_bootstrap(returns, fn, block_size, n_iter, seed)
        if block_size
        else iid_bootstrap(returns, fn, n_iter, seed)
    )
    return {
        "samples": samples,
        "ci95": confidence_interval(samples),
        "method": "block" if block_size else "iid",
    }
