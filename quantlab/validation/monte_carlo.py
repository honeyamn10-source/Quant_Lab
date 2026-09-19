"""Monte Carlo reshuffle / null-model engine.

If a strategy routinely "outperforms" reshuffled signals, the research pipeline
is broken. These routines build null distributions from the observed data and
return p-values for the observed statistic.
"""

from __future__ import annotations

import numpy as np

from quantlab.validation.metrics import sharpe_ratio


def shuffle_pvalue(
    returns, stat_fn, n_iter: int = 500, seed: int = 0, greater: bool = True
) -> dict:
    """p-value of the observed statistic under a reshuffle null (no serial edge)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    rng = np.random.default_rng(seed)
    observed = float(stat_fn(r))
    n = len(r)
    if n < 2:
        return {"pvalue": 1.0, "observed": observed, "null_mean": 0.0}
    null = np.empty(n_iter)
    for i in range(n_iter):
        shuffled = r[rng.permutation(n)]
        null[i] = float(stat_fn(shuffled))
    pvalue = float(np.mean(null >= observed)) if greater else float(np.mean(null <= observed))
    return {
        "pvalue": max(min(pvalue, 1.0), 1.0 / n_iter),
        "observed": observed,
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null)),
    }


def monte_carlo_null(returns, n_iter: int = 500, seed: int = 0) -> dict:
    """Null distribution for the Sharpe ratio from return reshuffles."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    rng = np.random.default_rng(seed)
    n = len(r)
    null = np.empty(n_iter)
    for i in range(n_iter):
        null[i] = sharpe_ratio(r[rng.permutation(n)])
    return {
        "mean_null_sharpe": float(np.mean(null)),
        "std_null_sharpe": float(np.std(null)),
        "p95_null_sharpe": float(np.quantile(null, 0.95)),
        "samples": null,
    }


def monte_carlo_paths(
    n_paths: int = 1000, n_periods: int = 252, mu: float = 0.0, sigma: float = 0.05, seed: int = 0
) -> np.ndarray:
    """Simulate return paths under a Gaussian null model."""
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, size=(n_paths, n_periods))
