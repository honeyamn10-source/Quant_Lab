"""Kelly position sizing: optimal and uncertainty-aware betting fractions."""

from __future__ import annotations

import numpy as np


def kelly_fraction(win_prob: float, payoff_ratio: float) -> float:
    """Binary Kelly fraction f = p - (1 - p) / b, clamped to [0, 1)."""
    if not 0.0 < win_prob < 1.0:
        raise ValueError(f"win_prob must lie in (0, 1), got {win_prob}")
    if payoff_ratio <= 0.0:
        raise ValueError(f"payoff_ratio must be positive, got {payoff_ratio}")
    f = win_prob - (1.0 - win_prob) / payoff_ratio
    return float(min(max(f, 0.0), float(np.nextafter(1.0, 0.0))))


def fractional_kelly(full_kelly: float, fraction: float) -> float:
    """Scaled Kelly fraction, clamped to [0, 1)."""
    return float(min(max(full_kelly * fraction, 0.0), float(np.nextafter(1.0, 0.0))))


def optimal_f_from_history(returns: np.ndarray, grid: int = 400) -> float:
    """Grid argmax of mean log(1 + f*r) over f in [0, 1).

    Any f for which a realised return drives 1 + f*r non-positive is treated as
    infinitely bad (log growth collapses to -inf).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0 or int(grid) < 2:
        return 0.0
    f_grid = np.linspace(0.0, 1.0, int(grid), endpoint=False)
    one_plus = 1.0 + np.outer(f_grid, r)
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = np.log(np.where(one_plus > 0.0, one_plus, 0.0))
    means = np.mean(growth, axis=1)
    return float(f_grid[int(np.argmax(means))])


def kelly_continuous(returns: np.ndarray) -> float:
    """Gaussian continuous Kelly (mu_annual - rf) / sigma_annual^2, clamped to [0, 1).

    The per-year scaling cancels in the ratio, so the daily mean and volatility
    measured over the sample determine the fraction; rf is taken as zero.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 2:
        return 0.0
    var = float(np.var(r, ddof=1))
    if var <= 0.0:
        return 0.0
    f = float(np.mean(r)) / var
    return float(min(max(f, 0.0), float(np.nextafter(1.0, 0.0))))


def edge_uncertainty_kelly(
    returns: np.ndarray, n_draws: int = 300, seed: int = 0, quantile: float = 0.05
) -> dict:
    """Bootstrap distribution of the growth-optimal fraction on re-drawn returns.

    A conservative plain-vanilla policy is to bet fractional_kelly at the
    kelly_p5 point of this distribution.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = r.size
    n_draws = int(n_draws)
    if n == 0 or n_draws <= 0:
        return {"kelly_p5": 0.0, "kelly_median": 0.0, "kelly_mean": 0.0, "n_draws": 0}
    rng = np.random.default_rng(seed)
    fs = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        sample = r[rng.integers(0, n, size=n)]
        fs[i] = optimal_f_from_history(sample)
    return {
        "kelly_p5": float(np.quantile(fs, quantile)),
        "kelly_median": float(np.median(fs)),
        "kelly_mean": float(np.mean(fs)),
        "n_draws": n_draws,
    }
