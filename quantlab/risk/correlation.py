"""Correlation analytics across a returns panel (rows = time, cols = assets).

Classic pairwise Pearson measures plus downside- and tail-conditioned
variants.  All helpers are NaN-safe: rows with missing returns are dropped
before aggregation and degenerate subsets yield NaN rather than raising.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from quantlab.risk.drawdown import drawdown_series

__all__ = [
    "conditional_correlation",
    "downside_correlation",
    "drawdown_correlation",
    "rolling_correlation",
    "tail_dependence",
]


def _valid_rows(x: np.ndarray) -> np.ndarray:
    """Drop rows containing any NaN and coerce to float64."""
    arr = np.asarray(x, dtype=float)
    return arr[~np.isnan(arr).any(axis=1)]


def _scalar_corr(subset: np.ndarray) -> float:
    """Pearson correlation of a two-column subset, NaN when degenerate."""
    if subset.shape[0] < 2:
        return np.nan
    corr = np.corrcoef(subset[:, 0], subset[:, 1])
    return float(corr[0, 1])


def rolling_correlation(returns: np.ndarray, window: int) -> np.ndarray:
    """Mean pairwise Pearson correlation in each trailing window of ``returns``.

    ``returns`` has shape (T, N); the result is a length-T vector where
    position ``t`` holds the mean off-diagonal correlation of the ``window``
    rows ending at ``t``.  The first ``window - 1`` entries are NaN (warmup).
    Windows with fewer than two assets yield NaN.

    Vectorized via the correlation matrix: every trailing window is centered
    in one strided pass and the off-diagonal correlation entries are averaged.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    t, n_assets = r.shape
    window = int(window)
    out = np.full(t, np.nan)
    if n_assets < 2 or window < 2 or t < window:
        return out
    blocks = sliding_window_view(r, window, axis=0).transpose(0, 2, 1)
    centered = blocks - blocks.mean(axis=1, keepdims=True)
    var = np.einsum("twi,twi->ti", centered, centered) / (window - 1)
    cov = np.einsum("twi,twj->tij", centered, centered) / (window - 1)
    std = np.sqrt(var)
    denom = std[:, :, None] * std[:, None, :]
    corr = cov / np.where(denom > 0, denom, np.nan)
    mask = ~np.eye(n_assets, dtype=bool)[None, :, :]
    mean = np.nanmean(np.where(mask, corr, np.nan), axis=(1, 2))
    out[window - 1 :] = mean
    return out


def downside_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over the joint downside subset (both returns < 0).

    Returns NaN when fewer than five joint-downside pairs are available.
    """
    joint = _valid_rows(np.column_stack([a, b]))
    subset = joint[(joint[:, 0] < 0) & (joint[:, 1] < 0)]
    if subset.shape[0] < 5:
        return np.nan
    return _scalar_corr(subset)


def drawdown_correlation(returns: np.ndarray) -> np.ndarray:
    """(N, N) correlation matrix of the per-asset drawdown series."""
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    dd = np.column_stack([drawdown_series(r[:, j]) for j in range(r.shape[1])])
    return np.corrcoef(dd, rowvar=False)


def conditional_correlation(
    a: np.ndarray,
    b: np.ndarray,
    condition_fn: Callable[[np.ndarray, np.ndarray], np.ndarray | bool],
) -> float:
    """Pearson correlation of ``a`` and ``b`` where ``condition_fn(a, b)`` holds.

    The condition is evaluated element-wise over the NaN-free rows.  Returns
    NaN when fewer than two qualifying rows remain.
    """
    joint = _valid_rows(np.column_stack([a, b]))
    select = np.asarray(condition_fn(joint[:, 0], joint[:, 1]), dtype=bool)
    return _scalar_corr(joint[select])


def tail_dependence(returns: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Empirical lower-tail dependence matrix over the return panel.

    ``tau[a, b]`` is the empirical probability that asset ``b`` sits below its
    ``threshold``-quantile given that asset ``a`` does too, symmetrized as
    ``(tau_ab + tau_ba) / 2``.  The diagonal is exactly 1.  Quantiles are the
    empirical ``threshold`` percentiles of each asset.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    valid = _valid_rows(r)
    n_assets = valid.shape[1]
    quantiles = np.quantile(valid, threshold, axis=0)
    below = (valid <= quantiles).astype(float)
    counts = below.sum(axis=0)
    joint = below.T @ below
    tau_ab = joint / np.where(counts > 0, counts, np.nan)[:, None]
    tau = (tau_ab + tau_ab.T) / 2.0
    if n_assets == 1:
        return np.eye(1)
    return tau
