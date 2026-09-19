"""Allocation schemes: dollar sizing, vol targeting and risk constraints."""

from __future__ import annotations

import numpy as np


def fixed_fraction(equity: float, fraction: float) -> float:
    """Dollar allocation equal to equity times the betting fraction."""
    if equity < 0.0:
        raise ValueError(f"equity must be non-negative, got {equity}")
    if fraction < 0.0:
        raise ValueError(f"fraction must be non-negative, got {fraction}")
    return float(equity * fraction)


def vol_target_weight(
    annualized_vol: float, target_vol: float, max_weight: float = 2.0
) -> float:
    """Weight scaling an asset's annualized vol up to the target, capped."""
    if annualized_vol <= 0.0:
        return float(max_weight)
    return float(min(max(target_vol / annualized_vol, 0.0), max_weight))


def risk_parity_weights(
    cov: np.ndarray, max_leverage: float = 1.5, seed: int = 0
) -> np.ndarray:
    """Risk-parity-style weights from the inverse-covariance diagonal.

    The non-negative diagonal of a ridge-stabilized inverse covariance is scaled
    to a gross exposure of max_leverage. Invalid or singular covariance matrices
    fall back to equal weights.
    """
    c = np.asarray(cov, dtype=float)
    if c.ndim == 1 and c.size > 0:
        n = c.size
    elif c.ndim == 2:
        n = c.shape[0]
    else:
        return np.array([], dtype=float)
    equal = np.full(n, float(max_leverage) / n)
    valid = (
        c.ndim == 2
        and c.shape[0] == c.shape[1]
        and n > 0
        and np.isfinite(c).all()
    )
    if not valid:
        return equal
    try:
        eps = max(float(np.mean(np.diag(c))) * 1e-6, 1e-12)
        precision = np.linalg.inv(c + eps * np.eye(n))
        w = np.maximum(np.diag(precision), 0.0)
    except np.linalg.LinAlgError:
        return equal
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        return equal
    return (w / total) * float(max_leverage)


def portfolio_cvar(weights: np.ndarray, returns: np.ndarray, alpha: float = 0.05) -> float:
    """Empirical CVaR (expected shortfall) of the weighted portfolio at level alpha.

    The value is the mean loss in the worst alpha-fraction of weighted returns,
    reported as a positive number.
    """
    w = np.asarray(weights, dtype=float).ravel()
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or w.size == 0 or r.shape[1] != w.size:
        raise ValueError(f"returns {r.shape} not aligned with {w.size} weights")
    rows = np.isfinite(r).all(axis=1)
    if not rows.any():
        return 0.0
    port = r[rows] @ w
    var = float(np.quantile(port, alpha))
    tail = port[port <= var]
    return float(-np.mean(tail)) if tail.size else 0.0


def cvar_constrained_weights(
    returns: np.ndarray, target_cvar: float, max_leverage: float = 1.0, seed: int = 0
) -> np.ndarray:
    """Long-only CVaR-constrained weights via a greedy CVaR-guided tilt.

    Heuristic: start from equal weights at gross exposure max_leverage and move
    money in small deterministic steps from the asset with the worst
    return-per-unit-downside to the best one (downside = asset CVaR at the 95%
    level). Each candidate tilt must keep the weighted portfolio's empirical
    CVaR at or below target_cvar; the first infeasible tilt ends the search, so
    the result is deterministic and long-only. No randomness is used.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or r.shape[0] < 2 or r.shape[1] < 1:
        return np.array([], dtype=float)
    rows = np.isfinite(r).all(axis=1)
    r = r[rows]
    n = r.shape[1]
    if r.shape[0] < 2:
        return np.array([], dtype=float)
    if target_cvar < 0.0:
        raise ValueError(f"target_cvar must be non-negative, got {target_cvar}")
    w = np.full(n, float(max_leverage) / n)
    if n == 1:
        return w
    unit = np.eye(n, dtype=float)
    asset_cvar = np.array([portfolio_cvar(unit[i], r) for i in range(n)])
    means = np.mean(r, axis=0)
    score = means / np.maximum(asset_cvar, 1e-12)
    order = np.argsort(-score, kind="stable")
    step = max(float(max_leverage) / (n * 40), 1e-9)
    for _ in range(n * 40):
        moved = False
        for tgt in order:
            if w[tgt] >= float(max_leverage) - 1e-12:
                continue
            for src in np.flip(order):
                if src == tgt or w[src] <= 1e-12:
                    continue
                amt = min(step, w[src], float(max_leverage) - w[tgt])
                if amt < step * 1e-3:
                    continue
                cand = w.copy()
                cand[tgt] += amt
                cand[src] -= amt
                if portfolio_cvar(cand, r) <= target_cvar + 1e-12:
                    w = cand
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break
    return w
