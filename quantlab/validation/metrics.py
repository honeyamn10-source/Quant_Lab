"""Performance metrics on a returns array (period frequency = daily default).

All functions return plain floats; individual metrics are reported separately,
never fused into one "strategy quality" number.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def to_array(returns) -> np.ndarray:
    return np.asarray(returns, dtype=float)


def _valid(r: np.ndarray) -> np.ndarray:
    return r[~np.isnan(r)]


def total_return(returns) -> float:
    r = to_array(returns)
    if len(r) == 0:
        return 0.0
    return float(np.prod(1.0 + r) - 1.0)


def volatility(returns, periods_per_year: int = 252) -> float:
    r = _valid(to_array(returns))
    return float(np.std(r, ddof=1) * math.sqrt(periods_per_year)) if len(r) > 1 else 0.0


def sharpe_ratio(returns, rf: float = 0.0, periods_per_year: int = 252) -> float:
    r = _valid(to_array(returns))
    if len(r) < 2:
        return 0.0
    excess = r - rf / periods_per_year
    sd = float(np.std(excess, ddof=1))
    if sd == 0:
        return 0.0
    return float(np.mean(excess) / sd * math.sqrt(periods_per_year))


def sortino_ratio(returns, rf: float = 0.0, periods_per_year: int = 252) -> float:
    r = _valid(to_array(returns))
    if len(r) < 2:
        return 0.0
    excess = r - rf / periods_per_year
    downside = float(np.std(excess[excess < 0], ddof=1)) if (excess < 0).any() else 0.0
    if downside == 0:
        return 0.0 if float(np.mean(excess)) <= 0 else float("inf")
    return float(np.mean(excess) / downside * math.sqrt(periods_per_year))


def max_drawdown(returns) -> float:
    """Max drawdown as a positive fraction of peak equity (0 = no drawdown)."""
    r = to_array(returns)
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = 1.0 - eq / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd)) if len(dd) else 0.0


def calmar_ratio(returns, periods_per_year: int = 252) -> float:
    ann = annualized_return(returns, periods_per_year)
    mdd = max_drawdown(returns)
    return ann / mdd if mdd > 0 else (ann if ann > 0 else 0.0)


def annualized_return(returns, periods_per_year: int = 252) -> float:
    r = to_array(returns)
    if len(r) == 0:
        return 0.0
    tr = np.prod(1.0 + r)
    if tr <= 0:
        return -1.0
    return float(tr ** (periods_per_year / len(r)) - 1.0)


def cagr(equity_curve: list[float], n_periods: int) -> float:
    if not equity_curve or n_periods <= 0:
        return 0.0
    start, end = equity_curve[0], equity_curve[-1]
    if start <= 0:
        return -1.0
    return float((end / start) ** (1.0 / n_periods) - 1.0)


def omega_ratio(returns, threshold: float = 0.0) -> float:
    """Omega = gains above threshold / |losses below threshold|."""
    r = _valid(to_array(returns)) - threshold
    gains = float(np.sum(r[r > 0]))
    losses = float(-np.sum(r[r < 0]))
    return gains / losses if losses > 0 else float("inf")


def hit_rate(returns) -> float:
    r = to_array(returns)
    if len(r) == 0:
        return 0.0
    return float(np.mean(r > 0))


def profit_factor(returns) -> float:
    r = to_array(returns)
    gross = float(np.sum(r[r > 0]))
    loss = float(-np.sum(r[r < 0]))
    return gross / loss if loss > 0 else float("inf")


def turnover(weights_changes: list[float], weights_levels: list[float]) -> float:
    """Average absolute weight change normalized by average gross exposure."""
    if not weights_changes:
        return 0.0
    avg_exp = float(np.mean(np.abs(weights_levels))) if weights_levels else 1.0
    return float(np.mean(np.abs(weights_changes))) / avg_exp if avg_exp > 0 else 0.0


MTX: dict[str, Callable[..., float]] = {
    "sharpe": sharpe_ratio,
    "sortino": sortino_ratio,
    "calmar": calmar_ratio,
    "omega": omega_ratio,
    "annualized_return": annualized_return,
    "volatility": volatility,
    "max_drawdown": max_drawdown,
    "hit_rate": hit_rate,
    "profit_factor": profit_factor,
    "total_return": total_return,
}


def all_metrics(returns) -> dict[str, float]:
    return {name: fn(returns) for name, fn in MTX.items()}
