"""Volatility estimation models.

All estimators operate on a 1-D array of simple returns and produce an array
of the same length. Series-based estimators leave warm-up bars as NaN so that
downstream consumers can never observe data that was not available in real
time. Every stochastic function accepts a ``seed`` and uses
``numpy.random.default_rng(seed)`` for exact reproducibility.
"""

from __future__ import annotations

import math

import numpy as np

from quantlab.volatility.garch import garch_family

_DEFAULT_PPY = 252


def _as_float_series(returns: np.ndarray) -> np.ndarray:
    return np.asarray(returns, dtype=np.float64).ravel()


def _trailing_ssq(r: np.ndarray, window: int) -> np.ndarray:
    """Sum of squared returns over the trailing window; NaN until defined."""
    out = np.full(len(r), np.nan)
    if window <= 0:
        raise ValueError("window must be positive")
    if len(r) >= window:
        cum = np.concatenate([[0.0], np.cumsum(r * r)])
        out[window - 1 :] = cum[window:] - cum[:-window]
    return out


def historical_vol(returns: np.ndarray, window: int = 20, ppy: int = 252) -> np.ndarray:
    """Rolling sample standard deviation (ddof=1) over a trailing window.

    The result is annualized by ``sqrt(ppy)`` and is NaN until ``window``
    observations have been seen.
    """
    r = _as_float_series(returns)
    out = np.full(len(r), np.nan)
    if window <= 0:
        raise ValueError("window must be positive")
    if len(r) >= window:
        view = np.lib.stride_tricks.sliding_window_view(r, window)
        out[window - 1 :] = np.std(view, axis=1, ddof=1) * math.sqrt(ppy)
    return out


def ewma_vol(returns: np.ndarray, lam: float = 0.94, ppy: int = 252) -> np.ndarray:
    """Exponentially weighted volatility of squared returns.

    The recursive estimator is ``var_t = lam * var_{t-1} + (1 - lam) * r_t^2``.
    To get a well-defined first value the recursion is seeded with the sample
    variance of the first ``min(10, len(returns))`` observations. Output is
    annualized by ``sqrt(ppy)``.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must lie in (0, 1)")
    r = _as_float_series(returns)
    n = len(r)
    if n == 0:
        return np.asarray([], dtype=np.float64)
    warm = min(10, n)
    if warm >= 2:
        seed_var = float(np.var(r[:warm], ddof=1))
    elif warm == 1:
        seed_var = float(r[0] * r[0])
    else:
        seed_var = 0.0
    var = np.empty(n)
    var[0] = seed_var
    for t in range(1, n):
        var[t] = lam * var[t - 1] + (1.0 - lam) * r[t] * r[t]
    return np.sqrt(np.maximum(var, 0.0)) * math.sqrt(ppy)


def realized_vol(returns: np.ndarray, window: int = 20, ppy: int = 252) -> np.ndarray:
    """Realized volatility from a trailing window of squared returns.

    ``rv = sqrt(sum(r^2) * ppy / window)``, i.e. the sum of squared returns
    over ``window`` bars annualized to a per-year scale. NaN before ``window``
    observations exist.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    r = _as_float_series(returns)
    rv = _trailing_ssq(r, window)
    return np.sqrt(rv * (ppy / window))


def har_vol(
    returns: np.ndarray,
    horizon_days: int = 1,
    window_d: int = 1,
    window_w: int = 5,
    window_m: int = 22,
) -> np.ndarray:
    """Heterogeneous autoregressive (HAR) realized-volatility forecast.

    Daily realized variance at bar ``t`` is the sum of squared returns over the
    trailing ``window_d`` bars. The log of that quantity is regressed on an
    intercept plus the logs of the daily, weekly and monthly realized variances
    measured at bar ``t - horizon_days`` (windows ending there), so every
    forecast uses only information available before the bar it predicts.

    The model is fitted with ordinary least squares (``numpy.linalg.lstsq``):
    coefficients are estimated on the first 60% of the usable sample and
    one-step-ahead forecasts are produced for the remaining 40%. Annualized
    forecasts are placed at the bar they predict; warm-up bars with no data (or
    too few usable observations) remain NaN. Annualization assumes 252 bars per
    year.
    """
    r = _as_float_series(returns)
    n = len(r)
    out = np.full(n, np.nan)
    if n < 2 or horizon_days < 1 or window_d <= 0 or window_w <= 0 or window_m <= 0:
        return out
    rv_d = _trailing_ssq(r, window_d)
    rv_w = _trailing_ssq(r, window_w)
    rv_m = _trailing_ssq(r, window_m)
    feat = np.arange(n - horizon_days)
    tgt = feat + horizon_days
    ok = (rv_w[feat] > 0) & (rv_m[feat] > 0) & (rv_d[feat] > 0) & (rv_d[tgt] > 0)
    feat, tgt = feat[ok], tgt[ok]
    if len(feat) < 6:
        return out
    design = np.column_stack(
        [
            np.ones(len(feat)),
            np.log(rv_d[feat]),
            np.log(rv_w[feat]),
            np.log(rv_m[feat]),
        ]
    )
    target = np.log(rv_d[tgt])
    split = int(0.6 * len(feat))
    if split < 4 or len(feat) - split < 1:
        return out
    beta, *_ = np.linalg.lstsq(design[:split], target[:split], rcond=None)
    pred = design[split:] @ beta
    out[tgt[split:]] = np.sqrt(np.exp(pred) * (_DEFAULT_PPY / window_d))
    return out


def forecast(model: str, returns: np.ndarray, **kwargs) -> np.ndarray:
    """Per-bar one-step-ahead volatility forecasts aligned to the input length.

    Historical, EWMA and realized volatility estimates use returns up to and
    including bar ``t``, so the forecast for bar ``t`` is the corresponding
    estimate made at bar ``t - 1`` (the first value is NaN). HAR and the
    GARCH-family models already produce forecasts of bar ``t`` from data before
    ``t`` and are returned as-is. ``kwargs`` are forwarded to the underlying
    estimator.
    """
    m = str(model).lower()
    if m in {"historical", "ewma", "realized", "har"}:
        from collections.abc import Callable

        estimators: dict[str, Callable[..., np.ndarray]] = {
            "historical": historical_vol,
            "ewma": ewma_vol,
            "realized": realized_vol,
            "har": har_vol,
        }
        estimator = estimators[m]
        estimated = estimator(returns, **kwargs)
        if m == "har":
            return estimated
        out = np.full(len(estimated), np.nan)
        out[1:] = estimated[:-1]
        return out
    if m in {"garch", "egarch", "gjr"}:
        fit = garch_family(m, returns, **kwargs)
        return np.sqrt(fit["sigma2"])
    raise ValueError(f"unknown volatility model: {model}")
