"""Per-bar regime feature extraction from a close-price path.

All features are returned as float64 arrays with the same length as the input
`closes`. Rolling statistics use a lookup window of ``WINDOW`` bars and report
NaN (``np.nan``) for the warm-up region where fewer than the full window of
finite observations is available. Only closes are supplied, so the volume-based
and cross-sectional measures rely on documented proxies:

    trend            rolling mean of daily log returns (drift over the window)
    realized_vol     rolling sample standard deviation of daily log returns
    vol_of_vol       rolling standard deviation of the rolling realized vol
    liquidity_proxy  rolling mean absolute daily return as a flow proxy
    dispersion       rolling std of returns relative to rolling mean absolute
                     return (single-series approximation of a cross-section)
    correlation_proxy rolling lag-1 autocorrelation of returns computed from
                     rolling auto-covariance over rolling variance
    drawdown         rolling maximum drawdown relative to the rolling peak
    momentum_ratio   rolling 5-day mean return normalised by 20-day vol
    tail_stress      rolling count of days with |return| above 3-sigma
"""

from __future__ import annotations

import numpy as np

WINDOW = 20
_EPS = 1e-12


def _returns(closes: np.ndarray) -> np.ndarray:
    """Daily log returns aligned to bars 1..T-1; bar 0 is NaN."""
    closes = np.asarray(closes, dtype=np.float64)
    logs = np.log(np.where(closes > 0.0, closes, np.nan))
    r = np.diff(logs)
    return np.concatenate(([np.nan], r))


def _rolling(x: np.ndarray, window: int, fn) -> np.ndarray:
    """Rolling aggregation over sliding windows with NaN warm-up.

    Any NaN inside a window propagates to the output (full-window validity).
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.size >= window:
        view = np.lib.stride_tricks.sliding_window_view(x, window)
        out[window - 1 :] = fn(view, axis=1)
    return out


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling arithmetic mean with full-window validity."""
    return _rolling(x, window, np.mean)


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling sample standard deviation (ddof=1) with full-window validity."""
    return _rolling(x, window, lambda a, axis: np.std(a, axis=axis, ddof=1))


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling sum with full-window validity."""
    return _rolling(x, window, np.sum)


def _rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling maximum with full-window validity."""
    return _rolling(x, window, np.max)


def regime_features(closes: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the per-bar regime feature dictionary for a close-price path.

    Returns one float64 array of the same length as `closes` per key:
    ``trend``, ``realized_vol``, ``vol_of_vol``, ``liquidity_proxy``,
    ``dispersion``, ``correlation_proxy``, ``drawdown``, ``momentum_ratio``,
    ``tail_stress``. Warm-up bars are NaN.
    """
    closes = np.asarray(closes, dtype=np.float64)
    r = _returns(closes)
    w = WINDOW

    realized_vol = _rolling_std(r, w)
    rol_abs = _rolling_mean(np.abs(r), w)
    trend = _rolling_mean(r, w)

    auto_cov = _rolling_mean(np.concatenate(([np.nan], r[1:] * r[:-1])), w)
    variance = realized_vol**2
    correlation_proxy = np.where(
        variance > _EPS,
        (auto_cov - trend**2) / np.where(variance > _EPS, variance, np.nan),
        np.nan,
    )

    sigma = realized_vol
    multiplier = 3.0
    tail_flag = np.where(np.abs(r) > multiplier * sigma, 1.0, 0.0)

    return {
        "trend": trend,
        "realized_vol": realized_vol,
        "vol_of_vol": _rolling_std(realized_vol, w),
        "liquidity_proxy": rol_abs,
        "dispersion": realized_vol / np.maximum(rol_abs, _EPS),
        "correlation_proxy": correlation_proxy,
        "drawdown": (closes - _rolling_max(closes, w)) / _rolling_max(closes, w),
        "momentum_ratio": _rolling_mean(r, 5) / np.maximum(realized_vol, _EPS),
        "tail_stress": _rolling_sum(tail_flag, w),
    }


def normalize_features(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Z-score each feature over its finite values.

    NaN entries stay NaN, and a feature that is constant (or fully NaN) on the
    finite region maps to all zeros.
    """
    out: dict[str, np.ndarray] = {}
    for key, series in features.items():
        values = np.asarray(series, dtype=np.float64)
        finite = np.isfinite(values)
        normalized = np.full_like(values, np.nan, dtype=np.float64)
        if np.any(finite):
            mu = np.nanmean(values)
            sd = np.nanstd(values)
            if sd > _EPS:
                normalized[finite] = (values[finite] - mu) / sd
            else:
                normalized = np.zeros_like(values, dtype=np.float64)
        out[key] = normalized
    return out
