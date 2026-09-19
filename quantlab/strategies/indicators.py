"""Purely-functional indicator helpers (depend only on already-arrived data).

All functions take 1-D float sequences; callers are responsible for passing only
information that was available at decision time (the BarStream guarantees that).
"""

from __future__ import annotations

import math

import numpy as np


def sma(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    out = np.full(len(x), np.nan)
    if len(x) < window:
        return out
    cum = np.cumsum(x)
    out[window - 1 :] = (cum[window - 1 :] - np.concatenate([[0.0], cum[:-window]])) / window
    return out


def ema(x: np.ndarray, span: int) -> np.ndarray:
    if span <= 0:
        raise ValueError("span must be positive")
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(x))
    if len(x) == 0:
        return out
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1 - alpha) * out[t - 1]
    return out


def rsi(x: np.ndarray, period: int = 14) -> np.ndarray:
    if len(x) < period + 1:
        return np.full(len(x), np.nan)
    delta = np.diff(x, prepend=x[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full(len(x), np.nan)
    avg_loss = np.full(len(x), np.nan)
    eg, el = gains[0], losses[0]
    for t in range(len(x)):
        if t == 0:
            avg_gain[t], avg_loss[t] = np.nan, np.nan
            continue
        if t < period:
            eg += gains[t]
            el += losses[t]
            avg_gain[t], avg_loss[t] = np.nan, np.nan
        elif t == period:
            avg_gain[t], avg_loss[t] = eg / period, el / period
        else:
            avg_gain[t] = (avg_gain[t - 1] * (period - 1) + gains[t]) / period
            avg_loss[t] = (avg_loss[t - 1] * (period - 1) + losses[t]) / period
    rs = np.full_like(avg_gain, np.nan)
    pos = (avg_loss > 0) & ~np.isnan(avg_gain)
    rs[pos] = avg_gain[pos] / avg_loss[pos]
    rs[~np.isnan(avg_gain) & (avg_loss == 0) & (avg_gain > 0)] = float("inf")
    rs[~np.isnan(avg_gain) & (avg_loss == 0) & (avg_gain == 0)] = 1.0  # flat stretch
    return 100.0 - 100.0 / (1.0 + rs)


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    if len(closes) < 2:
        return np.full(len(closes), np.nan)
    pc = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - pc), np.abs(lows - pc)))
    out = np.full(len(closes), np.nan)
    if len(tr) < period:
        return out
    out[period - 1] = float(np.mean(tr[1 : period + 1]))
    for t in range(period, len(tr)):
        out[t] = (out[t - 1] * (period - 1) + tr[t]) / period
    return out


def zscore(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score of a series against its own trailing window."""
    if window <= 0:
        raise ValueError("window must be positive")
    out = np.full(len(x), np.nan)
    if len(x) < 2:
        return out
    for t in range(1, len(x)):
        lo = max(0, t - window)
        seg = x[lo:t]
        sd = float(np.std(seg))
        if sd > 0:
            out[t] = (x[t] - float(np.mean(seg))) / sd
    return out


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    out = np.full(len(x), np.nan)
    if len(x) < 2:
        return out
    for t in range(1, len(x)):
        seg = x[max(0, t - window) : t]
        out[t] = float(np.std(seg))
    return out


def rolling_corr(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    n = min(len(a), len(b))
    out = np.full(n, np.nan)
    for t in range(1, n):
        lo = max(0, t - window)
        sa, sb = a[lo:t], b[lo:t]
        if len(sa) < 2:
            continue
        ca = np.cov(sa, sb)
        if ca[0, 0] > 0 and ca[1, 1] > 0:
            out[t] = ca[0, 1] / math.sqrt(ca[0, 0] * ca[1, 1])
    return out
