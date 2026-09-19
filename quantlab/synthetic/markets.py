"""Synthetic market generators with known ground truth.

These are the honest baseline: if Quant Lab routinely "discovers alpha" in a
random-walk generator, the research pipeline is broken. Each generator returns a
:class:`BarSeries` produced from a seeded RNG (deterministic).
"""

from __future__ import annotations

import math

import numpy as np

from quantlab.data.schemas import Bar, BarSeries

_PROCESSES = (
    "random_walk",
    "trend",
    "mean_reversion",
    "vol_clustering",
    "regime_switch",
    "structural_break",
    "jump",
)
_PAIRS_LABEL = "pairs"


def _make_bars(symbol: str, closes: np.ndarray, start_ts: int = 0) -> BarSeries:
    """Build OHLCV bars from a close-price path (deterministic)."""
    bars: list[Bar] = []
    prev_close = closes[0]
    for i, close in enumerate(closes):
        gap = max(abs(close - prev_close) * 0.5, prev_close * 1e-4)
        high = max(close, prev_close) + gap * 0.35
        low = min(close, prev_close) - gap * 0.35
        volume = float(abs(np.random.RandomState(int(abs(close) * 1e7) % 1_000_000).normal(2_000_000.0, 400_000.0)))
        bars.append(
            Bar(
                symbol=symbol,
                ts=start_ts + i * 86_400_000,
                interval_seconds=86_400,
                open=float(prev_close),
                high=float(max(high, prev_close, close)),
                low=float(min(low, prev_close, close)),
                close=float(close),
                volume=volume,
                available_time=start_ts + i * 86_400_000,
                source="synthetic",
            )
        )
        prev_close = close
    return BarSeries(symbol, bars, 86_400)


def _path(rng: np.random.Generator, n: int, drifts: np.ndarray, vols: np.ndarray) -> np.ndarray:
    """Return close path from daily drift/vol processes."""
    shocks = rng.standard_normal(n)
    rets = drifts + vols * shocks
    price = np.empty(n)
    price[0] = 100.0
    price[1:] = 100.0 * np.exp(np.cumsum(rets[:-1]))
    return price


def generate(process: str, n: int = 504, seed: int = 7, symbol: str = "SYN") -> BarSeries:
    """Generate a single-symbol synthetic series of `process` kind."""
    process = process.lower()
    if process not in _PROCESSES:
        raise ValueError(f"unknown process {process!r}; choose from {sorted(_PROCESSES)}")
    rng = np.random.default_rng(seed)
    mu, sigma = 0.04 / 252.0, 0.12 / math.sqrt(252)

    if process == "random_walk":
        closes = _path(rng, n, np.full(n, mu), np.full(n, sigma))
    elif process == "trend":
        closes = _path(rng, n, np.full(n, 0.06 / 252.0), np.full(n, sigma))
    elif process == "mean_reversion":
        k = 0.05
        price = np.empty(n)
        price[0] = 100.0
        for t in range(1, n):
            ret = -k * math.log(price[t - 1] / 100.0) + sigma * rng.standard_normal()
            price[t] = price[t - 1] * math.exp(ret)
        closes = price
    elif process == "vol_clustering":
        vols = np.full(n, sigma)
        for t in range(20, n):
            vols[t] = math.sqrt(0.94 * vols[t - 1] ** 2 + 0.05 * aug_ret2(rng, vols[t - 1]))
        closes = _path(rng, n, np.full(n, mu), vols)
    elif process == "regime_switch":
        vols = np.where(np.arange(n) % 126 < 63, sigma, 3.0 * sigma)
        drifts = np.where(np.arange(n) % 126 < 63, mu, -mu)
        closes = _path(rng, n, drifts, vols)
    elif process == "structural_break":
        vol = np.full(n, sigma)
        vol[n // 2 :] = 2.5 * sigma
        closes = _path(rng, n, np.full(n, mu), vol)
    elif process == "jump":
        rets = mu + sigma * rng.standard_normal(n)
        jmp = rng.random(n) < 0.02
        rets[jmp] += rng.normal(0.0, 0.08, size=int(jmp.sum()))
        path = np.empty(n)
        path[0] = 100.0
        path[1:] = 100.0 * np.exp(np.cumsum(rets[:-1]))
        closes = path
    return _make_bars(symbol, closes)


def aug_ret2(rng: np.random.Generator, prev_vol: float) -> float:
    """Vol shock used by the clustering generator."""
    return (prev_vol * rng.standard_normal()) ** 2


def generate_pairs(n: int = 504, seed: int = 7, symbols: tuple[str, str] = ("A", "B")) -> tuple[BarSeries, BarSeries]:
    """Cointegrated pair: A random-walks, B = k*A + stationary residual."""
    rng = np.random.default_rng(seed)
    k, sigma_a, omega = 1.0, 0.12 / math.sqrt(252), 0.05 * math.sqrt(1 / 252)
    a_path = _path(rng, n, np.full(n, 0.0), np.full(n, sigma_a))
    resid = np.zeros(n)
    resid[0] = 0.0
    for t in range(1, n):
        resid[t] = 0.9 * resid[t - 1] + omega * rng.standard_normal()
    b_path = k * a_path + resid
    return (
        _make_bars(symbols[0], a_path),
        _make_bars(symbols[1], b_path),
    )


def available_processes() -> tuple[str, ...]:
    return _PROCESSES
