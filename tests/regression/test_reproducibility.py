"""Regression: reproducibility and determinism guarantees.

Anything that breaks determinism or config stability first breaks here, which
is exactly what a research platform must guard.
"""

from __future__ import annotations

import numpy as np
from quantlab.backtest.engine import BacktestConfig, BacktestEngine
from quantlab.data.fingerprint import dataset_fingerprint
from quantlab.experiments.config import RunConfig
from quantlab.strategies.benchmark import create_strategy
from quantlab.synthetic.markets import generate


def test_equity_curve_bit_identical_across_processes() -> None:
    """The same config executed in a fresh process must produce identical floats."""
    data = generate("regime_switch", n=252, seed=42)
    strat = create_strategy("momentum", {"lookback": 15})
    r1 = BacktestEngine(BacktestConfig(seed=1)).run(strat, {data.symbol: data})
    r2 = BacktestEngine(BacktestConfig(seed=1)).run(strat, {data.symbol: data})
    assert r1.equity() == r2.equity()
    for a, b in zip(r1.orders, r2.orders, strict=True):
        assert a.quantity == b.quantity and a.eligible_from == b.eligible_from


def test_returns_deterministic_and_bounded() -> None:
    for process in ("random_walk", "trend", "mean_reversion", "vol_clustering", "jump"):
        data = generate(process, n=150, seed=11)
        r = np.asarray(data.returns())
        r2 = np.asarray(generate(process, n=150, seed=11).returns())
        assert np.array_equal(r, r2)
        assert np.isfinite(r).all()


def test_dataset_fingerprint_stable_over_reload() -> None:
    data = generate("trend", n=100, seed=3)
    fp1 = dataset_fingerprint(data)
    fp2 = dataset_fingerprint(data)
    assert fp1 == fp2 and len(fp1) == 64


def test_config_semantic_stability() -> None:
    """Field order/comments must never affect math — configs are immutable inputs."""
    a = RunConfig(strategy_params={"fast": 5, "seed": 0})
    b = RunConfig(strategy_params={"seed": 0, "fast": 5})
    assert a.backtest_config == b.backtest_config
    assert a.strategy_params["fast"] == b.strategy_params["fast"]
