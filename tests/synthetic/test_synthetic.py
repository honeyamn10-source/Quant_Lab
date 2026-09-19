"""Synthetic-known-ground-truth tests.

Ground truth is baked into each generator: a strategy that extracts multiple
standard-deviation Sharpe ratios from a random walk is the pipeline's canary
failing. Buy-and-hold must track the market; vol targeting must cut risk; pairs
must profit when its hypothesis (cointegration) is literally true.
"""

from __future__ import annotations

import numpy as np
from quantlab.backtest.engine import BacktestConfig, BacktestEngine
from quantlab.strategies.benchmark import BuyAndHold, RandomStrategy, VolTarget, create_strategy
from quantlab.strategies.pairs import PairsSpread
from quantlab.synthetic.markets import available_processes, generate, generate_pairs
from quantlab.validation.metrics import sharpe_ratio, volatility


def test_generators_cover_spec() -> None:
    want = {
        "random_walk", "trend", "mean_reversion", "vol_clustering",
        "regime_switch", "structural_break", "jump",
    }
    assert want <= set(available_processes())


def test_generators_deterministic_per_seed() -> None:
    a = generate("random_walk", n=100, seed=7)
    b = generate("random_walk", n=100, seed=7)
    assert a.closes() == b.closes()
    c = generate("random_walk", n=100, seed=8)
    assert a.closes() != c.closes()


def test_random_strategy_sharpe_not_spectacular() -> None:
    """The canary: randomized signals must NOT discover several-SD alpha."""
    data = generate("random_walk", n=504, seed=1)
    result = BacktestEngine(BacktestConfig()).run(
        RandomStrategy({"seed": 3}), {data.symbol: data}
    )
    # randomization is costly: turnover-driven drag plus no information.
    assert sharpe_ratio(result.returns) < 1.0


def test_sma_cross_on_random_walk_not_spectacular() -> None:
    data = generate("random_walk", n=504, seed=5)
    result = BacktestEngine(BacktestConfig()).run(
        create_strategy("sma_crossover", {"fast": 10, "slow": 30}), {data.symbol: data}
    )
    assert sharpe_ratio(result.returns) < 1.0


def test_buy_and_hold_tracks_market() -> None:
    data = generate("trend", n=504, seed=2)
    result = BacktestEngine(BacktestConfig()).run(BuyAndHold(), {data.symbol: data})
    closes = data.closes()
    market = closes[-1] / closes[1]  # first fill lands at bar-2 open = bar-1 close
    got = result.final_equity() / result.config.initial_cash
    assert np.isclose(got, market, rtol=2e-2)


def test_vol_target_tames_volatility() -> None:
    data = generate("vol_clustering", n=504, seed=6)
    result = BacktestEngine(BacktestConfig(profile="optimistic", participation_cap=0.5)).run(
        VolTarget({"target_vol": 0.15, "window": 20, "max_weight": 1.0}),
        {data.symbol: data},
    )
    mkt_vol = volatility(data.returns())
    strat_vol = volatility(result.returns)
    assert strat_vol <= mkt_vol * 1.2


def test_pairs_profits_on_cointegrated_pair() -> None:
    a, b = generate_pairs(n=504, seed=11)
    result = BacktestEngine(BacktestConfig(max_leverage=2.0)).run(
        PairsSpread({"window": 40, "z_enter": 1.0, "leverage": 1.0}),
        {a.symbol: a, b.symbol: b},
    )
    assert result.final_equity() > result.config.initial_cash


def test_pairs_known_beta() -> None:
    from quantlab.strategies.pairs import ols_beta

    x = np.linspace(1.0, 5.0, 50)
    y = 2.0 * x + 0.3 + 0.01 * np.random.default_rng(1).normal(size=50)
    beta, r2 = ols_beta(x, y)
    assert abs(beta - 2.0) < 1e-2
    assert r2 > 0.99


def test_mean_reversion_profits_in_mean_reversion_market() -> None:
    data = generate("mean_reversion", n=1000, seed=3)
    result = BacktestEngine(BacktestConfig(profile="optimistic", participation_cap=0.5)).run(
        create_strategy("mean_reversion", {"window": 20, "z_enter": 1.2}),
        {data.symbol: data},
    )
    assert result.final_equity() > result.config.initial_cash
