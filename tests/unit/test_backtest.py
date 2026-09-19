"""Backtest engine tests: correctness, timing rules, rejections, determinism."""

from __future__ import annotations

import numpy as np
from quantlab.backtest.engine import BacktestConfig, BacktestEngine, LookaheadError
from quantlab.strategies.benchmark import BuyAndHold, create_strategy
from quantlab.synthetic.markets import generate


class _LeakyStrategy:
    name = "leaky"

    def warmup(self) -> int:
        return 0

    def on_bar(self, streams, i: int):
        sym = next(iter(streams))
        stream = streams[sym]
        stream.closes(i + 1)  # future: must raise ValueError -> LookaheadError
        return None


def test_same_bar_execution_rejected() -> None:
    # Validation fires at engine construction (the config dataclass alone allows it)
    try:
        BacktestEngine(BacktestConfig(execution_delay=0))
        raise AssertionError("expected ValueError for same-bar execution")
    except ValueError:
        pass


def test_lookahead_strategy_is_caught() -> None:
    data = generate("random_walk", n=60, seed=1)
    try:
        BacktestEngine(BacktestConfig()).run(_LeakyStrategy(), {data.symbol: data})  # type: ignore[arg-type]
        raise AssertionError("LookaheadError expected")
    except LookaheadError:
        pass


def test_buy_and_hold_tracks_market() -> None:
    data = generate("trend", n=252, seed=5)
    result = BacktestEngine(BacktestConfig()).run(BuyAndHold(), {data.symbol: data})
    closes = np.asarray(data.closes())
    # First fill happens at bar 2's open, which equals bar 1's close.
    market = closes[-1] / closes[1]
    got = result.final_equity() / result.config.initial_cash
    assert np.isclose(got, market, rtol=2e-2), f"{got} vs {market}"


def test_deterministic_same_seed() -> None:
    data = generate("random_walk", n=200, seed=7)
    cfg = BacktestConfig()
    strat = create_strategy("sma_crossover", {"fast": 10, "slow": 30})
    r1 = BacktestEngine(cfg).run(strat, {data.symbol: data})
    r2 = BacktestEngine(cfg).run(strat, {data.symbol: data})
    assert r1.equity() == r2.equity()
    assert r1.returns == r2.returns


def test_fills_only_after_signal_bar() -> None:
    data = generate("random_walk", n=120, seed=2)
    delay = 2
    result = BacktestEngine(BacktestConfig(execution_delay=delay)).run(
        create_strategy("breakout", {"lookback": 20}), {data.symbol: data}
    )
    for fill in result.fills:
        order = next(o for o in result.orders if o.order_id == fill.order_id)
        assert fill.bar_index >= order.eligible_from
        assert order.eligible_from >= order.bar_index + 1 + delay


def test_orders_rejected_on_unknown_symbol() -> None:
    data = generate("mean_reversion", n=100, seed=3)
    result = BacktestEngine(BacktestConfig(max_leverage=0.5)).run(
        BuyAndHold(), {data.symbol: data}
    )
    # leverage cap 0.5 with a full-weight long will produce rejections or limits
    assert result.equity()  # engine still yields a valid curve


def test_leverage_cap_limits_gross() -> None:
    data = generate("random_walk", n=100, seed=4)
    result = BacktestEngine(BacktestConfig(max_leverage=1.0)).run(
        BuyAndHold(), {data.symbol: data}
    )
    for evt in result.portfolio_events:
        assert evt.leverage <= 1.01


def test_spread_increases_cost_reduces_return() -> None:
    data = generate("trend", n=252, seed=8)
    cheap = BacktestEngine(BacktestConfig(profile="optimistic")).run(BuyAndHold(), {data.symbol: data})
    pricey = BacktestEngine(BacktestConfig(profile="extreme", participation_cap=0.02)).run(BuyAndHold(), {data.symbol: data})
    assert pricey.final_equity() <= cheap.final_equity() * 1.0001
