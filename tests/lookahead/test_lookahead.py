"""Look-ahead / timing-integrity suite.

These tests exist to *prove* the pipeline leaks nothing: a strategy that touches
future data must be rejected by the framework itself, never silently accepted.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantlab.backtest.engine import BacktestConfig, BacktestEngine, LookaheadError
from quantlab.data.barstream import BarStream
from quantlab.data.schemas import Bar, BarSeries
from quantlab.strategies.base import Signal, Strategy

pytestmark = pytest.mark.lookahead


def _bars(n: int, delay_every: int | None = None) -> list[Bar]:
    bars = []
    for i in range(n):
        k = int(i // delay_every) if delay_every else i
        bars.append(
            Bar(
                symbol="T",
                ts=i,
                interval_seconds=1,
                open=100,
                high=101,
                low=99,
                close=100 + 0.1 * i,
                volume=10,
                available_time=k,
            )
        )
    return bars


class _UsesNextBar(Strategy):
    name = "uses_next_bar"

    def on_bar(self, streams, i):
        streams["T"].closes(i + 1)  # future: must raise ValueError -> LookaheadError
        return [Signal(symbol="T", target_weight=1.0, ts=streams["T"].bars[i].ts, reason="leak")]


class _CallsSnapshotBeyondCursor(Strategy):
    name = "snapshot_beyond"

    def on_bar(self, streams, i):
        if i in (3, 7):
            streams["T"].snapshot(i + 2)  # not yet available
        return None


def test_strategy_future_access_raises_lookahead() -> None:
    data = BarSeries("T", _bars(40), 1)
    engine = BacktestEngine(BacktestConfig())
    try:
        engine.run(_UsesNextBar(), {"T": data})  # type: ignore[arg-type]
        raise AssertionError("LookaheadError was expected")
    except LookaheadError:
        pass


def test_leak_mode_does_not_cover_for_us() -> None:
    """allow_same_bar exists to make *config-level* leaks possible, but it must
    never weaken the BarStream cursor guard: even in leak mode a strategy that
    peeks one bar ahead is still rejected."""
    cfg = BacktestConfig(allow_same_bar=True, execution_delay=0)  # config-level leak permitted
    assert BacktestConfig().allow_same_bar is False
    engine = BacktestEngine(cfg)
    data = BarSeries("T", _bars(40), 1)
    try:
        engine.run(_UsesNextBar(), {"T": data})  # type: ignore[arg-type]
        raise AssertionError("LookaheadError was expected even in leak mode")
    except LookaheadError:
        pass


def test_snapshot_beyond_cursor_rejected() -> None:
    data = BarSeries("T", _bars(40), 1)
    engine = BacktestEngine(BacktestConfig())
    try:
        engine.run(_CallsSnapshotBeyondCursor(), {"T": data})  # type: ignore[arg-type]
        raise AssertionError("LookaheadError was expected")
    except LookaheadError:
        pass


def test_delay_guarantees_no_same_bar_fill() -> None:
    data = generate_rw(60, seed=1)
    engine = BacktestEngine(BacktestConfig(execution_delay=3))
    result = engine.run(flip_on_schedule(), {"T": data})
    for fill in result.fills:
        order = next(o for o in result.orders if o.order_id == fill.order_id)
        assert fill.bar_index > order.bar_index  # strictly later bar


def generate_rw(n: int, seed: int) -> BarSeries:
    rng = np.random.default_rng(seed)
    path = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return BarSeries("T", _bars_from_path(path), 1)


def _bars_from_path(path: np.ndarray) -> list[Bar]:
    bars = []
    prev = path[0]
    for i, close in enumerate(path):
        lo, hi = min(prev, close), max(prev, close)
        bars.append(
            Bar(
                symbol="T",
                ts=i,
                interval_seconds=1,
                open=prev,
                high=hi,
                low=lo,
                close=float(close),
                volume=100,
                available_time=i,
            )
        )
        prev = close
    return bars


class _FlipOnSchedule(Strategy):
    name = "flip_schedule"

    def __init__(self) -> None:
        super().__init__({})

    def on_bar(self, streams, i):
        # deliberately switches position every 10 bars to force execution
        if i % 10 == 0:
            w = 1.0 if (i // 10) % 2 == 0 else -1.0
            return [Signal(symbol="T", target_weight=w, ts=streams["T"].bars[i].ts, reason="flip")]
        return None


def flip_on_schedule():
    return _FlipOnSchedule()


def test_stream_never_exposes_after_midnight_reporting() -> None:
    """A bar reported with available_time after the decision bar must NEVER be
    consumed by an earlier decision, even via closes() with cursor inside."""
    bars = []
    for i in range(30):
        av = i
        if i == 20:
            av = 25  # late report: usable only from bar 25 onward
        bars.append(
            Bar(
                symbol="T",
                ts=i,
                interval_seconds=1,
                open=100,
                high=101,
                low=99,
                close=100 + i,
                volume=10,
                available_time=av,
            )
        )
    stream = BarStream(bars, "T")
    # decision at bar 24 (cursor 0 -> 24): bar 20 must still be invisible
    while stream.cursor < 24:
        stream.advance()
    stream.snapshot(24)
    closes = stream.closes(24)
    idx20 = np.where(closes == 120.0)[0]
    assert len(closes) == 24  # bars 0..24 flow but bar 20 (av=25) stays hidden
    assert idx20.size == 0
