"""Deterministic event-driven backtest kernel.

Loop over bars; each bar produces:
    MarketEvent -> strategy signals -> OrderEvents (eligible later) ->
    fills at the earliest eligible bar -> PortfolioEvent (mark to market).

Timing rule: a signal computed from bar `t` becomes executable no earlier than
bar `t + 1 + execution_delay` (+ latency converted to bars). Same-bar execution
is only allowed when ``allow_same_bar=True``, which exists exclusively for
look-ahead tests so the engine can prove the leak exists.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from quantlab.backtest.accounting import AccountingError, Portfolio
from quantlab.backtest.events import (
    Event,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderKind,
    OrderSide,
    PortfolioEvent,
    RejectEvent,
    SignalEvent,
)
from quantlab.backtest.fills import FillModel
from quantlab.data.barstream import BarStream
from quantlab.data.schemas import BarSeries
from quantlab.strategies.base import Signal, Strategy


class LookaheadError(Exception):
    """Raised when a strategy touches information ahead of its cursor."""


@dataclass
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    execution_delay: int = 1  # bars after the signal bar before an order may fill
    execution_mode: str = "next_open"  # or "next_close"
    profile: str = "normal"
    max_leverage: float = 2.0
    max_position_weight: float = 1.0
    rebalance_tolerance: float = 0.005
    borrow_rate_year: float = 0.0
    commission_bps: float = 5.0
    min_commission: float = 1.0
    participation_cap: float = 0.10
    dividends_per_bar: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    allow_same_bar: bool = False
    seed: int = 0

    def validate(self) -> None:
        if self.execution_delay < 1 and not self.allow_same_bar:
            raise ValueError(
                "execution_delay < 1 would allow same-bar execution (look-ahead). "
                "Set allow_same_bar=True ONLY in explicit leak tests."
            )
        if self.execution_mode not in ("next_open", "next_close"):
            raise ValueError("execution_mode must be 'next_open' or 'next_close'")


@dataclass
class BacktestResult:
    config: BacktestConfig
    signals: list[SignalEvent] = field(default_factory=list)
    orders: list[OrderEvent] = field(default_factory=list)
    fills: list[FillEvent] = field(default_factory=list)
    rejects: list[RejectEvent] = field(default_factory=list)
    portfolio_events: list[PortfolioEvent] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    equity_curve: list[tuple[int, float]] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    n_bars: int = 0

    def to_dict(self) -> dict:
        return {
            "config": self.config.__dict__,
            "n_bars": self.n_bars,
            "signals": [s.to_dict() for s in self.signals],
            "orders": [o.to_dict() for o in self.orders],
            "fills": [f.to_dict() for f in self.fills],
            "rejects": [r.to_dict() for r in self.rejects],
            "portfolio_events": [p.to_dict() for p in self.portfolio_events],
            "equity_curve": [(ts, eq) for ts, eq in self.equity_curve],
            "returns": list(self.returns),
        }

    def equity(self) -> list[float]:
        return [eq for _, eq in self.equity_curve]

    def final_equity(self) -> float:
        return self.equity_curve[-1][1] if self.equity_curve else self.config.initial_cash


class BacktestEngine:
    """Runs a strategy over a dict of symbol -> BarSeries/BarStream."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self.config.validate()

    @staticmethod
    def _to_streams(data: Mapping[str, BarSeries | BarStream]) -> dict[str, BarStream]:
        return {
            symbol: (src if isinstance(src, BarStream) else BarStream(src, symbol))
            for symbol, src in data.items()
        }

    def _latency_bars(self, bar_interval_ms: int) -> int:
        return int(math.ceil(max(0.0, self.config.latency_ms) / max(bar_interval_ms, 1)))

    def _bar_ts(self, streams: dict[str, BarStream], i: int) -> int:
        first = next(iter(streams.values()))
        return first.bars[i].ts if first.total > 0 else 0

    def run(
        self,
        strategy: Strategy,
        data: Mapping[str, BarSeries | BarStream],
    ) -> BacktestResult:
        streams = self._to_streams(data)
        n = min((s.total for s in streams.values()), default=0)
        result = BacktestResult(config=self.config, n_bars=n)
        portfolio = Portfolio(
            initial_cash=self.config.initial_cash,
            max_leverage=self.config.max_leverage,
            max_position_weight=self.config.max_position_weight,
        )
        fill_model = FillModel(
            self.config.profile,
            self.config.commission_bps,
            self.config.min_commission,
            self.config.participation_cap,
            execution_mode=self.config.execution_mode,
        )
        pending: list[OrderEvent] = []
        order_id = 0
        prev_equity = float(self.config.initial_cash)
        first_streams = next(iter(streams.values()), None)
        interval_ms = (
            first_streams.bars[0].interval_seconds * 1000
            if first_streams and first_streams.bars
            else 86_400_000
        )
        extra_bars = self._latency_bars(interval_ms)

        for i in range(n):
            ts_i = self._bar_ts(streams, i)
            close_prices = {}
            for sym, s in streams.items():
                close_prices[sym] = float(s.bars[i].close)
                while s.cursor < i:
                    s.advance()
                assert s.cursor <= i  # engine-managed cursor can never exceed decision bar

            # 1. MarketEvent for this bar
            result.events.append(MarketEvent(ts=ts_i, bar_index=i))

            # 2. Process eligible pending orders (at the OPEN of bar i)
            remaining: list[OrderEvent] = []
            for order in pending:
                if order.eligible_from > i:
                    remaining.append(order)
                    continue
                bar = streams[order.symbol].bars[i]
                fill = fill_model.fill(order, bar, prev_equity)
                if fill is None:
                    remaining.append(order)  # resting limit/stop not yet triggered
                    continue
                try:
                    # Clamp at the portfolio's position/leverage boundary (price drift
                    # between signal and fill can overshoot the target), then re-check.
                    allowed = portfolio.clamp_fill(
                        order.symbol, fill.quantity, fill.price, prev_equity
                    )
                    if allowed == 0:
                        raise AccountingError(
                            f"position weight/leverage exceeded with zero clamp headroom for {order.symbol}"
                        )
                    portfolio.assert_limits(order.symbol, allowed, fill.price, prev_equity)
                    portfolio.apply_fill(order.symbol, allowed, fill.price)
                except AccountingError as exc:
                    result.rejects.append(
                        RejectEvent(
                            ts=bar.ts,
                            bar_index=i,
                            order_id=order.order_id,
                            symbol=order.symbol,
                            reason=exc.reason,
                        )
                    )
                    result.events.append(result.rejects[-1])
                    continue
                fill_event = FillEvent(
                    ts=bar.ts,
                    bar_index=i,
                    order_id=order.order_id,
                    symbol=order.symbol,
                    quantity=allowed,
                    price=fill.price,
                    commission=fill.commission,
                    requested=order.quantity,
                    partial=fill.partial or abs(allowed) < abs(order.quantity) - 1e-12,
                )
                result.fills.append(fill_event)
                result.events.append(fill_event)
                if fill.partial and fill.quantity > 0:  # keep the unfilled residual alive
                    residual = abs(order.quantity) - fill.quantity
                    if residual > 1e-12:
                        remaining.append(
                            OrderEvent(
                                ts=order.ts,
                                bar_index=order.bar_index,
                                order_id=order.order_id,
                                symbol=order.symbol,
                                side=order.side,
                                quantity=residual,
                                kind=order.kind,
                                limit_price=order.limit_price,
                                stop_price=order.stop_price,
                                eligible_from=order.eligible_from,
                                order_ts=order.order_ts,
                            )
                        )
            pending = remaining

            # 3. Strategy decision at CLOSE of bar i
            curr_equity = portfolio.equity_at(close_prices)
            try:
                strategy_out = strategy.on_bar(streams, i)
            except ValueError as exc:  # BarStream raises on future access
                raise LookaheadError(
                    f"strategy attempted future access at bar {i}: {exc}"
                ) from None
            emitted = (
                strategy_out
                if isinstance(strategy_out, list)
                else ([strategy_out] if strategy_out is not None else [])
            )
            for sig in emitted:
                if not isinstance(sig, Signal):
                    continue
                sym, target = sig.symbol, float(sig.target_weight)
                close = close_prices.get(sym)
                if close is None:
                    result.rejects.append(
                        RejectEvent(
                            ts=ts_i,
                            bar_index=i,
                            order_id=-1,
                            symbol=sym,
                            reason=f"unknown symbol {sym}",
                        )
                    )
                    continue
                result.signals.append(
                    SignalEvent(
                        ts=sig.ts,
                        bar_index=i,
                        symbol=sym,
                        target_weight=target,
                        reason=sig.reason,
                        params=sig.params,
                    )
                )
                result.events.append(result.signals[-1])
                desired_qty = target * curr_equity / close
                # Net against position AND any unfilled pending orders for this
                # symbol: a strategy that re-signals every bar (buy_and_hold,
                # vol_target) must not stack duplicate orders, while flips still
                # net correctly. If an adjustment is genuinely needed, replace
                # the symbol's pending orders with a single net order.
                open_qty = sum(
                    (o.quantity if o.side == OrderSide.BUY else -o.quantity)
                    for o in pending
                    if o.symbol == sym
                )
                committed = portfolio.qty(sym) + open_qty
                delta = desired_qty - committed
                if abs(delta) / max(1.0, abs(committed)) <= self.config.rebalance_tolerance:
                    continue  # existing (or empty) commitment already matches target
                pending = [o for o in pending if o.symbol != sym]  # replace, don't stack
                current_qty = portfolio.qty(sym)
                delta = desired_qty - current_qty
                if abs(delta) / max(1.0, abs(current_qty)) <= self.config.rebalance_tolerance:
                    continue
                order_id += 1
                order = OrderEvent(
                    ts=sig.ts,
                    bar_index=i,
                    order_id=order_id,
                    symbol=sym,
                    side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                    quantity=abs(delta),
                    kind=OrderKind.MARKET,
                    eligible_from=i + 1 + self.config.execution_delay + extra_bars,
                    order_ts=sig.ts,
                )
                result.orders.append(order)
                result.events.append(order)
                pending.append(order)

            # 4. Financing + mark to market at close of bar i
            fin = portfolio.financing(close_prices, self.config.borrow_rate_year)
            if fin:
                portfolio.cash -= fin
            eq = portfolio.equity_at(close_prices)
            result.portfolio_events.append(
                PortfolioEvent(
                    ts=ts_i,
                    bar_index=i,
                    equity=eq,
                    cash=portfolio.cash,
                    positions={s: p.quantity for s, p in portfolio.positions.items()},
                    leverage=portfolio.leverage_at(close_prices),
                )
            )
            result.events.append(result.portfolio_events[-1])
            result.equity_curve.append((ts_i, eq))
            if len(result.returns) < len(result.equity_curve) - 1:
                result.returns.append(result.equity_curve[-1][1] / result.equity_curve[-2][1] - 1.0)
            prev_equity = eq

        return result
