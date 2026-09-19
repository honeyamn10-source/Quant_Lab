"""Event model for the backtest kernel.

The engine dispatches: MarketEvent -> SignalEvent -> OrderEvent -> FillEvent
(which may be a RejectEvent) -> PortfolioEvent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderKind(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class EventType(StrEnum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    REJECT = "REJECT"
    PORTFOLIO = "PORTFOLIO"


@dataclass
class Event:
    type: EventType
    ts: int
    bar_index: int

    def to_dict(self) -> dict:
        return {"type": self.type.value, "ts": self.ts, "bar_index": self.bar_index}


@dataclass
class MarketEvent(Event):
    symbol: str = ""
    bar: Any = None

    def __init__(self, ts: int, bar_index: int, symbol: str = "", bar: Any = None) -> None:
        super().__init__(EventType.MARKET, ts, bar_index)
        self.symbol = symbol
        self.bar = bar


@dataclass
class SignalEvent(Event):
    symbol: str = ""
    target_weight: float = 0.0
    reason: str = ""
    params: dict = field(default_factory=dict)

    def __init__(
        self,
        ts: int,
        bar_index: int,
        symbol: str,
        target_weight: float,
        reason: str = "",
        params: dict | None = None,
    ) -> None:
        super().__init__(EventType.SIGNAL, ts, bar_index)
        self.symbol = symbol
        self.target_weight = target_weight
        self.reason = reason
        self.params = dict(params or {})

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(
            {
                "symbol": self.symbol,
                "target_weight": self.target_weight,
                "reason": self.reason,
                "params": self.params,
            }
        )
        return d


@dataclass
class OrderEvent(Event):
    order_id: int = 0
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    kind: OrderKind = OrderKind.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    eligible_from: int = 0  # bar index at which this order may first execute
    order_ts: int = 0

    def __init__(
        self,
        ts: int,
        bar_index: int,
        order_id: int,
        symbol: str,
        side: OrderSide,
        quantity: float,
        kind: OrderKind = OrderKind.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        eligible_from: int = 0,
        order_ts: int = 0,
    ) -> None:
        super().__init__(EventType.ORDER, ts, bar_index)
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.kind = kind
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.eligible_from = eligible_from
        self.order_ts = order_ts

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(
            {
                "order_id": self.order_id,
                "symbol": self.symbol,
                "side": self.side.value,
                "quantity": self.quantity,
                "kind": self.kind.value,
                "limit_price": self.limit_price,
                "stop_price": self.stop_price,
                "eligible_from": self.eligible_from,
            }
        )
        return d


@dataclass
class FillEvent(Event):
    order_id: int = 0
    symbol: str = ""
    quantity: float = 0.0
    price: float = 0.0
    commission: float = 0.0
    requested: float = 0.0
    partial: bool = False

    def __init__(
        self,
        ts: int,
        bar_index: int,
        order_id: int,
        symbol: str,
        quantity: float,
        price: float,
        commission: float = 0.0,
        requested: float = 0.0,
        partial: bool = False,
    ) -> None:
        super().__init__(EventType.FILL, ts, bar_index)
        self.order_id = order_id
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.commission = commission
        self.requested = requested
        self.partial = partial

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(
            {
                "order_id": self.order_id,
                "symbol": self.symbol,
                "quantity": self.quantity,
                "price": self.price,
                "commission": self.commission,
                "partial": self.partial,
            }
        )
        return d


@dataclass
class RejectEvent(Event):
    order_id: int = 0
    symbol: str = ""
    reason: str = ""

    def __init__(self, ts: int, bar_index: int, order_id: int, symbol: str, reason: str) -> None:
        super().__init__(EventType.REJECT, ts, bar_index)
        self.order_id = order_id
        self.symbol = symbol
        self.reason = reason

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"order_id": self.order_id, "symbol": self.symbol, "reason": self.reason})
        return d


@dataclass
class PortfolioEvent(Event):
    equity: float = 0.0
    cash: float = 0.0
    positions: dict = field(default_factory=dict)
    leverage: float = 0.0

    def __init__(
        self, ts: int, bar_index: int, equity: float, cash: float, positions: dict, leverage: float
    ) -> None:
        super().__init__(EventType.PORTFOLIO, ts, bar_index)
        self.equity = equity
        self.cash = cash
        self.positions = dict(positions)
        self.leverage = leverage

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(
            {
                "equity": self.equity,
                "cash": self.cash,
                "positions": dict(self.positions),
                "leverage": self.leverage,
            }
        )
        return d
