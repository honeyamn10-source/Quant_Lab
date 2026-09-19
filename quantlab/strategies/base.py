"""Strategy API.

A strategy observes only :class:`BarStream` objects whose cursors are advanced
by the backtest engine one bar at a time. Producing a signal for any bar it is
not allowed to see raises ``ValueError`` from the stream — structural look-ahead
protection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from quantlab.data.barstream import BarStream


@dataclass
class Signal:
    """A decision to hold `target_weight` of equity in one symbol.

    target_weight > 0 long, < 0 short, 0 flat. Magnitudes above 1 imply leverage.
    `ts` is the decision timestamp (the bar that closed before this signal).
    """

    symbol: str
    target_weight: float
    ts: int
    reason: str = ""
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "target_weight": self.target_weight,
            "ts": self.ts,
            "reason": self.reason,
            "params": self.params,
        }


class Strategy(ABC):
    """Base class for strategy implementations."""

    name: str = "base"

    def __init__(self, params: dict | None = None) -> None:
        self.params: dict = dict(params or {})

    def warmup(self) -> int:
        """Bars of data a strategy needs before it can decide."""
        return 0

    @abstractmethod
    def on_bar(self, streams: dict[str, BarStream], i: int) -> list[Signal] | Signal | None:
        """Called when decision bar `i` (0-indexed) has closed.

        ``streams[symbol].cursor == i`` when invoked, so future data is
        unreachable structurally. Return one or more Signals (target weights),
        or ``None`` to stay flat.
        """
        raise NotImplementedError
