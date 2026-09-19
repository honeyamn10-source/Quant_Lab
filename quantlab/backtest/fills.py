"""Fill model: how an order turns into (possibly partial) executions.

Defaults model spread/slippage on top of the reference price. Partial fills
happen when the requested notional exceeds a realistic per-bar participation
fraction of available volume.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantlab.backtest.events import OrderEvent, OrderKind, OrderSide
from quantlab.execution.cost_model import ExecutionProfile, get_profile


@dataclass
class FillResult:
    price: float
    quantity: float
    commission: float
    partial: bool


class FillModel:
    """Deterministic fill execution against a bar's OHLC + execution profile."""

    def __init__(
        self,
        profile_name_or_obj: str | ExecutionProfile,
        commission_bps: float = 5.0,
        min_commission: float = 1.0,
        participation_cap: float = 0.10,
        execution_mode: str = "next_open",
    ) -> None:
        self.profile = (
            get_profile(profile_name_or_obj)
            if isinstance(profile_name_or_obj, str)
            else profile_name_or_obj
        )
        self.commission_bps = float(commission_bps)
        self.min_commission = float(min_commission)
        self.participation_cap = float(participation_cap)
        self.execution_mode = execution_mode

    def commission(self, notional: float) -> float:
        c = notional * self.commission_bps / 10_000.0
        return max(c, self.min_commission if c > 0 else 0.0)

    def _reference_price(self, bar) -> float:
        if self.execution_mode == "next_close":
            return float(bar.close)
        return float(bar.open)

    def fill(self, order: OrderEvent, bar, equity: float) -> FillResult | None:
        """Try to fill `order` at `bar`. Returns None if a resting limit/stop
        should remain unfilled (it stays alive)."""
        price = self._reference_price(bar)
        filled_qty = order.quantity

        # Partial fill: cap participation against the bar's traded volume.
        if order.quantity > 0 and bar.volume > 0:
            cap_qty = self.participation_cap * bar.volume
            if cap_qty < abs(order.quantity):
                filled_qty = cap_qty

        half_spread = self.profile.spread_bps / 10_000.0 / 2.0
        effective = price * (1 + half_spread if order.side == OrderSide.BUY else 1 - half_spread)

        if order.kind == OrderKind.MARKET:
            exec_price = effective
            partial = filled_qty < abs(order.quantity)
        elif order.kind == OrderKind.LIMIT:
            limit = float(order.limit_price or 0.0)
            if order.side == OrderSide.BUY and effective <= limit:
                exec_price = min(effective, limit)
            elif order.side == OrderSide.SELL and effective >= limit:
                exec_price = max(effective, limit)
            else:
                return None
            partial = filled_qty < abs(order.quantity)
        elif order.kind == OrderKind.STOP:
            stop = float(order.stop_price or 0.0)
            if order.side == OrderSide.SELL and max(float(bar.high), float(bar.close)) >= stop:
                exec_price = max(stop, effective)
            elif order.side == OrderSide.BUY and min(float(bar.low), float(bar.close)) <= stop:
                exec_price = min(stop, effective)
            else:
                return None
            partial = filled_qty < abs(order.quantity)
        else:
            raise ValueError(f"unsupported order kind {order.kind}")

        notional = abs(filled_qty) * exec_price
        return FillResult(
            price=exec_price,
            quantity=filled_qty,
            commission=self.commission(notional),
            partial=partial,
        )
