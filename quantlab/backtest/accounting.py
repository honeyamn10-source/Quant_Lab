"""Portfolio accounting: cash, positions, equity, leverage, position limits."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    quantity: float = 0.0
    avg_price: float = 0.0


class AccountingError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Portfolio:
    """Long/short accounting with cash, leverage and position limits."""

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        max_leverage: float = 2.0,
        max_position_weight: float = 1.0,
    ) -> None:
        self.cash = float(initial_cash)
        self.positions: dict[str, Position] = {}
        self.max_leverage = float(max_leverage)
        self.max_position_weight = float(max_position_weight)

    def qty(self, symbol: str) -> float:
        return self.positions.get(symbol, Position()).quantity

    def gross_notional(self, prices: dict[str, float]) -> float:
        return sum(abs(p.quantity) * prices.get(sym, p.avg_price) for sym, p in self.positions.items())

    def short_notional(self, prices: dict[str, float]) -> float:
        return sum(abs(p.quantity) * prices.get(sym, p.avg_price) for sym, p in self.positions.items() if p.quantity < 0)

    def equity_at(self, prices: dict[str, float]) -> float:
        mv = sum(p.quantity * prices.get(sym, p.avg_price) for sym, p in self.positions.items())
        return self.cash + mv

    def leverage_at(self, prices: dict[str, float]) -> float:
        eq = self.equity_at(prices)
        return self.gross_notional(prices) / eq if eq > 0 else 0.0

    def position_weight(self, symbol: str, price: float, equity: float) -> float:
        qty = self.qty(symbol)
        if equity <= 0:
            return float("inf") if qty != 0 else 0.0
        return abs(qty * price) / equity

    def apply_fill(self, symbol: str, quantity: float, price: float) -> None:
        """Apply a fill. quantity > 0 buys, < 0 sells. Raises on cash violation."""
        if abs(quantity) < 1e-12:
            return
        px = float(price)
        if px <= 0:
            raise AccountingError("non-positive fill price")
        pos = self.positions.get(symbol, Position())
        new_qty = pos.quantity + quantity

        if self.cash - quantity * px < -1e-9:
            raise AccountingError("insufficient cash")

        if new_qty * pos.quantity <= 0 or pos.quantity == 0:
            # new / closed / flipped position
            self.positions[symbol] = Position(new_qty, px)
        else:
            old_notional = abs(pos.quantity) * pos.avg_price
            add_notional = abs(quantity) * px
            pos.quantity = new_qty
            pos.avg_price = (old_notional + add_notional) / (abs(pos.quantity) + abs(quantity))
            self.positions[symbol] = pos
        self.cash -= quantity * px
        if abs(new_qty) < 1e-9:
            del self.positions[symbol]

    # tolerance absorbs half-spread / reference-price noise at the boundary so a
    # target exactly at the cap (e.g. 2.0x) is not rejected by a rounding overshoot
    _WEIGHT_EPSILON = 1e-3

    def clamp_fill(self, symbol: str, signed_qty: float, price: float, equity: float) -> float:
        """Largest |signed_qty|-bounded quantity at `price` that keeps position
        weight and leverage within limits. Returns 0 when nothing fits."""
        if abs(signed_qty) < 1e-12 or price <= 0 or equity <= 0:
            return 0.0
        sign = 1.0 if signed_qty > 0 else -1.0
        cur = self.qty(symbol)
        max_qty = abs(signed_qty)
        if sign > 0:  # long buys must be cash-backed in this cash account
            max_qty = min(max_qty, max(self.cash, 0.0) / price)
        if self.max_position_weight >= 0:
            margin = self.max_position_weight + self._WEIGHT_EPSILON
            # adding same-side: |cur| + q <= margin*equity/price
            bound_same = margin * equity / price - abs(cur)
            # flipping/reducing opposite side with q > |cur|: q - |cur| <= margin*equity/price
            bound_flip = margin * equity / price + abs(cur)
            max_qty = min(max_qty, max(bound_same, bound_flip))
        if self.max_leverage >= 0:
            margin = self.max_leverage + self._WEIGHT_EPSILON
            other_gross = sum(abs(p.quantity) * (price if s == symbol else p.avg_price)
                              for s, p in self.positions.items() if s != symbol)
            max_qty = min(max_qty, (margin * equity - other_gross) / price)
        return sign * max(max_qty, 0.0)

    def assert_limits(self, symbol: str, signed_qty: float, price: float, equity: float) -> None:
        """Raise AccountingError if adding `signed_qty` at `price` breaches limits."""
        projected_qty = self.qty(symbol) + signed_qty
        if abs(projected_qty) < 1e-12:
            return
        w = abs(projected_qty) * price / equity if equity > 0 else float("inf")
        if self.max_position_weight >= 0 and w > self.max_position_weight + self._WEIGHT_EPSILON:
            raise AccountingError(f"position weight {w:.2%} exceeds limit {self.max_position_weight:.2%}")
        gross = self.gross_notional({symbol: price}) + abs(signed_qty) * price
        if equity > 0 and gross / equity > self.max_leverage + self._WEIGHT_EPSILON:
            raise AccountingError(f"leverage {gross / equity:.2f}x exceeds limit {self.max_leverage}x")

    def dividends(self, symbol: str, per_share: float) -> None:
        self.cash += self.qty(symbol) * per_share

    def financing(self, prices: dict[str, float], borrowing_rate_year: float) -> float:
        """Per-bar cost of borrowing short positions."""
        short_mv = self.short_notional(prices)
        return short_mv * float(borrowing_rate_year) / 252.0
