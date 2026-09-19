"""Execution reality engine.

Execution Cost = Commission + Spread + Slippage + Market Impact + Financing + Borrow Cost

Four built-in profiles: optimistic, normal, stress, extreme. The optimistic
model is never the default production assessment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import BaseModel, Field


class ExecutionProfile(BaseModel):
    """Assumptions that apply to fills and cost waterfalls."""

    name: str = "normal"
    spread_bps: float = Field(default=2.0, ge=0.0)
    slippage_bps: float = Field(default=1.0, ge=0.0)
    impact_bps: float = Field(default=2.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    commission_bps: float = Field(default=5.0, ge=0.0)
    fill_participation: float = Field(default=0.10, gt=0.0, le=1.0)
    borrow_rate_year: float = Field(default=0.0, ge=0.0)
    description: str = ""


PROFILES: dict[str, ExecutionProfile] = {
    "optimistic": ExecutionProfile(
        name="optimistic", spread_bps=0.5, slippage_bps=0.0, impact_bps=0.0,
        latency_ms=0.0, commission_bps=1.0, fill_participation=0.25,
        description="lower bound reference; never the production assessment",
    ),
    "normal": ExecutionProfile(
        name="normal", spread_bps=2.0, slippage_bps=1.0, impact_bps=2.0,
        latency_ms=50.0, commission_bps=5.0, fill_participation=0.10,
        description="default conservative baseline",
    ),
    "stress": ExecutionProfile(
        name="stress", spread_bps=6.0, slippage_bps=5.0, impact_bps=8.0,
        latency_ms=250.0, commission_bps=8.0, fill_participation=0.05,
        description="deteriorating market conditions",
    ),
    "extreme": ExecutionProfile(
        name="extreme", spread_bps=20.0, slippage_bps=20.0, impact_bps=25.0,
        latency_ms=1000.0, commission_bps=12.0, fill_participation=0.02,
        description="crisis / illiquid conditions",
    ),
}


def get_profile(name: str) -> ExecutionProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}") from None


@dataclass(frozen=True)
class CostBreakdown:
    spread: float
    slippage: float
    impact: float
    commission: float
    financing: float
    borrow: float

    @property
    def total(self) -> float:
        return self.spread + self.slippage + self.impact + self.commission + self.financing + self.borrow

    def to_dict(self) -> dict:
        return {
            "spread": self.spread,
            "slippage": self.slippage,
            "impact": self.impact,
            "commission": self.commission,
            "financing": self.financing,
            "borrow": self.borrow,
            "total": self.total,
        }


class ExecutionCostModel:
    """Computes round-trip cost estimates per unit of capital deployed.

    Market impact follows a square-root participation model:
        impact_bps · sqrt(participation)  where participation = size / ADV
    """

    def __init__(self, profile_name: str | ExecutionProfile = "normal", adv_liquidity: float = 1e7) -> None:
        self.profile = get_profile(profile_name) if isinstance(profile_name, str) else profile_name
        self.adv = float(adv_liquidity)

    def _impact(self, order_notional: float) -> float:
        if self.adv <= 0 or order_notional <= 0:
            return 0.0
        participation = min(1.0, order_notional / self.adv)
        return self.profile.impact_bps / 10_000.0 * math.sqrt(participation)

    def cost_per_unit_notional(self, order_notional: float, holding_bars: int = 1) -> CostBreakdown:
        """One-way cost, in fractions of order notional."""
        half_spread = self.profile.spread_bps / 10_000.0 / 2.0 + self.profile.slippage_bps / 10_000.0
        impact = self._impact(order_notional)
        commission = self.profile.commission_bps / 10_000.0
        financing = self.profile.borrow_rate_year / 252.0 * holding_bars if self.profile.borrow_rate_year > 0 else 0.0
        return CostBreakdown(
            spread=half_spread,
            slippage=self.profile.slippage_bps / 10_000.0,
            impact=impact,
            commission=commission,
            financing=financing,
            borrow=self.profile.borrow_rate_year / 252.0 if self.profile.borrow_rate_year > 0 else 0.0,
        )

    def round_trip_cost(self, order_notional: float) -> float:
        one_way = self.cost_per_unit_notional(order_notional)
        return 2.0 * (one_way.spread + one_way.slippage + one_way.commission) + one_way.impact

    def capacity(self, net_annual_return: float, order_notional: float, turnover_x: float = 20.0) -> float:
        """Estimate the capital at which trading costs consume the gross return."""
        if net_annual_return <= 0:
            return 0.0
        cost = self.round_trip_cost(order_notional) * turnover_x
        if cost <= 0:
            return float("inf")
        # scale: costs scale as sqrt(participation); solve for the turnover-adjusted break-even
        gross_per_turnover = net_annual_return / turnover_x
        breakeven_cost = gross_per_turnover
        bps0 = self.profile.impact_bps / 10_000.0
        if bps0 <= 0:
            return float("inf")
        participation_needed = (breakeven_cost / bps0) ** 2
        if participation_needed <= 0 or participation_needed >= 1.0:
            return 0.0
        return self.adv * participation_needed


def waterfall(gross_alpha: float, cost: CostBreakdown) -> dict:
    """Gross Alpha -> Spread -> ... -> Net Alpha."""
    running = gross_alpha
    steps = []
    for label, value in (
        ("spread", cost.spread),
        ("commission", cost.commission),
        ("slippage", cost.slippage),
        ("impact", cost.impact),
        ("financing", cost.financing),
        ("borrow", cost.borrow),
    ):
        running -= value
        steps.append({"stage": label, "cost": value, "cumulative": running})
    steps.append({"stage": "net_alpha", "cost": 0.0, "cumulative": running})
    return {"gross_alpha": gross_alpha, "net_alpha": running, "steps": steps}
