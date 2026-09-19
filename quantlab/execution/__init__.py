"""Execution reality engine (spread, slippage, impact, latency, profiles, capacity)."""

from quantlab.execution.cost_model import (
    CostBreakdown,
    ExecutionCostModel,
    ExecutionProfile,
    get_profile,
    waterfall,
)

PROFILE_NAMES = ("optimistic", "normal", "stress", "extreme")

__all__ = [
    "CostBreakdown",
    "ExecutionCostModel",
    "ExecutionProfile",
    "PROFILE_NAMES",
    "get_profile",
    "waterfall",
]
