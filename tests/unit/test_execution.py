"""Execution cost-model tests: profiles, waterfall, impact, capacity."""

from __future__ import annotations

from quantlab.execution.cost_model import (
    PROFILES,
    ExecutionCostModel,
    ExecutionProfile,
    get_profile,
    waterfall,
)


def test_profiles_available_and_ordered() -> None:
    assert {"optimistic", "normal", "stress", "extreme"} <= set(PROFILES)
    by_spread = sorted(PROFILES, key=lambda n: PROFILES[n].spread_bps)
    assert by_spread == ["optimistic", "normal", "stress", "extreme"]


def test_cost_breakdown_components_nonnegative() -> None:
    model = ExecutionCostModel(profile_name="normal", adv_liquidity=1e7)
    b = model.cost_per_unit_notional(order_notional=1e5)
    assert b.spread >= 0 and b.slippage >= 0 and b.impact >= 0 and b.commission >= 0
    # documented model: spread embeds half-spread + slippage
    assert abs(b.spread - (2.0 / 2.0 + 1.0) / 10_000.0) < 1e-12
    assert abs(b.slippage - 1.0 / 10_000.0) < 1e-12
    assert abs(b.impact - 2.0 / 10_000.0 * (1e5 / 1e7) ** 0.5) < 1e-12
    assert abs(b.commission - 5.0 / 10_000.0) < 1e-12
    assert b.total == b.spread + b.slippage + b.impact + b.commission + b.financing + b.borrow


def test_round_trip_at_least_one_way() -> None:
    model = ExecutionCostModel(profile_name="normal", adv_liquidity=1e7)
    rt = model.round_trip_cost(order_notional=1e5)
    one_way = model.cost_per_unit_notional(order_notional=1e5).total
    assert rt >= one_way


def test_larger_order_higher_impact_per_unit() -> None:
    model = ExecutionCostModel(profile_name="normal", adv_liquidity=1e7)
    small = model.cost_per_unit_notional(order_notional=1e4).impact
    large = model.cost_per_unit_notional(order_notional=1e6).impact
    assert large > small


def test_extreme_profile_costs_more() -> None:
    normal = ExecutionCostModel(profile_name="normal", adv_liquidity=1e7)
    extreme = ExecutionCostModel(profile_name="extreme", adv_liquidity=1e7)
    assert extreme.round_trip_cost(order_notional=1e5) > normal.round_trip_cost(order_notional=1e5)


def test_capacity_finite_and_positive() -> None:
    model = ExecutionCostModel(profile_name="normal", adv_liquidity=1e7)
    cap = model.capacity(net_annual_return=0.001, order_notional=1e5, turnover_x=10.0)
    assert 0 < cap <= 1e7


def test_capacity_zero_for_nonpositive_return() -> None:
    model = ExecutionCostModel(profile_name="normal", adv_liquidity=1e7)
    assert model.capacity(net_annual_return=0.0, order_notional=1e5) == 0.0


def test_waterfall_consumes_gross_alpha() -> None:
    model = ExecutionCostModel(profile_name="stress", adv_liquidity=1e6)
    cost = model.cost_per_unit_notional(order_notional=1e4)
    out = waterfall(gross_alpha=0.20, cost=cost)
    assert out["net_alpha"] < out["gross_alpha"]
    labels = [s["stage"] for s in out["steps"]]
    assert labels[0] == "spread"
    assert labels[-1] == "net_alpha"
    # cumulative strictly decreases through cost stages
    cums = [s["cumulative"] for s in out["steps"]]
    assert cums == sorted(cums, reverse=True)


def test_latency_tracked_per_profile() -> None:
    assert PROFILES["optimistic"].latency_ms == 0.0
    assert PROFILES["extreme"].latency_ms > PROFILES["normal"].latency_ms


def test_get_profile_invalid_raises() -> None:
    try:
        get_profile("nonexistent_profile")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_optimistic_never_default() -> None:
    assert ExecutionProfile().name == "normal"
    assert ExecutionCostModel().profile.name == "normal"
