"""Unit tests for the adversarial robustness battery."""

from __future__ import annotations

import types

import quantlab.adversarial.robustness as robustness
from quantlab.adversarial.perturbation import (
    cost_perturbation_sweep,
    param_perturbation_sweep,
    scaled_params,
)
from quantlab.adversarial.randomization import (
    drop_random_trades,
    random_entry_delay,
    shuffle_signal_order,
)
from quantlab.adversarial.robustness import degradation_slope, robustness_surface


class FakeResult:
    """Minimal result-like object exposing returns/signals/fills/equity."""

    def __init__(
        self,
        returns: list[float],
        signals: list[object] | None = None,
        fills: list[object] | None = None,
        equity: list[float] | None = None,
        metric: float = 0.0,
    ) -> None:
        self.returns = list(returns)
        self.signals = list(signals or [])
        self.fills = list(fills or [])
        self._equity = list(equity) if equity is not None else [1_000_000.0]
        self.metric = metric

    def equity(self) -> list[float]:
        return list(self._equity)


def _sig(weight: float, bar: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(target_weight=weight, bar_index=bar)


def _fill(qty: float, price: float, bar: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(quantity=qty, price=price, bar_index=bar)


def _params_sweep(scales: list[float], metrics: list[float]) -> list[dict]:
    return [
        {"params": {"perturbation": {"scale": scale}}, "result": FakeResult([], metric=m)}
        for scale, m in zip(scales, metrics, strict=True)
    ]


def test_scaled_params_variants() -> None:
    base = {"fast": 20, "slow": 50, "name": "sma"}
    scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    variants = scaled_params(base, scales)
    assert len(variants) == len(scales)
    for variant, scale in zip(variants, scales, strict=True):
        assert variant["name"] == base["name"]
        assert variant["fast"] == base["fast"] * scale
        assert variant["slow"] == base["slow"] * scale
        assert variant["perturbation"] == {"scale": scale}


def test_param_perturbation_sweep_calls_run_fn() -> None:
    calls: list[dict] = []

    def run_fn(overrides: dict) -> FakeResult:
        calls.append(overrides)
        return FakeResult([0.01, -0.005, 0.02, 0.0, 0.015])

    factors = [0.8, 0.9, 1.0, 1.1, 1.2]
    entries = param_perturbation_sweep(run_fn, {"fast": 20, "slow": 50}, factors)
    assert len(calls) >= len(factors)
    assert len(entries) == len(factors)
    assert all({"params", "result"} <= set(entry) for entry in entries)
    assert entries[0]["params"]["fast"] == 16
    assert entries[0]["params"]["slow"] == 40


def test_cost_perturbation_sweep_sorted() -> None:
    calls: list[dict] = []

    def run_fn(overrides: dict) -> FakeResult:
        calls.append(overrides)
        return FakeResult([0.01, 0.02, 0.01])

    entries = cost_perturbation_sweep(run_fn)
    factors = [entry["factor"] for entry in entries]
    assert factors == sorted(factors)
    assert 3.0 in factors
    assert calls[-1] == {"commission_bps": 15.0, "spread_factor": 3.0}


def test_shuffle_signal_order_deterministic() -> None:
    signals = [_sig(w, i) for i, w in enumerate([1.0, -1.0, 0.5, 0.5])]
    result = FakeResult([0.01, -0.02, 0.03, 0.0, 0.005], signals=signals)
    first = shuffle_signal_order(result, seed=7)
    second = shuffle_signal_order(result, seed=7)
    assert first == second
    assert set(first) == {"n_signals", "shuffled_net_return", "natural_net_return"}
    assert first["n_signals"] == len(signals)


def test_drop_random_trades_deterministic() -> None:
    fills = [_fill(500.0, 100.0, i) for i in range(10)]
    result = FakeResult([0.01] * 10, fills=fills)
    first = drop_random_trades(result, drop_frac=0.2, seed=3)
    second = drop_random_trades(result, drop_frac=0.2, seed=3)
    assert first == second
    assert set(first) == {"removed", "net_impact_estimate"}
    assert first["removed"] == 2


def test_random_entry_delay_deterministic() -> None:
    fills = [_fill(500.0, 100.0, i) for i in range(5)]
    result = FakeResult([0.01] * 10, fills=fills)
    first = random_entry_delay(result, delay_bars=2, seed=1)
    second = random_entry_delay(result, delay_bars=2, seed=1)
    assert first == second
    assert set(first) == {"n_shifted", "impact_estimate"}
    assert first["n_shifted"] == 5


def test_robustness_surface_ratings(monkeypatch) -> None:
    monkeypatch.setattr(robustness, "compute_sharpe", lambda result: float(result.metric))

    fragile = robustness_surface({"params": _params_sweep([1.0, 1.01, 1.02], [2.0, 1.5, 1.0])})
    assert fragile["rating"] == "FRAGILE"
    assert fragile["slopes"]["params"] < -0.30

    moderate = robustness_surface({"params": _params_sweep([1.0, 1.02], [0.4, 0.2])})
    assert moderate["rating"] == "MODERATE"

    robust = robustness_surface({"params": _params_sweep([1.0, 1.02], [1.0, 1.6])})
    assert robust["rating"] == "ROBUST"
    assert robust["worst"] == {"dimension": "params", "metric": 1.0}
    assert set(robust) == {"rating", "worst", "slopes", "table"}


def test_degradation_slope_flat() -> None:
    returns = [0.01, 0.02, 0.015, 0.01, 0.02]
    results = [{"factor": factor, "result": FakeResult(returns)} for factor in (1.0, 1.5, 2.0, 3.0)]
    slope = degradation_slope(results, dimension="cost")
    assert slope["dimension"] == "cost"
    assert isinstance(slope["slope_per_1pct"], float)
    assert abs(slope["slope_per_1pct"]) < 1e-12
    assert abs(slope["best_metric"] - slope["worst_metric"]) < 1e-12
