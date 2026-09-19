"""Indicator & strategy tests with known answers."""

from __future__ import annotations

import numpy as np
from quantlab.strategies.benchmark import ALL_STRATEGIES, create_strategy, list_strategies
from quantlab.strategies.indicators import ema, rolling_std, rsi, sma, zscore


def test_sma_known_answer() -> None:
    import pytest

    x = np.arange(1.0, 10.0)
    assert sma(x, 3)[-1] == pytest.approx(8.0)


def test_ema_weights_expansion() -> None:
    import pytest

    x = np.array([1.0, 2.0, 3.0, 4.0])
    e = ema(x, 2)
    span = 2 / (2 + 1)  # alpha
    assert e[3] == pytest.approx(span * 4.0 + (1 - span) * e[2])


def test_rolling_std_zero_for_constant() -> None:
    x = np.full(10, 5.0)
    assert np.all(np.isnan(rolling_std(x, 3)[:2])) | (rolling_std(x, 3)[2:] == 0).all()


def test_zscore_known() -> None:
    x = np.array([1.0, 2.0, 3.0, 2.0])
    z = zscore(x, 4)
    last = z[-1]
    mu, sd = np.mean(x), np.std(x, ddof=1)
    assert abs(last - (2.0 - mu) / sd) < 1e-9


def test_rsi_bounded_and_reverses() -> None:
    up = np.linspace(100.0, 130.0, 30)
    down = np.linspace(130.0, 100.0, 30)
    r_up = rsi(up, 14)
    r_down = rsi(down, 14)
    assert r_up[-1] >= 50 and r_down[-1] <= 50
    assert not np.isnan(r_up[-1]) and r_up[-1] <= 100.0
    assert not np.isnan(r_down[-1]) and r_down[-1] >= 0.0
    assert r_up[-1] - r_down[-1] > 20


def test_create_strategy_and_listing() -> None:
    assert "sma_crossover" in list_strategies()
    s = create_strategy("sma_crossover", {"fast": 5, "slow": 20})
    assert s.name == "sma_crossover"
    assert s.warmup() == 21
    try:
        create_strategy("does_not_exist")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_all_strategies_have_defaults() -> None:
    for name, cls in ALL_STRATEGIES.items():
        assert cls.DEFAULT is not None, name
        inst = cls.__call__()
        assert inst.name == name


def test_momentum_warmup_correct() -> None:
    m = create_strategy("momentum", {"lookback": 10})
    assert m.warmup() == 11


def test_vol_target_weight_capped() -> None:
    vt = create_strategy("vol_target", {"target_vol": 0.2, "max_weight": 2.0})
    assert vt.params["max_weight"] == 2.0
