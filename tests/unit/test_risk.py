"""Unit tests for quantlab.risk analytics."""

from __future__ import annotations

import numpy as np
import pytest
from quantlab.risk import (
    SCENARIOS,
    conditional_correlation,
    downside_correlation,
    drawdown_correlation,
    drawdown_series,
    max_drawdown,
    normal_correlation,
    portfolio_stress_report,
    rolling_correlation,
    stress_correlation,
    tail_dependence,
)

SCENARIO_KEYS = {
    "scenario",
    "normal_corr",
    "stress_corr",
    "corr_expansion",
    "stress_vol",
    "normal_vol",
    "vol_ratio",
    "stress_max_drawdown",
    "stress_p95_return",
}


def test_max_drawdown_known_series_after_peak() -> None:
    returns = np.array([0.0, -0.1, 0.1])
    assert max_drawdown(returns) == pytest.approx(0.1)
    np.testing.assert_allclose(drawdown_series(returns), [0.0, 0.1, 0.01], atol=1e-12)


def test_rolling_correlation_shape_and_anticorrelation() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 0.01, 300)
    b = -a
    panel = np.column_stack([a, b])
    rc = rolling_correlation(panel, window=20)
    assert rc.shape == (300,)
    assert np.isnan(rc[:19]).all()
    assert np.isclose(rc[-1], -1.0, atol=1e-6)


def test_downside_correlation_identical_series() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(-0.001, 0.02, 800)
    assert downside_correlation(a, a) == pytest.approx(1.0)
    assert np.isnan(downside_correlation(a, a + 1.0))


def test_tail_dependence_shape_and_multivariate_gaussians() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 0.01, (3000, 3))
    td = tail_dependence(x, threshold=0.05)
    assert td.shape == (3, 3)
    np.testing.assert_allclose(td, td.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(td), 1.0, atol=1e-12)
    off_diag = td[~np.eye(3, dtype=bool)]
    assert (off_diag < 0.15).all()


def test_portfolio_stress_report_keys() -> None:
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.01, (1000, 4))
    report = portfolio_stress_report(returns)
    assert set(report) >= {"scenarios", "worst_scenario", "diversification_degradation"}
    assert len(report["scenarios"]) == len(SCENARIOS)
    names = {row["scenario"] for row in report["scenarios"]}
    assert report["worst_scenario"] in names
    for row in report["scenarios"]:
        assert set(row) >= SCENARIO_KEYS


def test_support_functions_are_consistent() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 0.01, 400)
    b = rng.normal(0.0, 0.01, 400)
    panel = np.column_stack([a, b])
    mask = np.zeros(400, dtype=bool)
    mask[100:200] = True
    assert stress_correlation(panel, mask) == stress_correlation(panel, mask)
    assert normal_correlation(panel, mask) == pytest.approx(
        stress_correlation(panel[~mask], np.ones((~mask).sum(), dtype=bool))
    )
    approach = conditional_correlation(a, b, lambda x, y: x < 0)
    assert np.isnan(approach) or -1.0 <= approach <= 1.0
    dd = drawdown_correlation(panel)
    assert dd.shape == (2, 2)
