"""Unit tests for the volatility module."""

from __future__ import annotations

import numpy as np
import pytest
from quantlab.volatility import (
    benchmark_models,
    ewma_vol,
    forecast,
    garch_family,
    historical_vol,
    vol_metrics,
)

TRUE_ANNUALIZED_VOL = 0.12


def _constant_vol_returns(n: int = 1500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sd = TRUE_ANNUALIZED_VOL / np.sqrt(252.0)
    return rng.normal(0.0, sd, n)


def _clustered_returns(n: int = 800, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = np.full(n, 0.01)
    sigma[250:550] = 0.05
    sigma[600:700] = 0.035
    return rng.normal(0.0, sigma)


def test_historical_vol_tracks_true_annualized_vol():
    r = _constant_vol_returns()
    hv = historical_vol(r, window=50, ppy=252)
    assert np.isnan(hv[:49]).all()
    assert np.isfinite(hv[200:]).all()
    assert 0.08 < float(np.mean(hv[200:])) < 0.16


def test_ewma_warmup_and_positivity():
    r = _constant_vol_returns()
    ew = ewma_vol(r, lam=0.94, ppy=252)
    assert np.isfinite(ew[0])
    assert np.isfinite(ew[-1])
    assert (ew[200:] > 0.0).all()
    assert not np.isnan(ew).any()


def test_garch_family_garch_on_clustered_returns():
    pytest.importorskip("scipy", reason="GARCH MLE requires scipy (install 'quantlab[ml]')")
    r = _clustered_returns(n=1150)
    fit = garch_family("garch", r)
    assert fit["model"] == "GARCH"
    assert fit["params"]["omega"] > 0.0
    persistence = fit["params"]["alpha"][0] + fit["params"]["beta"][0]
    assert 0.0 < persistence <= 0.999 + 1e-2
    assert np.isfinite(fit["loglik"])
    assert np.isfinite(fit["sigma2"]).all()
    assert (fit["sigma2"] > 0.0).all()
    assert fit["converged"] is True
    assert "start_params" in fit


def test_vol_metrics_all_keys_finite():
    r = _constant_vol_returns()
    fc = forecast("historical", r, window=20, ppy=252)
    metrics = vol_metrics(fc, r * r)
    assert set(metrics) == {"mae", "rmse", "qlike", "calibration_slope", "direction_accuracy"}
    assert all(np.isfinite(v) for v in metrics.values())


def test_benchmark_models_shape_and_winner():
    pytest.importorskip("scipy", reason="GARCH MLE requires scipy (install 'quantlab[ml]')")
    r = _clustered_returns(n=750)
    out = benchmark_models(r)
    models = {"historical", "ewma", "realized", "har", "garch", "egarch", "gjr"}
    assert set(out) == models | {"winner"}
    assert out["winner"] in models
