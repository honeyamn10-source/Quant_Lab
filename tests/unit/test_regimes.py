"""Unit tests for the quantlab regimes module."""

from __future__ import annotations

import numpy as np
import pytest
from quantlab.regimes.features import normalize_features, regime_features
from quantlab.regimes.models import (
    change_point_regimes,
    ensemble_regimes,
    gaussian_mixture_regimes,
    hmm_regimes,
    rule_based_regimes,
)


@pytest.fixture(scope="module")
def synthetic_returns() -> np.ndarray:
    """Deterministic concatenated two-means return series with vol jump."""
    rng = np.random.default_rng(1)
    n = 250
    first = rng.normal(0.001, 0.005, size=n)
    second = rng.normal(-0.001, 0.02, size=n)
    return np.concatenate([first, second])


@pytest.fixture(scope="module")
def synthetic_features(synthetic_returns: np.ndarray) -> dict[str, np.ndarray]:
    """Feature dict derived from a synthetic cumulative close path."""
    closes = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], synthetic_returns])))
    return regime_features(closes)


def test_rule_based_regimes_shape(synthetic_features) -> None:
    labels, names = rule_based_regimes(synthetic_features)
    assert labels.dtype.kind == "i"
    assert np.isin(np.unique(labels), [0, 1, 2, 3]).all()
    assert len(names) == 4


def test_rule_based_regimes_valid_region(synthetic_features) -> None:
    labels, _ = rule_based_regimes(synthetic_features)
    warmup = np.flatnonzero(np.isnan(synthetic_features["trend"]))
    assert np.all(labels[warmup] == 0)
    assert np.all(labels[~np.isnan(synthetic_features["trend"])] >= 0)


def test_gaussian_mixture_separates_means(synthetic_returns) -> None:
    labels, responsibilities = gaussian_mixture_regimes(
        synthetic_returns, n_states=2, seed=0, n_iter=100
    )
    n = len(synthetic_returns) // 2
    first_mean = float(np.mean(labels[:n]))
    second_mean = float(np.mean(labels[n:]))
    assert first_mean != second_mean
    assert responsibilities.shape == (len(synthetic_returns), 2)
    assert np.allclose(responsibilities.sum(axis=1), 1.0)


def test_hmm_regimes_deterministic(synthetic_returns) -> None:
    labels_a, posterior_a = hmm_regimes(synthetic_returns, n_states=3, seed=7, n_iter=20)
    labels_b, posterior_b = hmm_regimes(synthetic_returns, n_states=3, seed=7, n_iter=20)
    assert np.array_equal(labels_a, labels_b)
    assert np.array_equal(posterior_a, posterior_b, equal_nan=True)
    assert posterior_a.shape == (len(synthetic_returns), 3)


def test_hmm_regimes_rejects_nan() -> None:
    r = np.array([0.001, 0.002, np.nan, -0.001, 0.0])
    labels, posterior = hmm_regimes(r, n_states=2, seed=0, n_iter=10)
    assert labels[2] == -1
    assert np.isnan(posterior[2]).all()
    assert np.all(posterior[np.isfinite(r)].sum(axis=1) > 0.99)


def test_change_point_finds_segments() -> None:
    rng = np.random.default_rng(11)
    first = rng.normal(0.02, 0.005, size=300)
    second = rng.normal(-0.02, 0.005, size=300)
    series = np.concatenate([first, second])
    labels = change_point_regimes(series, max_changes=3, penalty=1e-4)
    assert len(np.unique(labels)) >= 2
    assert len(labels) == len(series)


def test_normalize_features_moments(synthetic_features) -> None:
    normalized = normalize_features(synthetic_features)
    assert set(normalized.keys()) == set(synthetic_features.keys())
    for _key, series in normalized.items():
        finite = np.isfinite(series)
        raw = np.asarray(synthetic_features[_key], dtype=np.float64)
        if float(np.nanstd(raw)) < 1e-12:
            assert np.all(series == 0.0)
        else:
            assert np.abs(float(np.mean(series[finite]))) < 1e-6
            assert np.abs(float(np.std(series[finite])) - 1.0) < 1e-6


def test_normalize_features_constant_is_zero() -> None:
    const = np.full(10, 3.5)
    out = normalize_features({"const": const})
    assert np.array_equal(out["const"], np.zeros(10))


def test_regime_features_keys_and_lengths() -> None:
    rng = np.random.default_rng(3)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=60)))
    features = regime_features(closes)
    expected = {
        "trend",
        "realized_vol",
        "vol_of_vol",
        "liquidity_proxy",
        "dispersion",
        "correlation_proxy",
        "drawdown",
        "momentum_ratio",
        "tail_stress",
    }
    assert set(features.keys()) == expected
    for series in features.values():
        assert series.shape == closes.shape
        assert series.dtype == np.float64


def test_ensemble_regimes_contract(synthetic_returns, synthetic_features) -> None:
    aligned = (
        {key: series[1:] for key, series in synthetic_features.items()}
        if len(next(iter(synthetic_features.values()))) == len(synthetic_returns) + 1
        else synthetic_features
    )
    assert all(len(v) == len(synthetic_returns) for v in aligned.values())
    result = ensemble_regimes(synthetic_returns, aligned, seed=0)
    assert result["regime_names"] == [
        "low_vol_trend",
        "high_vol_trend",
        "high_vol_sideways",
        "crisis_stress",
    ]
    for key in (
        "labels_rule",
        "labels_gmm",
        "labels_hmm",
        "labels_segment",
        "ensemble_labels",
    ):
        assert result[key].shape == (len(synthetic_returns),)
    probabilities = result["probabilities"]
    assert probabilities.shape == (len(synthetic_returns), 4)
    warmup = np.flatnonzero(np.isnan(aligned["trend"]))
    finite_rows = np.setdiff1d(np.arange(len(synthetic_returns)), warmup)
    assert np.allclose(probabilities[finite_rows].sum(axis=1), 1.0)
    assert 0.0 <= result["consensus"] <= 1.0
    assert np.isin(np.unique(result["ensemble_labels"]), [0, 1, 2, 3]).all()
