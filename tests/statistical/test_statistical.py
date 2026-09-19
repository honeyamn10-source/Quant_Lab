"""Known-answer statistical tests: PSR/DSR/CSCV/PBO/bootstrap/monte-carlo."""

from __future__ import annotations

import math

import numpy as np
import pytest
from quantlab.validation.bootstrap import (
    block_bootstrap,
    bootstrap_sharpe,
    confidence_interval,
    iid_bootstrap,
    stationary_bootstrap,
)
from quantlab.validation.cscv import cscv_logits, probability_of_backtest_overfitting
from quantlab.validation.dsr import deflated_sharpe_ratio, expected_max_sharpe
from quantlab.validation.metrics import sharpe_ratio
from quantlab.validation.monte_carlo import monte_carlo_null, monte_carlo_paths
from quantlab.validation.psr import probabilistic_sharpe_ratio, psi_estimate

pytestmark = pytest.mark.statistical


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def test_psr_matches_analytic_gaussian() -> None:
    n = 300
    rng = np.random.default_rng(0)
    r = rng.normal(0.0003, 0.01, size=n)  # tiny mean; exercises skew/kurtosis terms
    out = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    sr_pp = out["sr_hat"]  # per-period Sharpe
    # analytic PSR using the Lo (2002) skew-adjusted standard error
    psi = psi_estimate(sr_pp, out["n"], out["skew"], out["kurtosis"])
    expected = _normal_cdf(sr_pp * math.sqrt(n - 1) / psi)
    assert abs(out["psr"] - expected) < 1e-9
    assert out["n"] == n


def test_psr_saturates() -> None:
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.005, size=500)
    assert probabilistic_sharpe_ratio(r, benchmark_sr=0.0)["psr"] > 0.99
    assert probabilistic_sharpe_ratio(r, benchmark_sr=5.0)["psr"] < 0.01


def test_psr_benchmark_at_observed_sr_is_half() -> None:
    # PSR(observed Sharpe) must be exactly 0.5, regardless of sampling noise.
    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 0.01, size=400)
    out = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    benchmark = out["sr_hat"] * math.sqrt(252)  # annualize the observed period SR
    assert abs(probabilistic_sharpe_ratio(r, benchmark_sr=benchmark)["psr"] - 0.5) < 1e-9


def test_dsr_with_single_trial_equals_psr_zero() -> None:
    rng = np.random.default_rng(3)
    r = rng.normal(0.0002, 0.01, size=300)
    psr0 = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)["psr"]
    dsr = deflated_sharpe_ratio(r, n_trials=1)["dsr"]
    assert abs(dsr - psr0) < 1e-6


def test_expected_max_sharpe_grows_with_trials() -> None:
    lo, hi = expected_max_sharpe(10), expected_max_sharpe(1000)
    assert lo > 0 and hi > lo
    assert expected_max_sharpe(1) == 0.0


def test_dsr_deflates_below_psr() -> None:
    rng = np.random.default_rng(4)
    r = rng.normal(0.0002, 0.01, size=400)
    psr0 = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)["psr"]
    dsr = deflated_sharpe_ratio(r, n_trials=50)["dsr"]
    assert dsr < psr0


def test_cscv_dominant_strategy_low_pbo() -> None:
    rng = np.random.default_rng(5)
    rows, n_strategies = 1200, 5
    matrix = rng.normal(0.0005, 0.01, size=(rows, n_strategies))
    matrix[:, 0] = 0.002 + 0.004 * rng.normal(size=rows)  # dominant and stable
    out = probability_of_backtest_overfitting(matrix, num_splits=6)
    assert out["pbo"] <= 0.2
    assert out["n_combinations"] == 20  # C(6,3)


def test_cscv_noise_matrix_high_pbo() -> None:
    rng = np.random.default_rng(6)
    matrix = rng.normal(0.0, 0.01, size=(1200, 8))
    out = probability_of_backtest_overfitting(matrix, num_splits=4)
    assert out["n_combinations"] == 6
    assert 0.0 <= out["pbo"] <= 1.0
    assert len(cscv_logits(matrix, num_splits=4)["oos_ranks"]) == 6


def test_bootstrap_ci_contains_sample_stat() -> None:
    rng = np.random.default_rng(7)
    r = rng.normal(0.0002, 0.01, size=252)
    out = bootstrap_sharpe(r, n_iter=400, seed=10)
    lo, hi = out["ci95"]["lo"], out["ci95"]["hi"]
    assert lo <= sharpe_ratio(r) <= hi


def test_bootstrap_methods_produce_sample_distributions() -> None:
    rng = np.random.default_rng(8)
    r = rng.normal(0.0, 0.01, size=200)
    iid = iid_bootstrap(r, lambda x: float(np.mean(x)), n_iter=50, seed=1)
    blk = block_bootstrap(r, lambda x: float(np.mean(x)), n_iter=50, seed=1)
    st = stationary_bootstrap(r, lambda x: float(np.mean(x)), n_iter=50, seed=1)
    assert iid.shape == (50,) and blk.shape == (50,) and st.shape == (50,)
    ci = confidence_interval(iid, alpha=0.05)
    assert ci["lo"] <= ci["hi"]


def test_shuffle_null_centered_at_zero() -> None:
    rng = np.random.default_rng(9)
    r = rng.normal(0.0, 0.01, size=300)
    out = monte_carlo_null(r, n_iter=100, seed=1)
    assert abs(out["mean_null_sharpe"]) < 0.5
    assert out["std_null_sharpe"] > 0
    assert len(out["samples"]) == 100


def test_monte_carlo_paths_shape() -> None:
    paths = monte_carlo_paths(n_paths=50, n_periods=120, mu=0.0, sigma=0.05, seed=0)
    assert paths.shape == (50, 120)


def test_psi_estimate_gaussian_unit() -> None:
    psi = psi_estimate(0.0, 1000, skew=0.0, kurtosis=3.0)
    assert abs(psi - 1.0 / math.sqrt(999)) < 1e-6
