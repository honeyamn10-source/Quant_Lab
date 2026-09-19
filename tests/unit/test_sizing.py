"""Unit tests for the sizing module."""

from __future__ import annotations

import numpy as np
from quantlab.sizing.kelly import kelly_fraction, optimal_f_from_history
from quantlab.sizing.robust import corrupted_edge_report, evaluate_policy, simulate_growth


def test_kelly_fraction():
    assert kelly_fraction(0.5, 2.0) == 0.25


def test_optimal_f_from_history():
    rng = np.random.default_rng(3)
    returns = np.where(rng.random(2000) < 0.6, 1.0, -1.0)
    f = optimal_f_from_history(returns)
    assert abs(f - 0.2) < 0.05


def test_simulate_growth_deterministic():
    a = simulate_growth(0.55, 1.0, 0.25, n_steps=100, n_paths=50, seed=7)
    b = simulate_growth(0.55, 1.0, 0.25, n_steps=100, n_paths=50, seed=7)
    np.testing.assert_array_equal(a["terminal_log_growth"], b["terminal_log_growth"])
    np.testing.assert_array_equal(a["min_wealth"], b["min_wealth"])


def test_half_kelly_reduces_ruin():
    p, b = 0.55, 1.0
    kelly_true = kelly_fraction(p, b)
    full = evaluate_policy(p, b, 0.25, kelly_true, n_steps=300, n_paths=800, seed=1)
    half = evaluate_policy(p, b, 0.125, kelly_true, n_steps=300, n_paths=800, seed=1)
    assert full["prob_of_ruin"] > half["prob_of_ruin"]


def test_corrupted_edge_report():
    report = corrupted_edge_report(
        0.6, 2.0, 0.125, [-0.2, 0.0, 0.2], n_steps=200, n_paths=200, seed=5
    )
    assert isinstance(report["table"], list)
    assert len(report["table"]) == 4
    assert report["verdict"] in {"ROBUST", "FRAGILE"}
    for row in report["table"]:
        assert "win_prob_used" in row
        assert "prob_of_ruin" in row
