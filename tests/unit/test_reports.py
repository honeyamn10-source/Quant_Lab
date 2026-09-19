"""Tests for quantlab.reports renderers."""

from __future__ import annotations

import numpy as np
from quantlab.reports.html import render_graveyard_html, render_html_report
from quantlab.reports.json import render_json_report, render_validation_report


class FakeConfig:
    def __init__(self) -> None:
        self.initial_cash = 1000.0


class FakeResult:
    def __init__(
        self,
        returns: list[float],
        equity_curve: list[tuple] | None = None,
    ) -> None:
        self.config = FakeConfig()
        self.returns = list(returns)
        if equity_curve is None:
            arr = np.asarray(returns, dtype=float)
            self.equity_curve = [(i, float(v)) for i, v in enumerate(np.cumprod(1.0 + arr))]
        else:
            self.equity_curve = list(equity_curve)
        self.signals: list = []
        self.fills: list = []
        self.orders: list = []
        self.rejects: list = []

    def equity(self) -> list[float]:
        return [eq for _, eq in self.equity_curve]


def test_render_json_report_shape() -> None:
    returns = [0.01, -0.005, 0.02, 0.0, 0.015, -0.01]
    report = render_json_report(FakeResult(returns), experiment_id="EXP-0001")
    assert report["report"] == "QUANT-LAB-VALIDATION-REPORT"
    assert report["experiment_id"] == "EXP-0001"
    assert "metrics" in report
    assert "final_equity" in report
    assert "equity_curve" in report
    assert "status" in report
    assert "concerns" in report
    assert report["status"] in {"INSUFFICIENT_EVIDENCE", "REJECTED", "SURVIVED_BATTERY"}
    assert report["returns"] == returns
    assert len(report["equity_curve"]) == len(returns)
    assert report["final_equity"] == report["equity_curve"][-1][1]
    assert report["signals_n"] == 0
    assert report["fills_n"] == 0
    assert report["orders_n"] == 0
    assert report["rejects_n"] == 0


def test_render_json_report_status_logic() -> None:
    returns = [0.001] * 12
    rejected = render_json_report(FakeResult(returns), validation={"dsr": 0.1, "pbo": 0.9})
    assert rejected["status"] == "REJECTED"
    assert rejected["concerns"]
    survived = render_json_report(FakeResult(returns), validation={"dsr": 0.9, "pbo": 0.05})
    assert survived["status"] == "SURVIVED_BATTERY"
    assert survived["concerns"] == []
    no_evidence = render_json_report(FakeResult(returns))
    assert no_evidence["status"] == "INSUFFICIENT_EVIDENCE"


def test_render_validation_report_grouped() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0005, 0.01, 60)
    validation = render_validation_report(returns, n_trials=50)
    assert {"metrics", "psr", "dsr", "pbo", "bootstrap"} <= set(validation)
    assert "psr" in validation["psr"]
    assert "dsr" in validation["dsr"]
    assert "pbo" in validation["pbo"]
    assert "ci95" in validation["bootstrap"]


def test_render_validation_report_passthrough() -> None:
    original = {"dsr": 0.9, "pbo": 0.05}
    assert render_validation_report(np.zeros(6), n_trials=1, validation=original) == original


def test_render_html_report_markup() -> None:
    returns = [0.001] * 30
    html = render_html_report(
        FakeResult(returns),
        validation={"dsr": 0.9, "pbo": 0.05},
        experiment_id="EXP-TEST-1",
    )
    assert "<svg" in html
    assert "</svg>" in html
    assert "EXP-TEST-1" in html
    assert "QUANT LAB" in html
    assert "<script" not in html
    assert "\n<script>" not in html
    assert "<!doctype html>" in html


def test_render_html_report_deterministic() -> None:
    returns = [0.001, -0.001, 0.002] * 10
    first = render_html_report(FakeResult(returns), experiment_id="EXP-DET")
    second = render_html_report(FakeResult(returns), experiment_id="EXP-DET")
    assert first == second


def test_render_html_report_escapes_text() -> None:
    result = FakeResult([0.001] * 10)
    html = render_html_report(
        result,
        validation={"regime_dependency": "<script>alert('x')</script>"},
        experiment_id="EXP-<b>HTML</b>",
    )
    assert "<b>HTML</b>" not in html
    assert "&lt;b&gt;HTML&lt;/b&gt;" in html
    assert "<script>alert('x')</script>" not in html
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" not in html
    assert "regime dependency" in html


def test_render_graveyard_html() -> None:
    entries = [
        {
            "id": "EXP-0001",
            "status": "REJECTED",
            "primary_failure": "deflated sharpe",
            "raw_sharpe": 0.4,
            "net_sharpe": 0.1,
        },
        {
            "id": "EXP-0002",
            "status": "INSUFFICIENT_EVIDENCE",
            "primary_failure": "elevated overfit probability",
            "raw_sharpe": 0.7,
            "net_sharpe": None,
        },
    ]
    html = render_graveyard_html(entries)
    assert "EXP-0001" in html
    assert "EXP-0002" in html
    assert "REJECTED" in html
    assert "raw_sharpe" in html
