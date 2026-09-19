"""JSON-serializable validation reports for backtest results.

The report encodes a falsification verdict (``status``) plus a ``concerns``
list derived from the validation battery.  Thresholds (documented here):

    ``dsr < 0.25``        hard failure              -> "REJECTED"
    ``pbo > 0.50``        hard failure              -> "REJECTED"
    ``dsr < 0.50``        weak deflated Sharpe      -> INSUFFICIENT_EVIDENCE
    ``pbo > 0.20``        elevated overfit          -> INSUFFICIENT_EVIDENCE
    ``psr < 0.50``        marginal probabilistic SR -> INSUFFICIENT_EVIDENCE
    ``regime_dependency`` truthy                    -> INSUFFICIENT_EVIDENCE
    bootstrap 95% CI spanning zero                  -> INSUFFICIENT_EVIDENCE

A missing validation dict means no affirmative evidence exists, so the
verdict defaults to ``INSUFFICIENT_EVIDENCE``.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np

from quantlab.validation.bootstrap import bootstrap_sharpe
from quantlab.validation.cscv import probability_of_backtest_overfitting
from quantlab.validation.dsr import deflated_sharpe_ratio
from quantlab.validation.metrics import all_metrics
from quantlab.validation.psr import probabilistic_sharpe_ratio

_DSR_HARD = 0.25
_DSR_WEAK = 0.50
_PBO_HARD = 0.50
_PBO_MODERATE = 0.20
_PSR_MARGINAL = 0.50
_PBO_SEED = 42

REPORT_NAME = "QUANT-LAB-VALIDATION-REPORT"
ALLOWED_STATUSES = ("SURVIVED_BATTERY", "INSUFFICIENT_EVIDENCE", "REJECTED")


class ReportableResult(Protocol):
    """Minimal structural interface consumed by the report renderers."""

    returns: list[float]
    equity_curve: list[tuple]
    signals: list
    fills: list
    orders: list
    rejects: list
    config: object

    def equity(self) -> list[float]: ...


def _read(validation: dict, key: str) -> float | None:
    """Read a scalar statistic out of a flat or grouped validation dict."""
    value = validation.get(key)
    if isinstance(value, dict):
        value = value.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _concerns(validation: dict | None) -> list[str]:
    """Map validation statistics to human-readable concern strings."""
    if not validation:
        return []
    concerns: list[str] = []
    psr = _read(validation, "psr")
    if psr is not None and psr < _PSR_MARGINAL:
        concerns.append("marginal probabilistic Sharpe ratio (< 0.5)")
    dsr = _read(validation, "dsr")
    if dsr is not None and dsr < _DSR_WEAK:
        concerns.append("weak deflated Sharpe (< 0.5)")
    pbo = _read(validation, "pbo")
    if pbo is not None and pbo > _PBO_MODERATE:
        concerns.append("elevated overfit probability (> 0.2)")
    if validation.get("regime_dependency"):
        concerns.append("regime dependency")
    bootstrap = validation.get("bootstrap")
    ci = bootstrap.get("ci95") if isinstance(bootstrap, dict) else validation.get("ci95")
    if isinstance(ci, dict):
        lo, hi = ci.get("lo"), ci.get("hi")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo <= 0 <= hi:
            concerns.append("bootstrap confidence interval spans zero")
    return concerns


def classify_validation(validation: dict | None) -> tuple[str, list[str]]:
    """Return the battery verdict and its concern list for a validation dict."""
    concerns = _concerns(validation)
    if validation is None:
        return "INSUFFICIENT_EVIDENCE", []
    dsr = _read(validation, "dsr")
    pbo = _read(validation, "pbo")
    if pbo is not None and pbo > _PBO_HARD:
        return "REJECTED", concerns
    if dsr is not None and dsr < _DSR_HARD:
        return "REJECTED", concerns
    if concerns:
        return "INSUFFICIENT_EVIDENCE", concerns
    return "SURVIVED_BATTERY", []


def _json_safe(value: Any) -> Any:
    """Collapse non-finite floats so the report stays strictly JSON-safe."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _final_equity(result: ReportableResult) -> float:
    equity = getattr(result, "equity", None)
    eqs = [float(value) for value in equity()] if callable(equity) else []
    if eqs:
        return eqs[-1]
    config = getattr(result, "config", None)
    return float(getattr(config, "initial_cash", 0.0))


def render_json_report(
    result: ReportableResult,
    validation: dict | None = None,
    experiment_id: str = "",
) -> dict:
    """Build a JSON-serializable validation report for a backtest result."""
    returns = [float(value) for value in result.returns]
    status, concerns = classify_validation(validation)
    return {
        "report": REPORT_NAME,
        "experiment_id": str(experiment_id),
        "metrics": _json_safe(all_metrics(returns)),
        "validation": _json_safe(dict(validation)) if validation else {},
        "signals_n": len(getattr(result, "signals", []) or []),
        "fills_n": len(getattr(result, "fills", []) or []),
        "orders_n": len(getattr(result, "orders", []) or []),
        "rejects_n": len(getattr(result, "rejects", []) or []),
        "final_equity": _final_equity(result),
        "returns": [_json_safe(value) for value in returns],
        "equity_curve": [
            [_json_safe(ts), float(eq)] for ts, eq in getattr(result, "equity_curve", [])
        ],
        "status": status,
        "concerns": concerns,
    }


def _cheap_pbo(returns: np.ndarray, n_trials: int, seed: int = _PBO_SEED) -> dict:
    """Cheap CSCV-style PBO: treat shuffled copies of the returns as trials."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n_returns = int(len(r))
    if n_returns < 6:
        return {
            "pbo": None,
            "n_returns": n_returns,
            "n_trials": int(n_trials),
            "method": "shuffle-fallback",
            "note": "insufficient returns for CSCV (need >= 6)",
        }
    rng = np.random.default_rng(seed)
    n_cols = int(max(2, min(int(n_trials), 16)))
    cols = [r]
    cols += [rng.permutation(r) for _ in range(n_cols - 1)]
    matrix = np.column_stack(cols)
    out = probability_of_backtest_overfitting(matrix, num_splits=6)
    out["method"] = "shuffle-fallback"
    out["n_trials"] = int(n_trials)
    return out


def render_validation_report(
    returns: np.ndarray,
    n_trials: int,
    validation: dict | None = None,
) -> dict:
    """Compute the full validation battery, grouped for embedding in reports.

    When ``validation`` is supplied it is returned unchanged; otherwise every
    statistic is recomputed.  The PBO needs a (T x N) matrix, so a cheap
    deterministic fallback builds N columns from the returns plus seeded
    shuffled copies.
    """
    if validation is not None:
        return dict(validation)
    arr = np.asarray(returns, dtype=float)
    n_trials = int(n_trials)
    return {
        "metrics": all_metrics(arr),
        "psr": probabilistic_sharpe_ratio(arr, benchmark_sr=0.0),
        "dsr": deflated_sharpe_ratio(arr, n_trials=n_trials),
        "pbo": _cheap_pbo(arr, n_trials),
        "bootstrap": bootstrap_sharpe(arr),
        "n_trials": n_trials,
    }
