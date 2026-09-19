"""Robustness scoring: degradation slopes, a surface rating and a report table.

Each sweep result list maps a perturbation size to a ``result`` object. Metrics are
measured by default with the annualized Sharpe of ``result.returns``; slopes are
ordinary least squares fits of the metric against relative perturbation size,
reported per one percentage point (0.01) of perturbation.
"""

from __future__ import annotations

import numpy as np

from quantlab.validation.metrics import sharpe_ratio

GRADES: dict[str, dict[str, float | None | str]] = {
    "ROBUST": {
        "max_slope_per_1pct": -0.15,
        "min_worst_metric": 0.5,
        "meaning": "performance holds under cost, execution and parameter perturbation",
    },
    "MODERATE": {
        "max_slope_per_1pct": -0.30,
        "min_worst_metric": 0.2,
        "meaning": "material but not catastrophic degradation under stress",
    },
    "FRAGILE": {
        "max_slope_per_1pct": None,
        "min_worst_metric": 0.2,
        "meaning": "collapses under mild perturbation; treat the edge as suspect",
    },
}


def compute_sharpe(result) -> float:
    """Annualized Sharpe of a result's return series."""
    returns = getattr(result, "returns", [])
    return float(sharpe_ratio(returns))


def _relative_size(entry: dict) -> float:
    """Recover the relative perturbation size from a sweep entry.

    Factor-based sweeps (cost/slippage/delay/liquidity) record ``factor``;
    parameter sweeps record ``params["perturbation"]["scale"]``.
    """
    scale = float((entry.get("params") or {}).get("perturbation", {}).get("scale", 1.0))
    factor = float(entry.get("factor", 1.0))
    if "params" in entry:
        return scale - 1.0
    return factor - 1.0


def degradation_slope(results: list[dict], dimension: str = "params") -> dict:
    """OLS slope of the metric vs relative perturbation size for one sweep.

    Returns ``dimension``, ``slope_per_1pct`` (metric change per 0.01 perturbation),
    ``r2`` of the fit, and the worst/best metric across the sweep.
    """
    xs = np.asarray([_relative_size(entry) for entry in results], dtype=float)
    ys = np.asarray([compute_sharpe(entry["result"]) for entry in results], dtype=float)
    if len(xs) < 2 or float(np.ptp(xs)) <= 1e-12:
        slope, r2 = 0.0, 0.0
    else:
        slope, intercept = np.polyfit(xs, ys, 1)
        resid = ys - (slope * xs + intercept)
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r2 = 1.0 if ss_tot <= 1e-12 else float(1.0 - np.sum(resid**2) / ss_tot)
        slope = float(slope)
    return {
        "dimension": dimension,
        "slope_per_1pct": float(slope / 100.0),
        "r2": float(r2),
        "worst_metric": float(ys.min()) if len(ys) else 0.0,
        "best_metric": float(ys.max()) if len(ys) else 0.0,
    }


def robustness_surface(sweeps: dict[str, list[dict]]) -> dict:
    """Grade a strategy across multiple perturbation dimensions.

    ``sweeps`` maps a dimension name to that dimension's sweep results. Returns the
    rating (ROBUST / MODERATE / FRAGILE), the worst dimension and metric, the per
    dimension slopes and a full report table.
    """
    table = [degradation_slope(results, dimension=dimension) for dimension, results in sweeps.items()]
    slopes = {row["dimension"]: row["slope_per_1pct"] for row in table}
    worst = min(table, key=lambda row: row["worst_metric"], default=None)
    if worst is None:
        rating = "ROBUST"
        worst_slot = None
    else:
        worst_slot = {"dimension": worst["dimension"], "metric": worst["worst_metric"]}
        worst_metric = worst["worst_metric"]
        steepest = min(slopes.values(), default=0.0)
        if steepest < -0.30 or worst_metric < 0.2:
            rating = "FRAGILE"
        elif steepest < -0.15 or worst_metric < 0.5:
            rating = "MODERATE"
        else:
            rating = "ROBUST"
    return {
        "rating": rating,
        "worst": worst_slot,
        "slopes": slopes,
        "table": table,
    }
