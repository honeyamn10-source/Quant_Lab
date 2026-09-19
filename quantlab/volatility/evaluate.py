"""Evaluation of volatility forecasts against realized (squared-return) targets."""

from __future__ import annotations

import math

import numpy as np

from quantlab.volatility.models import forecast

MODELS = ["historical", "ewma", "realized", "har", "garch", "egarch", "gjr"]
METRIC_KEYS = ["mae", "rmse", "qlike", "calibration_slope", "direction_accuracy"]
_NAN = float("nan")


def _aligned(forecasts: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(forecasts, dtype=np.float64).ravel()
    t = np.asarray(targets, dtype=np.float64).ravel()
    if len(f) != len(t):
        raise ValueError("forecasts and targets must have the same length")
    finite = np.isfinite(f) & np.isfinite(t)
    return f[finite], t[finite]


def vol_metrics(forecasts: np.ndarray, targets: np.ndarray) -> dict:
    """Score a volatility forecast against realized (squared-return) targets.

    Forecasts and targets are aligned on the region where both are finite, then
    scored with: mean absolute error, root mean squared error, the quasi
    likelihood QLIKE loss, the OLS slope of target on forecast, and the share
    of consecutive bars where the forecast and target move in the same
    direction. Scores are NaN when the aligned region is empty or degenerate.
    """
    f, t = _aligned(forecasts, targets)
    if len(f) == 0:
        return dict.fromkeys(METRIC_KEYS, _NAN)
    err = f - t
    scores = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(float(np.mean(err * err)))),
    }
    positive = (f > 0) & (t > 0)
    if positive.any():
        pf, pt = f[positive], t[positive]
        ratio = pt / (pf * pf)
        scores["qlike"] = float(np.mean(ratio - np.log(ratio) - 1.0))
    else:
        scores["qlike"] = _NAN
    if len(f) > 1:
        var_f = float(np.var(f, ddof=1))
        if var_f > 0:
            cov = float(np.mean((f - np.mean(f)) * (t - np.mean(t))))
            scores["calibration_slope"] = cov / var_f
        else:
            scores["calibration_slope"] = _NAN
        scores["direction_accuracy"] = float(
            np.mean(np.sign(np.diff(f)) == np.sign(np.diff(t)))
        )
    else:
        scores["calibration_slope"] = _NAN
        scores["direction_accuracy"] = _NAN
    return scores


def benchmark_models(returns: np.ndarray) -> dict:
    """Evaluate every volatility model's forecast against squared returns.

    Each model in ``MODELS`` produces one-step-ahead forecasts which are scored
    against the squared-return realized targets with :func:`vol_metrics`. The
    dictionary additionally holds a ``winner`` key naming the model with the
    lowest RMSE.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    targets = r * r
    results: dict = {model: vol_metrics(forecast(model, r), targets) for model in MODELS}
    ranked = [
        (model, results[model]["rmse"])
        for model in MODELS
        if math.isfinite(results[model]["rmse"])
    ]
    winner = min(ranked, key=lambda pair: pair[1])[0] if ranked else MODELS[0]
    results["winner"] = winner
    return results
