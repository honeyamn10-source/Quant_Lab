"""Probabilistic Sharpe Ratio (Bailey & Lopez de Prado).

PSR = P( true SR > SR* | observed returns ) using the skew-adjusted
standard error of the Sharpe estimator.
"""

from __future__ import annotations

import math

import numpy as np

from quantlab.validation.metrics import sharpe_ratio


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Small, accurate-enough inverse normal CDF (Acklam algorithm)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]

    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p > 0.97575:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    # One Newton refinement step.
    e = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


def psi_estimate(sr_hat: float, n: int, skew: float, kurtosis: float) -> float:
    """Standard error of the Sharpe estimator using non-normal returns."""
    var = (1.0 - skew * sr_hat + ((kurtosis - 1.0) / 4.0) * sr_hat**2) / max(n - 1, 1)
    return math.sqrt(max(var, 0.0))


def probabilistic_sharpe_ratio(returns, benchmark_sr: float = 0.0, periods_per_year: int = 252) -> dict:
    """PSR = Phi( (SR_hat - SR*) * sqrt(n-1) / psi ), with skew/kurtosis-standard error."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 3:
        return {"psr": 0.0, "n": n, "sr_hat": 0.0, "skew": 0.0, "kurtosis": 3.0}
    sr_hat = sharpe_ratio(r, periods_per_year=periods_per_year) / math.sqrt(periods_per_year)
    mu = float(np.mean(r))
    std = float(np.std(r, ddof=1))
    if std <= 0:
        return {"psr": 1.0 if benchmark_sr <= 0 else 0.0, "n": n, "sr_hat": 0.0, "skew": 0.0, "kurtosis": 3.0}
    skew = float(np.mean((r - mu) ** 3) / std**3)
    kurtosis = float(np.mean((r - mu) ** 4) / std**4)
    sr_bench = benchmark_sr / math.sqrt(periods_per_year)
    psi = psi_estimate(sr_hat, n, skew, kurtosis)
    psr = _normal_cdf((sr_hat - sr_bench) * math.sqrt(n - 1) / psi) if psi > 0 else (1.0 if sr_hat >= sr_bench else 0.0)
    return {"psr": float(psr), "n": n, "sr_hat": float(sr_hat), "skew": float(skew), "kurtosis": float(kurtosis)}
