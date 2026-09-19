"""Deflated Sharpe Ratio (Bailey & Lopez de Prado).

DSR corrects the observed Sharpe for selection bias and backtest overfitting by
using the *expected maximum* Sharpe across the number of independent trials (N)
as the benchmark within PSR.
"""

from __future__ import annotations

import math

from quantlab.validation.psr import probabilistic_sharpe_ratio, psi_estimate

_EULER_GAMMA = 0.57721566490153286060651209008240243


def expected_max_sharpe(n_trials: int, degenerate: float = 1.0, variance_sr: float = 1.0) -> float:
    """The expected maximum of N IID Gaussians (annualized benchmark Sharpe).

    SR_0 = sqrt(Var(SR_hat)) * ((1 - gamma) * Z(1 - 1/N) + gamma * Z(1 - 1/(N e)))
    """
    if n_trials <= 1:
        return 0.0
    from quantlab.validation.psr import _normal_ppf

    z1 = _normal_ppf(1.0 - 1.0 / n_trials)
    z2 = _normal_ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(max(variance_sr, 0.0)) * degenerate * (
        (1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2
    )


def deflated_sharpe_ratio(returns, n_trials: int, benchmark_sr: float = 0.0, periods_per_year: int = 252) -> dict:
    """Compute PSR against the expected-maximum benchmark defined by N trials."""
    import numpy as np

    from quantlab.validation.metrics import sharpe_ratio

    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 3 or n_trials < 1:
        return {"dsr": 0.0, "sr_0": 0.0, "n_trials": int(n_trials), "n": n}
    sr_hat_rescaled = sharpe_ratio(r, periods_per_year=periods_per_year) / math.sqrt(periods_per_year)
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    skew = float(np.mean((r - mu) ** 3) / sd**3) if sd > 0 else 0.0
    kurt = float(np.mean((r - mu) ** 4) / sd**4) if sd > 0 else 3.0
    var_sr = psi_estimate(sr_hat_rescaled, n, skew, kurt) ** 2

    sr_0 = expected_max_sharpe(n_trials=n_trials, variance_sr=var_sr)
    sr_0_ann = sr_0 * math.sqrt(periods_per_year)  # expected-max per-period -> annualized
    psr_out = probabilistic_sharpe_ratio(r, benchmark_sr=sr_0_ann, periods_per_year=periods_per_year)
    psr_out["sr_0"] = float(sr_0_ann)
    psr_out["n_trials"] = int(n_trials)
    return {"dsr": psr_out["psr"], "sr_0": float(sr_0_ann), "n_trials": int(n_trials), "n": n}
