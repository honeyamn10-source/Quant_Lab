"""GARCH-family conditional volatility models.

Gaussian maximum-likelihood fits are implemented in pure NumPy with
``scipy.optimize`` (lazily imported inside the fit functions so the rest of the
package works without scipy). When the optional ``arch`` package is installed,
plain GARCH can alternatively be solved with it while keeping the same return
contract. All fits are deterministic for a given ``seed``; every stochastic
function uses ``numpy.random.default_rng(seed)``.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np

_SQRT_TWO_OVER_PI = math.sqrt(2.0 / math.pi)
_FLOOR = 1e-12


def has_arch() -> bool:
    """True when the optional ``arch`` package can be imported."""
    return importlib.util.find_spec("arch") is not None


def _lazy_scipy_optimize():
    try:
        from scipy.optimize import minimize  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("scipy required for GARCH MLE; pip install scipy") from exc
    return minimize


def _presample_var(e2: np.ndarray) -> float:
    if len(e2) == 0:
        return _FLOOR
    return max(float(np.mean(e2)), _FLOOR)


def _gaussian_loglik(e2: np.ndarray, sigma2: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        total = float(np.sum(np.log(2.0 * math.pi * sigma2) + e2 / sigma2))
    if not math.isfinite(total):
        return -math.inf
    return -0.5 * total


def _garch_variance(params: np.ndarray, e2: np.ndarray, p: int, q: int) -> np.ndarray:
    omega = float(params[0])
    alpha = np.asarray(params[1 : 1 + p], dtype=float)
    beta = np.asarray(params[1 + p : 1 + p + q], dtype=float)
    n = len(e2)
    if n == 0:
        return np.asarray([], dtype=float)
    s0 = _presample_var(e2)
    m = max(p, q)
    e2pad = np.full(m + n, s0)
    e2pad[m:] = e2
    spad = np.full(m + n, s0)
    sigma2 = np.empty(n)
    for t in range(n):
        idx = t + m
        var = omega + float(alpha @ e2pad[idx - p : idx][::-1])
        if q:
            var += float(beta @ spad[idx - q : idx][::-1])
        var = max(var, _FLOOR)
        sigma2[t] = var
        spad[idx] = var
    return sigma2


def _garch_neg_loglik(x: np.ndarray, e2: np.ndarray, p: int, q: int) -> float:
    omega = max(float(x[0]), _FLOOR)
    alpha = np.clip(x[1 : 1 + p], _FLOOR, 0.999)
    beta = np.clip(x[1 + p :], _FLOOR, 0.999)
    persistence = float(alpha.sum() + beta.sum())
    penalty = 1e6 * max(0.0, persistence - 0.999) ** 2
    sigma2 = _garch_variance(np.concatenate([[omega], alpha, beta]), e2, p, q)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        total = float(np.sum(np.log(2.0 * math.pi * sigma2) + e2 / sigma2))
    if not math.isfinite(total):
        return 1e12
    return 0.5 * total + penalty


def _jittered(base: np.ndarray, seed: int, scale: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return base * (1.0 + scale * rng.standard_normal(len(base)))


def _garch_start(e2: np.ndarray, p: int, q: int, seed: int) -> np.ndarray:
    base_var = _presample_var(e2)
    base = np.concatenate(
        [[base_var * 0.1], np.full(p, 0.09 / p), np.full(q, 0.81 / q)]
    )
    x0 = _jittered(base, seed)
    x0 = np.maximum(x0, 1e-9)
    persistence = float(np.sum(x0[1:]))
    if persistence > 0.97:
        x0[1:] *= 0.97 / persistence
    return x0


def fit_garch(returns, p: int = 1, q: int = 1, seed: int = 0, max_iter: int = 500) -> dict:
    """Gaussian GARCH(p, q) maximum-likelihood fit.

    The variance recursion is
    ``sigma2_t = omega + sum(a_i * e^2_{t-i}) + sum(b_j * sigma2_{t-j})`` and
    the Gaussian log-likelihood is maximized with L-BFGS-B under ``omega > 0``,
    ``a_i, b_j in (0, 1)`` and ``sum(a) + sum(b) <= 0.999`` (enforced with a
    quadratic penalty). Presample squared shocks and variances are seeded with
    the sample variance of the returns.
    """
    if p < 1 or q < 1:
        raise ValueError("p and q must be at least 1")
    r = np.asarray(returns, dtype=np.float64).ravel()
    e2 = r * r
    if len(e2) < 2:
        raise ValueError("at least 2 observations required for GARCH fitting")
    minimize = _lazy_scipy_optimize()
    x0 = _garch_start(e2, p, q, seed)
    bounds = [(1e-12, None)] + [(1e-12, 0.999)] * (p + q)
    res = minimize(
        _garch_neg_loglik,
        x0,
        args=(e2, p, q),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter},
    )
    omega = max(float(res.x[0]), 1e-12)
    alpha = np.clip(res.x[1 : 1 + p], 1e-12, 0.999).tolist()
    beta = np.clip(res.x[1 + p :], 1e-12, 0.999).tolist()
    sigma2 = _garch_variance(
        np.concatenate([[omega], np.asarray(alpha), np.asarray(beta)]), e2, p, q
    )
    return {
        "model": "GARCH",
        "p": p,
        "q": q,
        "params": {"omega": omega, "alpha": alpha, "beta": beta},
        "loglik": _gaussian_loglik(e2, sigma2),
        "sigma2": sigma2,
        "converged": bool(res.success),
    }


def _egarch_variance(params: np.ndarray, e: np.ndarray, n: int) -> np.ndarray:
    omega, alpha, gamma, beta = (float(v) for v in params)
    if n == 0:
        return np.asarray([], dtype=float)
    s0 = _presample_var(e * e)
    sigma2 = np.empty(n)
    log_var = math.log(s0)
    for t in range(n):
        if t > 0:
            prev_s = math.sqrt(sigma2[t - 1])
            z = float(e[t - 1]) / prev_s if prev_s > 0 else 0.0
            log_var = (
                omega
                + alpha * (abs(z) - _SQRT_TWO_OVER_PI)
                + gamma * z
                + beta * log_var
            )
        var = math.exp(min(log_var, 20.0))
        if var < _FLOOR:
            var = _FLOOR
            log_var = math.log(_FLOOR)
        sigma2[t] = var
    return sigma2


def _egarch_neg_loglik(x: np.ndarray, e: np.ndarray, n: int) -> float:
    omega = float(x[0])
    alpha = float(np.clip(x[1], 1e-12, 1.0))
    gamma = float(x[2])
    beta = float(np.clip(x[3], 1e-12, 0.999))
    sigma2 = _egarch_variance(np.asarray([omega, alpha, gamma, beta]), e, n)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        total = float(np.sum(np.log(2.0 * math.pi * sigma2) + e * e / sigma2))
    if not math.isfinite(total):
        return 1e12
    return 0.5 * total


def _egarch_start(seed: int) -> np.ndarray:
    return _jittered(np.asarray([0.0, 0.1, 0.0, 0.9]), seed)


def fit_egarch(returns, seed: int = 0, max_iter: int = 500) -> dict:
    """Gaussian EGARCH(1, 1) maximum-likelihood fit.

    The recursion is
    ``log sigma2_t = omega + alpha*(|e_{t-1}|/sigma_{t-1} - sqrt(2/pi))
    + gamma*e_{t-1}/sigma_{t-1} + beta*log sigma2_{t-1}``, fitted with L-BFGS-B
    under ``alpha in (0, 1)`` and ``beta in (0, 1)``.
    """
    e = np.asarray(returns, dtype=np.float64).ravel()
    n = len(e)
    if n < 2:
        raise ValueError("at least 2 observations required for EGARCH fitting")
    minimize = _lazy_scipy_optimize()
    x0 = _egarch_start(seed)
    bounds = [(None, None), (1e-12, 1.0), (None, None), (1e-12, 0.999)]
    res = minimize(
        _egarch_neg_loglik,
        x0,
        args=(e, n),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter},
    )
    omega = float(res.x[0])
    alpha = float(np.clip(res.x[1], 1e-12, 1.0))
    gamma = float(res.x[2])
    beta = float(np.clip(res.x[3], 1e-12, 0.999))
    sigma2 = _egarch_variance(np.asarray([omega, alpha, gamma, beta]), e, n)
    return {
        "model": "EGARCH",
        "params": {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta},
        "loglik": _gaussian_loglik(e * e, sigma2),
        "sigma2": sigma2,
        "converged": bool(res.success),
    }


def _gjr_variance(params: np.ndarray, e: np.ndarray, n: int) -> np.ndarray:
    omega, alpha, gamma, beta = (float(v) for v in params)
    s0 = _presample_var(e * e)
    sigma2 = np.empty(n)
    prev = s0
    for t in range(n):
        if t == 0:
            var = omega + alpha * s0 + beta * s0
        else:
            ep = float(e[t - 1])
            leverage = ep * ep if ep < 0 else 0.0
            var = omega + alpha * ep * ep + gamma * leverage + beta * prev
        var = max(var, _FLOOR)
        sigma2[t] = var
        prev = var
    return sigma2


def _gjr_neg_loglik(x: np.ndarray, e: np.ndarray, n: int) -> float:
    omega = max(float(x[0]), _FLOOR)
    alpha = float(np.clip(x[1], 1e-12, 0.999))
    gamma = float(np.clip(x[2], 1e-12, 0.999))
    beta = float(np.clip(x[3], 1e-12, 0.999))
    persistence = alpha + gamma + beta
    penalty = 1e6 * max(0.0, persistence - 0.999) ** 2
    sigma2 = _gjr_variance(np.asarray([omega, alpha, gamma, beta]), e, n)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        total = float(np.sum(np.log(2.0 * math.pi * sigma2) + e * e / sigma2))
    if not math.isfinite(total):
        return 1e12
    return 0.5 * total + penalty


def _gjr_start(e2: np.ndarray, seed: int) -> np.ndarray:
    base = np.asarray([0.1 * _presample_var(e2), 0.05, 0.05, 0.8])
    x0 = _jittered(base, seed)
    return np.maximum(x0, 1e-9)


def fit_gjr(returns, seed: int = 0, max_iter: int = 500) -> dict:
    """Gaussian GJR-GARCH(1, 1) maximum-likelihood fit.

    The recursion is
    ``sigma2_t = omega + alpha*e^2_{t-1} + gamma*I(e_{t-1} < 0)*e^2_{t-1}
    + beta*sigma2_{t-1}``, fitted with L-BFGS-B under
    ``alpha, gamma, beta in (0, 1)`` and ``alpha + gamma + beta <= 0.999``.
    """
    e = np.asarray(returns, dtype=np.float64).ravel()
    n = len(e)
    if n < 2:
        raise ValueError("at least 2 observations required for GJR fitting")
    minimize = _lazy_scipy_optimize()
    x0 = _gjr_start(e * e, seed)
    bounds = [(1e-12, None), (1e-12, 0.999), (1e-12, 0.999), (1e-12, 0.999)]
    res = minimize(
        _gjr_neg_loglik,
        x0,
        args=(e, n),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter},
    )
    omega = max(float(res.x[0]), 1e-12)
    alpha = float(np.clip(res.x[1], 1e-12, 0.999))
    gamma = float(np.clip(res.x[2], 1e-12, 0.999))
    beta = float(np.clip(res.x[3], 1e-12, 0.999))
    sigma2 = _gjr_variance(np.asarray([omega, alpha, gamma, beta]), e, n)
    return {
        "model": "GJR",
        "params": {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta},
        "loglik": _gaussian_loglik(e * e, sigma2),
        "sigma2": sigma2,
        "converged": bool(res.success),
    }


def _fit_garch_arch(returns, p: int, q: int, seed: int, max_iter: int) -> dict:
    try:
        from arch import arch_model  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - guarded by has_arch()
        raise RuntimeError("arch required when use_arch=True; pip install arch") from exc
    am = arch_model(returns, p=p, q=q, mean="Zero", vol="GARCH", dist="normal", rescale=False)
    res = am.fit(disp="off", show_warning=False, options={"maxiter": max_iter})
    params = res.params
    omega = float(params["omega"])
    alpha = [float(params[f"alpha[{i}]"]) for i in range(1, p + 1)]
    beta = [float(params[f"beta[{i}]"]) for i in range(1, q + 1)]
    sigma2 = np.asarray(res.conditional_volatility, dtype=np.float64) ** 2
    converged = int(res.convergence_flag) == 0
    return {
        "model": "GARCH",
        "p": p,
        "q": q,
        "params": {"omega": omega, "alpha": alpha, "beta": beta},
        "loglik": _gaussian_loglik(np.asarray(returns, dtype=np.float64) ** 2, sigma2),
        "sigma2": sigma2,
        "converged": bool(converged),
    }


def garch_family(model: str, returns, seed: int = 0, max_iter: int = 500, use_arch: bool = False, **kwargs) -> dict:
    """Dispatch to a GARCH-family fit by name: ``garch``, ``egarch`` or ``gjr``.

    Returns the fit dict enriched with the deterministic (seed-dependent)
    starting parameters used by the optimizer under the ``start_params`` key.
    For plain GARCH, ``use_arch=True`` prefers the ``arch`` package when it is
    installed and falls back to the scipy MLE otherwise.
    """
    key = str(model).lower().replace("_", "-")
    r = np.asarray(returns, dtype=np.float64).ravel()
    e2 = r * r
    if key == "garch":
        p = int(kwargs.get("p", 1))
        q = int(kwargs.get("q", 1))
        start = _garch_start(e2, p, q, seed)
        if has_arch() and use_arch:
            try:
                fit = _fit_garch_arch(r, p, q, seed, max_iter)
            except Exception:
                fit = fit_garch(r, p=p, q=q, seed=seed, max_iter=max_iter)
        else:
            fit = fit_garch(r, p=p, q=q, seed=seed, max_iter=max_iter)
        fit["start_params"] = {
            "omega": float(start[0]),
            "alpha": start[1 : 1 + p].tolist(),
            "beta": start[1 + p :].tolist(),
        }
    elif key == "egarch":
        start = _egarch_start(seed)
        fit = fit_egarch(r, seed=seed, max_iter=max_iter)
        fit["start_params"] = {
            "omega": float(start[0]),
            "alpha": float(start[1]),
            "gamma": float(start[2]),
            "beta": float(start[3]),
        }
    elif key in {"gjr", "gjr-garch"}:
        start = _gjr_start(e2, seed)
        fit = fit_gjr(r, seed=seed, max_iter=max_iter)
        fit["start_params"] = {
            "omega": float(start[0]),
            "alpha": float(start[1]),
            "gamma": float(start[2]),
            "beta": float(start[3]),
        }
    else:
        raise ValueError(f"unknown GARCH-family model: {model}")
    return fit
