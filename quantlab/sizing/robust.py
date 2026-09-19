"""Robustness of betting policies to mis-specified edges."""

from __future__ import annotations

import numpy as np

from quantlab.sizing.kelly import kelly_fraction


def simulate_growth(
    win_prob: float,
    payoff_ratio: float,
    policy_frac: float,
    n_steps: int = 500,
    n_paths: int = 1000,
    seed: int = 0,
) -> dict:
    """Simulate fixed-fraction betting paths and return terminal and path stats.

    Each step the stake is a fixed fraction of current wealth: a win multiplies
    wealth by (1 + b*f), a loss by (1 - f). The dict carries per-path terminal
    log-growth and minimum wealth fraction, plus the full wealth/peak curves.
    """
    if not 0.0 < win_prob < 1.0:
        raise ValueError(f"win_prob must lie in (0, 1), got {win_prob}")
    if payoff_ratio <= 0.0:
        raise ValueError(f"payoff_ratio must be positive, got {payoff_ratio}")
    if not 0.0 <= policy_frac <= 1.0:
        raise ValueError(f"policy_frac must lie in [0, 1], got {policy_frac}")
    n_steps = int(n_steps)
    n_paths = int(n_paths)
    rng = np.random.default_rng(seed)
    wins = rng.random((n_paths, n_steps)) < win_prob
    gains = np.where(wins, payoff_ratio * policy_frac, -policy_frac)
    wealth = np.cumprod(1.0 + gains, axis=1)
    peak = np.maximum(1.0, np.maximum.accumulate(wealth, axis=1))
    terminal = wealth[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_growth = np.log(np.where(terminal > 0.0, terminal, np.nan))
    log_growth = np.where(np.isfinite(log_growth), log_growth, -np.inf)
    min_wealth = np.minimum(np.min(wealth, axis=1), 1.0)
    return {
        "terminal_log_growth": log_growth,
        "min_wealth": min_wealth,
        "wealth": wealth,
        "peak": peak,
    }


def evaluate_policy(
    win_prob: float,
    payoff_ratio: float,
    policy_frac: float,
    kelly_true: float,
    n_steps: int = 500,
    n_paths: int = 1000,
    seed: int = 0,
) -> dict:
    """Batch of robustness metrics for a fixed-fraction policy on a coin model."""
    sim = simulate_growth(win_prob, payoff_ratio, policy_frac, n_steps, n_paths, seed)
    g = sim["terminal_log_growth"]
    wealth = sim["wealth"]
    peak = sim["peak"]
    n_steps = int(n_steps)
    median_growth = float(np.median(g))
    prob_of_ruin = float(np.mean(np.any(wealth <= 0.05 * peak, axis=1)))
    q = float(np.quantile(g, 0.05))
    tail = g[g <= q]
    cvar95 = float(np.mean(tail)) if tail.size else float(median_growth)
    median_max_drawdown = float(np.median(np.max(1.0 - wealth / peak, axis=1)))
    recovered = wealth >= 1.0
    first = np.argmax(recovered, axis=1) + 1
    first[~recovered.any(axis=1)] = n_steps
    recovery_time = float(np.median(first))
    if not 0.0 < kelly_true < 1.0:
        growth_ratio_to_optimal = float("nan")
    else:
        g_opt = win_prob * np.log(1.0 + payoff_ratio * kelly_true) + (
            1.0 - win_prob
        ) * np.log(1.0 - kelly_true)
        growth_ratio_to_optimal = (
            float(median_growth / (n_steps * g_opt)) if g_opt > 0.0 else float("nan")
        )
    return {
        "median_growth": median_growth,
        "prob_of_ruin": prob_of_ruin,
        "cvar95": cvar95,
        "median_max_drawdown": median_max_drawdown,
        "recovery_time": recovery_time,
        "growth_ratio_to_optimal": growth_ratio_to_optimal,
    }


def corrupted_edge_report(
    base_win_prob: float,
    payoff_ratio: float,
    policy_frac: float,
    edge_spread: list,
    n_steps: int = 500,
    n_paths: int = 500,
    seed: int = 0,
) -> dict:
    """Report policy behaviour when the estimated win probability is wrong.

    Each corrupted win prob (base*(1 + delta) plus the calibration-reversed
    value 1 - base) is evaluated with the base edge as the assumed optimum.
    The verdict is "ROBUST" only if prob_of_ruin stays below 0.1 at the worst
    tested probability, otherwise "FRAGILE".
    """
    if not 0.0 < base_win_prob < 1.0:
        raise ValueError(f"base_win_prob must lie in (0, 1), got {base_win_prob}")
    if payoff_ratio <= 0.0:
        raise ValueError(f"payoff_ratio must be positive, got {payoff_ratio}")
    kelly_true = kelly_fraction(base_win_prob, payoff_ratio)
    probs = [base_win_prob * (1.0 + d) for d in edge_spread]
    probs.append(1.0 - base_win_prob)
    probs = [min(max(float(p), 1e-6), 1.0 - 1e-6) for p in probs]
    table = []
    worst_ruin = 0.0
    for i, p in enumerate(probs):
        row = evaluate_policy(
            p, payoff_ratio, policy_frac, kelly_true, n_steps=n_steps, n_paths=n_paths, seed=seed + i
        )
        table.append({"win_prob_used": p, **row})
        worst_ruin = max(worst_ruin, row["prob_of_ruin"])
    verdict = "ROBUST" if worst_ruin < 0.1 else "FRAGILE"
    return {"win_probs_tested": probs, "table": table, "verdict": verdict}
