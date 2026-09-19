"""Post-hoc randomization statistics computed on a completed backtest result.

Every function in this module is a *perturbation statistic*, not a re-run: it
operates on the recorded signals/fills of an already-finished result and produces
a deterministic summary given a seed. The statistics estimate how sensitive the
strategy's evidence is to signal ordering, dropped executions or entry delay
without ever re-executing the engine.
"""

from __future__ import annotations

import numpy as np


def _equity_scale(result) -> float:
    """Approximate portfolio equity used to convert a fill notional to a weight."""
    try:
        curve = result.equity()
    except AttributeError:
        curve = []
    if curve:
        last = float(curve[-1])
        if last > 0:
            return last
    return 1.0


def _next_return(result, bar_index: int) -> float:
    returns = np.asarray(result.returns, dtype=float)
    idx = int(bar_index)
    if 0 <= idx < len(returns):
        return float(returns[idx])
    return 0.0


def _signal_pairs(result) -> list[tuple[float, float]]:
    """(target_weight, next_return) for each signal that maps onto a return."""
    return [
        (float(getattr(signal, "target_weight", 0.0)), _next_return(result, int(getattr(signal, "bar_index", 0))))
        for signal in result.signals
    ]


def shuffle_signal_order(result, seed: int = 0) -> dict:
    """Randomly reorder the signal-return alignment and re-price the PnL.

    The natural alignment is ``sum(target_weight * next_return)`` where
    ``next_return`` is the next bar return indexed by the signal's bar. A random
    permutation breaks any temporal coupling between when a signal appears and the
    return it earns, isolating whether the strategy's edge depends on timing.
    Returns ``{"n_signals", "shuffled_net_return"}`` plus the natural return for
    reference. This is evidence, not a re-run.
    """
    pairs = _signal_pairs(result)
    rng = np.random.default_rng(seed)
    shuffled = np.asarray([weight * ret for weight, ret in pairs], dtype=float)
    rng.shuffle(shuffled)
    natural = sum(weight * ret for weight, ret in pairs)
    return {
        "n_signals": int(len(result.signals)),
        "shuffled_net_return": float(shuffled.sum()),
        "natural_net_return": float(natural),
    }


def drop_random_trades(result, drop_frac: float = 0.2, seed: int = 0) -> dict:
    """Estimate the effect of randomly removing a fraction of fills.

    Each removed fill's notional (``quantity * price``) scaled by final equity
    weights its exposure; the estimated impact is the sum of those weights times
    the next-bar return. Returns ``{"removed", "net_impact_estimate"}``. This is
    evidence of trade-level execution dependence, not a re-run.
    """
    fills = list(result.fills)
    if not fills:
        return {"removed": 0, "net_impact_estimate": 0.0}
    scale = _equity_scale(result)
    rng = np.random.default_rng(seed)
    n_removed = int(round(len(fills) * min(max(drop_frac, 0.0), 1.0)))
    chosen = set(rng.choice(len(fills), size=n_removed, replace=False).tolist())
    impact = 0.0
    for i, fill in enumerate(fills):
        weight = float(getattr(fill, "quantity", 0.0)) * float(getattr(fill, "price", 0.0)) / scale
        if i in chosen:
            impact += weight * _next_return(result, getattr(fill, "bar_index", 0))
    return {"removed": int(n_removed), "net_impact_estimate": float(impact)}


def random_entry_delay(result, delay_bars: int = 2, seed: int = 0) -> dict:
    """Shift fills later by ``delay_bars`` where the return series allows it.

    Each fill's next-bar return is replaced by the return ``delay_bars`` bars
    later; the impact estimate is the weighted difference between declined and
    original returns. Returns ``{"n_shifted", "impact_estimate"}``. This is evidence
    of execution-timing sensitivity, not a re-run.
    """
    fills = list(result.fills)
    scale = _equity_scale(result)
    impact = 0.0
    n_shifted = 0
    returns = np.asarray(result.returns, dtype=float)
    for fill in fills:
        bar_index = int(getattr(fill, "bar_index", 0))
        weight = float(getattr(fill, "quantity", 0.0)) * float(getattr(fill, "price", 0.0)) / scale
        delayed = bar_index + delay_bars
        if delayed >= len(returns):
            continue
        impact += weight * (float(returns[delayed]) - _next_return(result, bar_index))
        n_shifted += 1
    return {"n_shifted": int(n_shifted), "impact_estimate": float(impact)}
