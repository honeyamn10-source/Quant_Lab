"""Perturbation sweeps that stress a backtest along single dimensions.

Each sweep accepts a ``run_fn(overrides: dict)`` callable that re-runs the
backtest with the given configuration overrides and returns any result-like
object exposing ``.returns`` and ``.signals``/``.equity()``. All sweeps are
deterministic: any randomness is driven by an explicit ``seed``.
"""

from __future__ import annotations

import numpy as np


def _is_numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def scaled_params(base: dict, scales: list[float]) -> list[dict]:
    """Return one copy of ``base`` per scale with every numeric value scaled.

    Each variant keeps non-numeric values untouched and carries a ``"perturbation"``
    meta key of ``{"scale": s}`` recording the applied factor.
    """
    variants = []
    for scale in scales:
        variant = {
            key: (value * scale if _is_numeric(value) else value) for key, value in base.items()
        }
        variant["perturbation"] = {"scale": float(scale)}
        variants.append(variant)
    return variants


def _round_int_keys(variant: dict, base: dict) -> dict:
    """Round scaled values back to ``int`` for keys that are int-typed in ``base``."""
    fixed = dict(variant)
    for key, value in base.items():
        if type(value) is int:
            fixed[key] = int(round(fixed[key]))
    return fixed


def param_perturbation_sweep(
    run_fn, base_params: dict, factors: list[float] | None = None
) -> list[dict]:
    """Sweep every numeric strategy parameter at once across ``factors``.

    Defaults to ``[0.8, 0.9, 1.0, 1.1, 1.2]``. Each run passes
    ``{"strategy_params": variant}`` and returns ``{"params": variant, "result": ...}``.
    Values whose base type is ``int`` are rounded back to whole numbers.
    """
    factors = factors if factors is not None else [0.8, 0.9, 1.0, 1.1, 1.2]
    entries = []
    for variant in scaled_params(base_params, factors):
        variant = _round_int_keys(variant, base_params)
        entries.append({"params": variant, "result": run_fn({"strategy_params": variant})})
    return entries


def cost_perturbation_sweep(run_fn, cost_factors: list[float] | None = None) -> list[dict]:
    """Scale trading costs (commission and spread) together; default is the 5 bps baseline.

    For each factor ``f`` runs ``{"commission_bps": 5 * f, "spread_factor": f}`` and
    returns ``{"factor": f, "result": ...}``.
    """
    cost_factors = cost_factors if cost_factors is not None else [1.0, 1.5, 2.0, 3.0]
    return [
        {
            "factor": factor,
            "result": run_fn({"commission_bps": 5.0 * factor, "spread_factor": factor}),
        }
        for factor in cost_factors
    ]


def slippage_perturbation_sweep(run_fn, slip_factors: list[float] | None = None) -> list[dict]:
    """Stress slippage under the normal execution profile.

    Defaults to ``[1, 2, 3]``; each run passes ``{"profile": "normal", "slippage_mult": f}``.
    """
    slip_factors = slip_factors if slip_factors is not None else [1.0, 2.0, 3.0]
    return [
        {"factor": factor, "result": run_fn({"profile": "normal", "slippage_mult": factor})}
        for factor in slip_factors
    ]


def execution_delay_sweep(run_fn, delays: list[int] | None = None) -> list[dict]:
    """Lengthen the execution delay (bars between signal and fill eligibility).

    Defaults to ``[1, 2, 3]``; each run passes ``{"execution_delay": d}``.
    """
    delays = delays if delays is not None else [1, 2, 3]
    return [{"factor": delay, "result": run_fn({"execution_delay": delay})} for delay in delays]


def random_start(run_fn, n_tries: int = 20, seed: int = 0) -> list[dict]:
    """Sample randomized starting dates, plus shuffled-universe variants.

    Runs ``{"start_offset": k}`` for ``k`` in ``range(n_tries)`` and additionally
    reruns a deterministic subset (``n_tries // 5``) with
    ``{"start_offset": k, "universe": "shuffled"}``. Every entry is
    ``{"start_offset": k, "shuffled_universe": bool, "result": ...}``.
    """
    rng = np.random.default_rng(seed)
    entries = [
        {
            "start_offset": int(k),
            "shuffled_universe": False,
            "result": run_fn({"start_offset": int(k)}),
        }
        for k in range(n_tries)
    ]
    n_shuffled = max(0, n_tries // 5)
    if n_shuffled:
        for k in rng.choice(n_tries, size=n_shuffled, replace=False):
            k = int(k)
            entries.append(
                {
                    "start_offset": k,
                    "shuffled_universe": True,
                    "result": run_fn({"start_offset": k, "universe": "shuffled"}),
                }
            )
    return entries


def universe_variants(run_fn, base_universe: list[str], alternatives: dict) -> list[dict]:
    """Run the strategy on the base universe plus each named alternative.

    ``alternatives`` maps a label to a list of symbols. Each entry is
    ``{"label": label, "universe": [...], "result": ...}``.
    """
    entries = [
        {
            "label": "base",
            "universe": list(base_universe),
            "result": run_fn({"universe": list(base_universe)}),
        }
    ]
    entries.extend(
        {"label": label, "universe": list(universe), "result": run_fn({"universe": list(universe)})}
        for label, universe in alternatives.items()
    )
    return entries


def liquidity_reduced(run_fn, reductions: list[float] | None = None) -> list[dict]:
    """Degrade participation capacity and ADV simultaneously.

    Defaults to ``[0.5, 0.25, 0.1]``; each run passes
    ``{"participation_cap": 0.1 * r, "adv_multiplier": r}``.
    """
    reductions = reductions if reductions is not None else [0.5, 0.25, 0.1]
    return [
        {"factor": r, "result": run_fn({"participation_cap": 0.1 * r, "adv_multiplier": r})}
        for r in reductions
    ]
