"""Adversarial battery: perturbation sweeps, randomized evidence, robustness grade.

The battery stresses a backtest along execution, cost, parameter, universe and
timing dimensions without modifying the underlying engine. Sweeps accept a
``run_fn(overrides: dict)`` callable; post-hoc statistics operate on finished
results and are seeded and deterministic.
"""

from quantlab.adversarial.perturbation import (
    cost_perturbation_sweep,
    execution_delay_sweep,
    liquidity_reduced,
    param_perturbation_sweep,
    random_start,
    scaled_params,
    slippage_perturbation_sweep,
    universe_variants,
)
from quantlab.adversarial.randomization import (
    drop_random_trades,
    random_entry_delay,
    shuffle_signal_order,
)
from quantlab.adversarial.robustness import (
    GRADES,
    compute_sharpe,
    degradation_slope,
    robustness_surface,
)

__all__ = [
    "GRADES",
    "compute_sharpe",
    "cost_perturbation_sweep",
    "degradation_slope",
    "drop_random_trades",
    "execution_delay_sweep",
    "liquidity_reduced",
    "param_perturbation_sweep",
    "random_entry_delay",
    "random_start",
    "robustness_surface",
    "scaled_params",
    "shuffle_signal_order",
    "slippage_perturbation_sweep",
    "universe_variants",
]
