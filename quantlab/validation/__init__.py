"""Statistical validation battery.

Each module attacks a distinct failure mode and reports its own evidence:
    PSR  - non-normal returns vs a benchmark
    DSR  - selection bias / backtest overfitting across N trials
    CSCV - combinatorial symmetric cross-validation
    PBO  - probability of backtest overfitting derived from CSCV
    bootstrap + Monte Carlo - sampling uncertainty
    walk-forward / purged CV / embargo - honest out-of-sample timing
"""

from quantlab.validation.bootstrap import (
    block_bootstrap,
    confidence_interval,
    iid_bootstrap,
    stationary_bootstrap,
)
from quantlab.validation.cscv import cscv_logits, probability_of_backtest_overfitting
from quantlab.validation.dsr import deflated_sharpe_ratio, expected_max_sharpe
from quantlab.validation.embargo import embargo_mask
from quantlab.validation.metrics import all_metrics, max_drawdown, sharpe_ratio
from quantlab.validation.monte_carlo import monte_carlo_null, shuffle_pvalue
from quantlab.validation.psr import probabilistic_sharpe_ratio
from quantlab.validation.purged_cv import purged_cv_indices
from quantlab.validation.walk_forward import walk_forward_folds

__all__ = [
    "all_metrics",
    "block_bootstrap",
    "confidence_interval",
    "cscv_logits",
    "deflated_sharpe_ratio",
    "embargo_mask",
    "expected_max_sharpe",
    "iid_bootstrap",
    "max_drawdown",
    "monte_carlo_null",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "purged_cv_indices",
    "sharpe_ratio",
    "shuffle_pvalue",
    "stationary_bootstrap",
    "walk_forward_folds",
]
