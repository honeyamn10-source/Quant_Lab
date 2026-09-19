"""Failure taxonomy for strategy falsification.

Standardizes why a strategy failed so reports, graveyard entries, and
postmortems share one vocabulary.
"""

from __future__ import annotations

from enum import Enum


class FailureCategory(str, Enum):  # noqa: UP042
    """Curated set of documented strategy failure modes."""
    """Curated set of documented strategy failure modes."""

    OVERFIT = "overfit"
    LOOKAHEAD = "lookahead"
    SURVIVORSHIP = "survivorship"
    SLIPPAGE = "slippage"
    MARKET_IMPACT = "market_impact"
    REGIME_DEPENDENT = "regime_dependent"
    PARAMETER_UNSTABLE = "parameter_unstable"
    DATA_ERROR = "data_error"
    CORRELATION_FAILURE = "correlation_failure"
    TAIL_RISK = "tail_risk"
    INSUFFICIENT_SAMPLE = "insufficient_sample"
    ALPHA_DECAY = "alpha_decay"
    NO_NET_EDGE = "no_net_edge"


FAILURE_DESCRIPTIONS: dict[FailureCategory, str] = {
    FailureCategory.OVERFIT: "The strategy fits backtest noise rather than exploitable signal.",
    FailureCategory.LOOKAHEAD: "Signal or execution leaked information unavailable at decision time.",
    FailureCategory.SURVIVORSHIP: "Results depend on an asset or universe set that survived selection bias.",
    FailureCategory.SLIPPAGE: "Assumed fills are materially better than realistic execution costs allow.",
    FailureCategory.MARKET_IMPACT: "Bet sizes are large enough to move prices against the strategy.",
    FailureCategory.REGIME_DEPENDENT: "Edge exists only in a narrow and uncommon market regime.",
    FailureCategory.PARAMETER_UNSTABLE: "Performance collapses under small, reasonable parameter perturbations.",
    FailureCategory.DATA_ERROR: "Evidence is tainted by bad, missing, or misaligned data.",
    FailureCategory.CORRELATION_FAILURE: "Assumed correlation structure across signals or assets did not hold.",
    FailureCategory.TAIL_RISK: "Rare-event losses dominate and normal backtest statistics hide them.",
    FailureCategory.INSUFFICIENT_SAMPLE: "Not enough observations to draw a reliable conclusion.",
    FailureCategory.ALPHA_DECAY: "Signal strength decays with post-publication or post-crowding time.",
    FailureCategory.NO_NET_EDGE: "Gross edge disappears once costs, impact, and slippage are applied.",
}


def verdict_for(category: FailureCategory) -> str:
    """Map a failure category to the research verdict.

    ``INSUFFICIENT_SAMPLE`` is inconclusive by definition: there is not enough
    evidence to reject, but nothing to bless either. Every other documented
    failure kills the strategy.
    """
    if category is FailureCategory.INSUFFICIENT_SAMPLE:
        return "INCONCLUSIVE"
    return "REJECTED"
