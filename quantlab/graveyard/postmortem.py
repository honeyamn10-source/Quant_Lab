"""Postmortem generation and failure classification.

Turns a bare experiment/validation pair into a standardized graveyard entry and
a deterministic, machine-parsable postmortem text block.
"""

from __future__ import annotations

from quantlab.graveyard.taxonomy import FailureCategory, verdict_for

_MIN_DEFENSIBLE_OBS: int = 250
_NEAR_ZERO_NET_SHARPE: float = 0.1
_MIN_GROSS_EDGE: float = 0.5
_OVERFIT_DSR: float = 0.5
_OVERFIT_PBO: float = 0.5
_MAJOR_COST_STRESS: float = 2.0
_WEAK_PARAM_STABILITY: float = 0.5


def classify_failure(evidence: dict) -> list[FailureCategory]:
    """Classify evidence into a deterministic, ordered list of failure categories.

    Heuristic thresholds (documented here so verdicts are reproducible):

    - ``dsr < 0.5`` or ``pbo > 0.5`` -> OVERFIT
    - ``lookahead`` truthy -> LOOKAHEAD
    - ``survivorship`` truthy -> SURVIVORSHIP
    - ``abs(net_sharpe) < 0.1`` while ``gross_sharpe > 0.5`` -> NO_NET_EDGE
    - ``cost_stress_ratio > 2.0`` -> SLIPPAGE
    - ``regime_dependency`` truthy -> REGIME_DEPENDENT
    - ``param_stability < 0.5`` -> PARAMETER_UNSTABLE
    - ``tail_loss_exceeds`` truthy -> TAIL_RISK
    - ``n_obs < 250`` -> INSUFFICIENT_SAMPLE

    Missing evidence fields fall back to the least-incriminating default, so a
    sparse record never fabricates a failure it does not support.
    """
    categories: list[FailureCategory] = []
    dsr = evidence.get("dsr", 1.0)
    pbo = evidence.get("pbo", 0.0)
    if isinstance(dsr, (int, float)) and dsr < _OVERFIT_DSR or isinstance(pbo, (int, float)) and pbo > _OVERFIT_PBO:
        categories.append(FailureCategory.OVERFIT)
    if evidence.get("lookahead", False):
        categories.append(FailureCategory.LOOKAHEAD)
    if evidence.get("survivorship", False):
        categories.append(FailureCategory.SURVIVORSHIP)
    net_sharpe = evidence.get("net_sharpe", 0.0)
    gross_sharpe = evidence.get("gross_sharpe", 0.0)
    if (
        isinstance(net_sharpe, (int, float))
        and isinstance(gross_sharpe, (int, float))
        and abs(net_sharpe) < _NEAR_ZERO_NET_SHARPE
        and gross_sharpe > _MIN_GROSS_EDGE
    ):
        categories.append(FailureCategory.NO_NET_EDGE)
    cost_stress_ratio = evidence.get("cost_stress_ratio", 1.0)
    if isinstance(cost_stress_ratio, (int, float)) and cost_stress_ratio > _MAJOR_COST_STRESS:
        categories.append(FailureCategory.SLIPPAGE)
    if evidence.get("regime_dependency", False):
        categories.append(FailureCategory.REGIME_DEPENDENT)
    param_stability = evidence.get("param_stability", 1.0)
    if isinstance(param_stability, (int, float)) and param_stability < _WEAK_PARAM_STABILITY:
        categories.append(FailureCategory.PARAMETER_UNSTABLE)
    if evidence.get("tail_loss_exceeds", False):
        categories.append(FailureCategory.TAIL_RISK)
    n_obs = evidence.get("n_obs", float("inf"))
    if isinstance(n_obs, (int, float)) and n_obs < _MIN_DEFENSIBLE_OBS:
        categories.append(FailureCategory.INSUFFICIENT_SAMPLE)
    return categories


def _coerce_category(value: object) -> FailureCategory:
    if isinstance(value, FailureCategory):
        return value
    text = str(value)
    try:
        return FailureCategory(text)
    except ValueError:
        return FailureCategory[text]


def _merge_dedupe(categories: list[FailureCategory]) -> list[FailureCategory]:
    merged: list[FailureCategory] = []
    for category in categories:
        if category not in merged:
            merged.append(category)
    return merged


def postmortem_from_evidence(
    experiment: dict,
    validation: dict,
    failure_reasons: list[str] | list[tuple[str, str]],
) -> dict:
    """Build a standardized graveyard entry from an experiment/validation pair.

    ``failure_reasons`` items are either free-text reasons or
    ``(category, description)`` tuples; listed categories are merged with the
    heuristic classification. The first reason text becomes ``primary_failure``,
    the second ``secondary_failure``. ``status`` derives from the categories
    (INCONCLUSIVE only for a sole INSUFFICIENT_SAMPLE, REJECTED otherwise).
    """
    merged_evidence = {**experiment, **validation}
    categories = classify_failure(merged_evidence)
    reason_texts: list[str] = []
    for reason in failure_reasons:
        if isinstance(reason, (tuple, list)) and len(reason) == 2:
            categories.append(_coerce_category(reason[0]))
            reason_texts.append(str(reason[1]))
        else:
            reason_texts.append(str(reason))
    categories = _merge_dedupe(categories)
    if FailureCategory.INSUFFICIENT_SAMPLE in categories and all(
        verdict_for(category) == "INCONCLUSIVE" for category in categories
    ):
        status = "INCONCLUSIVE"
    else:
        status = "REJECTED"
    return {
        "id": experiment.get("id"),
        "hypothesis": experiment.get("hypothesis", ""),
        "raw_sharpe": merged_evidence.get("raw_sharpe"),
        "net_sharpe": merged_evidence.get("net_sharpe"),
        "metrics": dict(experiment.get("metrics", {}) or {}),
        "validation": dict(validation),
        "failure_categories": categories,
        "primary_failure": reason_texts[0] if reason_texts else None,
        "secondary_failure": reason_texts[1] if len(reason_texts) > 1 else None,
        "status": status,
    }


def generate_postmortem(record: dict) -> str:
    """Render a record as a deterministic, line-oriented postmortem.

    Missing keys fall back to stable defaults (``n/a``, ``Unknown`` or
    ``STRATEGY-UNKNOWN``) so any record produces parseable output.
    """

    def _fmt(key: str, default: str = "n/a") -> str:
        value = record.get(key)
        if value is None:
            return default
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    return "\n".join(
        [
            _fmt("id", "STRATEGY-UNKNOWN"),
            f"Hypothesis: {_fmt('hypothesis', 'Unknown')}",
            f"Raw Sharpe: {_fmt('raw_sharpe')}",
            f"Realistic-cost Sharpe: {_fmt('net_sharpe')}",
            f"Main failure: {_fmt('primary_failure', 'Unknown')}",
            f"Secondary failure: {_fmt('secondary_failure', 'Unknown')}",
            f"PBO: {_fmt('pbo')}",
            f"Walk-forward: {_fmt('walk_forward')}",
            f"Status: {_fmt('status', 'REJECTED')}.",
        ]
    )
