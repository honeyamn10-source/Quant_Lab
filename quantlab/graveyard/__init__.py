"""Graveyard — the strategy coffin where falsified ideas are recorded."""

from __future__ import annotations

from quantlab.graveyard.postmortem import (
    classify_failure,
    generate_postmortem,
    postmortem_from_evidence,
)
from quantlab.graveyard.recorder import Graveyard
from quantlab.graveyard.taxonomy import FAILURE_DESCRIPTIONS, FailureCategory, verdict_for

__all__ = [
    "FAILURE_DESCRIPTIONS",
    "FailureCategory",
    "Graveyard",
    "classify_failure",
    "generate_postmortem",
    "postmortem_from_evidence",
    "verdict_for",
]
