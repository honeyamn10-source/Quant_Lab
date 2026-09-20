"""Data quality scoring.

Produces a 0-100 DataQualityScore plus graded risk dimensions so researchers
see *where* inputs are fragile rather than a single opaque number.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from quantlab.data.schemas import PointInTimeRow
from quantlab.data.validation import DataAudit, Severity, audit_dataset

# Each code maps to the risk dimensions it degrades.
_CODE_WEIGHTS: dict[str, float] = {
    # hard errors
    "DUPLICATE_TS": 2.0,
    "OUT_OF_ORDER": 2.0,
    "EMPTY": 5.0,
    "NEGATIVE_PRICE": 4.0,
    "BAD_HIGH": 3.0,
    "BAD_LOW": 3.0,
    "HIGH_LT_LOW": 3.0,
    "ZERO_PRICE": 3.0,
    "AVAIL_BEFORE_EVENT": 4.0,
    "BAD_ADJ_FACTOR": 3.0,
    "NEGATIVE_VOLUME": 2.0,
    # warnings
    "MISSING_OBS": 1.0,
    "SPIKE": 0.5,
    "CA_ANOMALY": 0.5,
    "STALE_PRICE": 0.5,
    # information
    "SPLIT_ANOMALY": 0.0,
}

_RISK_MAP: dict[str, tuple[str, ...]] = {
    "DUPLICATE_TS": ("lookahead", "missing"),
    "OUT_OF_ORDER": ("lookahead",),
    "EMPTY": ("missing",),
    "NEGATIVE_PRICE": ("quality", "lookahead"),
    "BAD_HIGH": ("quality",),
    "BAD_LOW": ("quality",),
    "HIGH_LT_LOW": ("quality",),
    "ZERO_PRICE": ("quality",),
    "AVAIL_BEFORE_EVENT": ("lookahead",),
    "BAD_ADJ_FACTOR": ("corporate_action", "lookahead"),
    "NEGATIVE_VOLUME": ("quality",),
    "MISSING_OBS": ("missing",),
    "SPIKE": ("quality",),
    "CA_ANOMALY": ("corporate_action",),
    "STALE_PRICE": ("quality",),
    "SPLIT_ANOMALY": ("corporate_action",),
}

RISK_DIMENSIONS = (
    "quality",
    "lookahead",
    "corporate_action",
    "missing",
    "survivorship",
    "source",
)


def _bounded(v: int) -> int:
    return max(0, min(100, v))


def _risk_grade(penalty: float) -> str:
    if penalty <= 0:
        return "LOW"
    if penalty <= 1.5:
        return "MODERATE"
    if penalty <= 3.0:
        return "ELEVATED"
    return "HIGH"


@dataclass
class DataQualityScore:
    """Numeric quality plus per-dimension risk grades."""

    score: int
    risks: dict[str, str] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"score": self.score, "risks": self.risks, "reasons": self.reasons}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        risks = " ".join(f"{k}={v}" for k, v in self.risks.items())
        return f"DataQualityScore({self.score} {risks})"


@dataclass
class DataQualityReport:
    """Full data-quality document."""

    audit: DataAudit
    score: DataQualityScore
    fingerprint: str = ""
    n_rows: int = 0

    def to_dict(self) -> dict:
        return {
            "score": self.score.to_dict(),
            "summarized_audit": self.audit.summary(),
            "issues": [
                {
                    "code": i.code,
                    "severity": i.severity.value,
                    "symbol": i.symbol,
                    "message": i.message,
                }
                for i in self.audit.issues
            ],
            "fingerprint": self.fingerprint,
            "n_rows": self.n_rows,
        }


def score_quality(
    audit: DataAudit,
    include_survivorship_penalty: bool = True,
    survivorship_hint: str = "",
) -> DataQualityScore:
    """Turn an audit into a 0-100 score and per-dimension grades.

    `survivorship_hint` may be one of "delisted","active","unknown"; it lets the
    caller encode universe knowledge (e.g. backfills that only keep listed names).
    """
    penalty = 0.0
    reasons: list[str] = []
    dimension_penalty: dict[str, float] = dict.fromkeys(RISK_DIMENSIONS, 0.0)

    for issue in audit.issues:
        weight = _CODE_WEIGHTS.get(issue.code, 1.0)
        if issue.severity == Severity.ERROR:
            penalty += weight if issue.code not in ("EMPTY",) else weight * 2
        elif issue.severity == Severity.WARNING:
            penalty += weight
        for dim in _RISK_MAP.get(issue.code, ()):
            if dim in dimension_penalty:
                dimension_penalty[dim] += (
                    weight if issue.severity == Severity.ERROR else weight * 0.5
                )
        if issue.severity == Severity.ERROR and issue.code not in ("SPLIT_ANOMALY",):
            reasons.append(f"{issue.code}@{issue.symbol}: {issue.message}")

    if include_survivorship_penalty and survivorship_hint == "delisted":
        dimension_penalty["survivorship"] = 3.0
        reasons.append("universe may exclude delisted/dormant names (survivorship risk)")
    elif include_survivorship_penalty and survivorship_hint == "active":
        dimension_penalty["survivorship"] = 0.0
        reasons.append("universe is known to include delisted names")

    score = _bounded(100 - int(round(penalty * 10)))
    risks = {d: _risk_grade(dimension_penalty[d]) for d in RISK_DIMENSIONS}
    return DataQualityScore(score=score, risks=risks, reasons=reasons)


def report_quality(
    rows: Iterable[PointInTimeRow], audit: DataAudit | None = None
) -> DataQualityReport:
    """Convenience: audit fresh rows and produce the full report + fingerprint."""
    from quantlab.data.fingerprint import fingerprint_pointintime

    row_list = list(rows)
    audit = audit if audit is not None else audit_dataset(row_list)
    fingerprint = fingerprint_pointintime(row_list)
    return DataQualityReport(
        audit=audit, score=score_quality(audit), fingerprint=fingerprint, n_rows=len(row_list)
    )
