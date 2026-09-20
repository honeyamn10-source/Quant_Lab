"""Point-in-time data integrity.

Schemas, validation, quality scoring and fingerprinting for market data.
The critical rule enforced throughout:

    available_time <= decision_time

A strategy can only access information that was genuinely available at the
moment it makes a decision.
"""

from quantlab.data.barstream import BarStream
from quantlab.data.fingerprint import dataset_fingerprint
from quantlab.data.quality import DataQualityReport, DataQualityScore, score_quality
from quantlab.data.schemas import Bar, BarSeries, DatasetManifest, PointInTimeRow
from quantlab.data.validation import AuditIssue, DataAudit, audit_bars, audit_dataset

__all__ = [
    "AuditIssue",
    "Bar",
    "BarSeries",
    "BarStream",
    "DataAudit",
    "DataQualityReport",
    "DataQualityScore",
    "DatasetManifest",
    "PointInTimeRow",
    "audit_bars",
    "audit_dataset",
    "dataset_fingerprint",
    "score_quality",
]
