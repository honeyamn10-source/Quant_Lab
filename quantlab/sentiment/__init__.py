"""Sentiment evidence with point-in-time timestamps and leakage audits.

Every headline is modelled as a :class:`NewsItem` whose event -> publication
-> retrieval -> decision chain makes look-ahead mechanically detectable. Event
studies refuse to capture the bar that produced the signal (``delay_bars``),
and :func:`compare_leakage` quantifies how much return the leaked variant
inflates when that guard is dropped.
"""

from quantlab.sentiment.event_study import compare_leakage, event_study, leaked_event_study
from quantlab.sentiment.timestamps import (
    NewsItem,
    availability_lag_profile,
    frame_to_returns,
    leakage_guard,
    validate_timeline,
)

__all__ = [
    "NewsItem",
    "availability_lag_profile",
    "compare_leakage",
    "event_study",
    "frame_to_returns",
    "leakage_guard",
    "leaked_event_study",
    "validate_timeline",
]
