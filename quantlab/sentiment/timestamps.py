"""Point-in-time news timestamps and leakage guardrails.

A :class:`NewsItem` carries four distinct times so the research pipeline can
audit where information originated and when it actually became usable. The
event-to-publication and publication-to-retrieval lags are the classic
sentiment-scoring leakage vectors: a model that is scored with retrieval-time
knowledge of a headline is silently using the future.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator

_DECISION_DELAY_KEY = "decision_delay"
_PUBLICATION_LAG_KEY = "publication_lag"
_RETRIEVAL_LAG_KEY = "retrieval_lag"


def _field_chain_message(
    field: str,
    value: int,
    prior_field: str,
    prior_value: int,
) -> str:
    return f"{field} {value} must not precede {prior_field} {prior_value}"


class NewsItem(BaseModel):
    """A headline observation with its full availability timeline.

    Timestamps are expected in epoch seconds. The chain
    ``event_time <= publication_time <= retrieval_time <= decision_time`` is
    enforced by model validation. ``decision_time`` is the moment at which the
    consuming strategy acts on the item; because ``retrieval_time`` is recorded
    after the fact by our own pipeline, treating it as known at ``decision_time``
    leaks the availability of that headline.
    """

    event_time: int = Field(ge=0)
    publication_time: int = Field(ge=0)
    retrieval_time: int = Field(ge=0)
    decision_time: int = Field(ge=0)
    headline: str
    entities: list[str] = Field(default_factory=list)
    sentiment: float | None = Field(default=None, ge=-1.0, le=1.0)
    source: str = ""

    @field_validator("publication_time")
    @classmethod
    def _publication_not_before_event(cls, value: int, info) -> int:
        event_time = info.data.get("event_time")
        if event_time is not None and value < event_time:
            raise ValueError(
                _field_chain_message("publication_time", value, "event_time", event_time)
            )
        return value

    @field_validator("retrieval_time")
    @classmethod
    def _retrieval_not_before_publication(cls, value: int, info) -> int:
        publication_time = info.data.get("publication_time")
        if publication_time is not None and value < publication_time:
            raise ValueError(
                _field_chain_message("retrieval_time", value, "publication_time", publication_time)
            )
        return value

    @field_validator("decision_time")
    @classmethod
    def _decision_not_before_retrieval(cls, value: int, info) -> int:
        retrieval_time = info.data.get("retrieval_time")
        if retrieval_time is not None and value < retrieval_time:
            raise ValueError(
                _field_chain_message("decision_time", value, "retrieval_time", retrieval_time)
            )
        return value


def validate_timeline(items: list[NewsItem]) -> list[dict]:
    """Re-validate a list of items and report every timeline violation.

    Returns a list of records ``{"index", "field", "message"}``, one per
    violated field, or an empty list when the timeline is clean. Items are
    reconstructed from their raw payloads so even instances built with
    :meth:`NewsItem.model_construct` (which skips validation) are audited.
    """
    violations: list[dict] = []
    for index, item in enumerate(items):
        try:
            NewsItem.model_validate(item.model_dump())
        except ValidationError as exc:
            for error in exc.errors():
                field = ".".join(str(part) for part in error.get("loc", ())) or "model"
                message = error.get("msg", "validation failed")
                if message.startswith("Value error, "):
                    message = message[len("Value error, ") :]
                violations.append({"index": index, "field": field, "message": message})
    return violations


def leakage_guard(items: list[NewsItem], decision_time: int) -> list[NewsItem]:
    """Return only the items a scanner could act on at ``decision_time``.

    A scanner can only know a headline once it has been published, so the gate
    is ``publication_time <= decision_time``. Using ``retrieval_time`` instead
    would be a leakage vector: the scanner cannot know a future scrape, and an
    item published on time but retrieved late would be wrongly gated out.
    """
    return [item for item in items if item.publication_time <= decision_time]


def availability_lag_profile(items: list[NewsItem]) -> dict:
    """Summarise publication and retrieval lag statistics in seconds.

    ``mean_pub_lag``/``median_pub_lag`` are event-to-publication gaps;
    ``mean_retrieval_lag``/``median_retrieval_lag``/``p95_retrieval_lag`` are
    publication-to-retrieval gaps, i.e. how long our pipeline took to notice
    an already-published headline.
    """
    if not items:
        return {
            "count": 0,
            "mean_pub_lag": 0.0,
            "median_pub_lag": 0.0,
            "mean_retrieval_lag": 0.0,
            "median_retrieval_lag": 0.0,
            "p95_retrieval_lag": 0.0,
        }
    pub_lags = [item.publication_time - item.event_time for item in items]
    retrieval_lags = [item.retrieval_time - item.publication_time for item in items]
    return {
        "count": len(items),
        "mean_pub_lag": float(np.mean(pub_lags)),
        "median_pub_lag": float(np.median(pub_lags)),
        "mean_retrieval_lag": float(np.mean(retrieval_lags)),
        "median_retrieval_lag": float(np.median(retrieval_lags)),
        "p95_retrieval_lag": float(np.percentile(retrieval_lags, 95)),
    }


def frame_to_returns(matches: list[tuple[NewsItem, float]]) -> dict:
    """Collapse ``(item, forward_return)`` pairs into event-study statistics.

    Returns ``{"n", "mean_return", "median_return", "hit_rate", "cumulative"}``
    for use by event studies; ``hit_rate`` is the fraction of positive returns.
    """
    returns = np.asarray([ret for _, ret in matches], dtype=float)
    if returns.size == 0:
        return {
            "n": 0,
            "mean_return": 0.0,
            "median_return": 0.0,
            "hit_rate": 0.0,
            "cumulative": 0.0,
        }
    return {
        "n": int(returns.size),
        "mean_return": float(returns.mean()),
        "median_return": float(np.median(returns)),
        "hit_rate": float((returns > 0).mean()),
        "cumulative": float(returns.sum()),
    }
