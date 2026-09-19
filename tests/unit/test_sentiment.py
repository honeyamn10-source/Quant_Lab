"""Unit tests for the sentiment timestamps and event-study leakage guards."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from quantlab.sentiment import (
    NewsItem,
    availability_lag_profile,
    compare_leakage,
    event_study,
    frame_to_returns,
    leakage_guard,
    leaked_event_study,
    validate_timeline,
)


def _item(**overrides: object) -> NewsItem:
    base = {
        "event_time": 100,
        "publication_time": 120,
        "retrieval_time": 140,
        "decision_time": 150,
        "headline": "sample headline",
    }
    base.update(overrides)
    return NewsItem(**base)


def test_newsitem_rejects_publication_before_event() -> None:
    with pytest.raises(ValidationError):
        _item(publication_time=90)
    with pytest.raises(ValidationError):
        _item(retrieval_time=110)
    with pytest.raises(ValidationError):
        _item(decision_time=130)
    with pytest.raises(ValidationError):
        _item(sentiment=1.5)


def test_clean_timeline_validates_clean() -> None:
    items = [
        NewsItem(event_time=0, publication_time=1, retrieval_time=2, decision_time=3, headline="a"),
        NewsItem(event_time=5, publication_time=6, retrieval_time=6, decision_time=8, headline="b"),
    ]
    assert validate_timeline(items) == []


def test_leakage_guard_filters_on_publication_time() -> None:
    decision_time = 125
    published_early = _item(publication_time=120, retrieval_time=130, decision_time=140)
    on_boundary = _item(publication_time=125, retrieval_time=140, decision_time=150)
    published_late = NewsItem(
        event_time=200,
        publication_time=260,
        retrieval_time=270,
        decision_time=280,
        headline="late",
    )
    kept = leakage_guard([published_early, on_boundary, published_late], decision_time)
    assert kept == [published_early, on_boundary]
    assert published_late not in kept


def test_validate_timeline_reports_field_violations() -> None:
    bad_items = [
        NewsItem.model_construct(
            event_time=100,
            publication_time=90,
            retrieval_time=110,
            decision_time=120,
            headline="pub before event",
        ),
        NewsItem.model_construct(
            event_time=100,
            publication_time=110,
            retrieval_time=105,
            decision_time=120,
            headline="retrieval before pub",
        ),
        NewsItem.model_construct(
            event_time=100,
            publication_time=110,
            retrieval_time=115,
            decision_time=114,
            headline="decision before retrieval",
        ),
        NewsItem.model_construct(
            event_time=100,
            publication_time=110,
            retrieval_time=115,
            decision_time=120,
            headline="sentiment out of range",
            sentiment=1.5,
        ),
    ]
    violations = validate_timeline(bad_items)
    assert [v["index"] for v in violations] == [0, 1, 2, 3]
    assert {v["field"] for v in violations} == {
        "publication_time",
        "retrieval_time",
        "decision_time",
        "sentiment",
    }
    assert all(v["message"] for v in violations)


def test_event_study_returns_horizons_and_skips_out_of_range() -> None:
    closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    events = [0, 1, 5, 6]
    study = event_study(closes, events, horizons=[1, 3], delay_bars=1)
    assert study["delay_bars"] == 1
    by_horizon = {row["horizon"]: row for row in study["results"]}
    assert set(by_horizon) == {1, 3}
    required = {"mean_return", "median_return", "hit_rate", "n_events", "std_error"}
    assert all(required <= set(row) for row in study["results"])
    assert by_horizon[1]["n_events"] == 2
    assert by_horizon[3]["n_events"] == 2
    assert by_horizon[1]["mean_return"] == pytest.approx(
        (102.0 / 101.0 - 1.0 + 103.0 / 102.0 - 1.0) / 2.0
    )


def test_event_study_delay_excludes_same_bar_move() -> None:
    closes = np.array([100.0, 100.0, 110.0, 110.0, 110.0])
    delayed = event_study(closes, [0], horizons=[1], delay_bars=1)
    assert delayed["results"][0]["mean_return"] == pytest.approx(110.0 / 100.0 - 1.0)
    extracted = leaked_event_study(closes, [0], horizons=[1])
    assert extracted["delay_bars"] == 0


def test_compare_leakage_flags_suspicious_horizon() -> None:
    closes = np.array([100.0, 110.0, 110.0, 110.0, 110.0])
    comparison = compare_leakage(closes, [0], horizons=[1])
    row = comparison["results"][0]
    assert row["horizon"] == 1
    assert row["flag"] == "LEAKAGE_SUSPICIOUS"
    assert "LEAKAGE_SUSPICIOUS" in comparison["flags"]
    assert row["leaked_mean_return"] > row["delayed_mean_return"]


def test_event_study_skips_events_past_end() -> None:
    closes = np.array([100.0, 101.0, 102.0, 103.0])
    study = event_study(closes, [0, 3, 4, -2], horizons=[2], delay_bars=1)
    row = study["results"][0]
    assert row["n_events"] == 1
    assert study["results"] and row["hit_rate"] in (0.0, 1.0)


def test_leaked_study_inflates_returns() -> None:
    closes = np.array([100.0, 110.0, 110.0, 110.0, 110.0])
    leaked = leaked_event_study(closes, [0], horizons=[1])["results"][0]["mean_return"]
    delayed = event_study(closes, [0], horizons=[1], delay_bars=1)["results"][0]["mean_return"]
    assert leaked > delayed


def test_availability_lag_profile_statistics() -> None:
    items = [
        NewsItem(event_time=0, publication_time=10, retrieval_time=20, decision_time=30, headline="a"),
        NewsItem(event_time=5, publication_time=15, retrieval_time=25, decision_time=35, headline="b"),
    ]
    profile = availability_lag_profile(items)
    assert profile["count"] == 2
    assert profile["mean_pub_lag"] == pytest.approx(10.0)
    assert profile["median_pub_lag"] == pytest.approx(10.0)
    assert profile["mean_retrieval_lag"] == pytest.approx(10.0)
    assert profile["median_retrieval_lag"] == pytest.approx(10.0)
    assert profile["p95_retrieval_lag"] == pytest.approx(10.0)
    assert availability_lag_profile([])["count"] == 0


def test_frame_to_returns_statistics() -> None:
    items = [
        _item(publication_time=120, headline="one"),
        _item(publication_time=120, retrieval_time=140, decision_time=150, headline="two"),
    ]
    frame = frame_to_returns([(items[0], 0.02), (items[1], -0.01)])
    assert frame["n"] == 2
    assert frame["mean_return"] == pytest.approx(0.005)
    assert frame["median_return"] == pytest.approx(0.005)
    assert frame["hit_rate"] == pytest.approx(0.5)
    assert frame["cumulative"] == pytest.approx(0.01)
    empty = frame_to_returns([])
    assert empty["n"] == 0 and empty["mean_return"] == 0.0
