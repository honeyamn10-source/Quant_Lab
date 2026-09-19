"""Point-in-time data integrity tests (schemas, audit, BarStream, fingerprint)."""

from __future__ import annotations

from quantlab.data.barstream import BarStream, audit_stream
from quantlab.data.fingerprint import dataset_fingerprint
from quantlab.data.quality import score_quality
from quantlab.data.schemas import Bar, BarSeries, PointInTimeRow
from quantlab.data.validation import audit_bars, audit_dataset


def _series(bars: list[Bar]) -> BarSeries:
    return BarSeries("T", bars)


def test_bar_series_sorts_rejects_out_of_order() -> None:
    b1 = Bar(symbol="T", ts=100, interval_seconds=86_400, open=1, high=1, low=1, close=1)
    b2 = Bar(symbol="T", ts=50, interval_seconds=86_400, open=1, high=1, low=1, close=1)
    try:
        _series([b1, b2])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_audit_catches_bad_ohlc() -> None:
    bars = [
        Bar(
            symbol="T",
            ts=i,
            interval_seconds=86_400,
            open=100,
            high=110,
            low=90,
            close=100 + i,
            volume=100,
            available_time=i,
        )
        for i in range(50)
    ]
    bars[10].high = 50  # below max(open, close)
    bars[20].close = -1  # negative price
    audit = audit_bars(bars)
    codes = {i.code for i in audit.issues}
    assert "BAD_HIGH" in codes
    assert "NEGATIVE_PRICE" in codes
    assert audit.has_errors
    score = score_quality(audit)
    assert score.score < 90


def test_audit_clean_data_passes() -> None:
    bars = [
        Bar(
            symbol="T",
            ts=i * 86_400_000,
            interval_seconds=86_400,
            open=100,
            high=101,
            low=99,
            close=100.5,
            volume=100,
            available_time=i * 86_400_000,
        )
        for i in range(120)
    ]
    audit = audit_bars(bars)
    assert not audit.has_errors
    assert score_quality(audit).score >= 95


def test_audit_dataset_pit_violation() -> None:
    # Construct WITHOUT validation (model_construct) to simulate a corrupt file row:
    # a row reporting available_time before its own event time.
    row = PointInTimeRow.model_construct(
        symbol="T",
        ts=100,
        available_time=50,
        ingestion_time=60,
        open=1,
        high=1,
        low=1,
        close=1,
    )
    audit = audit_dataset([row])
    assert any(i.code == "AVAIL_BEFORE_EVENT" for i in audit.issues)


def test_bar_stream_hides_unavailable_bar() -> None:
    bars = [
        Bar(
            symbol="T",
            ts=i,
            interval_seconds=1,
            open=100,
            high=101,
            low=99,
            close=100 + i,
            volume=10,
            available_time=i,
        )
        for i in range(10)
    ]
    bars[5] = Bar(
        symbol="T",
        ts=5,
        interval_seconds=1,
        open=100,
        high=101,
        low=99,
        close=100,
        volume=10,
        available_time=9,
    )  # late-reported: not usable until bar 9
    stream = BarStream(bars, "T")
    for i in range(0, 5):
        while stream.cursor < i:
            stream.advance()
        assert stream.closes(i).size == i + 1  # bar 5 is hidden at decision i < 9
    while stream.cursor < 9:
        stream.advance()
    assert "bar 5 exposed" not in "\n".join(audit_stream(stream))


def test_bar_stream_future_access_raises() -> None:
    bars = [
        Bar(
            symbol="T",
            ts=i,
            interval_seconds=1,
            open=100,
            high=101,
            low=99,
            close=100 + i,
            volume=10,
            available_time=i,
        )
        for i in range(10)
    ]
    stream = BarStream(bars, "T")
    try:
        stream.closes(3)  # cursor is 0; index 3 is in the future
        raise AssertionError("expected ValueError for future access")
    except ValueError:
        pass


def test_fingerprint_stable_and_sensitive() -> None:
    bars = [
        Bar(
            symbol="T",
            ts=i,
            interval_seconds=1,
            open=100,
            high=101,
            low=99,
            close=100 + i,
            volume=10,
            available_time=i,
        )
        for i in range(10)
    ]
    fp1 = dataset_fingerprint(_series(bars))
    fp2 = dataset_fingerprint(_series(bars))
    bars[3].close += 1.0
    fp3 = dataset_fingerprint(_series(bars))
    assert fp1 == fp2
    assert fp1 != fp3


def test_snapshot_bar_requires_available_time() -> None:
    from pydantic import ValidationError
    from quantlab.data.schemas import SnapshotBar

    try:
        Bar(
            symbol="T",
            ts=1,
            interval_seconds=1,
            open=1,
            high=1,
            low=1,
            close=1,
        )
    except ValidationError:
        raise AssertionError("plain Bar should allow missing available_time") from None
    try:
        SnapshotBar(
            symbol="T",
            ts=1,
            interval_seconds=1,
            open=1,
            high=1,
            low=1,
            close=1,
            available_time=None,  # explicit None triggers the field validator
        )
        raise AssertionError("SnapshotBar must reject a missing available_time")
    except ValidationError:
        pass
