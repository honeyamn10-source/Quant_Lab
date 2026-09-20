"""Canonical market-data and point-in-time schemas.

Every observation carries event/availability timestamps so look-ahead bias can
be ruled out mechanically. Snapshot observations for multi-asset research use
:class:`SnapshotBar` so a single time index works across the universe.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, field_validator

TIMESTAMP_MESSAGE = "Time must be monotonically non-decreasing"


def _is_sortable_ts(ts: int) -> bool:
    return isinstance(ts, int) and ts >= 0


class Bar(BaseModel):
    """A single OHLCV observation."""

    symbol: str
    ts: int  # event time, epoch ms
    interval_seconds: int = Field(ge=1)
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    available_time: int | None = Field(default=None, ge=0)  # first time info was usable
    source: str = "unknown"
    source_version: str = "0"

    @field_validator("ts")
    @classmethod
    def _ts(cls, v: int) -> int:
        if not _is_sortable_ts(v):
            raise ValueError(f"negative timestamp not allowed: {v}")
        return v

    @field_validator("high", "low", "open", "close")
    @classmethod
    def _nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"negative price not allowed: {v}")
        return v


class SnapshotBar(Bar):
    """A bar that is guaranteed to embed `available_time`.

    This is the point-in-time default when consuming research datasets.
    """

    @field_validator("available_time")
    @classmethod
    def _avail(cls, v: int | None) -> int:
        if v is None:
            raise ValueError("SnapshotBar requires an available_time")
        return v

    @field_validator("available_time")
    @classmethod
    def _avail_gte_ts(cls, v: int, info) -> int:
        ts = info.data.get("ts")
        if ts is not None and v < ts:
            raise ValueError(f"available_time {v} is before event_time {ts}")
        return v


@dataclass
class BarSeries:
    """An ordered list of bars with convenience helpers."""

    symbol: str
    bars: list[Bar] = field(default_factory=list)
    interval_seconds: int = 60 * 60 * 24

    def __post_init__(self) -> None:
        self._bars_are_sorted()

    def _bars_are_sorted(self) -> None:
        ts = [b.ts for b in self.bars]
        if ts != sorted(ts):
            raise ValueError("Bars must be sorted by timestamp")

    def append(self, bar: Bar) -> None:
        if self.bars and bar.ts <= self.bars[-1].ts:
            raise ValueError("cannot append bar out of chronological order")
        self.bars.append(bar)

    def __len__(self) -> int:
        return len(self.bars)

    def __getitem__(self, i: int) -> Bar:
        return self.bars[i]

    def closes(self) -> list[float]:
        return [b.close for b in self.bars]

    def returns(self) -> list[float]:
        """Simple period-over-period returns from closes (no corporate-action
        adjustments; use bar.available_time/audit for adjusted histories)."""
        closes = self.closes()
        if len(closes) < 2:
            return []
        return [closes[t] / closes[t - 1] - 1.0 for t in range(1, len(closes))]

    def times(self) -> list[int]:
        return [b.ts for b in self.bars]

    def volatilities(self) -> list[float]:
        return [b.volume for b in self.bars]

    def trimmed(self, start: int, end: int) -> BarSeries:
        return BarSeries(self.symbol, self.bars[start:end], self.interval_seconds)


@dataclass(frozen=True)
class DatasetManifest:
    """Identifier + provenance for a research dataset."""

    name: str
    version: str
    fingerprint: str
    n_rows: int
    n_symbols: int
    start_ts: int
    end_ts: int
    hash_algorithm: str = "sha256"
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_rows(cls, rows: list[PointInTimeRow]) -> DatasetManifest:
        fingerprint = dataset_fingerprint_pointintime(rows)
        tss = [r.ts for r in rows]
        return cls(
            name="dataset",
            version="1",
            fingerprint=fingerprint,
            n_rows=len(rows),
            n_symbols=len({r.symbol for r in rows}),
            start_ts=min(tss) if tss else 0,
            end_ts=max(tss) if tss else 0,
        )


class PointInTimeRow(BaseModel):
    """The canonical research observation.

    Records provenance fields that make audit trails possible.
    """

    symbol: str
    exchange: str = ""
    ts: int = Field(ge=0)  # event_time
    available_time: int = Field(ge=0)
    ingestion_time: int = Field(ge=0)
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    adjustment_factor: float = 1.0
    source: str = "unknown"
    source_version: str = "0"
    quality_flags: list[str] = Field(default_factory=list)

    @field_validator("available_time")
    @classmethod
    def _not_future(cls, v: int, info) -> int:
        ts = info.data.get("ts")
        if ts is not None and v < ts:
            raise ValueError(f"available_time {v} precedes event_time {ts}")
        return v


def dataset_fingerprint_pointintime(rows: list[PointInTimeRow]) -> str:
    """Content hash of point-in-time rows (stable, byte-exact)."""
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: (r.symbol, r.ts)):
        h.update(
            f"{r.symbol}|{r.ts}|{r.available_time}|{r.open}|{r.high}|{r.low}|{r.close}|{r.volume}|{r.adjustment_factor}".encode()
        )
    return h.hexdigest()
