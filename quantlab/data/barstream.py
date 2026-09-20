"""The look-ahead guard.

A strategy never sees raw bars; it sees a :class:`BarStream`, a cursor that only
moves forward and that hides any observation whose ``available_time`` is after
the decision time. This is the structural enforcement of:

    available_time <= decision_time

Property: calling ``see()`` more than one step "ahead" is impossible because the
cursor only advances via ``advance()`` (called by the engine), not by the strategy.
"""

from __future__ import annotations

import numpy as np

from quantlab.data.schemas import Bar, BarSeries
from quantlab.data.validation import audit_bars


class BarStream:
    """Sequential read access over a single symbol's bars, gated on availability."""

    def __init__(self, series: BarSeries | list[Bar], symbol: str = "") -> None:
        if isinstance(series, BarSeries):
            self.bars = series.bars
            self.symbol = symbol or series.symbol
        else:
            self.bars = series
            self.symbol = symbol or (series[0].symbol if series else "")
        self._cursor = 0  # index of the current decision bar (start inclusive)
        self._closes: np.ndarray | None = None

    # -- engine interface ---------------------------------------------------
    def advance(self) -> int:
        """Advance the decision cursor by one bar. Returns new cursor index."""
        if self._cursor >= len(self.bars):
            raise IndexError("no more bars")
        self._cursor += 1
        return self._cursor

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def total(self) -> int:
        return len(self.bars)

    # -- strategy interface (read-only, availability-gated) ------------------
    def _usable(self, i: int) -> list[Bar]:
        """Bars index <= i whose information was available by then."""
        if i > self._cursor:
            raise ValueError("cannot look ahead: decision cursor has not reached that bar")
        ts_i = self.bars[i].ts
        usable = []
        for b in self.bars[: i + 1]:
            avail = b.available_time if b.available_time is not None else b.ts
            if avail <= ts_i:
                usable.append(b)
        return usable

    def closes(self, i: int | None = None) -> np.ndarray:
        i = self._decide(i)
        return np.asarray([b.close for b in self._usable(i)], dtype=float)

    def returns(self, i: int | None = None) -> np.ndarray:
        closes = self.closes(i)
        if len(closes) < 2:
            return np.array([], dtype=float)
        return np.diff(closes) / closes[:-1]

    def volumes(self, i: int | None = None) -> np.ndarray:
        i = self._decide(i)
        return np.asarray([b.volume for b in self._usable(i)], dtype=float)

    def highs(self, i: int | None = None) -> np.ndarray:
        i = self._decide(i)
        return np.asarray([b.high for b in self._usable(i)], dtype=float)

    def lows(self, i: int | None = None) -> np.ndarray:
        i = self._decide(i)
        return np.asarray([b.low for b in self._usable(i)], dtype=float)

    def opens(self, i: int | None = None) -> np.ndarray:
        i = self._decide(i)
        return np.asarray([b.open for b in self._usable(i)], dtype=float)

    def last_bar(self, i: int | None = None) -> Bar | None:
        """The latest *usable* bar at decision time (not necessarily bar `i`)."""
        i = self._decide(i)
        usable = self._usable(i)
        return usable[-1] if usable else None

    def _decide(self, i: int | None) -> int:
        return self._cursor if i is None else int(i)

    def snapshot(self, i: int | None = None) -> dict:
        """Serialize what the strategy can legitimately see right now."""
        i = self._decide(i)
        usable = self._usable(i)
        return {
            "symbol": self.symbol,
            "decision_index": i,
            "n_usable_bars": len(usable),
            "last_close": usable[-1].close if usable else None,
            "last_ts": usable[-1].ts if usable else None,
        }


def audit_stream(stream: BarStream) -> list[str]:
    """Sanity-guard: replay the stream forward and confirm no future access."""
    violations: list[str] = []
    audit = audit_bars(stream.bars)
    if audit.has_errors:
        violations.extend(i.message for i in audit.errors)
    # walk full series; the snapshot must never expose a bar after cursor
    cursor_tracker = BarStream(stream.bars, stream.symbol)
    for i in range(len(stream.bars)):
        snapshot = cursor_tracker.snapshot(i)
        if snapshot["last_ts"] is not None and snapshot["last_ts"] > cursor_tracker.bars[i].ts:
            violations.append(f"bar {i} exposed information after decision time")
        cursor_tracker.advance()
    return violations
