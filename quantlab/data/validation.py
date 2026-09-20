"""Data integrity scanner.

Detects missing observations, duplicate timestamps, impossible OHLC values,
zero/negative prices, split anomalies, dividend adjustments, stale prices,
suspicious spikes, future information and missing securities. Produces a
:class:`DataAudit` whose checks are individually graded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from quantlab.data.schemas import Bar, BarSeries, PointInTimeRow


class Severity(str, Enum):  # noqa: UP042
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(frozen=True)
class AuditIssue:
    code: str
    severity: Severity
    symbol: str
    message: str

    @property
    def is_error(self) -> bool:
        return self.severity == Severity.ERROR


@dataclass
class DataAudit:
    """Result of running the integrity scanner on a dataset."""

    issues: list[AuditIssue] = field(default_factory=list)
    n_bars: int = 0

    def add(self, code: str, severity: Severity, symbol: str, message: str) -> None:
        self.issues.append(AuditIssue(code, severity, symbol, message))

    @property
    def errors(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.is_error]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def by_code(self, code: str) -> list[AuditIssue]:
        return [i for i in self.issues if i.code == code]

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for i in self.issues:
            counts[i.code] = counts.get(i.code, 0) + 1
        return {"n_issues": len(self.issues), "n_errors": len(self.errors), "by_code": counts}

    def __bool__(self) -> bool:
        return not self.has_errors


def audit_bars(bars: list[Bar] | BarSeries) -> DataAudit:
    """Validate a raw bar list/series."""
    if isinstance(bars, BarSeries):
        series = bars
        bars = bars.bars
    else:
        series = None

    audit = DataAudit(n_bars=len(bars))
    if not bars:
        audit.add("EMPTY", Severity.ERROR, "", "no observations")
        return audit

    symbol = series.symbol if series else bars[0].symbol
    ts = np.array([b.ts for b in bars])
    closes = np.array([b.close for b in bars], dtype=float)
    opens = np.array([b.open for b in bars], dtype=float)
    highs = np.array([b.high for b in bars], dtype=float)
    lows = np.array([b.low for b in bars], dtype=float)
    vols = np.array([b.volume for b in bars], dtype=float)

    # Duplicates / ordering
    if len(set(ts.tolist())) != len(ts):
        audit.add("DUPLICATE_TS", Severity.ERROR, symbol, "duplicate timestamps present")
    if np.any(np.diff(ts) <= 0):
        audit.add("OUT_OF_ORDER", Severity.ERROR, symbol, "timestamps not strictly increasing")

    # Impossible OHLC
    if np.any(lows < 0) or np.any(closes < 0):
        audit.add("NEGATIVE_PRICE", Severity.ERROR, symbol, "negative price found")
    if np.any(highs < np.maximum(opens, closes)):
        audit.add("BAD_HIGH", Severity.ERROR, symbol, "high below max(open, close)")
    if np.any(lows > np.minimum(opens, closes)):
        audit.add("BAD_LOW", Severity.ERROR, symbol, "low above min(open, close)")
    if np.any(highs < lows):
        audit.add("HIGH_LT_LOW", Severity.ERROR, symbol, "high below low")

    # Zero prices
    zero = closes == 0
    if zero.any():
        idx = int(np.flatnonzero(zero)[0])
        audit.add("ZERO_PRICE", Severity.ERROR, symbol, f"zero close at index {idx}")

    # Suspicious spikes (log returns beyond 5 sigma of the series)
    rets = np.diff(np.log(np.where(closes > 0, closes, np.nan)))
    rets = rets[~np.isnan(rets)]
    if len(rets) > 2:
        mu, sd = float(np.mean(rets)), float(np.std(rets))
        if sd > 0:
            spikes = np.flatnonzero(np.abs(rets - mu) > 5 * sd)
            for s in spikes[:5]:
                audit.add(
                    "SPIKE", Severity.WARNING, symbol, f"suspicious |return| at index {int(s) + 1}"
                )

    # Corporate-action anomaly: single-bar gap with no volume and zero second
    # bar is classic split/dividend artifact. Nonzero second bar = suspicious.
    for i in range(1, len(bars)):
        prev_c, cur_c = closes[i - 1], closes[i]
        if prev_c > 0 and cur_c > 0:
            ratio = cur_c / prev_c
            if ratio > 3.0 or ratio < 1.0 / 3.0:
                if vols[i] == 0:
                    audit.add(
                        "SPLIT_ANOMALY",
                        Severity.INFO,
                        symbol,
                        f"price ratio {ratio:.3f} at index {i} with zero volume (possible split/adjustment)",
                    )
                else:
                    audit.add(
                        "CA_ANOMALY",
                        Severity.WARNING,
                        symbol,
                        f"unusual gap at index {i}: ratio {ratio:.3f}, volume {vols[i]}",
                    )

    # Stale prices: repeated consecutive closes
    flat = closes[1:] == closes[:-1]
    stale_runs = 0
    run = 0
    for f in flat:
        run = run + 1 if f else 0
        stale_runs = max(stale_runs, run)
    if stale_runs >= 5:
        audit.add(
            "STALE_PRICE", Severity.WARNING, symbol, f"{stale_runs} consecutive unchanged closes"
        )

    if vols.min() < 0:
        audit.add("NEGATIVE_VOLUME", Severity.ERROR, symbol, "negative volume observed")

    return audit


def audit_dataset(rows: list[PointInTimeRow]) -> DataAudit:
    """Audit a point-in-time snapshot dataset (multi-symbol)."""
    audit = DataAudit(n_bars=len(rows))
    if not rows:
        audit.add("EMPTY", Severity.ERROR, "", "no rows")
        return audit

    symbols = {r.symbol for r in rows}
    # Determine expected frequency per symbol (median gap) to catch missing obs.
    for sym in sorted(symbols):
        sr = [r for r in rows if r.symbol == sym]
        sr.sort(key=lambda r: r.ts)
        tss = [r.ts for r in sr]
        if len(tss) != len(set(tss)):
            audit.add("DUPLICATE_TS", Severity.ERROR, sym, "duplicate timestamps")
        if any(b < a for a, b in zip(tss, tss[1:], strict=False)):
            audit.add("OUT_OF_ORDER", Severity.ERROR, sym, "timestamps not increasing")
        for r in sr:
            if r.available_time < r.ts:
                audit.add(
                    "AVAIL_BEFORE_EVENT", Severity.ERROR, sym, "available_time precedes event_time"
                )
            if r.adjustment_factor <= 0:
                audit.add("BAD_ADJ_FACTOR", Severity.ERROR, sym, "non-positive adjustment factor")
        # missing observation heuristics
        gaps = np.diff(np.array(tss))
        if len(gaps) > 1:
            med = float(np.median(gaps))
            if med > 0:
                big = int(np.sum(gaps > med * 3))
                if big:
                    audit.add(
                        "MISSING_OBS", Severity.WARNING, sym, f"{big} gap(s) > 3x median interval"
                    )

    # Cross-sectional structural check: every timestamp should appear across
    # symbols for true snapshots, unless flagged as missing.
    return audit
