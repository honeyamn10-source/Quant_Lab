"""Dataset fingerprinting.

Hashes bars/rows by content so an experiment can always verify that the exact
rows it consumed are reproducible. Hash ignores ordering metadata that doesn't
change information content but preserves total order.
"""

from __future__ import annotations

import hashlib
import json

from quantlab.data.schemas import Bar, BarSeries, PointInTimeRow


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256(blob: str) -> str:
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def fingerprint_bars(bars: list[Bar]) -> str:
    rows = [
        {
            "symbol": b.symbol,
            "ts": b.ts,
            "o": b.open,
            "h": b.high,
            "l": b.low,
            "c": b.close,
            "v": b.volume,
            "av": b.available_time,
            "src": b.source,
            "ver": b.source_version,
        }
        for b in sorted(bars, key=lambda b: (b.ts, b.symbol))
    ]
    return _sha256(_canonical(rows))


def fingerprint_series(series: BarSeries) -> str:
    return fingerprint_bars(series.bars)


def fingerprint_pointintime(rows: list[PointInTimeRow]) -> str:
    clean = [
        {
            "symbol": r.symbol,
            "ts": r.ts,
            "av": r.available_time,
            "ing": r.ingestion_time,
            "o": r.open,
            "h": r.high,
            "l": r.low,
            "c": r.close,
            "v": r.volume,
            "adj": r.adjustment_factor,
            "flags": sorted(r.quality_flags),
        }
        for r in sorted(rows, key=lambda r: (r.symbol, r.ts))
    ]
    return _sha256(_canonical(clean))


def dataset_fingerprint(payload) -> str:
    """Fingerprint anything serializable (bars, series, rows, dicts, arrays)."""
    if isinstance(payload, BarSeries):
        return fingerprint_series(payload)
    if isinstance(payload, Bar):
        return fingerprint_bars([payload])
    if isinstance(payload, PointInTimeRow):
        return fingerprint_pointintime([payload])
    if isinstance(payload, list):
        if all(isinstance(x, Bar) for x in payload):
            return fingerprint_bars(payload)
        if all(isinstance(x, PointInTimeRow) for x in payload):
            return fingerprint_pointintime(payload)
    return _sha256(_canonical(payload))
