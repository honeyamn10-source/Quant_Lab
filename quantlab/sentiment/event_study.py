"""Event-study forward returns with an explicit decision-delay buffer.

A signal is produced at the end of the bar in which it fires. Any honest
event study must let the signal be acted upon without capturing the same bar's
move, otherwise the study silently trades on the decade of the signal itself.
The ``delay_bars`` parameter is the anti-leakage measure; the leaky variant is
kept around purely to demonstrate how much return leakage inflates.
"""

from __future__ import annotations

import numpy as np


def _is_inflated(leaked_mean: float, delayed_mean: float) -> bool:
    """Return True when the leaked estimate materially exceeds the delayed one.

    Material here means at least a 2x multiple or a 25% relative improvement
    over the delayed (honest) estimate. A delayed estimate of zero that the
    leaked variant can beat is automatically considered inflated.
    """
    if leaked_mean <= delayed_mean:
        return False
    if delayed_mean == 0.0:
        return True
    return leaked_mean >= 2.0 * delayed_mean or (leaked_mean - delayed_mean) / abs(delayed_mean) >= 0.25


def event_study(
    closes: np.ndarray,
    event_indices: list[int],
    horizons: list[int],
    delay_bars: int = 1,
) -> dict:
    """Measure post-signal forward returns per horizon with a decision delay.

    The signal is available at the end of bar ``i``. Returns are measured from
    the close ``delay_bars`` bars later so the signal cannot harvest its own bar::

        r(i, delay, h) = close[i + delay_bars + h] / close[i + delay_bars] - 1

    Events whose measurement window runs past the end of ``closes`` are skipped.
    Returns a ``{"delay_bars", "results"}`` dict where ``results`` is one record
    ``{"horizon", "mean_return", "median_return", "hit_rate", "n_events",
    "std_error"}`` per horizon; ``std_error`` is the unbiased standard error of
    the mean.
    """
    prices = np.asarray(closes, dtype=float)
    results: list[dict] = []
    for horizon in horizons:
        returns: list[float] = []
        for event in event_indices:
            start = event + delay_bars
            end = start + horizon
            if start < 0 or end >= prices.size:
                continue
            returns.append(prices[end] / prices[start] - 1.0)
        values = np.asarray(returns, dtype=float)
        count = int(values.size)
        mean = float(values.mean()) if count else 0.0
        median = float(np.median(values)) if count else 0.0
        hit_rate = float((values > 0).mean()) if count else 0.0
        std_error = float(values.std(ddof=1) / np.sqrt(count)) if count > 1 else 0.0
        results.append(
            {
                "horizon": int(horizon),
                "mean_return": mean,
                "median_return": median,
                "hit_rate": hit_rate,
                "n_events": count,
                "std_error": std_error,
            }
        )
    return {"delay_bars": delay_bars, "results": results}


def leaked_event_study(
    closes: np.ndarray,
    event_indices: list[int],
    horizons: list[int],
    delay_bars: int = 0,
) -> dict:
    """Compute returns with no decision delay, deliberately leaking the signal.

    Same interface as :func:`event_study` but the signal is allowed to capture
    the very next bar (``delay_bars=0``). Used only to demonstrate how leakage
    inflates event-study returns. Do not use for evaluation.
    """
    return event_study(closes, event_indices, horizons, delay_bars=delay_bars)


def compare_leakage(
    closes: np.ndarray,
    events: list[int],
    horizons: list[int],
) -> dict:
    """Compare the leaky and delayed event studies and flag inflated horizons.

    Runs :func:`leaked_event_study` (delay 0) and :func:`event_study`
    (delay 1) and reports the leaked-vs-delayed mean return per horizon. A
    horizon is flagged ``"LEAKAGE_SUSPICIOUS"`` whenever the un-delayed
    estimate materially exceeds the delayed one (>= 2x or >= 25% relative).
    """
    leaked = leaked_event_study(closes, events, horizons)
    delayed = event_study(closes, events, horizons)
    rows: list[dict] = []
    flags: list[str] = []
    for leak_row, delay_row in zip(leaked["results"], delayed["results"], strict=True):
        leaked_mean = leak_row["mean_return"]
        delayed_mean = delay_row["mean_return"]
        inflation = leaked_mean - delayed_mean
        flag = "LEAKAGE_SUSPICIOUS" if _is_inflated(leaked_mean, delayed_mean) else None
        rows.append(
            {
                "horizon": leak_row["horizon"],
                "leaked_mean_return": leaked_mean,
                "delayed_mean_return": delayed_mean,
                "inflation": inflation,
                "flag": flag,
            }
        )
        if flag is not None:
            flags.append(flag)
    return {
        "delay_bars": {"leaked": 0, "delayed": 1},
        "results": rows,
        "flags": flags,
    }
