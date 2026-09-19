"""Drawdown analytics for equity curves.

A drawdown is the decline from a running peak, expressed as a positive
fraction of the peak equity: 0 at a new peak, approaching 1 at a total loss.
The battery metric ``max_drawdown`` lives in ``validation.metrics`` and is
re-exposed here without reimplementation.
"""

from __future__ import annotations

import numpy as np

from quantlab.validation.metrics import max_drawdown

__all__ = ["drawdown_series", "max_drawdown"]


def drawdown_series(returns: np.ndarray) -> np.ndarray:
    """Equity drawdown fraction series (positive, 0 at peaks).

    Computes the trailing peak of the cumulative equity and returns
    ``1 - equity / peak`` for every observation, so every new high-water mark
    is exactly 0.  NaN returns propagate into the series (peak tracking
    ignores NaN rows so a late NaN cannot corrupt earlier values).
    """
    r = np.asarray(returns, dtype=float)
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.where(equity > 0, equity, 0.0))
    return 1.0 - equity / np.where(peak > 0, peak, 1.0)
