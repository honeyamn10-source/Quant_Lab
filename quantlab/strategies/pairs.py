"""Pairs trading strategy for cointegrated pairs (benchmark + hypothesis)."""

from __future__ import annotations

import numpy as np

from quantlab.data.barstream import BarStream
from quantlab.strategies.base import Signal, Strategy


class PairsSpread(Strategy):
    """Fades a z-scored spread between two streams (pairs).

    Requires streams dict with exactly two symbols: lexicographically smaller
    symbol is leg A (long leg of the spread), the other is leg B.
    """

    name = "pairs"

    DEFAULT = {"window": 40, "z_enter": 1.5, "z_exit": 0.0, "leverage": 1.0}

    def __init__(self, params: dict | None = None) -> None:
        super().__init__({**self.DEFAULT, **(params or {})})

    def warmup(self) -> int:
        return int(self.params["window"]) + 1

    def on_bar(self, streams: dict[str, BarStream], i: int) -> list[Signal] | None:
        syms = sorted(streams)
        if len(syms) != 2:
            raise ValueError("pairs strategy requires exactly two streams")
        a, b = syms
        pa = streams[a].closes(i)
        pb = streams[b].closes(i)
        n = min(len(pa), len(pb))
        if n < 2:
            return None
        spread = pb[-n:] - pa[-n:]
        win = int(self.params["window"])
        seg = spread[-win:]
        if len(seg) < 5 or float(np.std(seg)) <= 0:
            return None
        z = (seg[-1] - float(np.mean(seg))) / float(np.std(seg))
        lev = float(self.params["leverage"])
        z_enter, z_exit = float(self.params["z_enter"]), float(self.params["z_exit"])
        if z > z_enter:
            # spread wide -> expect convergence: short B, long A
            wa, wb = lev, -lev
        elif z < -z_enter:
            wa, wb = -lev, lev
        elif abs(z) < z_exit:  # spread faded to fair -> close both legs
            wa = wb = 0.0
        else:
            return None
        ts = streams[a].bars[i].ts
        return [
            Signal(symbol=a, target_weight=wa, ts=ts, reason="pairs", params={"z": z}),
            Signal(symbol=b, target_weight=wb, ts=ts, reason="pairs", params={"z": z}),
        ]


def ols_beta(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Least-squares hedge ratio and R2 for b regressed on a."""
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("need equal-length series of >= 3 points")
    design = np.vstack([a, np.ones(len(a))]).T
    coef, _, _, _ = np.linalg.lstsq(design, b, rcond=None)
    beta, alpha = float(coef[0]), float(coef[1])
    pred = beta * a + alpha
    ss_res = float(np.sum((b - pred) ** 2))
    ss_tot = float(np.sum((b - np.mean(b)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2


__all__ = ["PairsSpread", "ols_beta"]
