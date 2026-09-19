"""Benchmark strategies.

Deliberately simple so every part of the pipeline can be audited. The random
strategy matters most: if Quant Lab routinely "discovers alpha" in randomized
signals, a research-pipeline bug exists.
"""

from __future__ import annotations

import math

import numpy as np

from quantlab.data.barstream import BarStream
from quantlab.strategies.base import Signal, Strategy
from quantlab.strategies.indicators import rsi, sma, zscore
from quantlab.strategies.pairs import PairsSpread


class BuyAndHold(Strategy):
    name = "buy_and_hold"
    DEFAULT: dict[str, float] = {}

    def on_bar(self, streams: dict[str, BarStream], i: int) -> list[Signal]:
        sym = next(iter(streams))
        return [Signal(symbol=sym, target_weight=1.0, ts=streams[sym].bars[i].ts, reason="hold")]


class RandomStrategy(Strategy):
    """Seeded random positions. Uses its own RNG — never price data."""

    name = "random"
    DEFAULT: dict[str, float] = {"seed": 42.0}

    def __init__(self, params: dict | None = None) -> None:
        super().__init__({**self.DEFAULT, **(params or {})})
        self._rng = np.random.default_rng(int(self.params.get("seed", 42)))
        self._positions: dict[int, float] = {}

    def on_bar(self, streams: dict[str, BarStream], i: int) -> list[Signal] | None:
        if i not in self._positions:
            w = self._rng.choice([-1.0, 0.0, 1.0], p=[0.25, 0.5, 0.25])
            self._positions[i] = float(w)
        sym = next(iter(streams))
        target = self._positions[i]
        if target == 0.0:
            return None
        return [
            Signal(symbol=sym, target_weight=target, ts=streams[sym].bars[i].ts, reason="random")
        ]


class SmaCrossover(Strategy):
    name = "sma_crossover"
    DEFAULT = {"fast": 20, "slow": 50}

    def __init__(self, params: dict | None = None) -> None:
        super().__init__({**self.DEFAULT, **(params or {})})

    def warmup(self) -> int:
        return int(self.params["slow"]) + 1

    def on_bar(self, streams: dict[str, BarStream], i: int) -> list[Signal] | None:
        sym = next(iter(streams))
        closes = streams[sym].closes(i)
        fast = sma(closes, int(self.params["fast"]))
        slow = sma(closes, int(self.params["slow"]))
        if math.isnan(fast[-1]) or math.isnan(slow[-1]):
            return None
        w = 1.0 if fast[-1] > slow[-1] else -1.0
        return [
            Signal(
                symbol=sym,
                target_weight=w,
                ts=streams[sym].bars[i].ts,
                reason="sma_cross",
                params={"fast": fast[-1], "slow": slow[-1]},
            )
        ]


class Momentum(Strategy):
    name = "momentum"
    DEFAULT = {"lookback": 20, "threshold": 0.0}

    def __init__(self, params: dict | None = None) -> None:
        super().__init__({**self.DEFAULT, **(params or {})})

    def warmup(self) -> int:
        return int(self.params["lookback"]) + 1

    def on_bar(self, streams: dict[str, BarStream], i: int) -> list[Signal] | None:
        sym = next(iter(streams))
        closes = streams[sym].closes(i)
        lb = int(self.params["lookback"])
        if len(closes) <= lb or closes[-lb - 1] <= 0:
            return None
        ret = closes[-1] / closes[-lb - 1] - 1.0
        thr = float(self.params["threshold"])
        if ret > thr:
            w = 1.0
        elif ret < -thr:
            w = -1.0
        else:
            return None
        return [
            Signal(
                symbol=sym,
                target_weight=w,
                ts=streams[sym].bars[i].ts,
                reason="momentum",
                params={"ret": ret},
            )
        ]


class MeanReversion(Strategy):
    name = "mean_reversion"
    DEFAULT = {"window": 20, "z_enter": 1.0, "z_exit": 0.0}

    def __init__(self, params: dict | None = None) -> None:
        super().__init__({**self.DEFAULT, **(params or {})})

    def warmup(self) -> int:
        return int(self.params["window"]) + 1

    def on_bar(self, streams: dict[str, BarStream], i: int) -> list[Signal] | None:
        sym = next(iter(streams))
        closes = streams[sym].closes(i)
        z = zscore(closes, int(self.params["window"]))
        if len(z) == 0 or math.isnan(z[-1]):
            return None
        z_enter, z_exit = float(self.params["z_enter"]), float(self.params["z_exit"])
        if z[-1] > z_enter:  # stretched above -> fade it short
            w = -1.0
        elif z[-1] < -z_enter:
            w = 1.0
        elif abs(z[-1]) < z_exit:  # faded back toward the mean -> exit
            w = 0.0
        else:  # between entry and exit bands: hold what we have
            return None
        return [
            Signal(
                symbol=sym,
                target_weight=w,
                ts=streams[sym].bars[i].ts,
                reason="mean_revert",
                params={"z": z[-1]},
            )
        ]


class Breakout(Strategy):
    name = "breakout"
    DEFAULT = {"lookback": 20}

    def __init__(self, params: dict | None = None) -> None:
        super().__init__({**self.DEFAULT, **(params or {})})

    def warmup(self) -> int:
        return int(self.params["lookback"]) + 1

    def on_bar(self, streams: dict[str, BarStream], i: int) -> list[Signal] | None:
        sym = next(iter(streams))
        lb = int(self.params["lookback"])
        highs = streams[sym].highs(i)
        lows = streams[sym].lows(i)
        closes = streams[sym].closes(i)
        if len(highs) < lb:
            return None
        c = closes[-1]
        if c > float(np.max(highs[-lb:-1])):
            w = 1.0
        elif c < float(np.min(lows[-lb:-1])):
            w = -1.0
        else:
            return None
        return [
            Signal(
                symbol=sym,
                target_weight=w,
                ts=streams[sym].bars[i].ts,
                reason="breakout",
                params={
                    "high_band": float(np.max(highs[-lb:-1])),
                    "low_band": float(np.min(lows[-lb:-1])),
                },
            )
        ]


class VolTarget(Strategy):
    name = "vol_target"
    DEFAULT = {"target_vol": 0.20, "window": 20, "max_weight": 2.0}

    def __init__(self, params: dict | None = None) -> None:
        super().__init__({**self.DEFAULT, **(params or {})})

    def warmup(self) -> int:
        return int(self.params["window"]) + 1

    def on_bar(self, streams: dict[str, BarStream], i: int) -> list[Signal]:
        sym = next(iter(streams))
        rets = streams[sym].returns(i)
        win = int(self.params["window"])
        rw = rets[-win:] if len(rets) >= win else rets
        real_vol = float(np.std(rw)) * math.sqrt(252) if len(rw) > 2 else float("nan")
        target = float(self.params["target_vol"])
        cap = float(self.params["max_weight"])
        w = min(cap, target / real_vol) if real_vol and real_vol > 0 else 0.0
        return [
            Signal(
                symbol=sym,
                target_weight=w,
                ts=streams[sym].bars[i].ts,
                reason="vol_target",
                params={"realized": real_vol},
            )
        ]


class MeanReversionRSI(Strategy):
    name = "mean_reversion_rsi"
    DEFAULT = {"period": 14, "low": 30.0, "high": 70.0}

    def __init__(self, params: dict | None = None) -> None:
        super().__init__({**self.DEFAULT, **(params or {})})

    def warmup(self) -> int:
        return int(self.params["period"]) + 1

    def on_bar(self, streams: dict[str, BarStream], i: int) -> list[Signal] | None:
        sym = next(iter(streams))
        closes = streams[sym].closes(i)
        r = rsi(closes, int(self.params["period"]))
        if len(r) == 0 or math.isnan(r[-1]):
            return None
        if r[-1] < float(self.params["low"]):
            w = 1.0
        elif r[-1] > float(self.params["high"]):
            w = -1.0
        else:
            return None
        return [
            Signal(
                symbol=sym,
                target_weight=w,
                ts=streams[sym].bars[i].ts,
                reason="rsi_fade",
                params={"rsi": r[-1]},
            )
        ]


ALL_STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls
    for cls in (
        BuyAndHold,
        RandomStrategy,
        SmaCrossover,
        Momentum,
        MeanReversion,
        Breakout,
        VolTarget,
        MeanReversionRSI,
        PairsSpread,
    )
}


def create_strategy(name: str, params: dict | None = None) -> Strategy:
    if name not in ALL_STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(ALL_STRATEGIES)}")
    return ALL_STRATEGIES[name](params)


def list_strategies() -> list[str]:
    return sorted(ALL_STRATEGIES)
