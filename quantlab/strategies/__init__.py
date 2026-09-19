"""Strategy module: API, indicators, benchmarks and factory."""

from quantlab.strategies.base import Signal, Strategy
from quantlab.strategies.benchmark import (
    ALL_STRATEGIES,
    Breakout,
    BuyAndHold,
    MeanReversion,
    MeanReversionRSI,
    Momentum,
    RandomStrategy,
    SmaCrossover,
    VolTarget,
    create_strategy,
    list_strategies,
)
from quantlab.strategies.indicators import atr, ema, rsi, sma, zscore
from quantlab.strategies.pairs import PairsSpread, ols_beta

__all__ = [
    "ALL_STRATEGIES",
    "Breakout",
    "BuyAndHold",
    "MeanReversion",
    "MeanReversionRSI",
    "Momentum",
    "PairsSpread",
    "RandomStrategy",
    "Signal",
    "SmaCrossover",
    "Strategy",
    "VolTarget",
    "atr",
    "create_strategy",
    "ema",
    "list_strategies",
    "ols_beta",
    "rsi",
    "sma",
    "zscore",
]
