"""Benchmark suite: run all benchmark strategies across all synthetic markets
and print a table. The RandomStrategy column is the canary test: if random
signals "discover alpha", the research pipeline has a bug.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")


from quantlab.backtest.engine import BacktestConfig, BacktestEngine
from quantlab.strategies.benchmark import create_strategy
from quantlab.synthetic.markets import available_processes, generate
from quantlab.validation.metrics import sharpe_ratio

BENCHMARK_NAMES = [
    "buy_and_hold",
    "random",
    "sma_crossover",
    "momentum",
    "mean_reversion",
    "breakout",
    "vol_target",
]


def main() -> int:
    print("Benchmark strategies x synthetic markets (annualized Sharpe)")
    header = f"{'process':<16}" + "".join(f"{n:>14}" for n in BENCHMARK_NAMES)
    print(header)
    print("-" * len(header))
    for process in available_processes():
        row = f"{process:<16}"
        for name in BENCHMARK_NAMES:
            series = generate(process, n=504, seed=1)
            strat = create_strategy(
                name,
                {"seed": 2}
                if name == "random"
                else {"fast": 10, "slow": 30}
                if name == "sma_crossover"
                else None,
            )
            # vol_target is a scaled-leverage strategy; let its 2.0 weight band through,
            # otherwise the default 1.0 portfolio position cap rejects every order.
            max_w = 2.0 if name == "vol_target" else 1.0
            result = BacktestEngine(BacktestConfig(max_position_weight=max_w)).run(
                strat, {series.symbol: series}
            )
            row += f"{sharpe_ratio(result.returns):>14.2f}"
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
