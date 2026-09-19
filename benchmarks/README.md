# Benchmarks

Cross-validation of Quant Lab against itself and against reference engines.

## Strategy x market benchmark

```bash
python benchmarks/benchmark_strategies.py
```

Runs every benchmark strategy (buy & hold, random, SMA cross, momentum, mean
reversion, breakout, vol targeting) on every synthetic market (random walk,
trend, mean reversion, vol clustering, regime switch, structural break, jump).

The **Random** column is the canary: if randomized signals earn a high Sharpe
here, the research pipeline has a bug.

## External references

- **vectorbt** — cross-validation reference for parameter sweeps and
  walk-forward. Apache 2.0 + Commons Clause / Fair Code licensing: use as a
  reference/benchmark adapter, never copied into this codebase verbatim.
- The `arch` package is an optional validated layer for GARCH-family estimation
  (`quantlab.volatility.garch` falls back to a scipy MLE when absent).
- MLflow is an optional experiment-tracker adapter behind `quantlab.experiments`.