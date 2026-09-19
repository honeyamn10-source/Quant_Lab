# Quant Lab

> **Don't prove your strategy works. Try to prove it doesn't.**

Quant Lab is an open quantitative research and strategy-falsification platform for
detecting backtest overfitting, data leakage, execution assumptions, regime
dependency and fragile trading strategies.

Quant Lab is **not** a system for finding a magical trading strategy. Its purpose is:

> Find trading hypotheses, test them honestly, attempt to destroy them
> statistically, simulate realistic execution, measure uncertainty, document
> failures, and only allow strategies with strong evidence to progress toward
> paper/live testing.

## The ten problems Quant Lab attacks

1. Backtest overfitting
2. False / persistent alpha
3. Market regime instability
4. Slippage and market impact
5. Survivorship / data-quality bias
6. Volatility forecasting
7. Position sizing under estimation uncertainty
8. News / sentiment leakage
9. Correlation breakdown during crises
10. Lack of systematic failed-strategy data

## Core scientific principle

```
Hypothesis
    ↓
Evidence
    ↓
Attempt to falsify
    ↓
Measure uncertainty
    ↓
Document failures
    ↓
Reproduce
```

Not:

```
Search thousands of configurations
    ↓
Find the prettiest chart
    ↓
Call it alpha
```

Every experiment answers a much harder question than *"How profitable was this
strategy historically?"* It answers: **how much evidence exists that this result is
real, how could the result be misleading, under what assumptions does it disappear,
what happens when market conditions change, and what evidence would be required
before risking capital?**

## Repository layout

```text
quantlab/            Python package (engine, validation, execution, regimes...)
datasets/            Raw / research datasets and manifests
docs/                Architecture, methodology, data integrity documentation
examples/            Strategy configs and runnable examples
benchmarks/          Benchmark comparisons against other engines
tests/               unit / integration / statistical / lookahead / synthetic
experiments/         Reproducible experiment outputs (gitignored)
```

## Quick start

```bash
pip install -e ".[dev]"

# Generate a synthetic market with known ground truth
quantlab data generate-synthetic --out datasets/ --seed 7

# Run a benchmark strategy
quantlab backtest examples/strategies/sma_crossover.yaml

# Adversarially stress an experiment
quantlab stress EXP-0001

# This returns PASS / FAIL / INCONCLUSIVE — not "93% confidence"
```

See `docs/architecture.md` and `docs/methodology.md` for details, and
`Makefile` for the standard `make lint`, `make typecheck`, `make test` targets.

## How a strategy survives Quant Lab

A candidate is never called a *"proven profitable strategy."* Quant Lab can only
say that it has survived a specified battery of tests:

- ✓ Point-in-time data audit
- ✓ Look-ahead detection
- ✓ Out-of-sample validation
- ✓ Walk-forward validation
- ✓ Purged CV
- ✓ DSR
- ✓ PBO
- ✓ Parameter perturbation
- ✓ Cost stress
- ✓ Slippage stress
- ✓ Regime testing
- ✓ Crisis testing
- ✓ Bootstrap
- ✓ Monte Carlo
- ✓ Paper trading

Still unknown:
**Future profitability.**

## Status

Alpha. Phases (data integrity, backtesting, experiment registry, statistical
validation, execution modelling, adversarial testing, regimes, volatility,
sizing, sentiment timing, graveyard, CLI) are implemented. See `CHANGELOG.md`
and `docs/architecture.md`.

## License

MIT. See `LICENSE`.