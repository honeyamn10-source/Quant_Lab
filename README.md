# Quant Lab

> **Don't prove your strategy works. Try to prove it doesn't.**

Quant Lab is an open quantitative research and strategy-falsification platform
for detecting backtest overfitting, data leakage, execution assumptions, regime
dependency and fragile trading strategies.

It is **not** a system for finding a magical trading strategy. Its purpose is to
find trading hypotheses, test them honestly, attempt to destroy them
statistically, simulate realistic execution, measure uncertainty, document
failures, and only allow strategies with strong evidence to progress toward
paper/live testing.

Each experiment answers a harder question than *"How profitable was this
strategy historically?"*: how much evidence exists that this result is real, how
could it be misleading, under what assumptions does it disappear, what happens
when market conditions change, and what evidence would be required before
risking capital?

## Features

- **Synthetic markets with known ground truth** — random walk, trend, mean
  reversion, volatility clustering, regime switch, structural break, jump and
  cointegrated pairs. If the pipeline finds "alpha" in a seeded random walk,
  it is broken, not inspired.
- **Point-in-time data model** — every bar carries an `available_time`; strategies
  only ever see a guarded `BarStream`, making future data inaccessible by
  construction.
- **Event-driven backtester** — market/limit/stop orders, partial fills,
  rejections, position limits, commissions, financing, dividends, configurable
  execution delay.
- **Statistical falsification** — PSR, Deflated Sharpe Ratio, CSCV/PBO, purged
  CV, walk-forward, IID/block/stationary bootstrap, Monte Carlo.
- **Execution reality** — spread/slippage/market-impact/latency cost model with
  four profiles and capacity estimation.
- **Adversarial lab** — parameter, timing, cost and universe perturbation with a
  robustness rating.
- **Regimes & volatility** — rule-based/GMM/HMM/ensemble regimes; historical,
  EWMA, HAR, realized, GARCH/EGARCH/GJR volatility.
- **Sizing** — fixed fraction, vol targeting, Kelly/fractional Kelly, CVaR,
  robust sizing with corrupted-edge evaluation.
- **Sentiment / event timing** — timestamp model with leakage guards.
- **Experiment ledger & graveyard** — permanent, immutable records of every run,
  including the failures. Failed strategies are evidence, never deleted.
- **CLI, reports and API** — reproducible `quantlab` commands, self-contained
  HTML validation reports, and a read-only FastAPI research dashboard.

## Requirements

- Python **3.12+**
- Core dependencies are minimal: `numpy`, `pydantic`, `PyYAML`.
- Optional feature groups (see **Installation**): ML/statistics, tracking,
  dataframe storage, web dashboard.

## Installation

```bash
# core only
pip install "."

# full development environment (tests, lint, typecheck, build)
pip install -e ".[dev]"

# optional feature groups
pip install -e ".[ml]"       # scipy, scikit-learn, statsmodels, arch, optuna
pip install -e ".[tracking]" # mlflow
pip install -e ".[storage]"  # pandas, polars, duckdb, pyarrow
pip install -e ".[web]"      # fastapi, uvicorn, jinja2, plotly
```

Verify the install:

```bash
quantlab --version
quantlab --strategies
```

> **Reproducibility note.** Statistical results are only trustworthy if the
> environment is pinned. Every experiment records its environment (git commit,
> Python version, seed) in the ledger; see `docs/reproducibility.md`.

## Quick start

```bash
# 1. Generate a synthetic market with known ground truth
quantlab data generate-synthetic --out datasets/ --seed 7 --symbols SYN

# 2. Audit and score the data
quantlab data audit --file datasets/SYN.json
quantlab data quality --file datasets/SYN.json

# 3. Run a benchmark strategy
quantlab backtest examples/strategies/sma_crossover.yaml

# 4. Run a full experiment: backtest + validation battery + ledger + report
quantlab run-experiment examples/strategies/sma_crossover.yaml

# 5. Reproduce it later, from the ledger alone
quantlab reproduce EXP-0001
```

Start with `examples/strategies/`, which contain `momentum.yaml`,
`mean_reversion.yaml`, `sma_crossover.yaml`, `pairs.yaml`, `random.yaml`, and
their ground-truth notes.

## Command reference

Every command is deterministic: same seed, same data, same result.

| Command | Purpose |
| --- | --- |
| `quantlab data generate-synthetic [--out DIR] [--process NAME] [--seed N] [--symbols A,B]` | Generate a synthetic market series |
| `quantlab data audit --file FILE` | Point-in-time audit (OHLCV rules, monotonicity, spikes) |
| `quantlab data quality --file FILE [--survivorship HINT]` | Quality score 0-100 with risk dimensions |
| `quantlab backtest CONFIG.yaml` | Run a config and print metrics |
| `quantlab run-experiment CONFIG.yaml [--report-dir DIR] [--ledger DIR]` | Backtest + validation + ledger + report |
| `quantlab validate RESULTS.json` | Run the validation battery on a results file |
| `quantlab stress EXPERIMENT_ID` | Adversarial perturbation + robustness rating |
| `quantlab regimes FILE.json [--seed N]` | Regime decomposition of a series |
| `quantlab volatility FILE.json` | Benchmark volatility models on a series |
| `quantlab report EXPERIMENT_ID` | Write the HTML validation report |
| `quantlab graveyard list` | List recorded failed experiments |
| `quantlab reproduce EXPERIMENT_ID [--ledger DIR]` | Re-run an experiment from its ledger record |
| `quantlab sizing [--simulate]` | Position-sizing quick checks |

## Configuration

A run config is a YAML file consumed by `backtest`, `run-experiment`, `stress`
and `reproduce`. Example (`examples/strategies/sma_crossover.yaml`):

```yaml
name: sma_crossover_on_random_walk
strategy: sma_crossover
strategy_params:
  fast: 10
  slow: 30
data:
  symbols: ["SYN"]
  process: random_walk
  n_bars: 504
  seed: 7
initial_cash: 1000000.0
execution_delay: 1
execution_mode: next_open
profile: normal
max_leverage: 2.0
max_position_weight: 1.0
rebalance_tolerance: 0.005
commission_bps: 5.0
participation_cap: 0.10
borrow_rate_year: 0.0
seed: 0
notes: "Baseline: SMA cross on a pure random walk. Expect near-zero edge — not 'alpha'."
```

Available strategies (see `quantlab --strategies`): `breakout`, `buy_and_hold`,
`mean_reversion`, `mean_reversion_rsi`, `momentum`, `pairs`, `random`,
`sma_crossover`, `vol_target`.

## Project layout

```text
quantlab/            Python package (engine, validation, execution, regimes...)
datasets/            Raw / research datasets and manifests (gitignored)
docs/                Architecture, methodology, data integrity documentation
examples/            Strategy configs and runnable examples
benchmarks/          Benchmark comparisons against other engines
tests/               unit / integration / statistical / lookahead / synthetic
experiments/         Reproducible experiment outputs (gitignored)
```

## How a strategy survives Quant Lab

A candidate is never called a *"proven profitable strategy"*. Quant Lab can only
say it has survived a specified battery of tests:

- Point-in-time data audit
- Look-ahead detection
- Out-of-sample and walk-forward validation
- Purged CV
- Deflated Sharpe Ratio
- Probability of Backtest Overfitting
- Parameter / cost / slippage perturbation
- Regime and crisis testing
- Bootstrap and Monte Carlo
- Paper trading

Still unknown: **future profitability.**

## Development

```bash
pip install -e ".[dev]"

make lint          # ruff
make format        # ruff check --fix + ruff format
make typecheck     # mypy
make test          # full pytest suite
make test-lookahead # pytest -m lookahead
make cov           # coverage report
make build         # sdist + wheel
```

Tests live under `tests/{unit,integration,statistical,lookahead,synthetic}` and
use the markers `lookahead`, `synthetic`, `statistical`, `slow`. The statistical
and synthetic suites are the platform's canaries: a strategy extracting multiple
standard deviations from a random walk fails them.

## Documentation

| Document | Covers |
| --- | --- |
| `docs/architecture.md` | Module map and design decisions |
| `docs/backtesting.md` | Event model, orders, accounting, timing rules |
| `docs/methodology.md` | The falsification methodology |
| `docs/data-integrity.md` | Point-in-time model, audit, quality, fingerprints |
| `docs/execution-model.md` | Cost waterfall and capacity estimation |
| `docs/statistical-tests.md` | PSR, DSR, CSCV/PBO, bootstrap, Monte Carlo |
| `docs/reproducibility.md` | Ledger, environment pinning, re-running runs |
| `docs/strategy-graveyard.md` | Failure taxonomy and post-mortems |
| `docs/initial-issues.md` | Known issues tracker |
| `datasets/README.md` | Dataset provenance rules |
| `benchmarks/README.md` | Benchmark methodology |

## Contributing

Please read `CONTRIBUTING.md` before opening a PR. In short: never help hide
negative results (failed experiments are permanent evidence), put known-answer
tests on every mathematical change, keep the core dependency-light, and keep
`make lint` + `make typecheck` green. Behaviour is governed by the text in
`CODE_OF_CONDUCT.md`.

Security issues: report privately — see `SECURITY.md`.

## Status

Alpha. Phases (data integrity, backtesting, experiment registry, statistical
validation, execution modelling, adversarial testing, regimes, volatility,
sizing, sentiment timing, graveyard, CLI, API) are implemented. See
`CHANGELOG.md`.

## Citation

If you use Quant Lab in research, please cite it as described in `CITATION.cff`.

## License

MIT. See `LICENSE`.