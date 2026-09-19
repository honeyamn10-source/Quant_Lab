<div align="center">

# Quant Lab

**An open quantitative research & strategy-falsification platform.**

> Don't prove your strategy works. Try to prove it doesn't.

[![CI](https://img.shields.io/github/actions/workflow/status/honeyamn10-source/Quant_Lab/ci.yml?branch=main&label=CI)](https://github.com/honeyamn10-source/Quant_Lab/actions)
[![License: MIT](https://img.shields.io/github/license/honeyamn10-source/Quant_Lab)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB)](https://www.python.org/)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20Windows-lightgrey)

</div>

---

## Table of contents

- [What is Quant Lab?](#what-is-quant-lab)
- [Why another backtesting platform?](#why-another-backtesting-platform)
- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Command line reference](#command-line-reference)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Development](#development)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Security](#security)
- [Versioning & status](#versioning--status)
- [License](#license)

---

## What is Quant Lab?

Quant Lab is a research platform built around a single, unfashionable idea:
**strategies are guilty until proven otherwise.** Instead of searching thousands
of configurations and calling the prettiest chart "alpha", Quant Lab runs each
hypothesis through an honest pipeline of data audits, look-ahead detection,
statistical falsification (PSR, DSR, PBO, walk-forward, bootstrap, Monte Carlo),
realistic execution costs and adversarial perturbation — and it **keeps the
failures** as permanent evidence in an immutable experiment ledger.

Every experiment answers the question *"How much evidence exists that this
result is real?"*, not *"How pretty is this equity curve?"*.

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

## Why another backtesting platform?

Most backtesters optimise for finding good-looking results. Quant Lab optimises
for **trusting** a result, which forces it to solve problems others skip:

- backtest overfitting and selection bias,
- data leakage between decision time and available time,
- slippage, market impact and execution latency assumptions,
- regime dependency and crisis breakdowns,
- survivorship and data-quality bias,
- and, most importantly, the systematic **lack of failed-strategy data** — the
  thing that makes every "successful" strategy look better than it is.

## Features

- **Synthetic markets with known ground truth** — random walk, trend, mean
  reversion, volatility clustering, regime switch, structural break, jump and
  cointegrated pairs. If the pipeline finds "alpha" in a seeded random walk, the
  pipeline is broken, not inspired.
- **Point-in-time data model** — bars carry an `available_time`; strategies only
  ever see a guarded `BarStream`, so future data is inaccessible by construction.
- **Event-driven backtester** — market/limit/stop orders, partial fills,
  rejections, position limits, commissions, financing, dividends and configurable
  execution delay.
- **Statistical falsification** — Probabilistic Sharpe Ratio, Deflated Sharpe
  Ratio, CSCV / Probability of Backtest Overfitting, purged CV with embargo,
  walk-forward, IID/block/stationary bootstrap and Monte Carlo reshuffles.
- **Execution reality** — spread/slippage/market-impact/latency cost model with
  four profiles (`optimistic`, `normal`, `stress`, `extreme`) and capacity
  estimation.
- **Adversarial lab** — parameter, timing, cost and universe perturbation with a
  single robustness rating.
- **Regimes & volatility** — rule-based/GMM/HMM/ensemble regime models; historical,
  EWMA, HAR, realized, GARCH/EGARCH/GJR volatility.
- **Sizing** — fixed fraction, vol targeting, Kelly / fractional Kelly, CVaR and
  robust sizing with corrupted-edge evaluation.
- **Sentiment / event timing** — timestamp model with leakage guards.
- **Experiment ledger & strategy graveyard** — every run is recorded immutably,
  including the failures. Negative results are evidence, never delete.
- **CLI, reports & API** — reproducible `quantlab` commands, self-contained HTML
  validation reports, and a read-only FastAPI research dashboard.

## Installation

Requires **Python 3.12+**. The core is deliberately dependency-light
(`numpy`, `pydantic`, `PyYAML`); heavier frameworks are optional extras.

```bash
# core only
pip install "."

# full development environment
pip install -e ".[dev]"

# optional feature groups
pip install -e ".[ml]"        # scipy, scikit-learn, statsmodels, arch, optuna
pip install -e ".[tracking]"  # mlflow
pip install -e ".[storage]"   # pandas, polars, duckdb, pyarrow
pip install -e ".[web]"       # fastapi, uvicorn, jinja2, plotly
```

Verify the install:

```bash
quantlab --version
quantlab --strategies
```

## Quick start

```bash
# 1. Generate a synthetic market with known ground truth
quantlab data generate-synthetic --out datasets/ --seed 7 --symbols SYN

# 2. Audit and score it
quantlab data audit --file datasets/SYN.json
quantlab data quality --file datasets/SYN.json

# 3. Run a benchmark strategy
quantlab backtest examples/strategies/sma_crossover.yaml

# 4. Run a full experiment: backtest + validation battery + ledger + report
quantlab run-experiment examples/strategies/sma_crossover.yaml

# 5. Reproduce it later, from the ledger alone
quantlab reproduce EXP-0001
```

A worked example of the full pipeline lives in `examples/`; the bundled strategy
configs (`sma_crossover.yaml`, `momentum.yaml`, `mean_reversion.yaml`,
`pairs.yaml`, `random.yaml`) each carry a ground-truth note explaining what an
honest result on that market actually looks like.

## Command line reference

Every command is deterministic: the same seed, the same data, the same result.

| Command | Purpose |
| --- | --- |
| `data generate-synthetic [--out DIR] [--process NAME] [--seed N] [--symbols A,B]` | Generate a synthetic market series |
| `data audit --file FILE` | Point-in-time audit (OHLCV rules, monotonicity, spikes) |
| `data quality --file FILE [--survivorship HINT]` | Quality score 0–100 with risk dimensions |
| `backtest CONFIG.yaml` | Run a config and print metrics |
| `run-experiment CONFIG.yaml [--report-dir DIR] [--ledger DIR]` | Backtest + validation + ledger + report |
| `validate RESULTS.json` | Run the validation battery on a results file |
| `stress EXPERIMENT_ID` | Adversarial perturbation + robustness rating |
| `regimes FILE.json [--seed N]` | Regime decomposition of a series |
| `volatility FILE.json` | Benchmark volatility models on a series |
| `report EXPERIMENT_ID` | Write the self-contained HTML validation report |
| `graveyard list` | List recorded failed experiments |
| `reproduce EXPERIMENT_ID [--ledger DIR]` | Re-run an experiment from its ledger record |
| `sizing [--simulate]` | Position-sizing quick checks |

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

Available strategies: `breakout`, `buy_and_hold`, `mean_reversion`,
`mean_reversion_rsi`, `momentum`, `pairs`, `random`, `sma_crossover`,
`vol_target`.

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

## Development

```bash
pip install -e ".[dev]"

make lint            # ruff
make format          # ruff check --fix + ruff format
make typecheck       # mypy
make test            # full pytest suite
make test-lookahead  # pytest -m lookahead
make cov             # coverage report
make build           # sdist + wheel
```

Tests live under `tests/{unit,integration,statistical,lookahead,synthetic}` and
use the markers `lookahead`, `synthetic`, `statistical`, `slow`. The statistical
and synthetic suites are the platform's canaries: a strategy extracting
multiple standard-deviation Sharpe ratios from a random walk fails them.

CI (GitHub Actions, see `.github/workflows/ci.yml`) runs lint, typecheck and the
full test suite on Python 3.12 and 3.13.

## Documentation

| Document | Covers |
| --- | --- |
| [`docs/architecture.md`](docs/architecture.md) | Module map and design decisions |
| [`docs/backtesting.md`](docs/backtesting.md) | Event model, orders, accounting, timing rules |
| [`docs/methodology.md`](docs/methodology.md) | The falsification methodology |
| [`docs/data-integrity.md`](docs/data-integrity.md) | Point-in-time model, audit, quality, fingerprints |
| [`docs/execution-model.md`](docs/execution-model.md) | Cost waterfall and capacity estimation |
| [`docs/statistical-tests.md`](docs/statistical-tests.md) | PSR, DSR, CSCV/PBO, bootstrap, Monte Carlo |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Ledger, environment pinning, re-running runs |
| [`docs/strategy-graveyard.md`](docs/strategy-graveyard.md) | Failure taxonomy and post-mortems |
| [`docs/initial-issues.md`](docs/initial-issues.md) | Known issues tracker |
| [`datasets/README.md`](datasets/README.md) | Dataset provenance rules |
| [`benchmarks/README.md`](benchmarks/README.md) | Benchmark methodology |

## Contributing

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a PR. In short:

1. No results-deletion PRs — failed experiments and graveyard entries are
   permanent evidence.
2. Tests first for numerics — every mathematical change ships a known-answer test.
3. Reproducibility is non-negotiable — new strategies must be fully specified by
   a config file + seed.
4. Keep `make lint` and `make typecheck` green.
5. Keep the core dependency-light; heavy frameworks belong behind optional
   adapters.

Behaviour is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Security

The research pipeline runs on **local data only** and requires no secrets. The
FastAPI dashboard is intended for **loopback / trusted networks** — configure
authentication and host binding before exposing it publicly. Datasets are
user-provided, so treat external data as untrusted input. Report vulnerabilities
**privately** — see [`SECURITY.md`](SECURITY.md).

## Versioning & status

**Status:** Alpha. Phases (data integrity, backtesting, experiment registry,
statistical validation, execution modelling, adversarial testing, regimes,
volatility, sizing, sentiment timing, graveyard, CLI, API) are implemented and
covered by tests.

Changes follow [Keep a Changelog](https://keepachangelog.com/) — see
[`CHANGELOG.md`](CHANGELOG.md). Versioning follows
[Semantic Versioning](https://semver.org/). If you use Quant Lab in research,
please cite it as described in [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).