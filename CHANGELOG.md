# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Repository foundation: `pyproject.toml`, packaging (`quantlab` console entry),
  `Makefile`, `Dockerfile`/`docker-compose.yml`, CI workflow, docs set, license,
  code of conduct, security policy, citation metadata.
- **Data integrity (Phase 1):** canonical point-in-time schema, bar stream model,
  data validator (OHLCV rules, duplicates, monotonic timestamps, spikes,
  corporate-action heuristics), quality scoring (0–100 + risk dimensions),
  dataset fingerprinting.
- **Synthetic markets:** generators with known ground truth (random walk, trend,
  mean reversion, volatility clustering, regime switch, structural break,
  jump process, pairs).
- **Strategies:** strategy API (`BarStream`-gated, no future access) and
  benchmark strategies (buy & hold, random, SMA crossover, momentum,
  mean reversion, breakout, volatility targeting, pairs).
- **Backtester (Phase 2):** event-driven engine (Market/Signal/Order/Fill/
  Portfolio events), market/limit/stop orders, partial fills, rejections,
  position limits, commissions, financing, dividends, timing rules with
  configurable execution delay, and look-ahead audit tests.
- **Experiment ledger (Phase 3):** registry, strategy configs (JSON), run
  configuration with reproducibility envelope (git commit, python version,
  dataset hash, seed).
- **Statistical validation (Phase 4):** metrics engine, PSR, DSR, CSCV, PBO,
  walk-forward, purged CV with embargo, IID/block/stationary bootstrap,
  Monte Carlo reshuffles.
- **Execution reality (Phase 5):** spread/slippage/market-impact/latency
  cost model, four profiles (optimistic/normal/stress/extreme), capacity
  estimation.
- **Adversarial lab (Phase 6):** parameter/timing/cost/universe perturbation,
  robustness surface computation.
- **Regimes (Phase 7):** regime feature extraction, rule-based clustering,
  Gaussian mixture, HMM, and ensemble regimes; volatility laboratory
  (historical/EWMA/HAR/realized, GARCH/EGARCH/GJR), correlation and crisis
  stress engine.
- **Sizing (Phase 8):** fixed fraction, vol targeting, Kelly / fractional Kelly,
  CVaR-constrained, robust sizing with corrupted-edge evaluation.
- **Sentiment/event timing (Phase 9):** timestamp model (event/publication/
  retrieval/decision), leakage guard, event-study engine.
- **Strategy Graveyard (Phase 10):** failure taxonomy, recorder, automatic
  post-mortems.
- **CLI + reports (Phase 11):** `quantlab` command surface, HTML/JSON report
  generation with equity-curve rendering; FastAPI research API + dashboard stub.

### Fixed

- HTML report rendering from JSON results (API and CLI): results stored as plain
  dicts now satisfy the renderer's contract via `ReportResult`
  (`quantlab/reports/html.py`).
- `quantlab volatility`: iterate the flat model-result dict instead of a
  nonexistent `models` key.
- `quantlab sizing --simulate`: runs a concrete Monte-Carlo quick check instead
  of being a silent no-op.
- CI typecheck job installs the full dev extras (mypy needs PyYAML type stubs);
  lint now covers `benchmarks/`.
- `build>=1.0` added to dev extras so the Makefile `build` target works.
- `scipy` added to dev extras and GARCH tests skip cleanly without it: the dev
  environment now covers the full test suite (CI was failing on the two GARCH
  tests because scipy lives only in the optional `ml` extra).
- `quantlab data generate-synthetic --process pairs` now writes the cointegrated
  pair instead of failing on an unknown process; CLI surface tests added.
- Makefile `test`/`test-all` run the full suite; `test-lookahead` uses the
  `lookahead` marker; lint/format cover `benchmarks/`.
- Test suite markers (`lookahead`, `synthetic`, `statistical`) now attached so
  `--strict-markers` and per-area selection work.
- CI collection crash (`ModuleNotFoundError: No module named 'quantlab.data'`):
  `.gitignore` patterns `data/` and `experiments/` shadowed the `quantlab.data`
  and `quantlab.experiments` source packages, so they were never committed and
  were missing from the wheel. Patterns are now root-anchored and the two
  packages (plus their ruff cleanups) are tracked.