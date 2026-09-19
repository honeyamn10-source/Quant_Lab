# Architecture

## Design principles

1. **Honesty over convenience.** Every result carries its uncertainty. A
   strategy is never "proven profitable"; it has *survived a specified battery
   of tests*.
2. **Point-in-time truth.** A strategy can only access information whose
   `available_time <= decision_time`. This one rule eliminates an entire class
   of accidental look-ahead bias.
3. **Backtest engine is ours.** The core engine is written in-house and kept
   deliberately small and auditable rather than delegating execution semantics
   to a framework.
4. **Failures are evidence.** Nothing is deleted. Failed trials are permanent
   research artifacts (the *graveyard*).
5. **Determinism by default.** Same inputs + same seed ⇒ same result.

## Package map

```text
quantlab/
├── data/         Point-in-time storage, validation, quality, fingerprinting
├── synthetic/    Market generators with known ground truth
├── strategies/   Strategy API + benchmark strategies
├── backtest/     Event-driven engine: events, orders, fills, accounting, timing
├── execution/    Spread / slippage / impact / latency / profiles / capacity
├── validation/   PSR, DSR, CSCV, PBO, walk-forward, purged CV, bootstrap, MC
├── adversarial/  Parameter/timing/cost/universe perturbation, robustness surface
├── regimes/      Regime features, clustering, model comparison
├── volatility/   Historical, EWMA, HAR, realized, GARCH/EGARCH/GJR
├── sizing/       Kelly variants, vol targeting, CVaR, robust sizing
├── sentiment/    Timestamp validation, temporal guarding, event studies
├── risk/         Correlation, tail dependence, drawdown, stress scenarios
├── graveyard/    Failure taxonomy, post-mortems, negative-results store
├── experiments/  Registry, run configs, reproducibility
├── reports/      Metrics, HTML/JSON report generation
├── cli.py        `quantlab ...` command line
└── api/          FastAPI research API + dashboard
```

## Data flow

```text
RAW → VALIDATED → NORMALIZED → POINT-IN-TIME → RESEARCH DATASET
```

Raw files are immutable. Every transformation is recorded with its
`source_version` and produces a dataset fingerprint, so a research result can
always be traced to a byte-exact input.

## The promotion pipeline

No strategy goes from backtest to live trading directly:

```text
IDEA → BASELINE → DATA AUDIT → BACKTEST → OVERFIT TESTS → PERTURBATION
→ EXECUTION STRESS → REGIME TEST → CRISIS TEST → OUT-OF-SAMPLE →
PAPER TRADING → SHADOW LIVE → LIMITED CAPITAL
```

Every stage returns `PASS` / `FAIL` / `INCONCLUSIVE`.

## Dependency strategy

- **Core (required):** `numpy`, `pydantic`, `PyYAML` — the research core runs
  almost anywhere.
- **Optional extras:** `arch`, `scipy`, `scikit-learn`, `statsmodels`, `optuna`
  (`[ml]`); `mlflow` (`[tracking]`); `pandas`/`polars`/`duckdb` (`[storage]`);
  `fastapi`/`uvicorn`/`plotly` (`[web]`).

Heavy frameworks are adapters, never the core.