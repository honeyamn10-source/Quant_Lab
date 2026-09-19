# Initial GitHub Issues (QL-001 .. QL-037)

Create these issues at https://github.com/honeyamn10-source/Quant_Lab/issues and
track them roughly in order. Each maps to a module that already exists in this
repository (see `docs/architecture.md`); issues track hardening/review work.

| # | Title | Covered by |
|---|-------|-----------|
| QL-001 | Canonical market-data schema | `quantlab/data/schemas.py` |
| QL-002 | Point-in-time information model | `quantlab/data/barstream.py` |
| QL-003 | Data integrity scanner | `quantlab/data/validation.py` |
| QL-004 | Dataset fingerprinting | `quantlab/data/fingerprint.py` |
| QL-005 | Event-driven backtest kernel | `quantlab/backtest/engine.py` |
| QL-006 | Portfolio accounting | `quantlab/backtest/accounting.py` |
| QL-007 | Order/fill simulator | `quantlab/backtest/fills.py` |
| QL-008 | Same-bar look-ahead detector | `quantlab/backtest/engine.py` + `tests/lookahead` |
| QL-009 | Experiment registry | `quantlab/experiments/registry.py` |
| QL-010 | Reproducible run configuration | `quantlab/experiments/reproducibility.py` |
| QL-011 | Metrics engine | `quantlab/validation/metrics.py` |
| QL-012 | Walk-forward validator | `quantlab/validation/walk_forward.py` |
| QL-013 | Purged cross-validation | `quantlab/validation/purged_cv.py` |
| QL-014 | Embargo implementation | `quantlab/validation/embargo.py` |
| QL-015 | Probabilistic Sharpe Ratio | `quantlab/validation/psr.py` |
| QL-016 | Deflated Sharpe Ratio | `quantlab/validation/dsr.py` |
| QL-017 | CSCV | `quantlab/validation/cscv.py` |
| QL-018 | Probability of Backtest Overfitting | `quantlab/validation/cscv.py` |
| QL-019 | Bootstrap framework | `quantlab/validation/bootstrap.py` |
| QL-020 | Monte Carlo engine | `quantlab/validation/monte_carlo.py` |
| QL-021 | Parameter perturbation engine | `quantlab/adversarial/perturbation.py` |
| QL-022 | Slippage model | `quantlab/execution/cost_model.py` |
| QL-023 | Spread model | `quantlab/execution/cost_model.py` |
| QL-024 | Market-impact model | `quantlab/execution/cost_model.py` |
| QL-025 | Partial-fill simulator | `quantlab/backtest/fills.py` |
| QL-026 | Regime feature engine | `quantlab/regimes/features.py` |
| QL-027 | Regime ensemble | `quantlab/regimes/models.py` |
| QL-028 | Volatility benchmark suite | `quantlab/volatility/evaluate.py` |
| QL-029 | Robust position sizing | `quantlab/sizing/robust.py` |
| QL-030 | Crisis correlation analysis | `quantlab/risk/stress.py` |
| QL-031 | Sentiment timestamp guard | `quantlab/sentiment/timestamps.py` |
| QL-032 | Event-study engine | `quantlab/sentiment/event_study.py` |
| QL-033 | Strategy Graveyard schema | `quantlab/graveyard/` |
| QL-034 | Automatic postmortem | `quantlab/graveyard/postmortem.py` |
| QL-035 | HTML research report | `quantlab/reports/html.py` |
| QL-036 | FastAPI research API | `quantlab/api/main.py` |
| QL-037 | Dashboard | `quantlab/api/main.py` `/overview` |

## Issue body template (used for each)

```text
## Goal
(pointer to the named module under docs/architecture.md)

## Acceptance criteria
- [ ] Unit/known-answer tests shipped
- [ ] Reproducibility handled (seed + dataset hash)
- [ ] Docs updated (docs/architecture.md or the relevant doc)

## Out of scope
No hidden changes to test periods. No deletion of failed trials.
```