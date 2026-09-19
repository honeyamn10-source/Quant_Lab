# Methodology

## The falsification loop

Every feature follows:

1. **Hypothesis** — a falsifiable statement about market behavior.
2. **Evidence** — measured on point-in-time data, out-of-sample whenever possible.
3. **Attempt to falsify** — adversarial perturbation, purged CV, bootstrap,
   synthetic markets.
4. **Measure uncertainty** — confidence intervals, DSR, PBO.
5. **Document failures** — permanent graveyard entry with a failure taxonomy.
6. **Reproduce** — a pinned run that can be re-executed byte-for-byte.

## Metrics that matter

Raw performance metrics (`Sharpe`, `Sortino`, `Calmar`, `CAGR`, ...) are
reported — and then attacked. The headline number a researcher should look at
is a *battery of falsification results*, never a single score. Quant Lab never
combines evidence into one fake universal number like `Strategy Quality = 93%`.

## Statistical framework

- **Probabilistic Sharpe Ratio (PSR):** probability that the true Sharpe exceeds
  a benchmark given the observed non-normal returns.
- **Deflated Sharpe Ratio (DSR):** corrects Sharpe evaluation for selection bias
  and backtest overfitting across the number of trials.
- **CSCV / Probability of Backtest Overfitting (PBO):** estimate, from S×N
  combinations, how often an out-of-sample-chosen strategy would have been
  selected in-sample.
- **Purged CV + embargo:** removes information overlap between train and
  validation folds caused by serial correlation of labels.

## Execution reality

Gross alpha is eroded step by step:

```text
Gross Alpha → Spread → Commission → Slippage → Impact → Financing → Net Alpha
```

Four profiles exist: `optimistic`, `normal`, `stress`, `extreme`. The
optimistic profile is never the default production assessment.

## Regimes

No single HMM is treated as truth. Independent features (trend, realized vol,
vol-of-vol, liquidity, breadth, dispersion, correlation, drawdown, momentum,
tail stress) feed *competing* regime models (rule-based, clustering, GMM, HMM,
change-point, ensemble) whose outputs are probabilities, and strategy
performance is reported **conditional on each regime**.

## Sizing

The objective of sizing research is not maximum theoretical growth; it is
finding policies that remain acceptable **when the inputs are wrong**. Estimated
edge is deliberately corrupted (volatility, correlations, win probability, payoff
ratio, tail probability) and growth, drawdown, ruin probability, CVaR and
recovery time are measured.

## Data integrity

See `data-integrity.md`. The critical rule:

```
available_time <= decision_time
```

## What "survived validation" means

A strategy that passes every funnel stage is said to have *survived a specified
battery of tests*. The final line of every report remains:

> Still unknown: future profitability.