# Statistical Tests

This document catalogs the validation battery. Each test is an independent
module so evidence can be displayed *individually* — never fused into a single
"quality" number.

## Performance metrics (`metrics.py`)

Sharpe, Sortino, Calmar, Omega, CAGR, volatility, maximum drawdown, turnover,
hit rate, profit factor. All operate on a returns array and return `floats`.

## Advanced validation (`validation/`)

- `psr.py` — **Probabilistic Sharpe Ratio** (Bailey & López de Prado): P(Sharpe >
  benchmark) under observed skew/kurtosis.
- `dsr.py` — **Deflated Sharpe Ratio**: corrects the observed Sharpe for the
  number of independent trials `N` (selection bias / overfitting).
- `cscv.py` — **Combinatorially Symmetric Cross-Validation**: partition returns
  into `S` submatrices; for each of the `C(S, S/2)` combinations, rank strategies
  in-sample and measure the logit of out-of-sample ranks.
- `pbo.py` — **Probability of Backtest Overfitting**: share of combinations
  whose out-of-sample rank is below the median (i.e., where the in-sample
  preference was inverted).
- `bootstrap.py` — IID, block, and stationary bootstrap CIs for Sharpe / tail
  metrics.
- `monte_carlo.py` — reshuffle and null-model simulations; a strategy that
  "outperforms" reshuffled signals signals a broken pipeline.
- `purged_cv.py` + `embargo.py` — folds that purge overlapping labels and
  embargo a gap after the test fold to prevent serial-correlation leakage.
- `walk_forward.py` — a fixed rolling train window against the next test window.

## Ground rules

- PSR/DSR/CSCV/PBO must always be reported together; each attacks a different
  failure mode.
- No universal "Strategy Quality = 93%" number is ever produced.
- Every statistical routine is itself tested (see `tests/statistical/`),
  including a known-answer test for PSR on generated Gaussian returns.