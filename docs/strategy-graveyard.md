# Strategy Graveyard

The signature Quant Lab feature: **failed strategies get a permanent post-mortem.**

A graveyard entry is generated automatically from the failure taxonomy plus the
collected evidence. Entries are never deleted.

## Failure taxonomy

```text
OVERFIT              performance collapses out-of-sample / under perturbation
LOOKAHEAD            strategy used information that was not available at decision time
SURVIVORSHIP         universe selection favored survivors
SLIPPAGE             real fills are far worse than modeled
MARKET_IMPACT        order size moves price beyond edge
REGIME_DEPENDENT     alpha exists in one regime only
PARAMETER_UNSTABLE   tiny parameter changes destroy performance
DATA_ERROR           input data was corrupt
CORRELATION_FAILURE  diversification claims vanished under stress
TAIL_RISK            acceptable average, ruinous tails
INSUFFICIENT_SAMPLE  not enough evidence to judge
ALPHA_DECAY          edge was consumed, typically quickly
NO_NET_EDGE          transaction costs exceeded gross alpha
```

## Example entry

```text
STRATEGY-00482
Hypothesis:       Short-term equity mean reversion.
Raw Sharpe:       2.07
Realistic-cost Sharpe: 0.43
Main failure:     Transaction costs.
Secondary failure: Parameter instability.
PBO:              Elevated.
Walk-forward:     Failed.
Status:           REJECTED.
```

## Value

Over time the graveyard becomes a public **negative-results dataset** — which
is the only way a field learns which hypotheses to stop re-testing.