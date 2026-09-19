# Backtesting

## Event model

The engine processes a single loop over time, dispatching:

```text
MarketEvent → SignalEvent → OrderEvent → FillEvent → PortfolioEvent
```

All five event types are plain, serializable objects carrying timestamps.

## Orders

- market orders (immediate, next-bar or same-bar-non-leak per configuration)
- limit orders
- stop orders
- partial fills (fill model may split at queue position)
- rejected orders (insufficient cash / limits / hard rules) — never silently
  ignored; each rejection is an event and a record.

## Portfolio accounting

- cash accounting
- leverage and long/short positions
- commissions
- borrowing costs / financing
- dividends and corporate actions
- position limits

## Timing rule (the critical one)

A signal generated using bar `t` cannot execute using unavailable information
from that same bar:

```text
Bar closes → Signal computed → Order submitted → Earliest eligible execution
→ Next valid execution event
```

By default, execution happens on the bar *after* the signal. Configurable
`execution_delay` supports `0` (next bar open), `+1`, `+2`, etc.

## Look-ahead defenses

1. **Input guard:** strategies only receive data through a `BarStream` that
   hides future values.
2. **Structural timing:** orders generated at bar `t` are queued and eligible
   only from bar `t + 1 + delay`.
3. **Audit tests:** deliberately contaminated strategies are constructed in
   `tests/lookahead/` and the engine is asserted to catch them.

## Determinism

Same seed ⇒ same RNG stream ⇒ identical equity curve. Seeds are recorded in the
experiment ledger (see `experiments`).