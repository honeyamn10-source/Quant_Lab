# Execution Model

## Cost equation

```text
Execution Cost =
  Commission
+ Spread
+ Slippage
+ Market Impact
+ Financing
+ Borrow Cost
```

Inputs:

```text
order size | ADV | spread | volatility | participation rate
time of day | liquidity | order type | latency
```

## Profiles

Four built-in profiles are applied to spread / slippage / impact assumptions:

| Profile      | Purpose                       |
| ------------ | ----------------------------- |
| `optimistic` | lower bound reference only    |
| `normal`     | default conservative baseline |
| `stress`     | deteriorating market          |
| `extreme`    | crisis / illiquid conditions  |

The optimistic model is **never** the default production assessment.

## Alpha erosion waterfall

```text
Gross Alpha → Spread → Commission → Slippage → Impact → Financing → Net Alpha
```

A strategy whose apparent alpha is an execution-model artifact is exposed by
running the waterfall under `normal`/`stress`/`extreme`.

## Market impact

A square-root participation model is used (`impact ∝ size/ADV ** 0.5`), with
volatility scaling. Capacity is estimated as the order size at which net alpha
crosses zero for a given profile.

## Latency

Execution timestamps are shifted by a configurable latency. Order eligibility in
the backtest engine encodes this shift.