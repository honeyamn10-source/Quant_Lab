# Data Integrity

Built before sophisticated strategy research, because nothing downstream can be
trusted if the input frame is wrong.

## The point-in-time observation

Every observation preserves:

```text
symbol | exchange | event_time | available_time | ingestion_time
open | high | low | close | volume
adjustment_factor | source | source_version | quality_flags
```

### Critical rule

A strategy can only access information whose:

```text
available_time <= decision_time
```

This single rule eliminates an entire class of accidental look-ahead bias.

## Storage pipeline

```text
RAW → VALIDATED → NORMALIZED → POINT-IN-TIME → RESEARCH DATASET
```

- Raw files are never overwritten.
- Every transformation is logged with `source_version`.
- Output datasets are fingerprinted (hash), so `dataset_hash` in experiments is
  traceable to bytes.

## Checks performed

- missing observations
- duplicate timestamps
- impossible OHLC values (high < max(open, close), low > min(open, close), negatives)
- zero / negative prices
- split anomalies (close → adjusted close ratio jumps)
- dividend adjustments (non-trade price jumps on ex-dates)
- ticker changes / delistings / missing securities
- timezone errors / stale prices
- suspicious spikes (returns beyond a threshold)
- future information (timestamps out of order or negative latencies)
- monotonic time violations

### Outputs

A `DataQualityScore` (0–100) plus graded risk dimensions:

```text
Survivorship Risk | Look-Ahead Risk | Corporate Action Risk
Missing Data Risk | Source Reliability
```

## Fingerprinting

Datasets are identified by content hash plus dimension metadata. Given
`dataset_hash` in an experiment, the exact rows used can be regenerated or
located. `quantlab reproduce EXP-...` verifies the hash before re-running.