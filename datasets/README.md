# datasets/

Raw and research datasets live here. They are deliberately kept **out of git**
(see `.gitignore`) because market data can be large and licenses vary.

## Regenerating the synthetic source of truth

```bash
quantlab data generate-synthetic --out datasets --process random_walk --seed 7 --symbols SYN
```

Each generated file carries `source: "synthetic"` and byte-stable timestamps for
a given seed, so `dataset_hash` in the experiment ledger is reproducible.

## Expected format

A single-symbol series file is JSON:

```json
{
  "symbol": "SYN",
  "interval_seconds": 86400,
  "bars": [
    {"ts": 0, "open": 100.0, "high": 101.2, "low": 99.1, "close": 100.5,
     "volume": 83.1, "available_time": 0}
  ]
}
```

## Provenance rules

- Raw files are immutable: never overwrite; write `*_v2.json` instead.
- Every dataset consumed by an experiment must be fingerprinted
  (`quantlab.data.fingerprint.dataset_fingerprint`) and that hash recorded in
  the experiment ledger.