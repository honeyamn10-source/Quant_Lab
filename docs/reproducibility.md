# Reproducibility

Every experiment result must be regenerable. The ledger records:

```text
experiment_id | strategy_id | strategy_version | git_commit
dataset_hash | dataset_version | parameters | features | universe
training_period | validation_period | testing_period | seed
execution_model | cost_model | optimization_method | number_of_trials
results | failure_reason
```

## Determinism guarantees

- Same random seed ⇒ same RNG stream ⇒ identical backtests.
- Same dataset hash ⇒ identical rows.
- CLI commands re-run the identical configuration with `--seed`.

## Reproducing a run

```bash
quantlab reproduce EXP-0001
```

Looks up the ledger, verifies the dataset hash still matches the expected
fingerprint, and re-executes the pinned run configuration. Any divergence in
hash, git commit, Python version or dependency set is reported, not hidden.

## Environment pinning

- `python` version recorded at run time.
- Dependencies pinned through the install tooling (`pip-tools`/lockfiles
  recommended in `Makefile`).
- The research output JSON embeds the environment block and the exact command
  that produced it.