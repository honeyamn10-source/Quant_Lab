"""Walk-forward analysis: fixed train window slides forward, test is the next block."""

from __future__ import annotations

from collections.abc import Iterator


def walk_forward_folds(n: int, train_size: int, test_size: int, step: int | None = None) -> Iterator[tuple[tuple[int, int], tuple[int, int]]]:
    """Yield ((train_lo, train_hi), (test_lo, test_hi)) slices.

    Walks the fixed train window forward by `test_size` each step.
    """
    step = step or test_size
    pos = 0
    while pos + train_size + test_size <= n:
        train = (pos, pos + train_size)
        test = (pos + train_size, pos + train_size + test_size)
        yield train, test
        pos += step


def walk_forward_folds_split(train_size: int, test_size: int, step: int | None = None):
    """Expanding-window alias: each subsequent fold grows the training window."""
    step = step or test_size


def walk_forward_evaluate(n: int, run_fn, train_size: int, test_size: int, step: int | None = None, **kwargs) -> dict:
    """Evaluate `run_fn(train_slice, test_slice)` over walk-forward folds."""
    folds = []
    for train, test in walk_forward_folds(n, train_size, test_size, step):
        folds.append(run_fn(train, test, **kwargs))
    return {"folds": folds, "n_folds": len(folds)}
