"""Purged K-fold cross-validation with label overlap control.

Labels are forward returns over ``horizon`` bars. A training label starting
within ``horizon`` of the test window end overlaps test labels and must be
purged; an embargo gap then removes any residual serial correlation.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def _purged_test_folds(n: int, n_splits: int) -> list[tuple[int, int]]:
    """Even contiguous test blocks."""
    folds: list[tuple[int, int]] = []
    split = int(np.ceil(n / n_splits))
    for k in range(n_splits):
        lo, hi = k * split, min(n, (k + 1) * split)
        if hi > lo:
            folds.append((lo, hi))
    return folds


def purged_cv_indices(n: int, n_splits: int = 5, horizon: int = 5, embargo: int = 5) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) per fold.

    - test fold: contiguous block ``[lo, hi)``.
    - training: everything outside, minus purged samples whose label overlaps
      the test window, minus the embargo gap after the test window.
    """
    all_idx = np.arange(n)
    for lo, hi in _purged_test_folds(n, n_splits):
        train = np.concatenate([all_idx[:lo], all_idx[hi:]]).astype(int)
        # purge: training samples that START within horizon of the test end leak
        purge_mask = np.ones(len(train), dtype=bool)
        for t, x in enumerate(train):
            if x >= lo - horizon and x <= hi - 1:
                purge_mask[t] = False
        train = train[purge_mask]
        # embargo after test fold
        from quantlab.validation.embargo import embargo_mask

        emask = embargo_mask(n, hi, hi, embargo)
        train = train[emask[train]]
        if len(train) == 0:
            continue
        yield train, all_idx[lo:hi]


def purged_cv_evaluate(n: int, run_fn, n_splits: int = 5, horizon: int = 5, embargo: int = 5, **kwargs) -> dict:
    """Run `run_fn(train_idx, test_idx)` per fold and aggregate metrics."""
    fold_results = []
    for train_idx, test_idx in purged_cv_indices(n, n_splits, horizon, embargo):
        fold_results.append(run_fn(train_idx, test_idx, **kwargs))
    return {"folds": fold_results, "n_folds": len(fold_results)}
