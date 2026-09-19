"""Embargo: keep a gap after the test fold so overlapping labels from serial
correlation cannot leak back into the training set."""

from __future__ import annotations

import numpy as np


def embargo_mask(n: int, test_start: int, test_end: int, embargo: int) -> np.ndarray:
    """Boolean mask of samples removed from training due to embargo.

    Samples in ``[test_end, test_end + embargo)`` are dropped because their
    forward-return labels overlap the test window.
    """
    lo = min(n, max(0, test_end))
    hi = min(n, test_end + max(0, int(embargo)))
    mask = np.ones(n, dtype=bool)
    if hi > lo:
        mask[lo:hi] = False
    return mask


def apply_embargo(train_idx: np.ndarray, test_start: int, test_end: int, embargo: int, n: int) -> np.ndarray:
    mask = embargo_mask(n, test_start, test_end, embargo)
    return train_idx[mask[train_idx]]
